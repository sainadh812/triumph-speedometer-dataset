"""V5 mixed-view training. Optimization/evaluation run only on a Colab GPU.

The unchanged V4 12-landmark graph transfers all heads. V5 has a new dataset,
input contract, optimizer and checkpoint family. Geometry uses predictions,
never the sample's rendering transform or target landmarks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np
import PIL
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
import torchvision

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from real_adaptation_v4 import train_landmarks as v4
from real_adaptation_v4.geometry import load_calibration, read_speed
from real_adaptation_v3 import train_needle as durable

LandmarkModel = v4.LandmarkModel
LandmarkDataset = v4.LandmarkDataset
training_loss = v4.training_loss
KEYPOINT_NAMES = v4.KEYPOINT_NAMES
ARCHITECTURE_ID = v4.ARCHITECTURE_ID
AVAILABILITY_SEMANTICS = v4.AVAILABILITY_SEMANTICS
PREPROCESSING_ID = 'gray512_fullframe_native_view_lcd128_mask_v5_imagenet_normalized'
CHECKPOINT_KIND = 'joint_needle_ticks_mixed_view_full_state_v5'
INFERENCE_KIND = 'joint_needle_ticks_mixed_view_inference_v5'
FORMAT_VERSION = 1
VIEWS = ('straight_v3', 'tilted_v4')
ROLLS = ('left', 'straight', 'right')
STRATA = tuple((view, roll) for view in VIEWS for roll in ROLLS)
SOURCE_V4_SHA256 = {
    'ed7f27c44e89f4caebefddc79fb1481786efc2e6fc3ddd103f7bcb3b2625857d',
    'da52003cc008f92327b563b5f4bd334b737b14e196db3b59d0d08805c0a9bf51',
}
_sha, _json, _cpu = durable._sha, durable._json, durable._cpu


@dataclass
class TrainConfig(v4.TrainConfig):
    epochs: int = 12
    batch_size: int = 12
    warmup_epochs: int = 1
    patience: int = 4
    needle_lr: float = 1e-4
    tick_lr: float = 1e-4
    encoder_lr: float = 1e-5
    device: str = 'cuda'
    epoch_samples: int | None = None
    progress_every_batches: int = 100


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _require_colab_cuda(config):
    try:
        colab = importlib.util.find_spec('google.colab') is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        colab = False
    colab = colab or bool(os.environ.get('COLAB_RELEASE_TAG'))
    if not colab or config.device not in ('cuda', 'auto') or not torch.cuda.is_available():
        raise RuntimeError('V5 training and model evaluation require a Google Colab CUDA runtime; no local fallback')
    return torch.device('cuda')


def verify_dataset(directory, *, progress=True):
    root = Path(directory).resolve()
    if progress:
        print('Verifying V5 dataset hashes, labels, masks and split provenance...', flush=True)
    annotation_path, manifest_path = root / 'annotations.json', root / 'manifest.json'
    annotations = json.loads(annotation_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    hashes = manifest.get('sha256', manifest.get('files'))
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('V5 manifest needs protected file hashes')
    expected_hashes = {name: value.get('sha256') if isinstance(value, dict) else value for name, value in hashes.items()}
    for required in ('annotations.json', 'calibration.json', 'preprocessing_config.json', 'mask_references.json'):
        if required not in hashes:
            raise ValueError('Unprotected V5 asset: ' + required)
    for index, (relative, expected) in enumerate(expected_hashes.items(), 1):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _sha(path) != expected:
            raise ValueError('Dataset checksum/path mismatch: ' + relative)
        if progress and index % 500 == 0:
            print(f'Verified {index}/{len(hashes)} dataset files', flush=True)
    preprocessing = manifest.get('preprocessing')
    if not isinstance(preprocessing, dict) or preprocessing != json.loads((root / 'preprocessing_config.json').read_text(encoding='utf-8')):
        raise ValueError('V5 manifest/preprocessing configuration disagree')
    declared_id = preprocessing.get('preprocessing_id', manifest.get('preprocessing_id'))
    if declared_id is not None and declared_id != PREPROCESSING_ID:
        raise ValueError('Wrong preprocessing family')
    expected = {'color': 'grayscale', 'input_size': [512, 512], 'lcd_mask_applied': True,
                'lcd_mask_value': 128, 'camera_stabilization': 'none',
                'input_geometry': 'full_frame_square_letterbox_then512_no_crop_no_canonical_warp',
                'mask_placement': 'image_only_multi_reference_polygon_projection'}
    if any(preprocessing.get(key) != value for key, value in expected.items()):
        raise ValueError('V5 requires native full-frame, no-crop, mask-only projection preprocessing')
    references = json.loads((root / 'mask_references.json').read_text(encoding='utf-8')).get('references', [])
    if {row.get('id') for row in references} != set(VIEWS):
        raise ValueError('Both source views need protected LCD reference assets')
    for reference in references:
        if any(reference.get(key) not in expected_hashes for key in ('reference_image', 'calibration')):
            raise ValueError('Unprotected LCD reference image or calibration')
        polygon = np.asarray(reference.get('lcd_polygon_xy'), dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
            raise ValueError('Invalid reference LCD polygon')
    if annotations.get('keypoint_names') != KEYPOINT_NAMES:
        raise ValueError('V5 landmark ordering differs')
    semantics = manifest.get('landmark_supervision', annotations.get('landmark_supervision', preprocessing.get('landmark_supervision')))
    if semantics != AVAILABILITY_SEMANTICS:
        raise ValueError('V5 requires amodal landmark availability supervision')
    partitions = {'train': [], 'validation': []}
    ids, paths = set(), set()
    groups, parents, pixel_hashes = ({split: set() for split in partitions} for _ in range(3))
    counts = Counter()
    for index, original in enumerate(annotations['samples'], 1):
        row = dict(original)
        split = row.get('split')
        stratum = (row.get('source_view'), row.get('roll_bucket'))
        if split not in partitions or stratum not in STRATA or row.get('parent_split') != split:
            raise ValueError('Invalid split/stratum or changed parent split')
        sample_id = row.get('sample_id')
        if not isinstance(sample_id, str) or sample_id in ids or not sample_id:
            raise ValueError('Duplicate/missing sample id')
        ids.add(sample_id)
        if not row.get('parent_sample_id') or not row.get('source_group'):
            raise ValueError('Missing parent/group provenance')
        groups[split].add((row['source_view'], row['source_group']))
        parents[split].add((row['source_view'], row['parent_sample_id']))
        points = np.asarray(row['points_xy'], dtype=float)
        visible = np.asarray(row['visibility'])
        if (points.shape != (12, 2) or visible.shape != (12,) or not np.isfinite(points).all()
                or not np.isin(visible, [True, False]).all()):
            raise ValueError('Invalid 12-landmark labels')
        visible = visible.astype(bool)
        if ((points[visible] < 0) | (points[visible] > 511)).any():
            raise ValueError('Available landmark outside model image')
        image_relative = row['image_path']
        mask_relative = row.get('lcd_mask_path', row.get('mask_path'))
        if image_relative in paths:
            raise ValueError('Duplicate model image path')
        paths.add(image_relative)
        for relative in (image_relative, mask_relative):
            if relative not in expected_hashes:
                raise ValueError('Unprotected image or LCD mask')
        image_path, mask_path = (root / image_relative).resolve(), (root / mask_relative).resolve()
        with Image.open(image_path) as image:
            if image.size != (512, 512) or image.mode != 'L':
                raise ValueError('V5 images must be 512x512 grayscale')
            pixels = np.asarray(image).copy()
        with Image.open(mask_path) as image:
            if image.size != (512, 512) or image.mode != 'L':
                raise ValueError('V5 masks must be 512x512 grayscale')
            mask = np.asarray(image)
        if not np.isin(mask, [0, 255]).all() or not (mask == 255).any() or (mask == 255).all():
            raise ValueError('LCD mask must be nonempty binary 0/255')
        hidden = mask == 255
        if not (pixels[hidden] == 128).all():
            raise ValueError('LCD mask pixels must remain exactly128')
        targets = np.zeros((512, 512), np.uint8)
        for xy in points[visible]:
            cv2.circle(targets, tuple(np.rint(xy).astype(int)), 2, 255, -1)
        if visible[:2].all():
            cv2.line(targets, tuple(np.rint(points[0]).astype(int)), tuple(np.rint(points[1]).astype(int)), 255, 3)
        if (hidden & (targets > 0)).any():
            raise ValueError('LCD mask overlaps available landmarks/needle')
        pixel_hashes[split].add(hashlib.sha256(pixels.tobytes()).hexdigest())
        if type(row.get('speed_valid', True)) is not bool:
            raise ValueError('speed_valid must be boolean')
        row['speed_valid'] = row.get('speed_valid', True)
        speed = float(row['speed_kmh'])
        if not math.isfinite(speed) or row['speed_valid'] and not 0 <= speed <= 180:
            raise ValueError('Invalid speed label')
        row['image'] = str(image_path)
        row['visibility'] = visible.tolist()
        partitions[split].append(row)
        counts[(split, *stratum)] += 1
        if progress and index % 500 == 0:
            print(f'Verified {index}/{len(annotations["samples"])} image/label/mask rows', flush=True)
    if groups['train'] & groups['validation'] or parents['train'] & parents['validation']:
        raise ValueError('Source-group or parent leakage across splits')
    if pixel_hashes['train'] & pixel_hashes['validation']:
        raise ValueError('Duplicate image pixels across splits')
    for split in partitions:
        for view, roll in STRATA:
            selected = [row for row in partitions[split] if (row['source_view'], row['roll_bucket']) == (view, roll)]
            if not selected or not any(row['speed_valid'] for row in selected):
                raise ValueError('Each split needs labeled speed samples in all six strata')
    load_calibration(root / 'calibration.json')
    integrity = {'manifest_sha256': _sha(manifest_path), 'annotation_sha256': _sha(annotation_path),
                 'calibration_sha256': _sha(root / 'calibration.json'), 'preprocessing': preprocessing,
                 'protected_files_digest': _digest(expected_hashes), 'protected_file_count': len(hashes),
                 'sample_counts': {split: len(rows) for split, rows in partitions.items()},
                 'stratum_counts': {split: {view + '/' + roll: counts[(split, view, roll)] for view, roll in STRATA} for split in partitions},
                 'landmark_supervision': AVAILABILITY_SEMANTICS,
                 'limitation': 'Source-group synthetic holdouts from existing sessions; not independent real-world accuracy'}
    if progress:
        print('V5 dataset verification passed: ' + json.dumps(integrity['sample_counts']), flush=True)
    return partitions, integrity


class BalancedStrataBatchSampler(Sampler):
    """Deterministic equal-stratum batches; minority pools cycle with reshuffling."""
    def __init__(self, records, batch_size, epoch_samples=None, seed=42):
        if batch_size < 6 or batch_size % 6:
            raise ValueError('Balanced batches must contain a multiple of six images')
        self.pools = {key: [] for key in STRATA}
        for index, row in enumerate(records):
            self.pools[(row['source_view'], row['roll_bucket'])].append(index)
        if any(not pool for pool in self.pools.values()):
            raise ValueError('Balanced sampler requires all six strata')
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        self.epoch_samples = int(epoch_samples if epoch_samples is not None else len(records) // batch_size * batch_size)
        if self.epoch_samples < batch_size or self.epoch_samples % batch_size:
            raise ValueError('epoch_samples must be a positive whole number of batches')
        self.order_digest = _digest([(row['sample_id'], row['source_view'], row['roll_bucket']) for row in records])

    def __len__(self):
        return self.epoch_samples // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def state_dict(self):
        return {'epoch': self.epoch, 'seed': self.seed, 'batch_size': self.batch_size,
                'epoch_samples': self.epoch_samples, 'dataset_order_digest': self.order_digest,
                'policy': 'equal six strata; shuffle within stratum, cycle smaller pools; epoch-boundary resume'}

    def load_state_dict(self, state):
        current = self.state_dict()
        if any(state.get(key) != value for key, value in current.items() if key != 'epoch'):
            raise ValueError('Balanced sampler identity mismatch')
        self.set_epoch(state['epoch'])

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + 7919 * self.epoch)
        queues = {key: [] for key in STRATA}
        for _ in range(len(self)):
            batch = []
            for key in STRATA:
                for _ in range(self.batch_size // 6):
                    if not queues[key]:
                        pool = self.pools[key]
                        queues[key] = [pool[i] for i in torch.randperm(len(pool), generator=rng).tolist()]
                    batch.append(queues[key].pop())
            order = torch.randperm(len(batch), generator=rng).tolist()
            yield [batch[i] for i in order]


def configure_stage(model, epoch, config, *, fresh=False):
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for layer in model.features:
        layer.eval()  # Freeze BatchNorm statistics as well as parameters.
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
    if epoch > config.warmup_epochs:
        for layer in (list(model.features) if fresh else list(model.features)[-4:]):
            layer.train()
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    for parameter in model.speed_head.parameters():
        parameter.requires_grad_(False)
    model.speed_head.eval()
    return 'both_decoders_warmup_frozen_encoder_bn' if epoch <= config.warmup_epochs else 'joint_late_encoder_finetune'


def load_transfer_checkpoint(model, path):
    digest = _sha(path)
    if digest not in SOURCE_V4_SHA256:
        raise ValueError('Transfer requires the verified complete V4 epoch17 weights')
    source, payload = v4.load_model(path, 'cpu')
    epoch = payload.get('best_epoch', payload.get('training_state', {}).get('epoch'))
    if epoch != 17:
        raise ValueError('V4 transfer checkpoint must be epoch17')
    model.load_state_dict(source.state_dict(), strict=True)
    return {'mode': 'transfer_v4', 'source_sha256': digest, 'source_epoch': 17,
            'source_architecture': ARCHITECTURE_ID, 'all_12_landmark_heads_preserved': True,
            'optimizer_restored': False, 'new_components': [],
            'baseline_scope': 'V4 weights evaluated through the new V5 native full-frame input contract'}


def _angle_error(prediction, target, available):
    if not all(available[:2]):
        return None
    predicted, actual = np.diff(np.asarray(prediction[:2]), axis=0)[0], np.diff(np.asarray(target[:2]), axis=0)[0]
    if min(np.linalg.norm(predicted), np.linalg.norm(actual)) <= 1e-6:
        return None
    delta = math.atan2(predicted[1], predicted[0]) - math.atan2(actual[1], actual[0])
    return abs(math.degrees(math.atan2(math.sin(delta), math.cos(delta))))


def aggregate_metrics(rows, geometry_penalty_px=20.0):
    if not rows:
        raise ValueError('Cannot score an empty validation group')
    per_point = {name: [] for name in KEYPOINT_NAMES}
    angle_errors, speed_errors, rejections = [], [], Counter()
    eligible = accepted = 0
    for row in rows:
        for name, error, valid in zip(KEYPOINT_NAMES, row['point_errors_px'], row['target_availability']):
            if valid:
                per_point[name].append(error)
        if row['needle_angle_error_input_deg'] is not None:
            angle_errors.append(row['needle_angle_error_input_deg'])
        eligible += int(row['speed_valid'])
        accepted += int(row['reading']['accepted'])
        if row['speed_error_kmh'] is not None:
            speed_errors.append(row['speed_error_kmh'])
        if not row['reading']['accepted']:
            rejections[row['reading']['reason']] += 1
    needle = per_point['pivot'] + per_point['needle_tip']
    ticks = [error for name in KEYPOINT_NAMES[2:] for error in per_point[name]]
    if not needle or not ticks or not eligible:
        raise ValueError('Every scored group requires needle/tick labels and eligible speeds')
    within = sum(error <= 2 for error in speed_errors) / eligible
    result = {'count': len(rows), 'needle_mean_error_px': float(np.mean(needle)),
              'needle_pivot_mean_error_px': float(np.mean(per_point['pivot'])) if per_point['pivot'] else None,
              'needle_tip_mean_error_px': float(np.mean(per_point['needle_tip'])) if per_point['needle_tip'] else None,
              'tick_mean_error_px': float(np.mean(ticks)), 'tick_p95_error_px': float(np.percentile(ticks, 95)),
              'per_landmark_mean_error_px': {name: float(np.mean(values)) if values else None for name, values in per_point.items()},
              'needle_angle_input_degrees': {'count': len(angle_errors), 'unavailable_or_degenerate_count': len(rows)-len(angle_errors),
                    'mean_absolute_error': float(np.mean(angle_errors)) if angle_errors else None,
                    'p95_absolute_error': float(np.percentile(angle_errors, 95)) if angle_errors else None,
                    'scope': 'circular needle-vector error in the model input plane, not rectified physical angle'},
              'primary_speed': {'accepted': accepted, 'rejected': len(rows)-accepted,
                    'coverage_percent': 100 * accepted / len(rows), 'eligible_count': eligible,
                    'within_2_kmh_all_eligible_inputs_percent': 100 * within,
                    'accepted_only': durable._summary(speed_errors), 'rejection_reasons': dict(rejections)}}
    result['selection_score'] = result['needle_mean_error_px'] + result['tick_mean_error_px'] + geometry_penalty_px * (1-within)
    return result


def grouped_metrics(rows, geometry_penalty_px=20.0):
    overall = aggregate_metrics(rows, geometry_penalty_px)
    by_view = {view: aggregate_metrics([row for row in rows if row['source_view'] == view], geometry_penalty_px) for view in VIEWS}
    by_roll = {roll: aggregate_metrics([row for row in rows if row['roll_bucket'] == roll], geometry_penalty_px) for roll in ROLLS}
    by_stratum = {view+'/'+roll: aggregate_metrics([row for row in rows if (row['source_view'], row['roll_bucket']) == (view, roll)], geometry_penalty_px)
                  for view, roll in STRATA}
    return {**overall, 'overall_pooled_selection_score': overall['selection_score'],
            'by_source_view': by_view, 'by_roll_bucket': by_roll, 'by_stratum': by_stratum,
            'selection_score': float(np.mean([group['selection_score'] for group in by_stratum.values()])),
            'worst_view_selection_score': max(group['selection_score'] for group in by_view.values()),
            'selection_definition': 'equal-weight mean of six source-view x roll-stratum landmark-plus-geometry scores',
            'worst_view_selection_definition': 'maximum score across the two source-view holdouts',
            'uses_target_transform_for_speed': False, 'uses_target_ticks_for_speed': False,
            'limitation': 'Synthetic same-source development validation, not independent real-video accuracy'}


@torch.no_grad()
def evaluate(model, loader, device, config, calibration):
    model.eval()
    rows, total_loss, count = [], 0., 0
    for batch_index, raw in enumerate(loader, 1):
        batch = durable._to_device(raw, device)
        output = model(batch['image'])
        loss, _ = training_loss(output, batch, config)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite V5 validation loss')
        total_loss += float(loss) * len(raw['index'])
        count += len(raw['index'])
        points = output['points'].float().cpu().numpy() * 511.
        availability = output['visibility'].float().cpu().numpy()
        for j, index in enumerate(raw['index']):
            record = loader.dataset.records[int(index)]
            reading = read_speed(points[j], visibility=availability[j], calibration=calibration)
            error = abs(reading['speed_kmh']-record['speed_kmh']) if reading['accepted'] and record['speed_valid'] else None
            rows.append({'sample_id': record['sample_id'], 'source_view': record['source_view'],
                         'roll_bucket': record['roll_bucket'], 'source_group': record['source_group'],
                         'parent_sample_id': record['parent_sample_id'], 'target_speed_kmh': record['speed_kmh'],
                         'speed_valid': record['speed_valid'], 'target_availability': record['visibility'],
                         'predicted_points_xy': points[j].tolist(), 'predicted_availability': availability[j].tolist(),
                         'point_errors_px': np.linalg.norm(points[j]-np.asarray(record['points_xy']), axis=1).tolist(),
                         'needle_angle_error_input_deg': _angle_error(points[j], record['points_xy'], record['visibility']),
                         'speed_error_kmh': error, 'reading': reading})
        if batch_index % config.progress_every_batches == 0:
            print(f'Validation {batch_index}/{len(loader)} batches', flush=True)
    metrics = grouped_metrics(rows, config.geometry_penalty_px)
    metrics['combined_loss'] = total_loss/count
    return metrics, rows


def validate_payload(payload):
    required = {'checkpoint_kind', 'format_version', 'identity', 'model_state_dict', 'optimizer_state_dict',
                'scaler_state_dict', 'scheduler_state_dict', 'rng_state', 'loader_generator_states', 'sampler_state',
                'training_state', 'history', 'before_metrics', 'current_metrics', 'best_state', 'worst_view_best_state', 'initialization', 'config'}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError('Incomplete V5 full-state checkpoint')
    if payload['checkpoint_kind'] != CHECKPOINT_KIND or payload['format_version'] != FORMAT_VERSION:
        raise ValueError('Wrong V5 checkpoint family')
    epoch = payload['training_state']['epoch']
    if type(epoch) is not int or epoch < 0 or [row['epoch'] for row in payload['history']] != list(range(1, epoch+1)):
        raise ValueError('Incomplete V5 epoch history')
    if payload['scheduler_state_dict'] is not None:
        raise ValueError('V5 uses constant per-group learning rates, no scheduler')
    if payload['sampler_state']['epoch'] != epoch:
        raise ValueError('Sampler epoch differs from checkpoint')
    if set(payload['loader_generator_states']) != {'train', 'evaluation'}:
        raise ValueError('Missing loader RNG states')
    for generator in payload['loader_generator_states'].values():
        torch.Generator().set_state(generator)
    if not {'python', 'numpy', 'torch', 'cuda'}.issubset(payload['rng_state']):
        raise ValueError('Missing random state')
    all_metrics = [payload['before_metrics']] + [row['validation_metrics'] for row in payload['history']]
    if payload['current_metrics'] != all_metrics[-1]:
        raise ValueError('Current checkpoint metrics differ from epoch history')
    if any(not math.isfinite(item['selection_score']) or not math.isfinite(item['worst_view_selection_score']) for item in all_metrics):
        raise ValueError('Nonfinite selection score')
    for name, key in (('best_state', 'selection_score'), ('worst_view_best_state', 'worst_view_selection_score')):
        best = payload[name]
        expected_epoch = min(range(len(all_metrics)), key=lambda index: all_metrics[index][key])
        if best['epoch'] != expected_epoch or best['selection_loss'] != all_metrics[expected_epoch][key]:
            raise ValueError('Best checkpoint selection differs from history')
    current, best = payload['model_state_dict'], payload['best_state']['model_state_dict']
    if not current or set(current) != set(best):
        raise ValueError('Current/best model tensors differ')
    for name, tensor in current.items():
        if tensor.shape != best[name].shape or tensor.dtype != best[name].dtype or not torch.isfinite(tensor).all() or not torch.isfinite(best[name]).all():
            raise ValueError('Invalid model tensor: ' + name)
    if not {'state', 'param_groups'}.issubset(payload['optimizer_state_dict']):
        raise ValueError('Missing optimizer state')
    if epoch and not payload['optimizer_state_dict']['state']:
        raise ValueError('Trained checkpoint lacks optimizer moments')


def save_checkpoint(payload, path):
    validate_payload(payload)
    marker = durable._durable_torch_save(payload, path)
    marker.update(epoch=payload['training_state']['epoch'], identity=payload['identity'],
                  selection_loss=payload['current_metrics']['selection_score'],
                  worst_view_selection_loss=payload['current_metrics']['worst_view_selection_score'])
    _json(Path(path).with_suffix('.complete.json'), {key: value for key, value in marker.items() if key != 'path'})
    return marker


def discover_checkpoints(root, identity=None):
    candidates, rejected = [], []
    for sidecar in sorted(Path(root).rglob('epoch_*.complete.json')):
        try:
            path = sidecar.with_name(sidecar.name[:-len('.complete.json')] + '.pt')
            payload, marker = durable._read_complete(path)
            validate_payload(payload)
            if marker.get('identity') != payload['identity'] or identity is not None and payload['identity'] != identity:
                raise ValueError('Incompatible V5 dataset/code/runtime/settings')
            if marker.get('epoch') != payload['training_state']['epoch'] or marker.get('selection_loss') != payload['current_metrics']['selection_score']:
                raise ValueError('Checkpoint selection metadata differs')
            candidates.append({**marker, 'path': str(path)})
        except Exception as exc:
            rejected.append({'path': str(sidecar), 'reason': str(exc)})
    if not candidates:
        raise ValueError('No verified compatible V5 checkpoint; refusing silent fresh fallback')
    return candidates, rejected


def select_resume_checkpoint(root, identity, mode):
    if mode not in ('resume_best', 'resume_latest'):
        raise ValueError('Resume selection requires resume_best or resume_latest')
    candidates, rejected = discover_checkpoints(root, identity)
    selected = (max(candidates, key=lambda row: row['created_at_ns']) if mode == 'resume_latest'
                else min(candidates, key=lambda row: (row['selection_loss'], row['epoch'])))
    return selected, rejected


def load_model(path, device='cpu'):
    payload, marker = durable._read_complete(path)
    if payload.get('checkpoint_kind') == CHECKPOINT_KIND:
        validate_payload(payload)
        architecture, preprocessing = payload['identity']['architecture'], payload['identity']['preprocessing']
    elif payload.get('checkpoint_kind') == INFERENCE_KIND:
        architecture, preprocessing = payload.get('architecture'), payload.get('preprocessing')
    else:
        raise ValueError('Not a V5 landmark checkpoint')
    if architecture != ARCHITECTURE_ID or preprocessing != PREPROCESSING_ID or payload.get('format_version') != FORMAT_VERSION:
        raise ValueError('Incompatible V5 architecture/preprocessing')
    model = LandmarkModel().to(device)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.eval()
    return model, payload


def train(config, mode='transfer_v4', source_checkpoint=None, search_root=None, stop_after_epochs=None):
    if isinstance(config, dict):
        config = TrainConfig(**config)
    device = _require_colab_cuda(config)
    if mode not in {'transfer_v4', 'fresh', 'resume_best', 'resume_latest'}:
        raise ValueError('Unknown V5 initialization mode')
    if config.input_size != 512 or config.num_workers != 0 or config.batch_size % 6 or config.batch_size < 6:
        raise ValueError('V5 requires512 inputs, num_workers0 and batch size divisible by6')
    if min(config.epochs, config.patience, config.progress_every_batches) < 1 or config.warmup_epochs < 0:
        raise ValueError('Invalid epoch/progress settings')
    if any(not math.isfinite(value) or value <= 0 for value in (config.needle_lr, config.tick_lr, config.encoder_lr, config.grad_clip, config.heatmap_sigma)):
        raise ValueError('Optimization settings must be positive finite')
    if mode == 'transfer_v4' and not source_checkpoint:
        raise ValueError('V4 transfer requires a source checkpoint')
    if mode != 'transfer_v4' and source_checkpoint is not None:
        raise ValueError('Only transfer_v4 may load source weights; fresh means random initialization')
    partitions, integrity = verify_dataset(config.data_dir)
    calibration = load_calibration(Path(config.data_dir)/'calibration.json')
    sampler = BalancedStrataBatchSampler(partitions['train'], config.batch_size, config.epoch_samples, config.seed)
    settings = asdict(config)
    for key in ('data_dir', 'output_dir', 'device', 'epochs'):
        settings.pop(key)
    identity = {'architecture': ARCHITECTURE_ID, 'preprocessing': PREPROCESSING_ID, 'dataset': integrity,
                'config': settings, 'sampler_policy': {key: value for key, value in sampler.state_dict().items() if key != 'epoch'},
                'torch_version': str(torch.__version__), 'torchvision_version': str(torchvision.__version__),
                'numpy_version': str(np.__version__), 'opencv_version': str(cv2.__version__), 'pillow_version': str(PIL.__version__),
                'device': device.type, 'device_name': torch.cuda.get_device_name(device),
                'implementation_hashes': {relative: _sha(PROJECT/relative) for relative in (
                    'mixed_view_v5/train_landmarks.py', 'real_adaptation_v4/train_landmarks.py', 'real_adaptation_v4/geometry.py',
                    'real_adaptation_v3/train_needle.py', 'training_pilot/train_core.py')},
                'resume_policy': 'same V5 dataset/code/runtime/optimization; total epoch limit may be extended',
                'scheduler_policy': 'constant learning rates; no scheduler'}
    candidate, recovery = None, []
    if mode.startswith('resume_'):
        if not search_root:
            raise ValueError('V5 resume requires search_root')
        candidate, recovery = select_resume_checkpoint(search_root, identity, mode)
    durable._seed(config.seed)
    model = LandmarkModel().to(device)
    initialization = load_transfer_checkpoint(model, source_checkpoint) if mode == 'transfer_v4' else {'mode': 'fresh', 'optimizer_restored': False}
    groups = {'encoder': [], 'needle': [], 'ticks': []}
    for name, parameter in model.named_parameters():
        groups['encoder' if name.startswith('features.') else 'ticks' if name.startswith(('tick_', 'visibility_head.')) else 'needle'].append(parameter)
    optimizer = torch.optim.AdamW([{'params': groups['encoder'], 'lr': config.encoder_lr, 'name': 'encoder'},
                                  {'params': groups['needle'], 'lr': config.needle_lr, 'name': 'needle'},
                                  {'params': groups['ticks'], 'lr': config.tick_lr, 'name': 'ticks'}], weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=config.amp)
    generators = {name: torch.Generator().manual_seed(config.seed+offset) for name, offset in (('train',1),('evaluation',2))}
    train_loader = DataLoader(LandmarkDataset(partitions['train']), batch_sampler=sampler, generator=generators['train'], num_workers=0, pin_memory=True)
    validation_loader = DataLoader(LandmarkDataset(partitions['validation']), batch_size=config.batch_size,
                                  shuffle=False, generator=generators['evaluation'], num_workers=0, pin_memory=True)
    state = {'epoch': 0, 'target_epoch': config.epochs, 'stage': 'initialization', 'training_complete': False,
             'stop_reason': None, 'epochs_without_improvement': 0}
    history, before, current, best, worst = [], None, None, None, None
    run_dir = Path(config.output_dir)/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    # Fail on an unwritable Drive destination before the baseline evaluation.
    _json(run_dir/'identity.json', identity)
    if candidate:
        payload, _ = durable._read_complete(candidate['path'])
        validate_payload(payload)
        if payload['training_state']['epoch'] >= config.epochs:
            raise ValueError('Selected checkpoint reached this total epoch limit; explicitly increase epochs to continue')
        model.load_state_dict(payload['model_state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        scaler.load_state_dict(payload['scaler_state_dict'])
        state, history, best, worst = deepcopy(payload['training_state']), payload['history'], payload['best_state'], payload['worst_view_best_state']
        state.update(target_epoch=config.epochs, training_complete=False, stop_reason=None)
        before, current, initialization = payload['before_metrics'], payload['current_metrics'], payload['initialization']
        sampler.load_state_dict(payload['sampler_state'])
        for name, generator in generators.items():
            generator.set_state(payload['loader_generator_states'][name])
        durable._restore_rng(payload['rng_state'])
    else:
        print('Evaluating epoch 0 transferred/random weights on all six V5 validation strata...', flush=True)
        before, _ = evaluate(model, validation_loader, device, config, calibration)
        current = before
        best = {'epoch': 0, 'selection_loss': before['selection_score'], 'metrics': before, 'model_state_dict': _cpu(model.state_dict()), 'checkpoint_path': None}
        worst = {'epoch': 0, 'selection_loss': before['worst_view_selection_score'], 'metrics': before, 'checkpoint_path': None}
    _json(run_dir/'dataset_integrity.json', integrity)
    _json(run_dir/'before_validation_metrics.json', before)
    _json(run_dir/'identity.json', identity)
    latest = candidate['path'] if candidate else None
    def checkpoint():
        nonlocal latest
        path = run_dir/f"epoch_{state['epoch']:05d}_{uuid.uuid4().hex[:10]}.pt"
        if best['epoch'] == state['epoch']:
            best['checkpoint_path'] = str(path)
        if worst['epoch'] == state['epoch']:
            worst['checkpoint_path'] = str(path)
        payload = {'format_version': FORMAT_VERSION, 'checkpoint_kind': CHECKPOINT_KIND,
                   'identity': identity, 'config': asdict(config), 'model_state_dict': _cpu(model.state_dict()),
                   'optimizer_state_dict': _cpu(optimizer.state_dict()), 'scaler_state_dict': scaler.state_dict(),
                   'scheduler_state_dict': None, 'rng_state': durable._rng_state(),
                   'loader_generator_states': {name: generator.get_state() for name, generator in generators.items()},
                   'sampler_state': sampler.state_dict(), 'training_state': dict(state), 'history': deepcopy(history),
                   'best_state': _cpu(best), 'worst_view_best_state': deepcopy(worst), 'before_metrics': before,
                   'current_metrics': current, 'initialization': initialization}
        marker = save_checkpoint(payload, path)
        latest = str(path)
        _json(run_dir/'latest_checkpoint.json', marker)
        for label, selected in (('best',best), ('best_worst_view',worst)):
            selected_path = Path(selected['checkpoint_path'])
            _json(run_dir/(label+'_checkpoint.json'), {'path': str(selected_path), 'sha256': _sha(selected_path),
                  'epoch': selected['epoch'], 'selection_loss': selected['selection_loss']})
        print(f"Saved full V5 epoch {state['epoch']} checkpoint with verified bytes", flush=True)
    if not candidate:
        checkpoint()
    completed = 0
    for epoch in range(state['epoch']+1, config.epochs+1):
        if stop_after_epochs is not None and completed >= stop_after_epochs:
            break
        sampler.set_epoch(epoch)
        stage = configure_stage(model, epoch, config, fresh=initialization['mode']=='fresh')
        total, count, started = 0., 0, time.monotonic()
        for batch_index, raw in enumerate(train_loader, 1):
            batch = durable._to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=config.amp):
                output = model(batch['image'])
                loss, _ = training_loss(output, batch, config)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite V5 loss; earlier complete checkpoints remain valid')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())*len(raw['index'])
            count += len(raw['index'])
            if batch_index % config.progress_every_batches == 0 or batch_index == 1:
                print(f'Epoch {epoch}/{config.epochs} batch {batch_index}/{len(train_loader)} loss={total/count:.4f} elapsed={time.monotonic()-started:.0f}s', flush=True)
        current, _ = evaluate(model, validation_loader, device, config, calibration)
        history.append({'epoch': epoch, 'stage': stage, 'train_loss': total/count, 'training_draws': count, 'validation_metrics': current})
        meaningful = current['selection_score'] < best['selection_loss']-config.min_delta
        if current['selection_score'] < best['selection_loss']:
            best = {'epoch': epoch, 'selection_loss': current['selection_score'], 'metrics': current,
                    'model_state_dict': _cpu(model.state_dict()), 'checkpoint_path': None}
        if current['worst_view_selection_score'] < worst['selection_loss']:
            worst = {'epoch': epoch, 'selection_loss': current['worst_view_selection_score'], 'metrics': current, 'checkpoint_path': None}
        state['epochs_without_improvement'] = 0 if meaningful else state['epochs_without_improvement']+1
        early = state['epochs_without_improvement'] >= config.patience and epoch > config.warmup_epochs
        state.update(epoch=epoch, stage=stage, training_complete=early or epoch==config.epochs,
                     stop_reason='early_stopping' if early else 'target_reached' if epoch==config.epochs else None)
        checkpoint()
        _json(run_dir/'history.json', history)
        completed += 1
        print(json.dumps({'epoch': epoch, 'macro_selection': current['selection_score'], 'worst_view_selection': current['worst_view_selection_score'],
                          'best_epoch': best['epoch'], 'best_worst_view_epoch': worst['epoch'],
                          'view_within_2_kmh_all_inputs_percent': {view: metrics['primary_speed']['within_2_kmh_all_eligible_inputs_percent'] for view,metrics in current['by_source_view'].items()}}), flush=True)
        if state['training_complete']:
            break
    model.load_state_dict(best['model_state_dict'], strict=True)
    after, rows = evaluate(model, validation_loader, device, config, calibration)
    _json(run_dir/'after_validation_metrics.json', after)
    _json(run_dir/'after_validation_predictions.json', rows)
    def export(path, weights, selected, selection):
        durable._durable_torch_save({'format_version': FORMAT_VERSION, 'checkpoint_kind': INFERENCE_KIND,
            'architecture': ARCHITECTURE_ID, 'preprocessing': PREPROCESSING_ID, 'keypoint_names': KEYPOINT_NAMES,
            'model_state_dict': weights, 'landmark_availability_semantics': AVAILABILITY_SEMANTICS,
            'best_epoch': selected['epoch'], 'metrics': selected['metrics'], 'identity': identity,
            'config': asdict(config), 'initialization': initialization, 'selection_rule': selection,
            'contains_optimizer_state': False}, path)
    inference_path = run_dir/'best_model.pt'
    export(inference_path, best['model_state_dict'], best, 'equal_macro_strata')
    worst_payload, _ = durable._read_complete(worst['checkpoint_path'])
    validate_payload(worst_payload)
    worst_path = run_dir/'best_worst_view_model.pt'
    export(worst_path, worst_payload['model_state_dict'], worst, 'minimum_worst_source_view')
    report = {'run_dir': str(run_dir), 'best_epoch': best['epoch'], 'best_worst_view_epoch': worst['epoch'],
              'completed_epoch': state['epoch'], 'target_epoch': config.epochs, 'training_complete': state['training_complete'],
              'stop_reason': state['stop_reason'], 'epochs_completed_this_call': completed,
              'best_model_path': str(inference_path), 'best_worst_view_model_path': str(worst_path),
              'best_checkpoint_path': best['checkpoint_path'], 'best_worst_view_checkpoint_path': worst['checkpoint_path'],
              'latest_checkpoint_path': latest, 'before_metrics': before, 'after_metrics': after,
              'dataset_integrity': integrity, 'initialization': initialization, 'identity': identity,
              'sampler': {**sampler.state_dict(), 'unique_train_images': len(partitions['train']),
                          'draws_per_stratum_per_epoch': sampler.epoch_samples//6},
              'resume_metadata': {'requested_mode': mode, 'source_checkpoint_path': candidate['path'] if candidate else None,
                    'source_epoch': candidate['epoch'] if candidate else None, 'full_state_restored': candidate is not None,
                    'optimizer_restored': candidate is not None, 'recovery_diagnostics': recovery},
              'selection_rule': after['selection_definition'],
              'source_view_change_from_epoch0': {view: {'within_2_kmh_all_inputs_percentage_point_change':
                    after['by_source_view'][view]['primary_speed']['within_2_kmh_all_eligible_inputs_percent']-before['by_source_view'][view]['primary_speed']['within_2_kmh_all_eligible_inputs_percent'],
                    'needle_mean_error_px_change': after['by_source_view'][view]['needle_mean_error_px']-before['by_source_view'][view]['needle_mean_error_px']}
                    for view in VIEWS},
              'limitation': 'Synthetic source-group holdouts only; all independent video testing must run in Colab'}
    _json(run_dir/'history.json', history)
    _json(run_dir/'training_report.json', report)
    _json(Path(config.output_dir)/'latest_run.json', {'run_dir': str(run_dir), 'report_path': str(run_dir/'training_report.json')})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', default='transfer_v4', choices=['transfer_v4','fresh','resume_best','resume_latest'])
    parser.add_argument('--source-checkpoint')
    parser.add_argument('--search-root')
    args = parser.parse_args()
    print(json.dumps(train(json.loads(Path(args.config).read_text(encoding='utf-8')), mode=args.mode,
                           source_checkpoint=args.source_checkpoint, search_root=args.search_root), indent=2))
