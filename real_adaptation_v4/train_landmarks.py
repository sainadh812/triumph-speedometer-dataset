"""Joint needle/tick training with verified, resumable epoch checkpoints.

Images are 512px grayscale with the reviewed current-camera LCD mask already
applied by the generator. No image-registration transform, target homography
or per-image label is passed to the landmark decoder or speed calculation.

The legacy-named ``visibility`` tensors mean amodal in-frame landmark
availability. A tick covered by the needle remains a coordinate target; this
head is neither literal stroke visibility nor calibrated prediction confidence.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid
import warnings

import cv2
import numpy as np
import PIL
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from training_pilot import train_core as legacy
from real_adaptation_v3 import train_needle as durable

KEYPOINT_NAMES = ['pivot', 'needle_tip'] + [f'tick_{speed}' for speed in range(0, 181, 20)]
ARCHITECTURE_ID = 'mobilenetv3small_shared_encoder_dual_decoder_needle2_ticks10_v4'
PREPROCESSING_ID = 'gray512_day0_stabilized_fixed_lcd128_mask_v4_imagenet_normalized'
CHECKPOINT_KIND = 'joint_needle_ticks_full_state_v4'
INFERENCE_KIND = 'joint_needle_ticks_inference_v4'
FORMAT_VERSION = 1
SOURCE_SHA256 = '8dcfa269848bac5a46c8ff39d14df4d3470e67905085fb0a662ee45b9bc99d10'
AVAILABILITY_SEMANTICS = 'amodal_in_frame_landmark_availability'
_sha, _json, _cpu = durable._sha, durable._json, durable._cpu


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    epochs: int = 20
    batch_size: int = 8
    warmup_epochs: int = 2
    patience: int = 6
    min_delta: float = .01
    tick_lr: float = 2e-4
    needle_lr: float = 2e-5
    encoder_lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 5.
    input_size: int = 512
    seed: int = 42
    device: str = 'auto'
    amp: bool = True
    num_workers: int = 0
    heatmap_sigma: float = 1.5
    heatmap_weight: float = 1.
    coordinate_weight: float = 5.
    needle_weight: float = 1.
    tick_weight: float = 1.
    visibility_weight: float = .05
    geometry_penalty_px: float = 20.


class LandmarkModel(legacy.SpeedometerModel):
    """V3 needle decoder is retained; a second decoder learns semantic ticks."""
    def __init__(self):
        super().__init__(pretrained=False)
        self.tick_lateral = deepcopy(self.lateral)
        self.tick_decode16 = deepcopy(self.decode16)
        self.tick_decode8 = deepcopy(self.decode8)
        self.tick_decode4 = deepcopy(self.decode4)
        self.tick_head = nn.Conv2d(16, 10, 1)
        self.visibility_head = nn.Linear(576, 12)
        nn.init.zeros_(self.visibility_head.weight)
        nn.init.constant_(self.visibility_head.bias, 2.)

    def forward(self, image):
        skips, value = {}, image
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in (1, 3, 8):
                skips[index] = value
        deep = value
        def decode(lateral, blocks):
            value = lateral(deep)
            for index, block in zip((8, 3, 1), blocks):
                value = F.interpolate(value, size=skips[index].shape[-2:], mode='bilinear', align_corners=True)
                value = block(torch.cat((value, skips[index]), dim=1))
            return value
        needle = decode(self.lateral, (self.decode16, self.decode8, self.decode4))
        ticks = decode(self.tick_lateral, (self.tick_decode16, self.tick_decode8, self.tick_decode4))
        logits = torch.cat((self.heatmap_head(needle), self.tick_head(ticks)), dim=1)
        points = legacy.heatmap_coordinates(logits)
        pooled = F.adaptive_avg_pool2d(deep, 1).flatten(1)
        needle_points = points[:, :2]
        speed_input = torch.cat((pooled, needle_points.flatten(1), needle_points[:, 1] - needle_points[:, 0]), dim=1)
        visibility_logits = self.visibility_head(pooled)
        return {'points': points, 'heatmap_logits': logits,
                'visibility_logits': visibility_logits, 'visibility': visibility_logits.sigmoid(),
                'speed': self.speed_head(speed_input).squeeze(1).sigmoid() * 180.}


def load_transfer_checkpoint(model, path):
    if _sha(path) != SOURCE_SHA256:
        raise ValueError('V3 transfer file is not the verified epoch8 checkpoint')
    source, payload = durable.load_model(path, 'cpu')
    if payload.get('best_epoch') != 8:
        raise ValueError('V3 transfer checkpoint must be epoch8')
    own = model.state_dict()
    state = source.state_dict()
    for name, value in state.items():
        if name not in own or own[name].shape != value.shape:
            raise ValueError('V3 tensor mismatch: ' + name)
        own[name] = value
    for block in ('lateral', 'decode16', 'decode8', 'decode4'):
        prefix = block + '.'
        for name, value in state.items():
            if name.startswith(prefix):
                own['tick_' + name] = value.clone()
    model.load_state_dict(own, strict=True)
    return {'mode': 'transfer', 'source_sha256': SOURCE_SHA256, 'source_epoch': 8,
            'source_architecture': durable.ARCHITECTURE_ID, 'optimizer_restored': False,
            'needle_weights_preserved_at_initialization': True,
            'new_components': ['tick_decoder', 'tick_heatmap_head', 'visibility_head']}


def configure_stage(model, epoch, config, full_encoder=False):
    """Warmup protects the working needle; later fine-tuning uses separate LRs."""
    warmup = epoch <= config.warmup_epochs
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('tick_', 'visibility_head.')) or not warmup)
    for parameter in model.features.parameters():
        parameter.requires_grad_(False)
    if not warmup:
        layers = list(model.features.children())
        for layer in (layers if full_encoder else layers[-4:]):
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    # The direct regression head is retained for compatibility, not optimized or
    # used for primary speed; geometry is solely predicted needle + predicted ticks.
    for parameter in model.speed_head.parameters():
        parameter.requires_grad_(False)
    model.train()
    return 'ticks_warmup' if warmup else 'joint_finetune'


def verify_dataset(directory):
    root = Path(directory).resolve()
    annotation_path = root / 'annotations.json'
    manifest_path = root / 'manifest.json'
    annotations = json.loads(annotation_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if annotations.get('keypoint_names') != KEYPOINT_NAMES:
        raise ValueError('Dataset keypoint ordering differs')
    if manifest.get('landmark_supervision', annotations.get('landmark_supervision')) != AVAILABILITY_SEMANTICS:
        raise ValueError('Dataset must explicitly supervise amodal in-frame landmark availability')
    hashes = manifest.get('sha256', manifest.get('files'))
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Dataset manifest must contain sha256/file hashes')
    for relative, expected in hashes.items():
        expected = expected.get('sha256') if isinstance(expected, dict) else expected
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _sha(path) != expected:
            raise ValueError('Dataset integrity mismatch: ' + relative)
    for required in ('annotations.json', 'calibration.json', 'preprocessing_config.json'):
        if required not in hashes:
            raise ValueError('Manifest must protect ' + required)
    partitions = {'train': [], 'validation': []}
    ids = set()
    for original in annotations['samples']:
        row = dict(original)
        if row['sample_id'] in ids:
            raise ValueError('Duplicate sample id')
        ids.add(row['sample_id'])
        split = row['split']
        if split not in partitions:
            raise ValueError('Unknown dataset split')
        points = np.asarray(row['points_xy'], dtype=float)
        visible = np.asarray(row['visibility'], dtype=bool)
        if points.shape != (12, 2) or visible.shape != (12,) or not np.isfinite(points).all():
            raise ValueError('Invalid landmark labels')
        if ((points[visible] < 0) | (points[visible] > 511)).any():
            raise ValueError('Visible landmark outside image')
        path = (root / row['image_path']).resolve()
        if not path.is_relative_to(root) or row['image_path'] not in hashes:
            raise ValueError('Unprotected/out-of-root image')
        with Image.open(path) as image:
            if image.size != (512, 512) or image.mode != 'L':
                raise ValueError('Dataset images must be 512×512 grayscale')
            pixels = np.asarray(image).copy()
        mask_relative = row.get('lcd_mask_path')
        if not isinstance(mask_relative, str) or mask_relative not in hashes:
            raise ValueError('Every sample must have a hash-protected LCD mask')
        mask_path = (root / mask_relative).resolve()
        if not mask_path.is_relative_to(root):
            raise ValueError('LCD mask path escaped dataset')
        with Image.open(mask_path) as mask_image:
            if mask_image.size != (512, 512) or mask_image.mode != 'L':
                raise ValueError('LCD masks must be512×512 grayscale')
            mask = np.asarray(mask_image)
        if not np.isin(mask, [0, 255]).all() or not (mask == 255).any() or (mask == 255).all():
            raise ValueError('LCD mask must be nonempty binary0/255 and retain image pixels')
        hidden = mask == 255
        if not (pixels[hidden] == 128).all():
            raise ValueError('LCD mask pixels are not exactly128 after augmentation')
        targets = np.zeros((512, 512), dtype=np.uint8)
        for xy in points[visible]:
            cv2.circle(targets, tuple(np.rint(xy).astype(int)), 2, 255, -1)
        if visible[:2].all():
            cv2.line(targets, tuple(np.rint(points[0]).astype(int)), tuple(np.rint(points[1]).astype(int)), 255, 3)
        if (hidden & (targets > 0)).any():
            raise ValueError('LCD mask overlaps a visible landmark or the needle shaft')
        row['image'] = str(path)
        row['speed_valid'] = bool(row.get('speed_valid', True))
        if row['speed_valid'] and not 0 <= float(row['speed_kmh']) <= 180:
            raise ValueError('Invalid indicated-speed label')
        partitions[split].append(row)
    if any(not rows for rows in partitions.values()):
        raise ValueError('Both train and validation must be nonempty')
    groups = {split: {row['source_group'] for row in rows} for split, rows in partitions.items()}
    if groups['train'] & groups['validation']:
        raise ValueError('Source groups overlap across dataset partitions')
    integrity = {'manifest_sha256': _sha(manifest_path), 'annotation_sha256': _sha(annotation_path),
                 'calibration_sha256': _sha(root / 'calibration.json'),
                 'sample_counts': {split: len(rows) for split, rows in partitions.items()},
                 'source_groups': {split: sorted(values) for split, values in groups.items()},
                 'preprocessing': manifest.get('preprocessing'),
                 'landmark_supervision': AVAILABILITY_SEMANTICS,
                 'limitation': 'Same-video synthetic development holdout, not independent camera accuracy'}
    preprocessing = integrity['preprocessing']
    if (not isinstance(preprocessing, dict) or preprocessing.get('color') != 'grayscale'
            or preprocessing.get('input_size') != [512, 512]
            or preprocessing.get('camera_stabilization') != 'image_only_day_reference_SIFT_static_scale_RANSAC'
            or preprocessing.get('lcd_mask_applied') is not True
            or preprocessing.get('lcd_mask_value') != 128
            or preprocessing.get('mask_before_geometric_augmentation') is not True):
        raise ValueError('Manifest must describe approved stabilized day-camera fixed LCD128 preprocessing')
    polygon = np.asarray(preprocessing.get('lcd_polygon_reference_xy'), dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3 or not np.isfinite(polygon).all():
        raise ValueError('Manifest LCD polygon is missing/malformed')
    if json.loads((root / 'preprocessing_config.json').read_text(encoding='utf-8')) != preprocessing:
        raise ValueError('Manifest and preprocessing configuration differ')
    for path_key, digest_key in (('fixed_mask_path', 'fixed_mask_sha256'), ('reference_image_path', 'reference_sha256')):
        relative, digest = preprocessing.get(path_key), preprocessing.get(digest_key)
        protected = hashes.get(relative)
        protected = protected.get('sha256') if isinstance(protected, dict) else protected
        if not relative or protected != digest:
            raise ValueError('Reference image/fixed mask is not protected by manifest')
    if preprocessing.get('canonical_calibration_sha256') != integrity['calibration_sha256']:
        raise ValueError('Canonical calibration differs from preprocessing configuration')
    if preprocessing.get('stabilize_mask_script_sha256') != _sha(PROJECT / 'real_adaptation_v4/stabilize_mask.py'):
        raise ValueError('Current camera stabilizer differs from dataset generation implementation')
    return partitions, integrity


class LandmarkDataset(Dataset):
    def __init__(self, records):
        self.records = list(records)
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        row = self.records[index]
        return {'image': legacy.preprocess_image(row['image'], 512),
                'points': torch.tensor(row['points_xy'], dtype=torch.float32) / 511.,
                'visibility': torch.tensor(row['visibility'], dtype=torch.float32),
                'speed': torch.tensor(row['speed_kmh'], dtype=torch.float32), 'index': index}


def training_loss(output, batch, config):
    logits, points = output['heatmap_logits'].float(), batch['points'].float()
    height, width = logits.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height, device=logits.device), torch.arange(width, device=logits.device), indexing='ij')
    centers = points * points.new_tensor([width - 1, height - 1])
    distance = (xx[None, None] - centers[:, :, 0, None, None]).square() + (yy[None, None] - centers[:, :, 1, None, None]).square()
    target = torch.exp(-distance / (2 * config.heatmap_sigma ** 2)).flatten(2)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
    heatmap = -(target * logits.flatten(2).log_softmax(-1)).sum(-1)
    coordinate = F.smooth_l1_loss(output['points'].float(), points, beta=.02, reduction='none').mean(-1)
    visible = batch['visibility'].float()
    needle_mask, tick_mask = visible[:, :2], visible[:, 2:]
    losses = config.heatmap_weight * heatmap + config.coordinate_weight * coordinate
    needle = (losses[:, :2] * needle_mask).sum() / needle_mask.sum().clamp_min(1.)
    ticks = (losses[:, 2:] * tick_mask).sum() / tick_mask.sum().clamp_min(1.)
    visibility = F.binary_cross_entropy_with_logits(output['visibility_logits'].float(), visible)
    total = config.needle_weight * needle + config.tick_weight * ticks + config.visibility_weight * visibility
    return total, {'needle_loss': float(needle.detach()), 'tick_loss': float(ticks.detach()),
                   'visibility_loss': float(visibility.detach())}


@torch.no_grad()
def evaluate(model, loader, device, config, calibration):
    from real_adaptation_v4.geometry import read_speed
    model.eval()
    rows, needle_errors, tick_errors, speed_errors = [], [], [], []
    point_errors = {name: [] for name in KEYPOINT_NAMES}
    rejections, total_loss, count, eligible, accepted = Counter(), 0., 0, 0, 0
    for raw in loader:
        batch = durable._to_device(raw, device)
        output = model(batch['image'])
        loss, _ = training_loss(output, batch, config)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite validation loss')
        total_loss += float(loss) * len(raw['index'])
        predictions = output['points'].float().cpu().numpy() * 511.
        visibility = output['visibility'].float().cpu().numpy()
        for j, index in enumerate(raw['index']):
            record = loader.dataset.records[int(index)]
            errors = np.linalg.norm(predictions[j] - np.asarray(record['points_xy']), axis=1)
            valid = np.asarray(record['visibility'], bool)
            for name, error_value, is_visible in zip(KEYPOINT_NAMES, errors, valid):
                if is_visible:
                    point_errors[name].append(float(error_value))
            needle_errors.extend(errors[:2][valid[:2]].tolist())
            tick_errors.extend(errors[2:][valid[2:]].tolist())
            reading = read_speed(predictions[j], visibility=visibility[j], calibration=calibration)
            count += 1
            accepted += int(reading['accepted'])
            eligible += int(record['speed_valid'])
            error = abs(reading['speed_kmh'] - record['speed_kmh']) if reading['accepted'] and record['speed_valid'] else None
            if error is not None:
                speed_errors.append(error)
            if not reading['accepted']:
                rejections[reading['reason']] += 1
            rows.append({'sample_id': record['sample_id'], 'target_speed_kmh': record['speed_kmh'],
                         'predicted_points_xy': predictions[j].tolist(), 'predicted_visibility': visibility[j].tolist(),
                         'point_errors_px': errors.tolist(), 'speed_error_kmh': error, 'reading': reading})
    if not count or not needle_errors or not tick_errors:
        raise ValueError('Validation requires needle and tick labels')
    within2 = 100. * sum(error <= 2 for error in speed_errors) / max(eligible, 1)
    metrics = {'count': count, 'combined_loss': total_loss / count,
               'needle_mean_error_px': float(np.mean(needle_errors)),
               'needle_pivot_mean_error_px': float(np.mean(point_errors['pivot'])) if point_errors['pivot'] else None,
               'needle_tip_mean_error_px': float(np.mean(point_errors['needle_tip'])) if point_errors['needle_tip'] else None,
               'needle_tip_p95_error_px': float(np.percentile(point_errors['needle_tip'], 95)) if point_errors['needle_tip'] else None,
               'per_landmark_mean_error_px': {name: float(np.mean(values)) if values else None for name, values in point_errors.items()},
               'tick_mean_error_px': float(np.mean(tick_errors)), 'tick_p95_error_px': float(np.percentile(tick_errors, 95)),
               'primary_speed': {'accepted': accepted, 'rejected': count - accepted, 'coverage_percent': 100 * accepted / count,
                                 'eligible_count': eligible, 'within_2_kmh_all_eligible_inputs_percent': within2,
                                 'accepted_only': durable._summary(speed_errors), 'rejection_reasons': dict(rejections)},
               'uses_target_transform_for_speed': False, 'uses_target_ticks_for_speed': False,
               'limitation': 'Same-video synthetic development check; not independent real accuracy'}
    metrics['selection_score'] = metrics['needle_mean_error_px'] + metrics['tick_mean_error_px'] + config.geometry_penalty_px * (1 - within2 / 100)
    return metrics, rows


def validate_payload(payload):
    required = {'format_version', 'checkpoint_kind', 'identity', 'config', 'model_state_dict', 'optimizer_state_dict',
                'scaler_state_dict', 'rng_state', 'loader_generator_states', 'training_state', 'history', 'best_state',
                'before_metrics', 'current_metrics', 'initialization'}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError('Incomplete V4 full-state checkpoint')
    if payload['format_version'] != FORMAT_VERSION or payload['checkpoint_kind'] != CHECKPOINT_KIND:
        raise ValueError('Wrong checkpoint family/version')
    state = payload['training_state']
    epoch, target = state['epoch'], state['target_epoch']
    if type(epoch) is not int or type(target) is not int or not 0 <= epoch <= target or target != payload['config']['epochs']:
        raise ValueError('Invalid epoch/target')
    if [item['epoch'] for item in payload['history']] != list(range(1, epoch + 1)):
        raise ValueError('Incomplete epoch history')
    if state['training_complete'] != (epoch == target or state['stop_reason'] == 'early_stopping'):
        raise ValueError('Invalid completion state')
    best = payload['best_state']
    if not 0 <= best['epoch'] <= epoch or not math.isfinite(best['selection_loss']):
        raise ValueError('Invalid best epoch/metric')
    if best['selection_loss'] != best['metrics']['selection_score']:
        raise ValueError('Best selection metric mismatch')
    current = payload['model_state_dict']
    if not current or set(current) != set(best['model_state_dict']):
        raise ValueError('Current/best tensor names differ')
    for name, value in current.items():
        other = best['model_state_dict'][name]
        if not isinstance(value, torch.Tensor) or value.shape != other.shape or value.dtype != other.dtype or not torch.isfinite(value).all() or not torch.isfinite(other).all():
            raise ValueError('Invalid model tensor: ' + name)
    if set(payload['loader_generator_states']) != {'train', 'evaluation'}:
        raise ValueError('Missing loader RNG')
    for generator in payload['loader_generator_states'].values():
        torch.Generator().set_state(generator)
    if not {'state', 'param_groups'}.issubset(payload['optimizer_state_dict']):
        raise ValueError('Missing optimizer state')
    for values in payload['optimizer_state_dict']['state'].values():
        for value in values.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError('Nonfinite optimizer state')


def save_checkpoint(payload, path):
    validate_payload(payload)
    marker = durable._durable_torch_save(payload, path)
    # V3's durable byte publisher is generic; add V4 selection metadata only
    # after it has serialized, reread and checksum-verified every destination.
    marker.update(epoch=payload['training_state']['epoch'], identity=payload['identity'],
                  selection_loss=payload['current_metrics']['selection_score'])
    _json(Path(path).with_suffix('.complete.json'), {k: v for k, v in marker.items() if k != 'path'})
    return marker


def discover_checkpoints(root, identity=None):
    candidates, rejected = [], []
    for marker_path in sorted(Path(root).rglob('epoch_*.complete.json')):
        try:
            path = marker_path.with_name(marker_path.name[:-len('.complete.json')] + '.pt')
            payload, marker = durable._read_complete(path)
            validate_payload(payload)
            if marker.get('identity') != payload['identity'] or identity is not None and payload['identity'] != identity:
                raise ValueError('Incompatible dataset/model/runtime/configuration')
            if marker.get('epoch') != payload['training_state']['epoch'] or marker.get('selection_loss') != payload['current_metrics']['selection_score']:
                raise ValueError('Checkpoint selection metadata differs')
            candidates.append({**marker, 'path': str(path)})
        except Exception as exc:
            rejected.append({'path': str(marker_path), 'reason': str(exc)})
    if not candidates:
        raise ValueError('No verified compatible V4 checkpoint; refusing silent fresh fallback')
    if rejected:
        warnings.warn(f'Skipped {len(rejected)} invalid/incompatible checkpoints', RuntimeWarning)
    return candidates, rejected


def load_model(path, device='cpu'):
    payload, _ = durable._read_complete(path)
    if payload.get('checkpoint_kind') == CHECKPOINT_KIND:
        validate_payload(payload)
        architecture = payload['identity']['architecture']
        preprocessing = payload['identity']['preprocessing']
    elif payload.get('checkpoint_kind') == INFERENCE_KIND:
        architecture, preprocessing = payload.get('architecture'), payload.get('preprocessing')
    else:
        raise ValueError('Not V4 joint landmark weights')
    if architecture != ARCHITECTURE_ID or preprocessing != PREPROCESSING_ID or payload.get('format_version') != FORMAT_VERSION:
        raise ValueError('Incompatible V4 model/preprocessing')
    model = LandmarkModel().to(device)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.eval()
    return model, payload


def train(config, mode='transfer', source_checkpoint=None, search_root=None, stop_after_epochs=None):
    from real_adaptation_v4.geometry import load_calibration
    if isinstance(config, dict):
        config = TrainConfig(**config)
    if mode not in {'transfer', 'fresh', 'resume_latest', 'resume_best'}:
        raise ValueError('Unknown training mode')
    if config.input_size != 512 or config.num_workers != 0:
        raise ValueError('Requires512px input and num_workers0 for exact epoch continuation')
    if min(config.epochs, config.batch_size, config.patience) < 1 or config.warmup_epochs < 0:
        raise ValueError('Invalid epoch/batch/warmup/patience settings')
    if any(not math.isfinite(x) or x <= 0 for x in (config.tick_lr, config.needle_lr, config.encoder_lr, config.heatmap_sigma, config.grad_clip)):
        raise ValueError('Learning rates/scales must be positive finite')
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 0):
        raise ValueError('stop_after_epochs must be nonnegative integer')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if config.device == 'auto' else config.device)
    partitions, integrity = verify_dataset(config.data_dir)
    calibration = load_calibration(Path(config.data_dir) / 'calibration.json')
    settings = asdict(config)
    for key in ('data_dir', 'output_dir', 'device'):
        settings.pop(key)
    identity = {'architecture': ARCHITECTURE_ID, 'preprocessing': PREPROCESSING_ID,
                'dataset': integrity, 'config': settings, 'torch_version': str(torch.__version__),
                'torchvision_version': str(torchvision.__version__), 'numpy_version': str(np.__version__),
                'opencv_version': str(cv2.__version__), 'pillow_version': str(PIL.__version__),
                'device': device.type, 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                'implementation_hashes': {relative: _sha(PROJECT / relative) for relative in (
                    'real_adaptation_v4/train_landmarks.py', 'real_adaptation_v4/geometry.py', 'real_adaptation_v4/stabilize_mask.py',
                    'real_adaptation_v3/train_needle.py', 'training_pilot/train_core.py')},
                'resume_policy': 'same_dataset_code_runtime_config_full_state_only'}
    candidate, recovery = None, []
    if mode.startswith('resume_'):
        if search_root is None:
            raise ValueError('Resume requires search_root')
        candidates, recovery = discover_checkpoints(search_root, identity)
        candidate = max(candidates, key=lambda row: row['created_at_ns']) if mode == 'resume_latest' else min(candidates, key=lambda row: (row['selection_loss'], -row['created_at_ns']))
    if mode == 'transfer' and not source_checkpoint:
        raise ValueError('Transfer requires verified V3 epoch8 checkpoint')
    durable._seed(config.seed)
    model = LandmarkModel().to(device)
    initialization = load_transfer_checkpoint(model, source_checkpoint) if mode == 'transfer' else {'mode': 'fresh', 'optimizer_restored': False}
    groups = {'encoder': [], 'needle': [], 'ticks': []}
    for name, parameter in model.named_parameters():
        groups['encoder' if name.startswith('features.') else 'ticks' if name.startswith(('tick_', 'visibility_head.')) else 'needle'].append(parameter)
    optimizer = torch.optim.AdamW([{'params': groups['encoder'], 'lr': config.encoder_lr, 'name': 'encoder'},
                                   {'params': groups['needle'], 'lr': config.needle_lr, 'name': 'needle'},
                                   {'params': groups['ticks'], 'lr': config.tick_lr, 'name': 'ticks'}], weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=bool(config.amp and device.type == 'cuda'))
    generators = {name: torch.Generator().manual_seed(config.seed + offset) for name, offset in (('train', 1), ('evaluation', 2))}
    train_loader = DataLoader(LandmarkDataset(partitions['train']), batch_size=config.batch_size, shuffle=True, generator=generators['train'], num_workers=0)
    validation_loader = DataLoader(LandmarkDataset(partitions['validation']), batch_size=config.batch_size, shuffle=False, generator=generators['evaluation'], num_workers=0)
    state = {'epoch': 0, 'target_epoch': config.epochs, 'stage': 'ticks_warmup', 'training_complete': False, 'stop_reason': None, 'epochs_without_improvement': 0}
    history, before, current, best = [], None, None, None
    if candidate:
        payload, _ = durable._read_complete(candidate['path'])
        validate_payload(payload)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        scaler.load_state_dict(payload['scaler_state_dict'])
        state, history, best = payload['training_state'], payload['history'], payload['best_state']
        before, current, initialization = payload['before_metrics'], payload['current_metrics'], payload['initialization']
        for name, generator in generators.items():
            generator.set_state(payload['loader_generator_states'][name])
        durable._restore_rng(payload['rng_state'])
    else:
        before, _ = evaluate(model, validation_loader, device, config, calibration)
        current = before
        best = {'epoch': 0, 'selection_loss': before['selection_score'], 'metrics': before,
                'model_state_dict': _cpu(model.state_dict()), 'checkpoint_path': None}
    run_dir = Path(config.output_dir) / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    _json(run_dir / 'dataset_integrity.json', integrity)
    _json(run_dir / 'before_validation_metrics.json', before)
    latest = candidate['path'] if candidate else None
    def checkpoint():
        nonlocal latest
        path = run_dir / f"epoch_{state['epoch']:05d}_{uuid.uuid4().hex[:10]}.pt"
        if best['epoch'] == state['epoch']:
            best['checkpoint_path'] = str(path)
        payload = {'format_version': FORMAT_VERSION, 'checkpoint_kind': CHECKPOINT_KIND,
                   'identity': identity, 'config': asdict(config), 'model_state_dict': _cpu(model.state_dict()),
                   'optimizer_state_dict': _cpu(optimizer.state_dict()), 'scaler_state_dict': scaler.state_dict(),
                   'rng_state': durable._rng_state(), 'loader_generator_states': {name: generator.get_state() for name, generator in generators.items()},
                   'training_state': dict(state), 'history': list(history), 'best_state': _cpu(best),
                   'before_metrics': before, 'current_metrics': current, 'initialization': initialization}
        marker = save_checkpoint(payload, path)
        latest = str(path)
        _json(run_dir / 'latest_checkpoint.json', marker)
        _json(run_dir / 'best_checkpoint.json', {'path': best['checkpoint_path'], 'epoch': best['epoch'], 'selection_loss': best['selection_loss']})
    if not candidate:
        checkpoint()
    completed = 0
    if not state['training_complete']:
        for epoch in range(state['epoch'] + 1, config.epochs + 1):
            if stop_after_epochs is not None and completed >= stop_after_epochs:
                break
            stage = configure_stage(model, epoch, config, full_encoder=initialization['mode'] == 'fresh')
            total, count = 0., 0
            for raw in train_loader:
                batch = durable._to_device(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=bool(config.amp and device.type == 'cuda')):
                    output = model(batch['image'])
                    loss, _ = training_loss(output, batch, config)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss; previous durable checkpoint remains available')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach()) * len(raw['index'])
                count += len(raw['index'])
            current, _ = evaluate(model, validation_loader, device, config, calibration)
            history.append({'epoch': epoch, 'stage': stage, 'train_loss': total / count, 'validation_metrics': current})
            meaningful_improvement = current['selection_score'] < best['selection_loss'] - config.min_delta
            if current['selection_score'] < best['selection_loss']:
                best = {'epoch': epoch, 'selection_loss': current['selection_score'], 'metrics': current,
                        'model_state_dict': _cpu(model.state_dict()), 'checkpoint_path': None}
            if meaningful_improvement:
                state['epochs_without_improvement'] = 0
            else:
                state['epochs_without_improvement'] += 1
            early = state['epochs_without_improvement'] >= config.patience and epoch > config.warmup_epochs
            state.update(epoch=epoch, stage=stage, training_complete=early or epoch == config.epochs,
                         stop_reason='early_stopping' if early else 'target_reached' if epoch == config.epochs else None)
            checkpoint()
            _json(run_dir / 'history.json', history)
            completed += 1
            primary = current['primary_speed']
            print(f"Epoch {epoch}/{config.epochs} {stage} | needle={current['needle_mean_error_px']:.2f}px | ticks={current['tick_mean_error_px']:.2f}px | accepted={primary['coverage_percent']:.1f}% | all-input±2={primary['within_2_kmh_all_eligible_inputs_percent']:.1f}% | best={best['epoch']}", flush=True)
            if state['training_complete']:
                break
    model.load_state_dict(best['model_state_dict'], strict=True)
    after, rows = evaluate(model, validation_loader, device, config, calibration)
    _json(run_dir / 'after_validation_metrics.json', after)
    _json(run_dir / 'after_validation_predictions.json', rows)
    inference_path = run_dir / 'best_model.pt'
    durable._durable_torch_save({'format_version': FORMAT_VERSION, 'checkpoint_kind': INFERENCE_KIND,
                               'architecture': ARCHITECTURE_ID, 'preprocessing': PREPROCESSING_ID,
                               'keypoint_names': KEYPOINT_NAMES, 'model_state_dict': best['model_state_dict'],
                               'landmark_availability_semantics': AVAILABILITY_SEMANTICS,
                               'best_epoch': best['epoch'], 'metrics': best['metrics'], 'identity': identity,
                               'config': asdict(config), 'initialization': initialization}, inference_path)
    report = {'run_dir': str(run_dir), 'best_epoch': best['epoch'], 'completed_epoch': state['epoch'],
              'target_epoch': config.epochs, 'training_complete': state['training_complete'], 'stop_reason': state['stop_reason'],
              'epochs_completed_this_call': completed, 'best_model_path': str(inference_path),
              'best_checkpoint_path': best['checkpoint_path'], 'latest_checkpoint_path': latest,
              'before_metrics': before, 'after_metrics': after, 'dataset_integrity': integrity,
              'initialization': initialization, 'resume_metadata': {'requested_mode': mode, 'source_checkpoint_path': candidate['path'] if candidate else None,
                 'source_epoch': candidate['epoch'] if candidate else None, 'full_state_restored': candidate is not None, 'optimizer_restored': candidate is not None, 'recovery_diagnostics': recovery},
              'selection_rule': 'minimum mean needle pixel error + mean tick pixel error +20*(1-all-eligible-input speed within2 fraction)',
              'limitation': 'Same-video synthetic development evaluation; independent real-video accuracy remains unmeasured'}
    _json(run_dir / 'history.json', history)
    _json(run_dir / 'training_report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', default='transfer', choices=['transfer', 'fresh', 'resume_latest', 'resume_best'])
    parser.add_argument('--source-checkpoint')
    parser.add_argument('--search-root')
    args = parser.parse_args()
    result = train(json.loads(Path(args.config).read_text(encoding='utf-8')), mode=args.mode,
                   source_checkpoint=args.source_checkpoint, search_root=args.search_root)
    print(json.dumps(result, indent=2))
