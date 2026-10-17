"""Isolated V3 needle fine-tuning with durable full-state epoch checkpoints.

Validation is a same-clip synthetic base-plate holdout with shared donor pixels.
Private real diagnostics use approximate visual labels and never select weights.
Reported primary speed uses predicted needle points plus image-only registration.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
import uuid
import warnings

import cv2
import numpy as np
import PIL
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from training_pilot import train_core as legacy
from real_adaptation_v3.needle_data import NeedleDataset, verify_dataset, read_speed, ENDPOINT_TOLERANCE_KMH

SpeedometerModel = legacy.SpeedometerModel
ARCHITECTURE_ID = legacy._ARCHITECTURE_ID
PREPROCESSING_ID = legacy._PREPROCESSING_ID
SOURCE_ARCHITECTURE_ID = 'dual_mobilenet_v3_small_frozen_needle_multiscale64_ticks6_arc512_v2'
APPROVED_SOURCE_SHA256 = '4f56b76d8b5b44a1d0a7250b52ed0b7cb3e37cd7a8baff0bdd21eff1221fa4c1'
FORMAT_VERSION = 1
CHECKPOINT_KIND = 'real_adaptation_needle_full_state_v3'
INFERENCE_KIND = 'real_adaptation_needle_inference_v3'


@dataclass
class NeedleTrainConfig:
    data_dir: str
    output_dir: str
    real_data_dir: str | None = None
    include_real_training: bool = False
    epochs: int = 25
    batch_size: int = 8
    decoder_lr: float = 1e-4
    encoder_lr: float = 2e-5
    weight_decay: float = 1e-4
    grad_clip: float = 5.
    input_size: int = 512
    seed: int = 42
    device: str = 'auto'
    amp: bool = True
    num_workers: int = 0
    pretrained: bool = False
    full_encoder: bool = False
    heatmap_weight: float = 1.
    coordinate_weight: float = 5.
    heatmap_sigma: float = 1.5
    auxiliary_speed_weight: float = .25
    geometry_penalty_px: float = 20.


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / ('.tmp_' + uuid.uuid4().hex[:12] + '.partial')
    try:
        with temporary.open('x', encoding='utf-8') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _rng_state():
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': {'kind': n[0], 'keys': n[1].tolist(),
            'position': n[2], 'has_gauss': n[3], 'cached_gaussian': n[4]},
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['kind'], np.array(n['keys'], dtype=np.uint32), n['position'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available():
            raise ValueError('Cannot restore CUDA state on CPU')
        torch.cuda.set_rng_state_all(state['cuda'])


def configure_model(model, full_encoder=False):
    """All needle decoder/head weights learn; transfer only limits early encoder."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    layers = list(model.features.children())
    for layer in layers:
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
    for layer in (layers if full_encoder else layers[-4:]):
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    model.train()


def load_transfer_checkpoint(model, path, expected_sha256=None):
    expected_sha256 = APPROVED_SOURCE_SHA256 if expected_sha256 is None else expected_sha256
    digest = _sha(path)
    if digest != expected_sha256:
        raise ValueError('Combined source checkpoint SHA256 differs from the approved epoch34 file')
    source = torch.load(path, map_location='cpu', weights_only=True)
    if (source.get('format_version') != 1 or source.get('checkpoint_kind') != 'augmented_ring_inference_weights_v2'
            or source.get('architecture') != SOURCE_ARCHITECTURE_ID
            or source.get('preprocessing') != PREPROCESSING_ID
            or source.get('config', {}).get('input_size') != 512 or source.get('best_epoch') != 34):
        raise ValueError('Combined source architecture/input/preprocessing/epoch differs')
    state = {name[len('needle_branch.'):]: tensor for name, tensor in source['model_state_dict'].items()
             if name.startswith('needle_branch.')}
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError('Extracted needle tensor names do not exactly match the legacy model')
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape
                or tensor.dtype != expected[name].dtype or not torch.isfinite(tensor).all()):
            raise ValueError('Invalid source needle tensor: ' + name)
    model.load_state_dict(state, strict=True)
    return {'mode': 'transfer', 'source_sha256': digest, 'source_epoch': 34,
            'source_branch': 'needle_branch', 'source_architecture': SOURCE_ARCHITECTURE_ID,
            'target_architecture': ARCHITECTURE_ID, 'optimizer_restored': False,
            'full_state_restored': False, 'epoch_numbering': 'New experiment starts at0',
            'frozen_needle_policy_removed': True}


def training_loss(output, batch, config):
    original, parts = legacy.training_loss(output, batch, config)
    individual_speed = F.smooth_l1_loss(output['speed'].float() / 10., batch['speed'].float() / 10.,
                                      beta=.2, reduction='none')
    valid = batch.get('speed_valid', torch.ones_like(individual_speed, dtype=torch.bool)).float()
    speed_loss = (individual_speed * valid).sum() / valid.sum().clamp_min(1.)
    total = original - individual_speed.mean() + config.auxiliary_speed_weight * speed_loss
    parts['speed_loss'] = float(speed_loss.detach())
    return total, parts


def _to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _summary(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'count': 0, 'mae_kmh': None, 'rmse_kmh': None, 'p95_kmh': None,
                'within_2_kmh_percent': None, 'within_5_kmh_percent': None}
    return {'count': len(a), 'mae_kmh': float(a.mean()), 'rmse_kmh': float(np.sqrt(np.mean(a * a))),
            'p95_kmh': float(np.percentile(a, 95)), 'within_2_kmh_percent': float(100 * np.mean(a <= 2)),
            'within_5_kmh_percent': float(100 * np.mean(a <= 5))}


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    rows, pivot_errors, tip_errors, direct_errors, geometry_errors = [], [], [], [], []
    rejected, loss_sum, count, accepted, eligible = Counter(), 0., 0, 0, 0
    clamped, clamped_eligible = 0, 0
    for raw in loader:
        batch = _to_device(raw, device)
        output = model(batch['image'])
        loss, _ = training_loss(output, batch, config)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite evaluation loss')
        loss_sum += float(loss) * len(batch['image'])
        predictions = output['points'].float().cpu().numpy() * 511
        speed = output['speed'].float().cpu().numpy()
        for j, index in enumerate(raw['index']):
            record = loader.dataset.records[int(index)]
            points = predictions[j]
            reading = read_speed(points, record['registration'], record['calibration'])
            valid_speed = bool(record['speed_valid'])
            gt_points = np.asarray(record['points_xy'])
            point_errors = np.linalg.norm(points - gt_points, axis=1)
            point_valid = record.get('confidence') != 'low'
            if point_valid:
                pivot_errors.append(float(point_errors[0]))
                tip_errors.append(float(point_errors[1]))
            count += 1
            eligible += int(valid_speed)
            accepted += int(reading['accepted'])
            clamped += int(reading['endpoint_clamped'])
            clamped_eligible += int(reading['endpoint_clamped'] and valid_speed)
            if not reading['accepted']:
                rejected[str(reading['reason'])] += 1
            direct_error = abs(float(speed[j]) - record['speed_kmh']) if valid_speed else None
            geometry_error = abs(reading['speed_kmh'] - record['speed_kmh']) if valid_speed and reading['accepted'] else None
            if direct_error is not None:
                direct_errors.append(direct_error)
            if geometry_error is not None:
                geometry_errors.append(geometry_error)
            rows.append({'sample_id': record['sample_id'], 'kind': record['kind'],
                         'source_frame': record['source_frame'], 'split': record['split'],
                         'target_indicated_speed_kmh': record['speed_kmh'], 'speed_target_eligible': valid_speed,
                         'label_status': record['label_status'], 'label_confidence': record.get('confidence'),
                         'predicted_pivot_xy': points[0].tolist(), 'predicted_tip_xy': points[1].tolist(),
                         'pivot_error_px': float(point_errors[0]) if point_valid else None,
                         'tip_error_px': float(point_errors[1]) if point_valid else None,
                         'direct_speed_kmh': float(speed[j]), 'direct_error_kmh': direct_error,
                         'primary_speed_kmh': reading['speed_kmh'], 'primary_error_kmh': geometry_error,
                         'raw_indicated_speed_kmh': reading['raw_indicated_speed_kmh'],
                         'endpoint_clamped': reading['endpoint_clamped'],
                         'endpoint_tolerance_kmh': reading['endpoint_tolerance_kmh'],
                         'geometry_accepted': reading['accepted'], 'geometry_reason': reading['reason'],
                         'image_registration': record['registration'],
                         'geometry_inputs': 'predicted pivot/tip plus accepted image-only registration and static reference calibration'})
    if not count or not pivot_errors:
        raise ValueError('Evaluation requires nonempty valid needle labels')
    within2 = 100. * sum(error <= 2 for error in geometry_errors) / max(eligible, 1)
    pixel_mean = (float(np.mean(pivot_errors)) + float(np.mean(tip_errors))) / 2
    metrics = {'count': count, 'combined_loss': loss_sum / count,
               'selection_score': pixel_mean + config.geometry_penalty_px * (1 - within2 / 100.),
               'needle_pivot_mean_error_px': float(np.mean(pivot_errors)),
               'needle_tip_mean_error_px': float(np.mean(tip_errors)),
               'needle_mean_error_px': pixel_mean, 'needle_tip_p95_error_px': float(np.percentile(tip_errors, 95)),
               'keypoint_scored_count': len(pivot_errors), 'keypoint_excluded_low_confidence_count': count - len(pivot_errors),
               'direct_speed': _summary(direct_errors),
               'primary_speed': {'accepted': accepted, 'rejected': count - accepted,
                                 'coverage_percent': 100. * accepted / count,
                                 'speed_target_eligible_count': eligible, 'excluded_speed_label_count': count - eligible,
                                 'endpoint_clamped_count': clamped, 'endpoint_clamped_eligible_count': clamped_eligible,
                                 'endpoint_tolerance_kmh': ENDPOINT_TOLERANCE_KMH,
                                 'endpoint_policy': 'Fixed development calibration tolerance; local linear extrapolation retained, small end-stop excursions clipped; not a model improvement',
                                 'within_2_kmh_all_eligible_inputs_percent': within2,
                                 'accepted_only': _summary(geometry_errors), 'rejection_reasons': dict(rejected)},
               'uses_ground_truth_camera_transform_for_speed': False,
               'uses_per_image_target_ticks_for_speed': False,
               'scope': sorted({record['split'] for record in loader.dataset.records}),
               'limitation': 'Same-clip synthetic base-plate holdout shares donor pixels; real labels estimate indicated speed. Neither establishes independent real-camera or road-speed accuracy.'}
    return metrics, rows


def _validate_payload(payload):
    required = {'format_version', 'checkpoint_kind', 'identity', 'config', 'model_state_dict',
                'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict', 'rng_state',
                'loader_generator_states', 'training_state', 'history', 'best_state',
                'before_metrics', 'before_predictions', 'before_real_metrics',
                'before_real_predictions', 'current_metrics', 'initialization'}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError('Incomplete needle full-state checkpoint')
    if payload['format_version'] != FORMAT_VERSION or payload['checkpoint_kind'] != CHECKPOINT_KIND:
        raise ValueError('Wrong needle checkpoint family/version')
    state = payload['training_state']
    epoch, target = state['epoch'], state['target_epoch']
    if type(epoch) is not int or type(target) is not int or not 0 <= epoch <= target or target != payload['config']['epochs']:
        raise ValueError('Invalid checkpoint epoch/fixed target')
    if [row['epoch'] for row in payload['history']] != list(range(1, epoch + 1)):
        raise ValueError('Checkpoint history does not cover completed epochs')
    if state['training_complete'] != (epoch == target) or state['stage'] != 'finetune':
        raise ValueError('Invalid saved completion/stage')
    if payload['scheduler_state_dict'] is not None:
        raise ValueError('This trainer uses fixed learning rates and no scheduler')
    if set(payload['loader_generator_states']) != {'train', 'evaluation'}:
        raise ValueError('Missing loader random states')
    if not {'python', 'numpy', 'torch', 'cuda'}.issubset(payload['rng_state']):
        raise ValueError('Missing random state')
    best = payload['best_state']
    if not 0 <= best['epoch'] <= epoch or not math.isfinite(best['selection_loss']):
        raise ValueError('Invalid checkpoint best state')
    if best['selection_loss'] != best['metrics']['selection_score'] or not math.isfinite(payload['current_metrics']['selection_score']):
        raise ValueError('Inconsistent checkpoint selection metric')
    if not {'state', 'param_groups'}.issubset(payload['optimizer_state_dict']):
        raise ValueError('Missing optimizer state')
    current_weights, best_weights = payload['model_state_dict'], best['model_state_dict']
    if not current_weights or set(current_weights) != set(best_weights):
        raise ValueError('Current and best model tensor names differ')
    for name, value in current_weights.items():
        other = best_weights[name]
        if (not isinstance(value, torch.Tensor) or not isinstance(other, torch.Tensor)
                or value.shape != other.shape or value.dtype != other.dtype
                or not torch.isfinite(value).all() or not torch.isfinite(other).all()):
            raise ValueError('Invalid current/best model tensor: ' + name)
    for optimizer_state in payload['optimizer_state_dict']['state'].values():
        for value in optimizer_state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError('Nonfinite optimizer tensor')
    random.Random().setstate(payload['rng_state']['python'])
    numpy_state = payload['rng_state']['numpy']
    if (numpy_state['kind'] != 'MT19937' or len(numpy_state['keys']) != 624
            or any(type(value) is not int or not 0 <= value < 2 ** 32 for value in numpy_state['keys'])
            or not 0 <= numpy_state['position'] <= 624
            or numpy_state['has_gauss'] not in (0, 1)
            or not math.isfinite(numpy_state['cached_gaussian'])):
        raise ValueError('Malformed NumPy random state')
    random_tensors = [payload['rng_state']['torch'], *payload['rng_state']['cuda'],
                      *payload['loader_generator_states'].values()]
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
           or value.ndim != 1 or value.numel() == 0 for value in random_tensors):
        raise ValueError('Malformed Torch/loader random state')
    for value in [payload['rng_state']['torch'], *payload['loader_generator_states'].values()]:
        torch.Generator().set_state(value)


def _durable_torch_save(payload, target):
    """Serialize locally, verify the destination bytes, then publish the marker."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.with_suffix('.complete.json').exists():
        raise FileExistsError('Refusing to replace a completed checkpoint')
    temporary = target.parent / ('.tmp_' + uuid.uuid4().hex[:12] + '.partial')
    try:
        with tempfile.TemporaryDirectory(prefix='needle_save_') as directory:
            local = Path(directory) / 'checkpoint.pt'
            with local.open('xb') as f:
                torch.save(_cpu(payload), f)
                f.flush()
                os.fsync(f.fileno())
            check = torch.load(local, map_location='cpu', weights_only=True)
            if payload['checkpoint_kind'] == CHECKPOINT_KIND:
                _validate_payload(check)
            checksum = _sha(local)
            with local.open('rb') as source, temporary.open('xb') as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            if _sha(temporary) != checksum:
                raise OSError('Drive copy checksum failed')
            os.replace(temporary, target)
            if _sha(target) != checksum:
                raise OSError('Published checkpoint checksum failed')
        marker = {'checkpoint': target.name, 'sha256': checksum, 'bytes': target.stat().st_size,
                  'created_at_ns': time.time_ns(), 'checkpoint_kind': payload['checkpoint_kind'],
                  'format_version': FORMAT_VERSION}
        if payload['checkpoint_kind'] == CHECKPOINT_KIND:
            marker.update(epoch=payload['training_state']['epoch'], identity=payload['identity'],
                          selection_loss=payload['current_metrics']['selection_score'])
        _json(target.with_suffix('.complete.json'), marker)
        return {**marker, 'path': str(target)}
    finally:
        temporary.unlink(missing_ok=True)


def _read_complete(path):
    path = Path(path)
    marker = json.loads(path.with_suffix('.complete.json').read_text(encoding='utf-8'))
    if marker.get('checkpoint') != path.name or marker.get('sha256') != _sha(path) or marker.get('bytes') != path.stat().st_size:
        raise ValueError('Checkpoint completion hash/size mismatch')
    return torch.load(path, map_location='cpu', weights_only=True), marker


def discover_checkpoints(search_root, identity=None, *, diagnostics=None):
    candidates, rejected = [], []
    root = Path(search_root)
    for sidecar in sorted(root.rglob('epoch_*.complete.json')):
        try:
            path = sidecar.with_name(sidecar.name[:-len('.complete.json')] + '.pt')
            payload, marker = _read_complete(path)
            _validate_payload(payload)
            if marker.get('identity') != payload['identity'] or identity is not None and payload['identity'] != identity:
                raise ValueError('Incompatible dataset/model/runtime/configuration identity')
            if marker.get('epoch') != payload['training_state']['epoch'] or marker.get('selection_loss') != payload['current_metrics']['selection_score']:
                raise ValueError('Completion selection metadata differs')
            candidates.append({**marker, 'path': str(path)})
        except Exception as error:
            rejected.append({'path': str(sidecar), 'reason': str(error)})
    for path in root.rglob('epoch_*.pt'):
        if not path.with_suffix('.complete.json').is_file():
            rejected.append({'path': str(path), 'reason': 'Incomplete save without completion marker'})
    if diagnostics is not None:
        diagnostics.extend(rejected)
    if not candidates:
        raise ValueError('No verified compatible needle full-state checkpoint; refusing fresh fallback')
    if rejected:
        warnings.warn(f'Skipped {len(rejected)} corrupt/incompatible/incomplete states; verified older states remain', RuntimeWarning)
    return candidates


def _select_checkpoint(search_root, mode, identity):
    diagnostics = []
    candidates = discover_checkpoints(search_root, identity, diagnostics=diagnostics)
    selected = (max(candidates, key=lambda row: row['created_at_ns']) if mode == 'resume_latest'
                else min(candidates, key=lambda row: (row['selection_loss'], -row['created_at_ns'])))
    return {**selected, 'recovery_diagnostics': diagnostics}


def load_model(path, device='cpu'):
    payload, _ = _read_complete(path)
    if payload.get('checkpoint_kind') == CHECKPOINT_KIND:
        _validate_payload(payload)
        architecture, preprocessing = payload['identity']['architecture'], payload['identity']['preprocessing']
    elif payload.get('checkpoint_kind') == INFERENCE_KIND:
        architecture, preprocessing = payload.get('architecture'), payload.get('preprocessing')
    else:
        raise ValueError('Not a V3 needle inference/full-state checkpoint')
    if payload.get('format_version') != FORMAT_VERSION or architecture != ARCHITECTURE_ID or preprocessing != PREPROCESSING_ID:
        raise ValueError('Incompatible needle architecture/preprocessing')
    model = SpeedometerModel(pretrained=False).to(device)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.eval()
    return model, payload


def _save_evaluation(directory, prefix, metrics, rows):
    _json(Path(directory) / (prefix + '_metrics.json'), metrics)
    _json(Path(directory) / (prefix + '_predictions.json'), rows)


def plot_history(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row['epoch'] for row in history]
    axes[0].plot(epochs, [r['validation_metrics']['needle_tip_mean_error_px'] for r in history], label='Tip error')
    axes[0].plot(epochs, [r['validation_metrics']['needle_pivot_mean_error_px'] for r in history], label='Pivot error')
    axes[0].set(xlabel='Completed epoch', ylabel='Pixels', title='Needle errors on synthetic validation')
    axes[0].legend()
    axes[1].plot(epochs, [r['validation_metrics']['primary_speed']['within_2_kmh_all_eligible_inputs_percent'] for r in history])
    axes[1].set(xlabel='Completed epoch', ylabel='All eligible inputs within ±2 km/h (%)', ylim=(0, 100), title='Image-registration-based indicated speed')
    fig.suptitle('Same-clip synthetic base-plate holdout; not independent real-world accuracy')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def train(config, mode='transfer', source_checkpoint=None, search_root=None, stop_after_epochs=None):
    if isinstance(config, dict):
        config = NeedleTrainConfig(**config)
    mode = {'latest': 'resume_latest', 'best': 'resume_best'}.get(mode, mode)
    if mode not in {'transfer', 'fresh', 'resume_latest', 'resume_best'}:
        raise ValueError('Unknown needle training mode')
    if config.input_size != 512 or config.num_workers != 0:
        raise ValueError('V3 requires512 input and num_workers=0 for epoch-exact continuation')
    if type(config.epochs) is not int or config.epochs <= 0 or type(config.batch_size) is not int or config.batch_size <= 0:
        raise ValueError('Positive integer epoch and batch settings required')
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 0):
        raise ValueError('stop_after_epochs must be a nonnegative integer')
    values = [config.decoder_lr, config.encoder_lr, config.grad_clip, config.heatmap_sigma]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Learning rates and scales must be positive finite values')
    if any(not math.isfinite(v) or v < 0 for v in [config.weight_decay, config.heatmap_weight, config.coordinate_weight,
                                                 config.auxiliary_speed_weight, config.geometry_penalty_px]):
        raise ValueError('Loss/optimizer weights must be finite and nonnegative')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if config.device == 'auto' else config.device)
    partitions, integrity = verify_dataset(config.data_dir, config.real_data_dir, config.include_real_training)
    settings = asdict(config)
    for key in ('data_dir', 'real_data_dir', 'output_dir', 'device'):
        settings.pop(key)
    identity = {'architecture': ARCHITECTURE_ID, 'preprocessing': PREPROCESSING_ID,
                'dataset': integrity, 'config': settings, 'torch_version': str(torch.__version__),
                'torchvision_version': str(torchvision.__version__), 'numpy_version': str(np.__version__),
                'pillow_version': str(PIL.__version__), 'opencv_version': str(cv2.__version__),
                'device': device.type, 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                'implementation_hashes': {relative: _sha(PROJECT / relative) for relative in (
                    'real_adaptation_v3/train_needle.py', 'real_adaptation_v3/needle_data.py',
                    'real_adaptation_v3/registration_image.py', 'training_pilot/train_core.py')},
                'optimizer_policy': 'AdamW_decoder_heads_1e-4_encoder_2e-5_fixed_lrs',
                'resume_policy': 'same_dataset_code_runtime_config_full_state_only'}
    candidate = None
    if mode.startswith('resume_'):
        if search_root is None:
            raise ValueError('Exact resume requires search_root')
        candidate = _select_checkpoint(search_root, mode, identity)
    if mode == 'transfer' and (source_checkpoint is None or not Path(source_checkpoint).is_file()):
        raise ValueError('Transfer requires the approved combined epoch34 checkpoint')
    _seed(config.seed)
    model = SpeedometerModel(pretrained=config.pretrained if mode == 'fresh' else False).to(device)
    initialization = {'mode': 'fresh', 'pretrained': config.pretrained, 'source_epoch': None,
                      'optimizer_restored': False, 'full_state_restored': False}
    if mode == 'transfer':
        initialization = load_transfer_checkpoint(model, source_checkpoint)
    initialization['full_encoder'] = bool(config.full_encoder or mode == 'fresh' and not config.pretrained)
    configure_model(model, initialization['full_encoder'])
    encoder = [p for name, p in model.named_parameters() if name.startswith('features.')]
    decoder = [p for name, p in model.named_parameters() if not name.startswith('features.')]
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': config.encoder_lr, 'name': 'encoder'},
                                   {'params': decoder, 'lr': config.decoder_lr, 'name': 'decoder_heads'}],
                                  weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=bool(config.amp and device.type == 'cuda'))
    generators = {'train': torch.Generator().manual_seed(config.seed + 1),
                  'evaluation': torch.Generator().manual_seed(config.seed + 2)}
    train_loader = DataLoader(NeedleDataset(partitions['train']), batch_size=config.batch_size,
                             shuffle=True, generator=generators['train'], num_workers=0)
    validation_loader = DataLoader(NeedleDataset(partitions['validation']), batch_size=config.batch_size,
                                  shuffle=False, generator=generators['evaluation'], num_workers=0)
    state = {'epoch': 0, 'target_epoch': config.epochs, 'stage': 'finetune', 'training_complete': False}
    history, before, before_rows, best = [], None, [], None
    before_real, before_real_rows = None, []
    diagnostic_loader = None
    if partitions.get('real_diagnostic'):
        diagnostic_loader = DataLoader(NeedleDataset(partitions['real_diagnostic']), batch_size=config.batch_size,
                                       shuffle=False, num_workers=0,
                                       generator=torch.Generator().manual_seed(config.seed + 3))
    resume = {'requested_mode': mode, 'full_state_restored': False, 'optimizer_restored': False,
              'source_epoch': None, 'source_checkpoint_path': None, 'recovery_diagnostics': []}
    if candidate:
        payload, _ = _read_complete(candidate['path'])
        _validate_payload(payload)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        initialization = payload['initialization']
        configure_model(model, initialization['full_encoder'])
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        scaler.load_state_dict(payload['scaler_state_dict'])
        state, history, best = payload['training_state'], payload['history'], payload['best_state']
        before, before_rows, current = payload['before_metrics'], payload['before_predictions'], payload['current_metrics']
        before_real, before_real_rows = payload['before_real_metrics'], payload['before_real_predictions']
        for key, generator in generators.items():
            generator.set_state(payload['loader_generator_states'][key])
        _restore_rng(payload['rng_state'])
        resume.update(full_state_restored=True, optimizer_restored=True, source_epoch=state['epoch'],
                      source_checkpoint_path=candidate['path'], recovery_diagnostics=candidate['recovery_diagnostics'])
    else:
        before, before_rows = evaluate(model, validation_loader, device, config)
        current = before
        best = {'epoch': 0, 'selection_loss': before['selection_score'], 'metrics': before,
                'model_state_dict': _cpu(model.state_dict()), 'checkpoint_path': None}
        if diagnostic_loader is not None:
            before_real, before_real_rows = evaluate(model, diagnostic_loader, device, config)
            before_real.update(used_for_model_selection=False, approved_real_train_frames=list(
                record['source_frame'] for record in partitions['train'] if record['kind'] == 'real'))
    run_dir = Path(config.output_dir) / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    _save_evaluation(run_dir, 'before_validation', before, before_rows)
    if before_real is not None:
        _save_evaluation(run_dir, 'before_real_diagnostic', before_real, before_real_rows)
    initialization['trainable_tensor_names'] = sorted(name for name, p in model.named_parameters() if p.requires_grad)
    _json(run_dir / 'dataset_integrity.json', integrity)
    _json(run_dir / 'initialization.json', initialization)
    latest = candidate['path'] if candidate else None

    def checkpoint():
        nonlocal latest
        name = f"epoch_{state['epoch']:05d}_{uuid.uuid4().hex[:10]}.pt"
        target = run_dir / name
        if best['epoch'] == state['epoch']:
            best['checkpoint_path'] = str(target)
        payload = {'format_version': FORMAT_VERSION, 'checkpoint_kind': CHECKPOINT_KIND,
                   'identity': identity, 'config': asdict(config), 'model_state_dict': _cpu(model.state_dict()),
                   'optimizer_state_dict': _cpu(optimizer.state_dict()), 'scheduler_state_dict': None,
                   'scaler_state_dict': scaler.state_dict(), 'rng_state': _rng_state(),
                   'loader_generator_states': {k: g.get_state() for k, g in generators.items()},
                   'training_state': dict(state), 'history': list(history), 'best_state': _cpu(best),
                   'before_metrics': before, 'before_predictions': before_rows,
                   'before_real_metrics': before_real, 'before_real_predictions': before_real_rows,
                   'current_metrics': current, 'initialization': initialization}
        _validate_payload(payload)
        marker = _durable_torch_save(payload, target)
        latest = marker['path']
        _json(run_dir / 'latest_checkpoint.json', marker)
        _json(run_dir / 'best_checkpoint.json', {'path': best['checkpoint_path'], 'epoch': best['epoch'],
                                               'selection_loss': best['selection_loss']})

    if not candidate:
        checkpoint()
    completed_here = 0
    for epoch in range(state['epoch'] + 1, state['target_epoch'] + 1):
        if stop_after_epochs is not None and completed_here >= stop_after_epochs:
            break
        configure_model(model, initialization['full_encoder'])
        total, n_samples = 0., 0
        for raw in train_loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=bool(config.amp and device.type == 'cuda')):
                output = model(batch['image'])
                loss, _ = training_loss(output, batch, config)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss; last complete checkpoint remains recoverable')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(batch['image'])
            n_samples += len(batch['image'])
        current, _ = evaluate(model, validation_loader, device, config)
        history.append({'epoch': epoch, 'train_combined_loss': total / n_samples, 'validation_metrics': current,
                        'decoder_lr': optimizer.param_groups[1]['lr'], 'encoder_lr': optimizer.param_groups[0]['lr']})
        if current['selection_score'] < best['selection_loss']:
            best = {'epoch': epoch, 'selection_loss': current['selection_score'], 'metrics': current,
                    'model_state_dict': _cpu(model.state_dict()), 'checkpoint_path': None}
        state.update(epoch=epoch, training_complete=epoch == state['target_epoch'])
        checkpoint()
        _json(run_dir / 'history.json', history)
        completed_here += 1
        primary = current['primary_speed']
        print(f"Epoch {epoch}/{state['target_epoch']} | tip={current['needle_tip_mean_error_px']:.3f}px | "
              f"coverage={primary['coverage_percent']:.1f}% | all-input ±2={primary['within_2_kmh_all_eligible_inputs_percent']:.1f}% | "
              f"best={best['epoch']}", flush=True)
    model.load_state_dict(best['model_state_dict'], strict=True)
    model.eval()
    after, after_rows = evaluate(model, validation_loader, device, config)
    _save_evaluation(run_dir, 'after_validation', after, after_rows)
    real_metrics = None
    if state['training_complete'] and diagnostic_loader is not None:
        real_metrics, real_rows = evaluate(model, diagnostic_loader, device, config)
        real_metrics.update(used_for_model_selection=False, approved_real_train_frames=list(
            record['source_frame'] for record in partitions['train'] if record['kind'] == 'real'))
        _save_evaluation(run_dir, 'real_diagnostic', real_metrics, real_rows)
        _save_evaluation(run_dir, 'after_real_diagnostic', real_metrics, real_rows)
    inference_path = run_dir / 'best_model.pt'
    _durable_torch_save({'format_version': FORMAT_VERSION, 'checkpoint_kind': INFERENCE_KIND,
                         'architecture': ARCHITECTURE_ID, 'preprocessing': PREPROCESSING_ID,
                         'model_state_dict': best['model_state_dict'], 'best_epoch': best['epoch'],
                         'metrics': best['metrics'], 'identity': identity, 'config': asdict(config),
                         'initialization': initialization, 'keypoint_order': ['pivot', 'tip'],
                         'speed_range_kmh': [0, 180]}, inference_path)
    _json(run_dir / 'history.json', history)
    plot_history(history, run_dir / 'training_history.png')
    report = {'run_dir': str(run_dir), 'mode': mode, 'best_epoch': best['epoch'], 'completed_epoch': state['epoch'],
              'target_epoch': state['target_epoch'], 'training_complete': state['training_complete'],
              'epochs_completed_this_call': completed_here, 'best_model_path': str(inference_path),
              'best_checkpoint_path': best['checkpoint_path'], 'latest_checkpoint_path': latest,
              'history_path': str(run_dir / 'history.json'), 'before_metrics': before, 'after_metrics': after,
              'real_diagnostics': real_metrics, 'before_real_metrics': before_real,
              'after_real_metrics': real_metrics, 'initialization': initialization, 'resume_metadata': resume,
              'dataset_integrity': integrity,
              'selection_rule': f'minimum mean needle pixel error +{config.geometry_penalty_px:g}*(1 - all-eligible-input primary speed within2 fraction), including epoch0',
              'limitation': 'All public validation and private diagnostics are same-clip development checks, not independent real-world accuracy.'}
    _json(run_dir / 'training_report.json', report)
    return report
