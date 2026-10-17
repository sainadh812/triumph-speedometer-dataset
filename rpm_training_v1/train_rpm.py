"""Small RPM-strip CNN with synthetic validation and durable epoch checkpoints.

This model predicts a rendered fill fraction and an uncalibrated readability
score. Multiplying the fraction by 12,000 uses the draft printed-scale mapping;
it does not establish physical engine RPM. A full-LCD startup gate remains an
external prerequisite because the cropped strip cannot identify engine state.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
from PIL import Image
import PIL
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ARCHITECTURE_ID = 'rpm_strip_gray256x192_small_cnn_fraction_readability_v1'
CHECKPOINT_KIND = 'rpm_strip_full_training_state_v1'
INFERENCE_KIND = 'rpm_strip_inference_weights_v1'
INPUT_HW = (256, 192)
RPM_SCALE = 12000.0
FORMAT_VERSION = 1


@dataclass
class TrainConfig:
    dataset: str
    output: str
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_gamma: float = 0.95
    readability_weight: float = 0.25
    smooth_l1_beta: float = 0.05
    patience: int = 8
    min_delta: float = 1e-5
    grad_clip: float = 5.0
    seed: int = 42
    device: str = 'auto'
    amp: bool = True
    num_workers: int = 0
    resume: str = 'fresh'
    checkpoint: str | None = None
    report_path: str | None = None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _safe_file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('Dataset paths must be nonempty relative paths')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Dataset path is missing or escapes its root: ' + relative)
    return path


def verify_dataset(directory):
    """Validate labels, pixels, provenance splits and every manifest checksum."""
    root = Path(directory).resolve()
    label_path = root / 'labels.jsonl'
    rows = [json.loads(line) for line in label_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    declared_hw = manifest.get('input', {}).get('input_size_hw')
    if declared_hw is not None and declared_hw != list(INPUT_HW):
        raise ValueError('Manifest input dimensions disagree with the model')
    checksums = manifest.get('sha256')
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError('manifest.json must contain a nonempty sha256 path mapping')
    verified = {}
    for relative, expected in checksums.items():
        actual = sha256(_safe_file(root, relative))
        if actual != expected:
            raise ValueError('Dataset checksum mismatch: ' + relative)
        verified[relative] = actual
    if verified.get('labels.jsonl') != sha256(label_path):
        raise ValueError('Manifest must protect labels.jsonl')
    identifiers, paths = set(), set()
    groups = {'train': set(), 'val': set()}
    image_hashes = {'train': set(), 'val': set()}
    counts = Counter()
    for row in rows:
        required = {'id', 'image', 'split', 'readable', 'fill_fraction', 'edge_xy',
                    'indicated_rpm_approx', 'label_origin', 'source_group', 'augmentation'}
        if not required.issubset(row):
            raise ValueError('Missing label fields: ' + str(required - set(row)))
        if row.get('training_eligible', True) is not True:
            raise ValueError('Diagnostic/ineligible rows cannot enter the training label file')
        if row.get('renderer', {}).get('numeric_target_from_existing_predictions', False) is not False:
            raise ValueError('Existing model predictions cannot supply numeric training targets')
        if not isinstance(row['label_origin'], str) or not row['label_origin']:
            raise ValueError('Labels need an explicit provenance label_origin')
        if not isinstance(row['id'], str) or not row['id'] or row['id'] in identifiers:
            raise ValueError('Dataset ids must be unique nonempty strings')
        if row['image'] in paths:
            raise ValueError('Duplicate image path')
        identifiers.add(row['id'])
        paths.add(row['image'])
        split = row['split']
        if split not in groups or type(row['readable']) is not bool:
            raise ValueError('Invalid split or readable label')
        if not isinstance(row['source_group'], str) or not row['source_group']:
            raise ValueError('source_group must identify the procedural/source family')
        groups[split].add(row['source_group'])
        counts[split] += 1
        counts[split + ('_readable' if row['readable'] else '_unreadable')] += 1
        q, rpm, point = row['fill_fraction'], row['indicated_rpm_approx'], row['edge_xy']
        if row['readable']:
            if (not isinstance(q, (int, float)) or isinstance(q, bool) or not math.isfinite(q)
                    or not 0 <= q <= 1 or not isinstance(rpm, (int, float)) or not math.isfinite(rpm)
                    or abs(rpm - q * RPM_SCALE) > 1e-3):
                raise ValueError('Readable rows need consistent finite fraction/RPM targets')
            if (not isinstance(point, list) or len(point) != 2
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point)
                    or not (0 <= point[0] < INPUT_HW[1] and 0 <= point[1] < INPUT_HW[0])):
                raise ValueError('Readable edge_xy must lie inside the input image')
        elif any(value is not None for value in (q, rpm, point)):
            raise ValueError('Unreadable rows must have null numeric and edge targets')
        path = _safe_file(root, row['image'])
        if row['image'] not in verified:
            raise ValueError('Image missing from manifest: ' + row['image'])
        with Image.open(path) as image:
            if image.mode != 'L' or image.size != (INPUT_HW[1], INPUT_HW[0]):
                raise ValueError('Inputs must be grayscale L, width 192, height 256')
            # Pixel hashes also catch duplicate PNG contents with different metadata.
            image_hashes[split].add(hashlib.sha256(image.tobytes()).hexdigest())
    if groups['train'] & groups['val']:
        raise ValueError('Source-group collision between train and validation')
    if image_hashes['train'] & image_hashes['val']:
        raise ValueError('Identical image pixels across train and validation')
    if any(counts[split + suffix] == 0 for split in groups for suffix in ('_readable', '_unreadable')):
        raise ValueError('Both splits need readable positives and unreadable negatives')
    identity = {'manifest_sha256': sha256(manifest_path), 'labels_sha256': sha256(label_path),
                'verified_files_digest': _digest(verified), 'verified_file_count': len(verified),
                'input_hw': list(INPUT_HW), 'counts': dict(counts),
                'preprocessing': manifest.get('input', manifest.get('preprocessing')),
                'protected_preprocessing_calibration_files': {name: digest for name, digest in verified.items()
                     if name == 'preprocess.py' or name.startswith('calibration/')},
                'provenance_limit': 'synthetic development data, not independent real engine-RPM ground truth'}
    return rows, identity


class RPMDataset(Dataset):
    def __init__(self, directory, rows, split):
        self.root = Path(directory)
        self.rows = [row for row in rows if row['split'] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['image']) as image:
            pixels = np.array(image, dtype=np.float32, copy=True) / 255.0
        return {'image': torch.from_numpy(pixels).unsqueeze(0),
                'readable': torch.tensor(float(row['readable'])),
                'fill_fraction': torch.tensor(float(row['fill_fraction'] or 0.0)),
                'id': row['id']}


class RPMModel(nn.Module):
    """Preserve vertical position in a small CNN instead of global pooling it away."""
    def __init__(self):
        super().__init__()
        layers = []
        previous = 1
        for channels in (8, 16, 32, 48):
            layers += [nn.Conv2d(previous, channels, 3, stride=2, padding=1, bias=False),
                       nn.GroupNorm(4, channels), nn.SiLU(inplace=True)]
            previous = channels
        self.features = nn.Sequential(*layers)
        self.shared = nn.Sequential(nn.AdaptiveAvgPool2d((16, 3)), nn.Flatten(),
                                    nn.Linear(48 * 16 * 3, 64), nn.SiLU(), nn.Dropout(0.1))
        self.fraction_head = nn.Linear(64, 1)
        self.readability_head = nn.Linear(64, 1)

    def forward(self, image):
        value = self.shared(self.features(image))
        logit = self.readability_head(value).squeeze(1)
        return {'fill_fraction': self.fraction_head(value).squeeze(1).sigmoid(),
                'readable_logit': logit, 'readability_score': logit.sigmoid()}


def loss_terms(output, targets, readability_weight=0.25, beta=0.05):
    """Negatives supervise readability only; they never imply zero fill."""
    mask = targets['readable'].bool()
    regression = (F.smooth_l1_loss(output['fill_fraction'][mask], targets['fill_fraction'][mask], beta=beta)
                  if mask.any() else output['fill_fraction'].sum() * 0.0)
    readability = F.binary_cross_entropy_with_logits(output['readable_logit'], targets['readable'])
    return {'loss': regression + readability_weight * readability,
            'regression_loss': regression, 'readability_loss': readability,
            'numeric_count': int(mask.sum().item())}


@torch.inference_mode()
def evaluate(model, loader, device, readability_weight=0.25):
    model.eval()
    fractions, scores, targets, labels, sample_ids = [], [], [], [], []
    for batch in loader:
        output = model(batch['image'].to(device))
        fractions.extend(output['fill_fraction'].cpu().tolist())
        scores.extend(output['readability_score'].cpu().tolist())
        targets.extend(batch['fill_fraction'].tolist())
        labels.extend(batch['readable'].bool().tolist())
        sample_ids.extend(batch['id'])
    q, target, score, truth = map(np.asarray, (fractions, targets, scores, labels))
    truth = truth.astype(bool)
    if not truth.any() or truth.all():
        raise ValueError('Validation needs both numeric positives and negatives')
    predicted = score >= 0.5
    error = np.abs(q[truth] - target[truth])
    tp, fp = int((predicted & truth).sum()), int((predicted & ~truth).sum())
    tn, fn = int((~predicted & ~truth).sum()), int((~predicted & truth).sum())
    bounded = np.clip(score.astype(np.float64), 1e-7, 1.0 - 1e-7)
    bce = float(-(truth * np.log(bounded) + (~truth) * np.log(1 - bounded)).mean())
    mae = float(error.mean())
    selected_positive = predicted & truth
    metrics = {
        'samples': len(labels), 'readable_targets': int(truth.sum()), 'unreadable_targets': int((~truth).sum()),
        'fill_fraction_mae': mae, 'fill_fraction_p95_abs_error': float(np.percentile(error, 95)),
        'approx_indicated_rpm_mae': mae * RPM_SCALE,
        'approx_indicated_rpm_p95_abs_error': float(np.percentile(error, 95) * RPM_SCALE),
        'readability_bce': bce, 'readability_accuracy': float((predicted == truth).mean()),
        'readability_precision': tp / (tp + fp) if tp + fp else None,
        'readability_recall': tp / (tp + fn), 'readability_specificity': tn / (tn + fp),
        'readability_balanced_accuracy': 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
        'readability_confusion': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
        'predicted_readable_count': int(predicted.sum()), 'abstained_count': int((~predicted).sum()),
        'numeric_output_coverage': float(predicted.mean()),
        'readable_target_coverage': tp / (tp + fn),
        'accepted_readable_approx_rpm_mae': float(np.abs(q[selected_positive] - target[selected_positive]).mean() * RPM_SCALE) if selected_positive.any() else None,
        'selection_score': mae + readability_weight * bce,
        'selection_definition': 'fill MAE on ALL readable targets + readability_weight * BCE on ALL targets',
        'readability_threshold': 0.5,
        'metric_scope': 'synthetic validation agreement with draft calibration; not real engine-RPM accuracy',
        'score_semantics': 'uncalibrated readability score; not running-state or physical-RPM confidence'
    }
    predictions = [{'id': sid, 'target_readable': bool(label),
                    'target_fill_fraction': float(t) if label else None,
                    'predicted_fill_fraction': float(p), 'readability_score': float(s),
                    'numeric_output_accepted': bool(s >= 0.5),
                    'indicated_rpm_approx': float(p * RPM_SCALE) if s >= 0.5 else None}
                   for sid, label, t, p, s in zip(sample_ids, labels, targets, fractions, scores)]
    return metrics, predictions


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temp.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Exact CUDA RNG resume requires the same number of CUDA devices')
        torch.cuda.set_rng_state_all(state['cuda'])


def _runtime():
    return {'torch': torch.__version__, 'numpy': np.__version__, 'pillow': PIL.__version__}


def _identity(config, dataset_identity):
    parameters = asdict(config)
    # Epoch limits can be extended; ExponentialLR does not depend on the horizon.
    for key in ('dataset', 'output', 'epochs', 'resume', 'checkpoint', 'report_path'):
        parameters.pop(key)
    return {'architecture': ARCHITECTURE_ID, 'dataset': dataset_identity,
            'implementation_sha256': sha256(__file__), 'training_parameters': parameters,
            'runtime': _runtime(), 'normalization': 'grayscale uint8 / 255, no other augmentation in loader',
            'rpm_scale': RPM_SCALE}


def _validate_checkpoint(payload):
    required = {'format_version', 'kind', 'identity', 'model_state_dict', 'optimizer_state_dict',
                'scheduler_state_dict', 'scaler_state_dict', 'rng_state', 'loader_generator_states',
                'epoch', 'history', 'best_epoch', 'best_score', 'bad_epochs', 'best_checkpoint', 'saved_at'}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError('Incomplete full-state RPM checkpoint')
    if payload['kind'] != CHECKPOINT_KIND or payload['format_version'] != FORMAT_VERSION:
        raise ValueError('Incompatible checkpoint kind; speed checkpoints cannot initialize this model')
    if not isinstance(payload['epoch'], int) or payload['epoch'] < 0:
        raise ValueError('Invalid completed epoch')
    history = payload['history']
    if not history or [row['epoch'] for row in history] != list(range(payload['epoch'] + 1)):
        raise ValueError('Checkpoint history does not cover its completed epochs')
    if any(not math.isfinite(row['validation']['selection_score']) for row in history):
        raise ValueError('Nonfinite validation selection score')
    best_epoch = payload['best_epoch']
    if not 0 <= best_epoch <= payload['epoch'] or not math.isfinite(payload['best_score']):
        raise ValueError('Invalid best epoch/score')
    if abs(history[best_epoch]['validation']['selection_score'] - payload['best_score']) > 1e-12:
        raise ValueError('Best epoch score disagrees with history')
    if best_epoch != min(history, key=lambda row: row['validation']['selection_score'])['epoch']:
        raise ValueError('Best epoch is not the minimum validation objective')
    if not {'python', 'numpy', 'torch', 'cuda'}.issubset(payload['rng_state']):
        raise ValueError('Missing RNG state')
    if set(payload['loader_generator_states']) != {'train', 'val'}:
        raise ValueError('Missing data-loader RNG states')
    if payload['epoch'] > 0 and not payload['optimizer_state_dict'].get('state'):
        raise ValueError('Trained checkpoint lacks optimizer moments')
    if payload['scheduler_state_dict'].get('last_epoch') != payload['epoch']:
        raise ValueError('Scheduler epoch disagrees with completed epoch')


def save_checkpoint(payload, path):
    """Flush, rename, hash, reload, then publish the completion marker last."""
    _validate_checkpoint(payload)
    path = Path(path)
    marker_path = path.with_suffix('.complete.json')
    if path.exists() or marker_path.exists():
        raise FileExistsError('Refusing to overwrite an epoch checkpoint: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temp.open('wb') as stream:
            torch.save(_cpu(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        _validate_checkpoint(torch.load(path, map_location='cpu', weights_only=False))
        marker = {'format_version': FORMAT_VERSION, 'kind': CHECKPOINT_KIND, 'filename': path.name,
                  'sha256': sha256(path), 'bytes': path.stat().st_size, 'epoch': payload['epoch'],
                  'identity_digest': _digest(payload['identity']), 'saved_at': payload['saved_at'],
                  'selection_score': payload['history'][-1]['validation']['selection_score']}
        _atomic_json(marker_path, marker)
        return {'path': str(path.resolve()), **marker}
    finally:
        temp.unlink(missing_ok=True)


def load_checkpoint(path, expected_identity=None):
    """Only load trusted local/Drive checkpoints with a matching completion hash."""
    path = Path(path).resolve()
    marker = json.loads(path.with_suffix('.complete.json').read_text(encoding='utf-8'))
    if (marker.get('kind') != CHECKPOINT_KIND or marker.get('filename') != path.name
            or marker.get('bytes') != path.stat().st_size or marker.get('sha256') != sha256(path)):
        raise ValueError('Checkpoint completion marker or checksum mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    _validate_checkpoint(payload)
    if marker.get('identity_digest') != _digest(payload['identity']) or marker.get('epoch') != payload['epoch']:
        raise ValueError('Checkpoint marker identity/epoch mismatch')
    if expected_identity is not None and payload['identity'] != expected_identity:
        raise ValueError('Resume identity mismatch: dataset, preprocessing, code, runtime or training settings changed')
    return payload


def resolve_checkpoint(output, mode, checkpoint=None, identity=None):
    if mode == 'fresh':
        if checkpoint is not None:
            raise ValueError('Fresh random initialization cannot take a checkpoint')
        return None
    if mode == 'explicit':
        if not checkpoint:
            raise ValueError('Explicit resume requires --checkpoint')
        path = Path(checkpoint).resolve()
    elif mode in ('latest', 'best'):
        if checkpoint is not None:
            raise ValueError('Use explicit mode with --checkpoint')
        manifest_path = Path(output) / (mode + '.json')
        if not manifest_path.is_file():
            raise FileNotFoundError('No verified ' + mode + ' checkpoint; use fresh explicitly for random initialization')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        path = Path(manifest['path']).resolve()
        if sha256(path) != manifest.get('sha256'):
            raise ValueError('Checkpoint pointer checksum mismatch')
    else:
        raise ValueError('Resume must be fresh, latest, best or explicit')
    load_checkpoint(path, identity)
    return path


def _best_reference(payload, source_path, identity):
    if payload['best_epoch'] == payload['epoch']:
        path = Path(source_path)
    else:
        reference = payload['best_checkpoint']
        if not reference:
            raise ValueError('Resume checkpoint is missing its earlier best full-state reference')
        path = Path(reference['path'])
        if sha256(path) != reference['sha256']:
            raise ValueError('Earlier best checkpoint checksum mismatch')
    best = load_checkpoint(path, identity)
    if best['epoch'] != payload['best_epoch'] or best['best_score'] != payload['best_score']:
        raise ValueError('Earlier best full-state checkpoint disagrees with resumed history')
    return {'path': str(path.resolve()), 'sha256': sha256(path), 'epoch': best['epoch'],
            'selection_score': best['best_score'], 'kind': CHECKPOINT_KIND}


def _choose_device(requested):
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested not in ('cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return torch.device(requested)


def train(config, *, stop_after_epochs=None):
    """Train to a total epoch limit. Every invocation creates a separate run directory."""
    if config.epochs < 1 or config.batch_size < 1 or config.num_workers < 0 or config.patience < 0:
        raise ValueError('Invalid training counts')
    if any(value <= 0 for value in (config.learning_rate, config.lr_gamma, config.smooth_l1_beta, config.grad_clip)):
        raise ValueError('Positive optimization settings required')
    if config.readability_weight <= 0 or config.min_delta < 0:
        raise ValueError('Readability weight must be positive and min_delta nonnegative')
    rows, dataset_identity = verify_dataset(config.dataset)
    device = _choose_device(config.device)
    identity = _identity(config, dataset_identity)
    identity['execution'] = {'device_type': device.type, 'amp_enabled': config.amp and device.type == 'cuda'}
    source_path = resolve_checkpoint(config.output, config.resume, config.checkpoint, identity)
    source = load_checkpoint(source_path, identity) if source_path else None
    if source and source['epoch'] >= config.epochs:
        raise ValueError('Checkpoint already reached requested total epochs; increase --epochs to continue')
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    probe = output / ('.write-probe-' + uuid.uuid4().hex)
    try:
        with probe.open('wb') as stream:
            stream.write(b'RPM checkpoint write probe\n')
            stream.flush()
            os.fsync(stream.fileno())
        if not probe.read_bytes():
            raise OSError('Output write verification failed')
    finally:
        probe.unlink(missing_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generators = {split: torch.Generator().manual_seed(config.seed + index)
                  for index, split in enumerate(('train', 'val'))}
    loaders = {split: DataLoader(RPMDataset(config.dataset, rows, split), batch_size=config.batch_size,
                                shuffle=split == 'train', num_workers=config.num_workers,
                                generator=generators[split], pin_memory=device.type == 'cuda')
               for split in ('train', 'val')}
    model = RPMModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)
    scaler = torch.amp.GradScaler('cuda', enabled=config.amp and device.type == 'cuda')
    start_epoch, history, bad_epochs = 0, [], 0
    best_epoch, best_score, best_reference = 0, math.inf, None
    if source:
        model.load_state_dict(source['model_state_dict'], strict=True)
        optimizer.load_state_dict(source['optimizer_state_dict'])
        scheduler.load_state_dict(source['scheduler_state_dict'])
        scaler.load_state_dict(source['scaler_state_dict'])
        start_epoch, history = source['epoch'], deepcopy(source['history'])
        best_epoch, best_score, bad_epochs = source['best_epoch'], source['best_score'], source['bad_epochs']
        best_reference = _best_reference(source, source_path, identity)
        for split, generator in generators.items():
            generator.set_state(source['loader_generator_states'][split])
        _restore_rng(source['rng_state'])
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:8]
    run = output / 'runs' / run_id
    run.mkdir(parents=True, exist_ok=False)
    _atomic_json(run / 'config.json', asdict(config))
    _atomic_json(run / 'identity.json', identity)
    _atomic_json(run / 'initialization.json', {'mode': config.resume,
                 'random_initialization': source is None, 'source_checkpoint': str(source_path) if source_path else None,
                 'source_sha256': sha256(source_path) if source_path else None, 'start_epoch': start_epoch,
                 'optimizer_restored': source is not None, 'scheduler_restored': source is not None,
                 'rng_restored': source is not None, 'parameter_count': sum(p.numel() for p in model.parameters()),
                 'device': str(device), 'amp_enabled': scaler.is_enabled()})

    def checkpoint(epoch, improved):
        nonlocal best_reference
        payload = {'format_version': FORMAT_VERSION, 'kind': CHECKPOINT_KIND, 'identity': identity,
                   'config': asdict(config), 'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                   'scaler_state_dict': scaler.state_dict(), 'rng_state': _rng_state(),
                   'loader_generator_states': {split: generator.get_state() for split, generator in generators.items()},
                   'epoch': epoch, 'history': deepcopy(history), 'best_epoch': best_epoch,
                   'best_score': best_score, 'bad_epochs': bad_epochs,
                   'best_checkpoint': None if improved else best_reference,
                   'saved_at': datetime.now(timezone.utc).isoformat()}
        reference = save_checkpoint(payload, run / 'checkpoints' / f'epoch_{epoch:04d}.pt')
        if improved:
            best_reference = reference
        for destination in (run, output):
            _atomic_json(destination / 'latest.json', reference)
            _atomic_json(destination / 'best.json', best_reference)
        _atomic_json(run / 'history.json', history)
        return reference

    if not source:
        baseline, _ = evaluate(model, loaders['val'], device, config.readability_weight)
        best_score = baseline['selection_score']
        history.append({'epoch': 0, 'training': None, 'validation': baseline})
        latest_reference = checkpoint(0, True)
    else:
        latest_reference = {'path': str(source_path), 'sha256': sha256(source_path), 'epoch': start_epoch,
                            'kind': CHECKPOINT_KIND, 'selection_score': history[-1]['validation']['selection_score']}
    epochs_this_invocation = 0
    stop_reason = 'epoch_limit'
    for epoch in range(start_epoch + 1, config.epochs + 1):
        model.train()
        total_loss, total_samples, numeric_samples = 0.0, 0, 0
        for batch in loaders['train']:
            image = batch['image'].to(device, non_blocking=True)
            targets = {key: batch[key].to(device, non_blocking=True) for key in ('readable', 'fill_fraction')}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                prediction = model(image)
                losses = loss_terms(prediction, targets, config.readability_weight, config.smooth_l1_beta)
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('Nonfinite training loss; last completed checkpoint remains usable')
            scaler.scale(losses['loss']).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(losses['loss'].detach()) * len(image)
            total_samples += len(image)
            numeric_samples += losses['numeric_count']
        scheduler.step()
        validation, _ = evaluate(model, loaders['val'], device, config.readability_weight)
        score = validation['selection_score']
        if not math.isfinite(score):
            raise FloatingPointError('Nonfinite validation score')
        history.append({'epoch': epoch, 'training': {'batch_weighted_loss': total_loss / total_samples,
                        'samples': total_samples, 'numeric_samples': numeric_samples,
                        'next_learning_rate': scheduler.get_last_lr()[0]}, 'validation': validation})
        significant = score < best_score - config.min_delta
        improved = score < best_score
        bad_epochs = 0 if significant else bad_epochs + 1
        if improved:
            best_score, best_epoch = score, epoch
        latest_reference = checkpoint(epoch, improved)
        print(json.dumps({'epoch': epoch, 'fill_mae': validation['fill_fraction_mae'],
                          'approx_rpm_mae': validation['approx_indicated_rpm_mae'],
                          'readability_balanced_accuracy': validation['readability_balanced_accuracy'],
                          'numeric_output_coverage': validation['numeric_output_coverage'],
                          'best_epoch': best_epoch}), flush=True)
        epochs_this_invocation += 1
        if stop_after_epochs is not None and epochs_this_invocation >= stop_after_epochs:
            stop_reason = 'requested_epoch_boundary'
            break
        if config.patience and bad_epochs >= config.patience:
            stop_reason = 'early_stopping'
            break
    best_payload = load_checkpoint(best_reference['path'], identity)
    model.load_state_dict(best_payload['model_state_dict'], strict=True)
    best_metrics, predictions = evaluate(model, loaders['val'], device, config.readability_weight)
    inference = {'format_version': FORMAT_VERSION, 'kind': INFERENCE_KIND, 'identity': identity,
                 'model_state_dict': _cpu(model.state_dict()), 'best_epoch': best_epoch,
                 'validation': best_metrics, 'source_full_checkpoint': best_reference,
                 'contains_optimizer_state': False}
    inference_path = run / 'best_model.pt'
    # Reuse write/flush/rename semantics; inference exports have a distinct marker kind.
    temp = inference_path.with_name(inference_path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temp.open('wb') as stream:
            torch.save(inference, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, inference_path)
    finally:
        temp.unlink(missing_ok=True)
    _atomic_json(inference_path.with_suffix('.complete.json'), {'kind': INFERENCE_KIND,
                 'sha256': sha256(inference_path), 'bytes': inference_path.stat().st_size,
                 'filename': inference_path.name, 'best_epoch': best_epoch,
                 'contains_optimizer_state': False})
    _atomic_json(run / 'validation_predictions.json', predictions)
    summary = {'run_dir': str(run), 'best_checkpoint': best_reference, 'latest_checkpoint': latest_reference,
               'best_model': str(inference_path), 'best_epoch': best_epoch,
               'completed_epoch': history[-1]['epoch'], 'stop_reason': stop_reason,
               'before_training': history[0]['validation'], 'best_validation': best_metrics,
               'identity': identity, 'metric_scope': best_metrics['metric_scope'],
               'all_epoch_full_states_saved': True, 'inference_export_has_optimizer': False}
    summary['report_path'] = str(run / 'training_report.json')
    _atomic_json(run / 'training_report.json', summary)
    _atomic_json(output / 'latest_run.json', {'run_dir': str(run), 'report_path': summary['report_path'],
                 'report_sha256': sha256(summary['report_path']), 'completed_epoch': history[-1]['epoch'],
                 'best_epoch': best_epoch, 'status': 'completed'})
    if config.report_path:
        _atomic_json(config.report_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--resume', choices=('fresh', 'latest', 'best', 'explicit'), default='fresh')
    parser.add_argument('--checkpoint')
    parser.add_argument('--report-path', help='Also write the completed report to this exact path')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--no-amp', action='store_true')
    args = vars(parser.parse_args())
    args['amp'] = not args.pop('no_amp')
    print(json.dumps(train(TrainConfig(**args)), indent=2))


if __name__ == '__main__':
    main()
