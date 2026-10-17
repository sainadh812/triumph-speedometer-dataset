"""Verified V3 needle inputs and image-registration-based speed decoding.

Synthetic camera transforms and per-image target ticks are deliberately omitted
from the records exposed to evaluation. Speed inference uses predicted needle
points and accepted image-only registration against the calibrated reference.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from training_pilot.train_core import preprocess_image
from real_adaptation_v3.registration_image import register_image as _register_image

REAL_TRAIN_FRAMES = (0, 36, 96, 108, 120, 132, 144, 156, 168, 180)
REAL_DIAGNOSTIC_FRAMES = (240, 264, 300, 360, 396, 408, 420, 432, 480)
ENDPOINT_TOLERANCE_KMH = 2.0


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    if not isinstance(relative, str):
        raise ValueError('Missing relative input path')
    p, w = PurePosixPath(relative), PureWindowsPath(relative)
    if p.is_absolute() or w.drive or '..' in p.parts or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe dataset path')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing dataset file: ' + relative)
    return path


def _point(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all() or np.any((a < 0) | (a > 511)):
        raise ValueError('Invalid needle coordinate')
    return a.tolist()


def _image(path):
    with Image.open(path) as image:
        if image.mode != 'L' or image.size != (512, 512):
            raise ValueError('Expected a grayscale 512x512 input')
        return np.asarray(image).copy()


def load_calibration(path):
    cal = json.loads(Path(path).read_text(encoding='utf-8'))
    pivot = np.asarray(_point(cal['pivot_xy']))
    ticks = sorted(cal['ticks'], key=lambda row: row['speed_kmh'])
    speeds = np.asarray([float(t['speed_kmh']) for t in ticks])
    points = np.asarray([_point(t['xy']) for t in ticks])
    angles = np.unwrap(np.arctan2(points[:, 1] - pivot[1], points[:, 0] - pivot[0]))
    if (speeds.tolist() != list(range(0, 181, 20)) or not np.all(np.diff(angles) > 0)):
        raise ValueError('Expected reviewed monotonic 0-180 real-dial calibration')
    return {'pivot_xy': pivot.tolist(), 'speeds': speeds.tolist(), 'angles': angles.tolist(),
            'ticks_xy': points.tolist(), 'raw': cal,
            'limitation': 'Manual real-dial draft calibration; estimates of indicated speed, not road-speed ground truth'}


def registration_from_record(row):
    metadata = dict(row.get('image_registration') or {})
    h = row.get('estimated_to_reference_h')
    # The generator must explicitly identify an image-only estimate. A missing
    # estimate rejects the reading; there is no use of reference_to_image_h.
    image_only = (metadata.get('uses_render_transform_or_landmark_labels') is False
                  and str(metadata.get('method', '')).startswith('image_only_'))
    accepted = metadata.get('accepted') is True and image_only
    return {**metadata, 'accepted': bool(accepted and h is not None),
            'image_to_reference_h': h,
            'reason': ('registration_not_identified_as_image_only' if not image_only else
                       metadata.get('reason') or ('image_registration_rejected' if not accepted else 'accepted'))}


def register_image(image, reference, calibration):
    """Use the exact same image-only helper as synthetic registration generation."""
    h, metadata = _register_image(np.asarray(image, np.uint8), np.asarray(reference, np.uint8), seed=0)
    return {**metadata, 'accepted': bool(metadata.get('accepted') and h is not None),
            'image_to_reference_h': h, 'reason': metadata.get('reason', 'image_registration_rejected')}


def read_speed(predicted_points_xy, registration, calibration):
    result = {'accepted': False, 'speed_kmh': None, 'reason': 'image_registration_rejected',
              'raw_indicated_speed_kmh': None, 'endpoint_clamped': False,
              'endpoint_tolerance_kmh': ENDPOINT_TOLERANCE_KMH}
    if not registration.get('accepted'):
        result['reason'] = registration.get('reason', result['reason'])
        return result
    h = np.asarray(registration.get('image_to_reference_h'), dtype=float)
    points = np.asarray(predicted_points_xy, dtype=float)
    if h.shape != (3, 3) or points.shape != (2, 2) or not np.isfinite(h).all() or not np.isfinite(points).all():
        result['reason'] = 'invalid_predicted_geometry'
        return result
    q = np.c_[points, np.ones(2)] @ h.T
    if np.any(np.abs(q[:, 2]) < 1e-8) or q[0, 2] * q[1, 2] <= 0:
        result['reason'] = 'needle_crosses_projective_horizon'
        return result
    p, t = q[:, :2] / q[:, 2, None]
    result.update(rectified_pivot_xy=p.tolist(), rectified_tip_xy=t.tolist())
    if np.linalg.norm(p - calibration['pivot_xy']) > 12:
        result['reason'] = 'pivot_inconsistent_with_reference'
        return result
    v = t - p
    if not 40 < np.linalg.norm(v) < 200:
        result['reason'] = 'implausible_needle_length'
        return result
    # Small manual pivot differences between plates are real label uncertainty.
    # Recompute the static reference tick directions about the predicted,
    # rectified pivot, as in the preview generator; do not substitute a target.
    ticks = np.asarray(calibration['ticks_xy'], dtype=float)
    angles = np.unwrap(np.arctan2(ticks[:, 1] - p[1], ticks[:, 0] - p[0]))
    if not np.all(np.diff(angles) > 0):
        result['reason'] = 'nonmonotonic_reference_tick_angles'
        return result
    theta = math.atan2(v[1], v[0])
    theta += round(((angles[0] + angles[-1]) / 2 - theta) / (2 * math.pi)) * 2 * math.pi
    speeds = np.asarray(calibration['speeds'])
    if theta < angles[0]:
        raw_speed = float(speeds[0] + (theta - angles[0]) * (speeds[1] - speeds[0]) / (angles[1] - angles[0]))
    elif theta > angles[-1]:
        raw_speed = float(speeds[-1] + (theta - angles[-1]) * (speeds[-1] - speeds[-2]) / (angles[-1] - angles[-2]))
    else:
        raw_speed = float(np.interp(theta, angles, speeds))
    result['raw_indicated_speed_kmh'] = raw_speed
    # Physical end-stop/calibration tolerance fixed during development. This
    # policy is explicit, preserves the raw estimate, and changes no image or
    # needle plausibility gates. It is not a model-accuracy improvement.
    numerical_tolerance = 1e-8
    if (raw_speed < speeds[0] - ENDPOINT_TOLERANCE_KMH - numerical_tolerance
            or raw_speed > speeds[-1] + ENDPOINT_TOLERANCE_KMH + numerical_tolerance):
        result['reason'] = 'needle_outside_calibrated_arc'
        return result
    speed = float(np.clip(raw_speed, speeds[0], speeds[-1]))
    clamped = bool(raw_speed < speeds[0] - numerical_tolerance or raw_speed > speeds[-1] + numerical_tolerance)
    result.update(accepted=True, speed_kmh=speed, endpoint_clamped=clamped,
                  reason='accepted_with_endpoint_tolerance' if clamped else 'accepted')
    return result


def verify_dataset(data_dir, real_data_dir=None, include_real_training=False):
    root = Path(data_dir).resolve()
    doc = json.loads((root / 'labels.json').read_text(encoding='utf-8'))
    samples = doc['samples']
    if len(samples) != 500 or Counter(r['split'] for r in samples) != {'train': 400, 'validation': 100}:
        raise ValueError('Expected exactly 400 training and 100 validation synthetic images')
    if Counter(r['source_frame'] for r in samples) != {0: 200, 180: 200, 204: 100}:
        raise ValueError('Expected the approved 200/200/100 source-plate allocation')
    cal = load_calibration(root / 'calibration.json')
    reference = _image(root / 'masked_reference.png')
    hashes = {'labels.json': sha256(root / 'labels.json'), 'calibration.json': sha256(root / 'calibration.json'),
              'masked_reference.png': sha256(root / 'masked_reference.png')}
    partitions = {'train': [], 'validation': [], 'real_diagnostic': []}
    ids, source_splits = set(), {}
    for row in samples:
        if row['id'] in ids:
            raise ValueError('Duplicate synthetic sample ID')
        ids.add(row['id'])
        source, split = row['source_frame'], row['split']
        if (split == 'train' and source not in (0, 180)) or (split == 'validation' and source != 204):
            raise ValueError('Synthetic source-plate split differs from approved partition')
        source_splits.setdefault(source, set()).add(split)
        path = safe_path(root, row['image'])
        digest = sha256(path)
        if digest != row['image_sha256']:
            raise ValueError('Synthetic image checksum mismatch')
        image = _image(path)
        hashes[row['image']] = digest
        for name in ('lcd_mask', 'needle_mask'):
            mask_path = safe_path(root, row[name])
            mask = _image(mask_path)
            if not np.isin(mask, (0, 255)).all():
                raise ValueError('Binary supervision mask required')
            hashes[row[name]] = sha256(mask_path)
            if name == 'lcd_mask' and not np.all(image[mask > 0] == 128):
                raise ValueError('LCD masking differs from the approved128 value')
        points = [_point(row['needle_position_px'][name]) for name in ('pivot', 'tip')]
        norm = [row['needle_position_normalized'][name] for name in ('pivot', 'tip')]
        if not np.allclose(np.asarray(points) / 511, norm, atol=1e-7, rtol=0):
            raise ValueError('Needle normalized labels disagree')
        speed = float(row['speed_kmh'])
        if not math.isfinite(speed) or not 0 <= speed <= 180:
            raise ValueError('Invalid synthetic speed')
        partitions[split].append({'sample_id': 'synthetic:' + row['id'], 'image': str(path),
                                  'points_xy': points, 'speed_kmh': speed, 'speed_valid': True,
                                  'split': split, 'kind': 'synthetic', 'source_frame': source,
                                  'registration': registration_from_record(row), 'calibration': cal,
                                  'label_status': row.get('label_status', 'synthetic_draft_calibration')})
    if any(len(s) != 1 for s in source_splits.values()):
        raise ValueError('Source plate crosses training and validation')
    private_hashes = None
    if include_real_training and not real_data_dir:
        raise ValueError('include_real_training requires explicit private real_data_dir')
    if real_data_dir:
        private = Path(real_data_dir).resolve()
        real_doc = json.loads((private / 'labels.json').read_text(encoding='utf-8'))
        real_rows = real_doc['frames']
        by_frame = {row['frame_index']: row for row in real_rows}
        if len(by_frame) != len(real_rows) or not set(REAL_TRAIN_FRAMES + REAL_DIAGNOSTIC_FRAMES).issubset(by_frame):
            raise ValueError('Private bundle lacks required real frames or contains duplicates')
        private_hashes = {'labels.json': sha256(private / 'labels.json')}
        approval_path = private / 'approval_manifest.json'
        if approval_path.is_file():
            approval = json.loads(approval_path.read_text(encoding='utf-8'))
            if (approval.get('train_frame_indices') != list(REAL_TRAIN_FRAMES)
                    or approval.get('diagnostic_frame_indices') != list(REAL_DIAGNOSTIC_FRAMES)):
                raise ValueError('Private approval manifest differs from the explicit frame partition')
            private_hashes['approval_manifest.json'] = sha256(approval_path)
        for index in sorted(set(REAL_TRAIN_FRAMES + REAL_DIAGNOSTIC_FRAMES)):
            row = by_frame[index]
            path = safe_path(private, row['image'])
            image = _image(path)
            mask_path = safe_path(private, row['lcd_mask'])
            mask = _image(mask_path)
            if not np.isin(mask, (0, 255)).all() or not np.all(image[mask > 0] == 128):
                raise ValueError('Private real LCD mask is invalid')
            private_hashes[row['image']] = sha256(path)
            private_hashes[row['lcd_mask']] = sha256(mask_path)
            points = [_point(row['pivot_xy']), _point(row['tip_xy'])]
            speed = row.get('speed_kmh_calibrated_estimate')
            speed_valid = row.get('confidence') != 'low' and speed is not None and math.isfinite(float(speed)) and 0 <= float(speed) <= 180
            split = 'train' if index in REAL_TRAIN_FRAMES else 'real_diagnostic'
            if split == 'train' and not include_real_training:
                continue
            if split == 'train' and not speed_valid:
                raise ValueError('Approved real training frame has no usable draft speed label')
            partitions[split].append({'sample_id': f'real:{index:04d}', 'image': str(path),
                                      'points_xy': points, 'speed_kmh': float(speed) if speed_valid else None,
                                      'speed_valid': speed_valid, 'split': split, 'kind': 'real',
                                      'source_frame': index, 'confidence': row['confidence'],
                                      'registration': register_image(image, reference, cal), 'calibration': cal,
                                      'label_status': 'manual_draft_indicated_speed_not_road_speed'})
    integrity = {'synthetic_labels_sha256': hashes['labels.json'],
                 'synthetic_content_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                 'private_labels_sha256': None if private_hashes is None else private_hashes['labels.json'],
                 'private_approval_sha256': None if private_hashes is None else private_hashes.get('approval_manifest.json'),
                 'private_content_sha256': None if private_hashes is None else hashlib.sha256(json.dumps(private_hashes, sort_keys=True).encode()).hexdigest(),
                 'counts': {key: len(value) for key, value in partitions.items()},
                 'train_ids': [r['sample_id'] for r in partitions['train']],
                 'validation_ids': [r['sample_id'] for r in partitions['validation']],
                 'diagnostic_ids': [r['sample_id'] for r in partitions['real_diagnostic']],
                 'validation_semantics': 'same_clip_synthetic_base_plate_holdout_shared_donor_pixels_not_independent_real_accuracy',
                 'real_diagnostic_semantics': 'same_clip_held_out_frames_with_approximate_visual_labels_not_independent_road_speed',
                 'source_plate_split': {'train': [0, 180], 'validation': [204]},
                 'calibration_sha256': hashes['calibration.json']}
    return partitions, integrity


class NeedleDataset(Dataset):
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return {'image': preprocess_image(row['image'], 512),
                'points': torch.tensor(row['points_xy'], dtype=torch.float32) / 511,
                'speed': torch.tensor(row['speed_kmh'] if row['speed_valid'] else 0., dtype=torch.float32),
                'speed_valid': torch.tensor(row['speed_valid'], dtype=torch.bool), 'index': int(index)}
