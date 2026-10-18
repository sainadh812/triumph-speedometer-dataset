"""Validated exported crops; no online geometry or unreviewed pseudo-labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

POINT_NAMES = {
    'localizer': ['lcd_tl', 'lcd_tr', 'lcd_br', 'lcd_bl',
                  'rpm_roi_tl', 'rpm_roi_tr', 'rpm_roi_br', 'rpm_roi_bl'],
    'speed': ['pivot', 'tip'] + [f'tick_{v}' for v in range(0, 181, 20)],
    'rpm': [f'rpm_{v}' for v in range(0, 13, 2)] + ['bar_end'],
}
DEFAULT_HW = {'localizer': (512, 512), 'speed': (512, 512), 'rpm': (384, 256)}
PREPROCESSING_ID = 'exported_gray3_imagenet_normalized_no_online_geometry_v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError('Missing or escaping dataset file: ' + str(relative))
    return candidate


def _hash_rows(manifest):
    files = manifest.get('files', manifest.get('file_sha256', manifest.get('hashes')))
    if isinstance(files, dict):
        rows = [{'path': path, 'sha256': value if isinstance(value, str) else value['sha256']}
                for path, value in files.items()]
    elif isinstance(files, list):
        rows = files
    else:
        raise ValueError('dataset_manifest.json needs a files list or path-to-SHA256 map')
    result = {}
    for row in rows:
        path, digest = row['path'], row['sha256']
        if path in result or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError('Duplicate path or invalid dataset hash')
        result[path] = digest.lower()
    return result


def validate_dataset(dataset_root):
    root = Path(dataset_root).resolve()
    manifest_path = root / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    schema_version = manifest.get('schema_version', 1)
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise ValueError('Unsupported dataset schema_version; expected 1 or 2')
    hashes = _hash_rows(manifest)
    if not hashes:
        raise ValueError('Empty dataset hash manifest')
    for relative, expected in hashes.items():
        if sha256(safe_path(root, relative)) != expected:
            raise ValueError('Dataset SHA256 mismatch: ' + relative)
    tasks, global_splits = {}, {}
    for task in POINT_NAMES:
        task_root = root / task
        samples_path = safe_path(task_root, 'samples.json')
        if task + '/samples.json' not in hashes:
            raise ValueError('Task sample list is not hash protected: ' + task)
        document = json.loads(samples_path.read_text(encoding='utf8'))
        rows = document if isinstance(document, list) else document['samples']
        if not rows:
            raise ValueError('Empty task: ' + task)
        dimensions = {(int(row['height']), int(row['width'])) for row in rows}
        if len(dimensions) != 1:
            raise ValueError('Task crops need one declared input shape')
        shape = dimensions.pop()
        task_manifest = manifest.get('tasks', {}).get(task, {})
        declared_size = task_manifest.get('input_size')
        document_size = document.get('input_size') if isinstance(document, dict) else None
        for size in (declared_size, document_size):
            if size is not None and list(size) != [shape[1], shape[0]]:
                raise ValueError('Manifest input_size must match exported width,height')
        names = document.get('point_names', POINT_NAMES[task]) if isinstance(document, dict) else POINT_NAMES[task]
        if len(names) != len(POINT_NAMES[task]) or len(set(names)) != len(names):
            raise ValueError('Invalid semantic point name inventory')
        if list(names) != POINT_NAMES[task]:
            raise ValueError('Task semantic point order differs from the model contract')
        if task_manifest.get('point_names', names) != names:
            raise ValueError('Manifest and task semantic point order differ')
        if min(shape) < 32 or any(value % 32 for value in shape):
            raise ValueError('Input dimensions must be positive multiples of32')
        ids = set()
        counts = {'train': 0, 'val': 0, 'test': 0}
        for row in rows:
            if schema_version == 2 and 'visibility_known' not in row:
                raise ValueError('Schema 2 requires visibility_known on every sample')
            if row['id'] in ids:
                raise ValueError('Duplicate sample ID')
            ids.add(row['id'])
            if row['split'] == 'validation':
                row['split'] = 'val'
            if row['split'] not in counts:
                raise ValueError('Unknown split: ' + row['split'])
            counts[row['split']] += 1
            previous = global_splits.setdefault(row['source_id'], row['split'])
            if previous != row['split']:
                raise ValueError('A source and its descendants cross dataset splits')
            points, visible = validate_points(row, len(POINT_NAMES[task]), shape)
            for key in ('image', 'lcd_mask') if task == 'localizer' else ('image',):
                relative = task + '/' + row[key]
                if relative not in hashes:
                    raise ValueError('Unprotected training asset: ' + relative)
                image = cv2.imread(str(safe_path(task_root, row[key])), cv2.IMREAD_UNCHANGED)
                if image is None or image.shape != shape or image.dtype != np.uint8:
                    raise ValueError('Expected exported uint8 grayscale input/mask at declared dimensions')
                if key == 'lcd_mask' and not np.isin(image, [0, 255]).all():
                    raise ValueError('LCD mask must contain only0/255')
            if task == 'localizer':
                box = np.asarray(row['dashboard_box'], float)
                if box.shape != (4,) or not np.isfinite(box).all() or row['present'] not in (0, 1):
                    raise ValueError('Invalid dashboard box/presence')
                if row['present']:
                    if not (0 <= box[0] < box[2] <= shape[1] and 0 <= box[1] < box[3] <= shape[0]):
                        raise ValueError('Dashboard box outside image or degenerate')
                elif visible.any() or image.any():
                    raise ValueError('Absent dashboard examples require unavailable localizer points and empty LCD mask')
        if not all(counts.values()):
            raise ValueError('Each task requires nonempty train, val and test splits')
        if task_manifest.get('counts', counts) != counts:
            raise ValueError('Manifest split counts do not match samples')
        tasks[task] = {'rows': rows, 'input_hw': shape, 'counts': counts,
                       'samples_sha256': sha256(samples_path), 'point_names': list(names)}
    return {'root': root, 'manifest_sha256': sha256(manifest_path), 'tasks': tasks,
            'preprocessing_id': PREPROCESSING_ID}


def validate_points(row, count, input_hw):
    points = np.asarray(row['points'], dtype=np.float32)
    visible = np.asarray(row['visible'], dtype=bool)
    if points.shape != (count, 2) or visible.shape != (count,) or not np.isfinite(points).all():
        raise ValueError('Invalid landmark arrays')
    if np.any(points[~visible] != 0):
        raise ValueError('Unavailable points must be[0,0]')
    validate_visibility_known(row, count, visible)
    if visible.any() and (np.any(points[visible] < 0) or
                         np.any(points[visible] > [input_hw[1] - 1, input_hw[0] - 1])):
        raise ValueError('Visible landmark outside image')
    return points, visible


def validate_visibility_known(row, count, visible):
    """Unknown labels are neither visibility positives nor negatives.

    Older archives omit this optional field and retain their original complete
    visibility supervision. New archives must supply actual Boolean entries.
    """
    known = np.asarray(row.get('visibility_known', [True] * count))
    if known.shape != (count,) or known.dtype != np.bool_:
        raise ValueError('visibility_known must be one Boolean per landmark')
    if np.any(np.asarray(visible, bool) & ~known):
        raise ValueError('A visible landmark must have known visibility')
    return known


def normalize_gray(image):
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('Expected uint8 grayscale image')
    rgb = np.repeat(image[None], 3, axis=0).astype(np.float32) / 255.
    return (rgb - np.asarray([.485, .456, .406], np.float32)[:, None, None]) / np.asarray([.229, .224, .225], np.float32)[:, None, None]


class TaskDataset:
    """Duck-typed PyTorch Dataset; importing this file does not import torch."""
    def __init__(self, validated, task, split):
        self.root = validated['root'] / task
        self.task, self.input_hw = task, validated['tasks'][task]['input_hw']
        self.rows = [row for row in validated['tasks'][task]['rows'] if row['split'] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(str(self.root / row['image']), cv2.IMREAD_GRAYSCALE)
        points, visible = validate_points(row, len(POINT_NAMES[self.task]), self.input_hw)
        known = validate_visibility_known(row, len(POINT_NAMES[self.task]), visible)
        result = {'image': normalize_gray(image), 'points': points, 'visible': visible.astype(np.float32),
                  'visibility_known': known.astype(np.float32),
                  'id': row['id'], 'source_id': row['source_id'],
                  'lighting': row.get('lighting') if row.get('lighting') in ('day', 'night') else 'unknown'}
        if self.task == 'localizer':
            mask = cv2.imread(str(self.root / row['lcd_mask']), cv2.IMREAD_GRAYSCALE)
            h, w = self.input_hw
            result.update(mask=mask[None].astype(np.float32) / 255., present=np.float32(row['present']),
                          box=np.asarray(row['dashboard_box'], np.float32) / [w, h, w, h])
            result['box'] = result['box'].astype(np.float32)
        return result
