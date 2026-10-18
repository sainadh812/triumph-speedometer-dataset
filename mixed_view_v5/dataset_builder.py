"""Combine reviewed straight/day sources with absolute left/straight/right roll.

Static dataset construction only. No video decoding, model inference, training,
new crops, mirrors, generative edits or source-file modifications are performed.
Each parent produces exactly three full-footprint similarity transforms. All
labels and masks move together; old parent splits and speed targets are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from real_adaptation_v4.geometry import LANDMARK_NAMES, load_calibration, read_speed

SIZE = 512
SEED = 202610180501
PREPROCESSING_ID = 'gray512_fullframe_native_view_lcd128_mask_v5_imagenet_normalized'
AVAILABILITY = 'amodal_in_frame_landmark_availability'
BUCKETS = {'left': (-25., -8.), 'straight': (-4., 4.), 'right': (8., 25.)}
SOURCE_SPECS = {
    'straight_v3': {'root': PROJECT/'real_adaptation_v3/dataset500', 'annotations': 'labels.json',
                    'reference': 'masked_reference.png', 'count': 500, 'recording': 'clip181323'},
    'tilted_v4': {'root': PROJECT/'real_adaptation_v4/dataset2000', 'annotations': 'annotations.json',
                  'reference': 'reference.png', 'count': 2000, 'recording': 'clip174359'},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def project(points, matrix):
    q = np.c_[np.asarray(points, float), np.ones(len(points))] @ np.asarray(matrix).T
    if not np.isfinite(q).all() or np.any(np.abs(q[:, 2]) < 1e-9):
        raise ValueError('Invalid static label transform')
    return q[:, :2]/q[:, 2:]


def roll_degrees(points):
    """Clockwise dial roll of the pivot-to-zero axis relative image-down.

    The reviewed straight V3 zero anchor lies directly below its hub. This
    semantic axis remains usable when the needle covers the physical zero tick.
    Perspective is retained; this is an explicit image-plane roll convention,
    not a recovered 3D camera attitude or a calibrated physical-angle label.
    """
    direction = np.asarray(points[2])-np.asarray(points[0])
    if np.linalg.norm(direction) < 20:
        raise ValueError('Degenerate pivot-to-zero axis')
    return float((math.degrees(math.atan2(direction[1], direction[0]))-90+180)%360-180)


def rotation_to_fit(delta_clockwise):
    theta = math.radians(delta_clockwise)
    scale = (SIZE-4)/(SIZE*(abs(math.cos(theta))+abs(math.sin(theta))))
    h = np.eye(3)
    h[:2] = cv2.getRotationMatrix2D(((SIZE-1)/2, (SIZE-1)/2), -delta_clockwise, scale)
    # Pixel-cell footprint, not merely the centers of the four corner pixels.
    corners = np.array([[-.5, -.5], [511.5, -.5], [511.5, 511.5], [-.5, 511.5]])
    mapped = project(corners, h)
    if mapped.min() < 1.49 or mapped.max() > 509.51 or np.linalg.det(h[:2, :2]) <= 0:
        raise ValueError('Full parent content would be clipped or mirrored')
    return h, float(scale), mapped


def image(path):
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None or data.shape != (SIZE, SIZE) or data.dtype != np.uint8:
        raise ValueError('Expected grayscale512 PNG: '+str(path))
    return data


def validate_mask_points(pixels, mask, points):
    if set(np.unique(mask)) - {0, 255} or np.any(pixels[mask > 0] != 128):
        return False
    points = np.asarray(points)
    if points.shape != (12, 2) or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 511):
        return False
    xy = np.rint(points).astype(int)
    if np.any(mask[xy[:, 1], xy[:, 0]]):
        return False
    needle = np.zeros_like(mask)
    cv2.line(needle, tuple(xy[0]), tuple(xy[1]), 255, 1)
    return not np.any((needle > 0) & (mask > 0))


def parent_records():
    for view, spec in SOURCE_SPECS.items():
        data = json.loads((spec['root']/spec['annotations']).read_text(encoding='utf-8'))
        rows = data['samples']
        if len(rows) != spec['count']:
            raise ValueError('Source dataset count changed: '+view)
        for row in rows:
            if view == 'straight_v3':
                ticks = sorted(row['tick_landmarks'], key=lambda t: t['speed_kmh'])
                if [t['speed_kmh'] for t in ticks] != list(range(0, 181, 20)):
                    raise ValueError('Incomplete source tick labels')
                points = [row['needle_position_px']['pivot'], row['needle_position_px']['tip']]+[t['xy'] for t in ticks]
                parent_id, path, mask = row['id'], row['image'], row['lcd_mask']
                group = row['source_family']
                needle_mask = image(spec['root']/row['needle_mask'])
                tick_xy = np.rint(points[2:]).astype(int)
                observed = [True, True]+[bool(needle_mask[y, x] == 0) for x, y in tick_xy]
            else:
                points = row['points_xy']
                parent_id, path, mask = row['sample_id'], row['image_path'], row['lcd_mask_path']
                group = row['source_group']
                observed = row['observed_visibility']
            p = np.asarray(points, float)
            source_pixels, source_mask = image(spec['root']/path), image(spec['root']/mask)
            if sha(spec['root']/path) != row['image_sha256']:
                raise ValueError('Parent image hash changed: '+parent_id)
            if not validate_mask_points(source_pixels, source_mask, p):
                raise ValueError('Parent geometry/mask validation failed: '+parent_id)
            yield {'source_view': view, 'parent_sample_id': parent_id, 'source_group': group,
                'source_recording_group': spec['recording'], 'source_frame': row['source_frame'],
                'donor_frame': row.get('donor_frame'), 'split': row['split'], 'points': p,
                'speed_kmh': float(row['speed_kmh']), 'observed': observed,
                'parent_image_path': path, 'parent_mask_path': mask,
                'parent_image_sha256': row['image_sha256'], 'parent_mask_sha256': sha(spec['root']/mask),
                'reference_to_parent_h': row['reference_to_image_h'],
                'pixels': source_pixels, 'mask': source_mask}


def decorate(pixels, points, mask, heading):
    panel = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contour, -1, (240, 160, 70), 1)
    xy = np.rint(points).astype(int)
    cv2.line(panel, tuple(xy[0]), tuple(xy[1]), (255, 230, 0), 2, cv2.LINE_AA)
    for index, point in enumerate(xy):
        cv2.circle(panel, tuple(point), 3, (60, 190, 255), 1, cv2.LINE_AA)
        label = ['P', 'N'][index] if index < 2 else str((index-2)*20)
        cv2.putText(panel, label, tuple(point+[4, -3]), cv2.FONT_HERSHEY_SIMPLEX, .32,
                    (60, 190, 255), 1, cv2.LINE_AA)
    panel = cv2.copyMakeBorder(panel, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(panel, heading, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1, cv2.LINE_AA)
    return panel


def _summary(errors):
    x = np.asarray(errors, float)
    return {'count': len(x), 'mean_abs_kmh': float(x.mean()), 'max_abs_kmh': float(x.max()),
            'p95_abs_kmh': float(np.quantile(x, .95))}


def build(output):
    if cv2.__version__ != '4.14.0':
        raise ValueError('Use the recorded OpenCV4.14.0 runtime for reproducible geometry QA')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Originals and existing builds are preserved; choose an unused output directory')
    started = time.perf_counter()
    output.mkdir(parents=True)
    for folder in ['images', 'masks', 'references', 'source_metadata', 'qa']:
        (output/folder).mkdir()
    dump(output/'BUILD_INCOMPLETE.json', {'status': 'incomplete', 'seed': SEED})
    references = []
    source_inputs = {}
    for view, spec in SOURCE_SPECS.items():
        target = output/'references'/view
        target.mkdir()
        shutil.copyfile(spec['root']/spec['reference'], target/'reference.png')
        shutil.copyfile(spec['root']/'calibration.json', target/'calibration.json')
        cal = json.loads((target/'calibration.json').read_text(encoding='utf-8'))
        references.append({'id': view, 'reference_image': f'references/{view}/reference.png',
            'calibration': f'references/{view}/calibration.json', 'lcd_polygon_xy': cal['lcd_polygon_xy'],
            'reference_lcd_already_masked': True})
        metadata = output/'source_metadata'/view
        metadata.mkdir()
        for name in [spec['annotations'], 'calibration.json', 'provenance.json']:
            shutil.copyfile(spec['root']/name, metadata/name)
        source_inputs[view] = {'source_dataset': spec['root'].relative_to(PROJECT).as_posix(),
            'parent_count': spec['count'], 'recording_group': spec['recording'],
            'annotation_sha256': sha(spec['root']/spec['annotations']),
            'calibration_sha256': sha(spec['root']/'calibration.json'),
            'reference_sha256': sha(spec['root']/spec['reference'])}
    shutil.copyfile(SOURCE_SPECS['tilted_v4']['root']/'calibration.json', output/'calibration.json')
    dump(output/'mask_references.json', {'schema_version': 1, 'references': references,
        'use': 'Image-only LCD polygon placement; no canonical input warp or cropping; no target annotation access at inference'})
    preprocessing = {'preprocessing_id': PREPROCESSING_ID, 'color': 'grayscale', 'input_size': [512, 512],
        'lcd_mask_applied': True, 'lcd_mask_value': 128,
        'input_geometry': 'full_frame_square_letterbox_then512_no_crop_no_canonical_warp',
        'camera_stabilization': 'none', 'mask_placement': 'image_only_multi_reference_polygon_projection',
        'mask_references_path': 'mask_references.json', 'canonical_calibration_path': 'calibration.json',
        'landmark_supervision': AVAILABILITY, 'network_normalization': 'grayscale_repeated_RGB_ImageNet_mean_std',
        'training_images': 'Already512px masked full-frame geometric descendants; load as-is without another alignment',
        'geometric_augmentation': 'absolute dial roll left[-25,-8],straight[-4,4],right[8,25] degrees; uniform full-footprint scale to fit',
        'mirroring': False, 'new_cropping': False, 'border_value': 0}
    dump(output/'preprocessing_config.json', preprocessing)
    canonical = load_calibration(output/'calibration.json')
    rows, parent_audit, hashes, contact = [], [], {}, {}
    group_splits, parent_splits = {}, {}
    oracle_errors = defaultdict(list)
    source_before = {view: sha(spec['root']/spec['annotations']) for view, spec in SOURCE_SPECS.items()}
    minimum_footprint_margin, maximum_roll_error = 512., 0.
    candidate_retries = 0
    for parent_index, parent in enumerate(parent_records()):
        view, pid, split = parent['source_view'], parent['parent_sample_id'], parent['split']
        if split not in ('train', 'validation'):
            raise ValueError('Unexpected parent split')
        group_key, parent_key = (view, parent['source_group']), (view, pid)
        if group_key in group_splits and group_splits[group_key] != split:
            raise ValueError('Source group crosses splits')
        if parent_key in parent_splits:
            raise ValueError('Duplicate parent ID')
        group_splits[group_key] = parent_splits[parent_key] = split
        existing_roll = roll_degrees(parent['points'])
        parent_audit.append({key: value for key, value in parent.items()
                            if key not in ('pixels', 'mask', 'points', 'observed')})
        for bucket, bounds in BUCKETS.items():
            seed = int.from_bytes(hashlib.sha256(f'{SEED}|{view}|{pid}|{bucket}'.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            for attempt in range(32):
                target_roll = float(rng.uniform(*bounds))
                delta = (target_roll-existing_roll+180)%360-180
                h, scale, corners = rotation_to_fit(delta)
                points = project(parent['points'], h)
                pixels = cv2.warpPerspective(parent['pixels'], h, (512, 512), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                mask = cv2.warpPerspective(parent['mask'], h, (512, 512), flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                pixels[mask > 0] = 128
                if validate_mask_points(pixels, mask, points):
                    break
            else:
                raise ValueError('Could not preserve source landmark/mask separation: '+pid)
            candidate_retries += attempt
            measured = roll_degrees(points)
            maximum_roll_error = max(maximum_roll_error, abs(measured-target_roll))
            if not bounds[0]-1e-7 <= measured <= bounds[1]+1e-7:
                raise ValueError('Absolute roll bucket incorrect')
            minimum_footprint_margin = min(minimum_footprint_margin, float(corners.min()+.5), float(511.5-corners.max()))
            name = f'v5_{view}_{pid}_{bucket}'
            image_path, mask_path = f'images/{name}.png', f'masks/{name}.png'
            for relative, content in [(image_path, pixels), (mask_path, mask)]:
                if not cv2.imwrite(str(output/relative), content, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise IOError('PNG write failed: '+relative)
                hashes[relative] = sha(output/relative)
            total_h = h @ np.asarray(parent['reference_to_parent_h'])
            row = {'sample_id': name, 'image_path': image_path, 'mask_path': mask_path,
                'lcd_mask_path': mask_path, 'split': split, 'parent_split': split,
                'points_xy': points.tolist(), 'visibility': [True]*12,
                'observed_visibility': list(parent['observed']),
                'visibility_semantics': 'amodal_label_available_and_inside_frame; observed_visibility separately estimates original needle occlusion',
                'speed_kmh': parent['speed_kmh'], 'speed_valid': True,
                'source_view': view, 'roll_bucket': bucket, 'parent_sample_id': pid,
                'source_group': parent['source_group'], 'source_recording_group': parent['source_recording_group'],
                'source_frame': parent['source_frame'], 'donor_frame': parent['donor_frame'],
                'source_calibration_id': view, 'parent_image_sha256': parent['parent_image_sha256'],
                'parent_mask_sha256': parent['parent_mask_sha256'], 'image_sha256': hashes[image_path],
                'mask_sha256': hashes[mask_path], 'parent_to_child_h': h.tolist(), 'total_H': total_h.tolist(),
                'reference_to_image_h': total_h.tolist(), 'existing_parent_roll_degrees_clockwise': existing_roll,
                'roll_target_degrees_clockwise': target_roll, 'measured_roll_degrees_clockwise': measured,
                'rotation_delta_degrees_clockwise': delta, 'full_footprint_scale': scale,
                'parent_pixel_cell_footprint_in_child_xy': corners.tolist(), 'seed': seed,
                'geometry': {'mirrored': False, 'cropped': False, 'uniform_similarity_only': True,
                             'sampling_attempts': attempt+1, 'rotation_degrees_clockwise': delta, 'scale': scale},
                'label_status': 'unchanged_parent_synthetic_speed_and_transformed_geometry_relative_to_manual_draft_calibration'}
            rows.append(row)
            reading = read_speed(points, visibility=np.ones(12), calibration=canonical)
            if not reading['accepted']:
                raise ValueError('Static known-label geometry rejected: '+name+' '+reading['reason'])
            oracle_errors[view].append(abs(reading['speed_kmh']-row['speed_kmh']))
            # Every source group gets its own three-direction visual QA row.
            ck = (view, split, parent['source_group'])
            if ck not in contact:
                contact[ck] = {'parent': pid, 'panels': {}, 'inputs': {}}
            if contact[ck]['parent'] == pid:
                heading = f'{view} {split} {bucket} roll{measured:+.1f} speed{row["speed_kmh"]:.1f}'
                contact[ck]['panels'][bucket] = decorate(pixels, points, mask, heading)
                raw = cv2.copyMakeBorder(cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR), 36, 0, 0, 0,
                                        cv2.BORDER_CONSTANT, value=(24, 24, 24))
                cv2.putText(raw, heading, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1, cv2.LINE_AA)
                contact[ck]['inputs'][bucket] = raw
        if (parent_index+1)%100 == 0:
            print(f'Built {len(rows)}/7500 images from {parent_index+1}/2500 immutable parents', flush=True)
    counts = Counter(row['split'] for row in rows)
    if len(rows) != 7500 or counts != {'train': 6000, 'validation': 1500}:
        raise ValueError('Unexpected final counts')
    for view, spec in SOURCE_SPECS.items():
        if sha(spec['root']/spec['annotations']) != source_before[view]:
            raise ValueError('Source annotations changed during build')
    for index, (key, item) in enumerate(sorted(contact.items())):
        name = '_'.join(key)
        cv2.imwrite(str(output/'qa'/f'labels_{name}.jpg'), np.hstack([item['panels'][b] for b in BUCKETS]))
        cv2.imwrite(str(output/'qa'/f'inputs_{name}.jpg'), np.hstack([item['inputs'][b] for b in BUCKETS]))
    annotations = {'schema_version': 1, 'keypoint_names': list(LANDMARK_NAMES),
        'landmark_supervision': AVAILABILITY, 'coordinate_system': '512x512 pixel centers xright ydown; normalize by511',
        'samples': rows}
    dump(output/'annotations.json', annotations)
    dump(output/'source_metadata/parents.json', {'parents': parent_audit,
        'note': 'Original files remain immutable in their existing datasets; source hashes identify every parent.'})
    oracle = {'scope': 'STATIC synthetic label-only geometry checks, not model/video inference or measured accuracy',
        'canonical_calibration_source': 'tilted_v4', 'canonical_calibration_sha256': sha(output/'calibration.json'),
        'accepted': len(rows), 'rejected': 0, 'by_source_view': {v: _summary(e) for v, e in oracle_errors.items()},
        'interpretation': 'Small cross-calibration differences are retained and disclosed; original requested speed labels are unchanged.'}
    dump(output/'qa/label_geometry_oracle.json', oracle)
    strata = Counter((r['split'], r['source_view'], r['roll_bucket']) for r in rows)
    report = {'status': 'built_and_static_checks_passed', 'count': len(rows), 'split_counts': dict(counts),
        'original_parent_count': len(parent_audit), 'new_descendants_per_parent': 3,
        'strata_counts': [{'split': k[0], 'source_view': k[1], 'roll_bucket': k[2], 'count': v}
                          for k, v in sorted(strata.items())],
        'sampler_requirement': 'Equal training probability for each of6 source_view x roll_bucket strata; validation retains every image and reports each view/roll separately',
        'roll_definition': 'Clockwise pivot-to-tick0 axis relative image-down; negativeleft/positiveright. Perspective retained. Not3Dcameraattitude.',
        'roll_buckets_degrees': BUCKETS, 'maximum_absolute_roll_target_error_deg': maximum_roll_error,
        'minimum_full_parent_pixel_footprint_margin_px': minimum_footprint_margin,
        'geometry_sampling_retries': candidate_retries,
        'source_inputs': source_inputs, 'parent_split_preserved': True, 'source_group_split_disjoint': True,
        'all_12_landmarks_finite_inside_frame_and_outside_lcd': True,
        'all_needle_segments_outside_lcd': True, 'all_masks_binary_and_masked_pixels_exact128': True,
        'no_new_crop_or_mirror': True, 'no_photometric_change_beyond_resampling_and_mask_reapplication': True,
        'source_images_modified': False, 'video_testing_performed': False, 'model_inference_performed': False,
        'label_geometry_oracle': oracle, 'opencv_version': cv2.__version__, 'numpy_version': np.__version__,
        'builder_sha256': sha(__file__), 'elapsed_seconds': time.perf_counter()-started,
        'limitation': 'Two existing development recordings. Synthetic source/parent holdouts are not independent real-camera accuracy. Straight clip181323 and dayclip174359 are now both source recordings, including shared donor/calibration dependence.'}
    dump(output/'qa/checks.json', report)
    provenance = {'sources': source_inputs, 'seed': SEED, 'builder_sha256': sha(__file__),
        'appearance': 'Existing source images preserved; deterministic full-frame rotations/scale plusLCDmask reapplication only',
        'source_calibration_uncertainty': 'Manual approximate2px; V3zero approximately3px and partlyinferred. Not measured statistical confidence.',
        'split_limit': report['limitation'], 'new_imagegen_calls': 0, 'new_video_frames_extracted': 0}
    dump(output/'provenance.json', provenance)
    (output/'README.md').write_text('''# V5 mixed straight and tilted dashboard dataset

7,500 grayscale512x512 synthetic images:6,000 training and1,500 validation.
The500 V3 straight-view parents and2,000 V4 daytime parents each produce left,
straight and right descendants. Original parent splits, source groups, synthetic
speed targets and source files are preserved. Equal sampling across the six
view-by-roll strata is required to prevent the larger daytime set dominating.

Roll means the clockwise angle of hub-to-zero tick relative to image-down:
left[-25,-8], straight[-4,4], right[8,25] degrees. Each parent's existing angle
is removed before choosing that absolute target. Uniform scaling fits the full
original512-square pixel footprint; no new cropping or mirroring occurs.
Original appearance/perspective remains except unavoidable resampling. The same
transform moves all12 labels and the binary LCD mask; masked pixels are128.

Landmarks are pivot, needle tip, then tick0 through180 in20km/h steps.
Availability is amodal in-frame label availability; observed_visibility is
separate. Reference/render transforms are provenance for training-label QA only,
never permitted inputs to real inference. Read512px training images as-is.
For real video, full-frame square letterboxing/resizing is retained; image-only
reference matching may project LCD polygons but may not warp/crop the input.
mask_references.json and its protected images/calibrations are inference assets.

The canonical speed geometry is the reviewed V4 calibration. The small retained
V3-to-V4 calibration discrepancy is quantified in qa/label_geometry_oracle.json.
Calibration is still a manual draft. Both recordings are development sources;
retesting either is not independent real accuracy. No local video/model inference
or training was performed by this builder. Video testing belongs in Colab.
''', encoding='utf-8')
    (output/'BUILD_INCOMPLETE.json').unlink()
    for path in sorted(output.rglob('*')):
        if path.is_file():
            relative = path.relative_to(output).as_posix()
            if relative not in hashes:
                hashes[relative] = sha(path)
    manifest = {'dataset': 'triumph_mixed_view_landmarks_v5', 'schema_version': 1, 'count': len(rows),
        'split_counts': dict(counts), 'landmark_supervision': AVAILABILITY,
        'preprocessing': preprocessing, 'source_views': list(SOURCE_SPECS), 'roll_buckets': list(BUCKETS),
        'sha256': hashes}
    dump(output/'manifest.json', manifest)
    report.update(manifest_sha256=sha(output/'manifest.json'), annotation_sha256=sha(output/'annotations.json'),
                  protected_file_count=len(hashes), total_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
    dump(ROOT/'DATASET_REPORT.json', report)
    print(json.dumps({k: report[k] for k in ['status', 'count', 'split_counts', 'manifest_sha256', 'annotation_sha256',
                                           'protected_file_count', 'total_bytes', 'elapsed_seconds']}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dataset')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    build(args.output)
