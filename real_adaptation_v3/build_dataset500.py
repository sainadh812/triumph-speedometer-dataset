"""Expand the approved real-background preview to 500 reproducible examples.

This module has no import-time file mutations. It reuses the approved rendering
helpers and plates without editing preview20. Train/validation are separated by
entire source plate. Validation is a same-recording synthetic diagnostic only.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from registration_image import ImageOnlyRegistrar
from generate_preview import (
    angle_table, appearance, draw_overlay, estimate_speed, font, norm, project,
)


ROOT = Path(__file__).resolve().parent
SEED = 2026101805
SIZE = 512
PROFILES = ['clean', 'dim', 'reflection', 'shadow', 'blur', 'motion', 'compression', 'bright']
SOURCE_PLAN = [(0, 'train', 200), (180, 'train', 200), (204, 'validation', 100)]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, image):
    assert cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6]), path


def speed_values(count, rng):
    """Stratify the full range, retaining anchors and fractional readings."""
    values = np.linspace(0., 180., count)
    values[1:-1] += rng.uniform(-.30, .30, count - 2) * (180 / (count - 1))
    for anchor in range(0, 181, 20):
        values[np.argmin(abs(values - anchor))] = float(anchor)
    values = np.round(np.sort(values), 2)
    assert values[0] == 0 and values[-1] == 180 and np.all(np.diff(values) > 0)
    return values


def camera_transform(rng, pivot, framing_profile):
    angle = float(rng.uniform(-3., 3.))
    scale = float(rng.uniform(.985, 1.015))
    shift_limit = 8. if framing_profile == 'near_reference' else 64.
    delta = rng.uniform(-shift_limit, shift_limit, 2)
    matrix = np.eye(3)
    matrix[:2] = cv2.getRotationMatrix2D(tuple(np.asarray(pivot, float)), angle, scale)
    matrix[:2, 2] += delta
    offsets = rng.uniform(-1.5, 1.5, (4, 2)).astype(np.float32)
    corners = np.array([[0, 0], [511, 0], [511, 511], [0, 511]], np.float32)
    matrix = cv2.getPerspectiveTransform(corners, corners + offsets) @ matrix
    return matrix, {
        'roll_degrees': angle, 'translation_xy_px': delta.tolist(), 'scale': scale,
        'framing_profile': framing_profile, 'translation_limit_px': shift_limit,
        'perspective_corner_offsets_px': offsets.tolist(),
    }


def check_outputs(out, rows, plate_rows, preview_hashes, recovered_errors):
    counts = Counter(row['split'] for row in rows)
    sources = {split: sorted({r['source_frame'] for r in rows if r['split'] == split})
               for split in ['train', 'validation']}
    seen = set()
    endpoint_distances = []
    registration_errors = []
    registration_status = Counter()
    profile_counts = Counter()
    split_profile_counts = {s: Counter() for s in sources}
    framing_counts = Counter()
    for row in rows:
        image = cv2.imread(str(out / row['image']), cv2.IMREAD_UNCHANGED)
        needle = cv2.imread(str(out / row['needle_mask']), 0)
        lcd = cv2.imread(str(out / row['lcd_mask']), 0)
        assert image.ndim == 2 and image.shape == (512, 512)
        assert image.dtype == np.uint8 and needle.shape == image.shape and lcd.shape == image.shape
        assert set(np.unique(needle)).issubset({0, 255})
        assert set(np.unique(lcd)).issubset({0, 255})
        assert np.all(image[lcd > 0] == 128), row['id']
        assert not np.any((lcd > 0) & (needle > 0)), row['id']
        digest = sha(out / row['image'])
        assert digest == row['image_sha256'] and digest not in seen and digest not in preview_hashes
        seen.add(digest)
        points = np.array([row['needle_position_px']['pivot'], row['needle_position_px']['tip']])
        assert np.all((points >= 0) & (points <= 511)), row['id']
        for point in points:
            assert lcd[tuple(np.rint(point).astype(int)[::-1])] == 0, row['id']
        mask_y, mask_x = np.nonzero(needle)
        assert len(mask_x) > 0
        for point in points:
            distance = float(np.min(np.hypot(mask_x - point[0], mask_y - point[1])))
            endpoint_distances.append(distance)
            assert distance <= 2.25, (row['id'], distance)
        camera = np.array(row['reference_to_image_h'])
        plate = plate_rows[row['source_frame']]
        p = np.array(plate['pivot_xy'])
        angle = math.radians(row['needle_angle_degrees_reference_unwrapped'])
        tip = p + row['needle_radius_reference_px'] * np.array([math.cos(angle), math.sin(angle)])
        assert np.max(abs(project([p, tip], camera) - points)) < 1e-8
        ticks = np.array([t['xy'] for t in row['tick_landmarks']])
        assert np.all((ticks >= 0) & (ticks <= 511))
        for point in ticks:
            assert lcd[tuple(np.rint(point).astype(int)[::-1])] == 0, row['id']
        registration_status[row['image_registration']['status']] += 1
        # This diagnostic compares a feature-estimated H with known geometry.
        # It is never fed back into estimation or substituted as a fallback.
        if row['image_registration']['status'] == 'accepted':
            estimated = np.array(row['estimated_to_reference_h'])
            recovered = project(ticks, estimated)
            expected = project(ticks, np.linalg.inv(camera))
            registration_errors.extend(np.linalg.norm(recovered - expected, axis=1).tolist())
        profile_counts[row['appearance']['profile']] += 1
        split_profile_counts[row['split']][row['appearance']['profile']] += 1
        framing_counts[row['geometry']['framing_profile']] += 1
    assert counts == {'train': 400, 'validation': 100} and len(rows) == 500
    assert sources == {'train': [0, 180], 'validation': [204]}
    assert len(set(sources['train']) & set(sources['validation'])) == 0
    assert max(recovered_errors) < 1e-8
    assert framing_counts == {'near_reference': 375, 'observed_framing_shift': 125}
    return {
        'status': 'passed', 'image_count': 500, 'splits': dict(counts),
        'source_frames_by_split': sources, 'no_source_plate_crosses_splits': True,
        'duplicate_image_count': 0, 'exact_approved_preview_duplicates': 0,
        'all_images_grayscale512_uint8': True, 'all_lcd_pixels_exact128': True,
        'all_masks_binary512': True, 'no_needle_or_required_landmark_inside_lcd_mask': True,
        'all_point_transforms_consistent': True,
        'max_endpoint_to_needle_mask_distance_px': max(endpoint_distances),
        'render_speed_roundtrip_max_error_kmh': max(recovered_errors),
        'profile_counts': dict(profile_counts),
        'profile_counts_by_split': {k: dict(v) for k, v in split_profile_counts.items()},
        'framing_profile_counts': dict(framing_counts),
        'estimated_registration_counts': dict(registration_status),
        'estimated_registration_vs_render_geometry_diagnostic': {
            'mean_tick_error_px': float(np.mean(registration_errors)) if registration_errors else None,
            'p95_tick_error_px': float(np.percentile(registration_errors, 95)) if registration_errors else None,
            'max_tick_error_px': max(registration_errors) if registration_errors else None,
            'note': 'Evaluation of image-only registration; rendering H not used by its estimator.',
        },
        'source_repairs_preserve_outside_mask': all(p['outside_repair_pixels_identical'] for p in plate_rows.values()),
        'calibration_is_visual_draft': True, 'independent_test_set': False,
        'training_performed': False, 'uploaded': False,
    }


def contact_sheets(out, rows):
    selected = []
    for source_ix, (source, _, _) in enumerate(SOURCE_PLAN):
        for profile_ix, profile in enumerate(PROFILES):
            candidates = [r for r in rows if r['source_frame'] == source and r['appearance']['profile'] == profile]
            desired = ((profile_ix * 24 + source_ix * 11) % 181)
            selected.append(min(candidates, key=lambda r: abs(r['speed_kmh'] - desired)))
    for labeled in [False, True]:
        sheet = Image.new('RGB', (6 * 256, 4 * 284), (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(selected):
            image = cv2.imread(str(out / row['image']), 0)
            if labeled:
                points = row['needle_position_px']
                tile = draw_overlay(image, points['pivot'], points['tip'],
                                    [(t['speed_kmh'], t['xy']) for t in row['tick_landmarks']],
                                    row['speed_kmh'], f"{row['speed_kmh']:.2f} km/h | {row['split']}")
            else:
                tile = Image.fromarray(image).convert('RGB')
            x, y = index % 6 * 256, index // 6 * 284
            sheet.paste(tile.resize((256, 256)), (x, y))
            draw.text((x + 4, y + 258),
                      f"{row['speed_kmh']:.2f} | {row['appearance']['profile']} | src{row['source_frame']}",
                      font=font(13), fill='white')
        sheet.save(out / 'qa' / ('contact_labels.jpg' if labeled else 'contact_inputs.jpg'), quality=94)
    dump(out / 'qa' / 'contact_samples.json', [r['id'] for r in selected])


def main():
    cv2.setNumThreads(1)
    out = ROOT / 'dataset500'
    for name in ['images', 'needle_masks', 'lcd_masks', 'qa']:
        (out / name).mkdir(parents=True, exist_ok=True)
    calibration_path = ROOT / 'annotations/reference_ticks.json'
    cal = json.loads(calibration_path.read_text())
    provenance = json.loads((ROOT / 'sources/provenance.json').read_text())
    plate_rows = {r['frame_index']: r for r in provenance['plates']}
    ref_pivot = np.array(cal['pivot_xy'], float)
    ref_ticks = np.array([t['xy'] for t in cal['ticks']], float)
    tick_speeds = [t['speed_kmh'] for t in cal['ticks']]
    raw_alpha = np.load(ROOT / 'sources/needle_alpha.npy')
    base_lcd = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(base_lcd, [np.rint(cal['lcd_polygon_xy']).astype(np.int32)], 255)
    base_lcd = cv2.dilate(base_lcd, np.ones((5, 5), np.uint8))
    reference = cv2.imread(str(ROOT / plate_rows[0]['plate']), 0)
    reference[base_lcd > 0] = 128
    save(out / 'masked_reference.png', reference)
    registrar = ImageOnlyRegistrar(reference)
    rows, roundtrip_errors = [], []
    for source, split, count in SOURCE_PLAN:
        source_rng = np.random.default_rng(SEED + source)
        values = speed_values(count, source_rng)
        profiles = np.resize(np.array(PROFILES), count)
        source_rng.shuffle(profiles)
        framings = np.array(['near_reference'] * (count * 3 // 4)
                            + ['observed_framing_shift'] * (count // 4))
        source_rng.shuffle(framings)
        plate_info = plate_rows[source]
        plate = cv2.imread(str(ROOT / plate_info['plate']), 0).astype(float)
        pivot = np.array(plate_info['pivot_xy'], float)
        angles, radii = angle_table(pivot, ref_ticks)
        assert np.all(np.diff(angles) > 0)
        for ordinal, (speed, profile, framing) in enumerate(zip(values, profiles, framings)):
            index = len(rows)
            sample_seed = SEED + source * 10000 + ordinal
            rng = np.random.default_rng(sample_seed)
            theta = float(np.interp(speed, tick_speeds, angles))
            radius = float(np.interp(speed, tick_speeds, radii) + 4.)
            target_tip = pivot + radius * np.array([math.cos(theta), math.sin(theta)])
            turn = theta - math.pi / 2
            scale = radius / 115.
            c, s = math.cos(turn) * scale, math.sin(turn) * scale
            needle_h = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]], float)
            needle_h[:2, 2] = pivot - needle_h[:2, :2] @ ref_pivot
            alpha = cv2.warpPerspective(raw_alpha, needle_h, (512, 512), flags=cv2.INTER_LINEAR)
            alpha = np.clip(alpha * (1.2 if source else 1.), 0., 1.)
            image = np.rint(plate * (1 - alpha) + 255 * alpha).clip(0, 255).astype(np.uint8)
            image[base_lcd > 0] = 128
            camera, geometry = camera_transform(rng, ref_pivot, str(framing))
            image = cv2.warpPerspective(image, camera, (512, 512), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            lcd = cv2.warpPerspective(base_lcd, camera, (512, 512), flags=cv2.INTER_NEAREST)
            needle_mask = cv2.warpPerspective((alpha > .14).astype(np.uint8) * 255,
                                              camera, (512, 512), flags=cv2.INTER_NEAREST)
            needle_mask[lcd > 0] = 0
            image, photo = appearance(image, rng, str(profile))
            image[lcd > 0] = 128
            points = project([pivot, target_tip], camera)
            ticks = project(ref_ticks, camera)
            h_estimate, registration = registrar.estimate(image, sample_seed)
            recovered = estimate_speed(pivot, target_tip, ref_ticks, tick_speeds)
            roundtrip_errors.append(abs(recovered - speed))
            name = f'v3_{index + 1:04d}_p{source:04d}'
            row = {
                'id': name, 'image': f'images/{name}.png', 'speed_kmh': float(speed),
                'speed_label_kind': 'requested_synthetic_indicated_speed',
                'label_status': 'render_geometry_exact_relative_to_draft_real_tick_calibration',
                'source_frame': source, 'source_family': f'clip181323_plate_{source:04d}',
                'source_recording_group': 'clip181323', 'split': split,
                'evaluation_scope': 'same_video_source_plate_holdout_diagnostic' if split == 'validation' else 'development_training',
                'width': 512, 'height': 512, 'grayscale': True,
                'needle_position_px': {'pivot': points[0].tolist(), 'tip': points[1].tolist()},
                'needle_position_normalized': {'pivot': norm(points[0]), 'tip': norm(points[1])},
                'needle_angle_degrees_image_clockwise_from_right': float(math.degrees(math.atan2(points[1, 1] - points[0, 1], points[1, 0] - points[0, 0])) % 360),
                'needle_angle_degrees_reference_unwrapped': float(math.degrees(theta)),
                'needle_radius_reference_px': radius,
                'needle_tip_radius_rule': 'interpolated observed tick radius plus4px; approved preview approximation, not metrology',
                'tick_landmarks': [{'speed_kmh': v, 'xy': point.tolist(), 'calibration_quality': item['quality']}
                                   for v, point, item in zip(tick_speeds, ticks, cal['ticks'])],
                'needle_mask': f'needle_masks/{name}.png', 'lcd_mask': f'lcd_masks/{name}.png',
                'lcd_polygon_xy': project(cal['lcd_polygon_xy'], camera).tolist(),
                'lcd_gray_value': 128, 'reference_to_image_h': camera.tolist(),
                'source_needle_to_reference_h': needle_h.tolist(),
                'estimated_to_reference_h': h_estimate, 'image_registration': registration,
                'geometry': geometry, 'appearance': photo, 'seed': sample_seed,
                'calibration_sha256': sha(calibration_path),
                'source_plate_sha256': sha(ROOT / plate_info['plate']),
                'source_needle_alpha_sha256': sha(ROOT / 'sources/needle_alpha.npy'),
            }
            save(out / row['image'], image)
            save(out / row['needle_mask'], needle_mask)
            save(out / row['lcd_mask'], lcd)
            row['image_sha256'] = sha(out / row['image'])
            rows.append(row)
            if len(rows) % 50 == 0:
                print(f'Generated {len(rows)}/500', flush=True)
    labels = {
        'dataset': 'real_dashboard_lcd_mask_dataset500_v3',
        'status': 'development_dataset_derived_from_user_approved_preview20',
        'coordinate_system': '512x512 pixel centers, x right y down; normalized by511',
        'calibration_uncertainty': 'Manual real tick/pivot visual drafts: about2px visually estimated; zero about3px. These are judgments, not measured error bars. Rendering roundtrip does not establish real accuracy.',
        'split_policy': {'train': {'source_frames': [0, 180], 'count': 400},
                         'validation': {'source_frames': [204], 'count': 100},
                         'test': None,
                         'warning': 'Source plates are from the same recording and share repair donors/calibration. Validation is a limited synthetic source-holdout diagnostic, not independent real-world accuracy.'},
        'samples': rows,
    }
    dump(out / 'labels.json', labels)
    with (out / 'labels.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'image', 'split', 'speed_kmh', 'pivot_x', 'pivot_y', 'tip_x', 'tip_y', 'source_frame', 'appearance'])
        for row in rows:
            writer.writerow([row['id'], row['image'], row['split'], row['speed_kmh'],
                             *row['needle_position_px']['pivot'], *row['needle_position_px']['tip'],
                             row['source_frame'], row['appearance']['profile']])
    preview_rows = json.loads((ROOT / 'preview20/labels.json').read_text())['samples']
    preview_hashes = {r['image_sha256'] for r in preview_rows}
    checks = check_outputs(out, rows, plate_rows, preview_hashes, roundtrip_errors)
    dump(out / 'qa/checks.json', checks)
    contact_sheets(out, rows)
    shutil.copyfile(calibration_path, out / 'calibration.json')
    source_provenance = {
        'status': '500_sample_expansion_after_user_approved_preview20', 'seed': SEED,
        'source_preparation': provenance,
        'generator_file': 'build_dataset500.py', 'generator_sha256': sha(Path(__file__)),
        'approved_rendering_helpers_file': 'generate_preview.py',
        'approved_rendering_helpers_sha256': sha(ROOT / 'generate_preview.py'),
        'image_registration_helper_file': 'registration_image.py',
        'image_registration_helper_sha256': sha(ROOT / 'registration_image.py'),
        'imagegen_calls_for_this_expansion': 0,
        'source_needle_extraction': 'real frame0 alpha; final17px taper reconstructed as recorded in source preparation',
        'source_plate_separation_does_not_remove_shared_calibration_or_repair_donor_dependence': True,
        'reference_for_image_registration': 'masked_reference.png',
        'reference_note': 'Needle-free training plate0 with LCD filled128. Image-only registration uses static annulus features.',
        'real_training_frames_included': False,
    }
    dump(out / 'provenance.json', source_provenance)
    (out / 'README.md').write_text(
        '# Real dashboard adaptation: 500 synthetic images\n\n'
        '500 grayscale 512x512 PNGs derived from three locally repaired real dashboard frames. '
        'This expands the approved 20-image preview without modifying that preview.\n\n'
        '- Training: 400 images, 200 each from source plates 0 and 180.\n'
        '- Validation: 100 images from source plate 204. No test split.\n'
        '- This validation is a same-video synthetic source-holdout diagnostic. The plates '
        'share calibration and repair donors; results cannot establish accuracy on independent videos.\n'
        '- Requested indicated speeds span 0-180 km/h with fractional readings and exact20km/h anchors. '
        'They are synthetic rendering targets, not road-speed measurements.\n'
        '- Calibration remains a manual visual estimate. Exact rendering labels do not establish '
        'exact physical dial calibration.\n'
        '- LCD pixels are filled with128 before augmentation and again afterward. Warning lights '
        'inside the needle sweep remain visible.\n'
        '- Appearance profiles: clear, dim, reflection, shadow, blur, motion, compression, bright. '
        'Camera variations use +/-3deg roll, .985-1.015scale and +/-1.5px perspective-corner offsets. '
        'Framing is exactly75% near-reference (+/-8px translation) and25% wider framing '
        '(+/-64px translation), per source plate. Real-frame registration found about50-69px '
        'corner displacement even in this nominally fixed-camera recording, motivating this range.\n'
        '- labels.json includes pivot/tip, tick anchors, masks, rendering homographies, provenance, '
        'and an independently image-estimated registration to masked_reference.png. A rejected '
        'registration must not be silently replaced by the known rendering homography for evaluation.\n'
        '- needle_mask is the geometric pre-photometric rendered support, thresholded at alpha>.14. '
        'Blur can spread visible appearance beyond that binary support.\n'
        '- No real labeled training examples or independent real test examples are included.\n\n'
        'Inspect qa/contact_inputs.jpg, qa/contact_labels.jpg and qa/checks.json.\n',
        encoding='utf-8')
    archive = ROOT / 'real_dashboard_lcd_mask_500_v3.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=2) as package:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                package.write(path, 'dataset500/' + path.relative_to(out).as_posix())
    assert archive.stat().st_size < 100_000_000, 'Dataset ZIP exceeds conservative100MB limit'
    manifest = {
        'dataset': labels['dataset'], 'status': 'generated_and_verified_no_training_or_upload',
        'archive': archive.name, 'archive_size_bytes': archive.stat().st_size,
        'archive_sha256': sha(archive), 'labels_sha256': sha(out / 'labels.json'),
        'seed': SEED, 'image_count': len(rows), 'split_counts': {'train': 400, 'validation': 100},
        'calibration_sha256': sha(calibration_path), 'validation_scope': 'same_video_synthetic_source_holdout_diagnostic',
        'independent_test_set': False, 'checks': checks,
    }
    dump(ROOT / 'dataset500_manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
