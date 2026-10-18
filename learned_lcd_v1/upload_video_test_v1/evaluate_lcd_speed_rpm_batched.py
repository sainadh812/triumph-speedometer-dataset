"""UNGATED DIAGNOSTIC ONLY: every source frame enters LCD9, speed11 and RPM9.

No reading-quality, confidence, availability, display-state or geometry gate
suppresses a model forward. Required RPM input geometry is estimated separately;
its fallback mappings and any mathematically invalid placeholder are explicit.
Weights and original pipeline files remain untouched. All model/video execution
is restricted to Colab CUDA. Numeric outputs are not validated measurements.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

CODE_COMMIT = 'af991b11529aeabd6355732b7562e85d857caa86'
PROVENANCE_SHA256 = '4e8d74042c0585e954a657e858643a749041fe1344d0edd04e8f7d55be17b875'
VIDEO_SHA256 = '076d4a01d96fb4fca3c98e425a5a3bf8a425c42441ffdc97b1edfe377ebe7d4b'
LCD_SHA256 = '23daf88c88f0a4ff9960be82591ab9089aecc71bb0339e30b70c1ff6c0d23f44'
SPEED_SHA256 = 'de08c9772c4eacc2b78f9037ff6c45c980d63ee6223739199b4ad28fc4777e1a'
RPM_SHA256 = 'b3ed93a01d9668cf823c32d4c80e7dcd9ede0e6b69e8f5a66942bb5faa1c6016'
CALIBRATION_SHA256 = '2581a049a418d0f337a9997dcd6d196aef7d49d4d9194acc2bc176f14441d2f5'
SOURCE_FRAMES, FPS, DIMENSIONS = 13518, 30, (848, 598)
TOP, BOTTOM = 80, 40
KEYPOINT_NAMES = ['pivot', 'needle_tip'] + [f'tick_{s}' for s in range(0, 181, 20)]
MODE = 'UNGATED_DIAGNOSTIC'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_json(value):
    if isinstance(value, np.ndarray): return safe_json(value.tolist())
    if isinstance(value, np.generic): return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe_json(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(safe_json(value), indent=2, allow_nan=False) + '\n', encoding='utf8')


def rate(count, seconds): return count / seconds if seconds > 0 else None


def batches(count, size):
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def verify_sources(repo, manifest_path):
    if sha256(manifest_path) != PROVENANCE_SHA256: raise ValueError('Unexpected saved-source manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest['status'] != 'complete_and_verified' or manifest['code_commit'] != CODE_COMMIT:
        raise ValueError('Unexpected source provenance')
    prefixes = ('learned_lcd_v1/', 'mixed_view_v5/', 'real_adaptation_v4/', 'real_adaptation_v3/',
                'training_pilot/', 'rpm_training_v1/', 'rpm_prototype/', 'rpm_exploration/')
    verified = {}
    for row in manifest['files']:
        if not row['path'].startswith('pipeline/'): continue
        name = row['path'][len('pipeline/'):]
        if not name.startswith(prefixes): continue
        if not (name.endswith('.py') or name == 'mixed_view_v5/calibration.json' or
                name.startswith(('rpm_prototype/qa/', 'rpm_exploration/'))): continue
        path = (repo / name).resolve()
        if not path.is_relative_to(repo) or not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Missing/changed protected runtime asset: ' + name)
        verified[name] = row['sha256']
    for name in ('learned_lcd_v1/inference.py', 'mixed_view_v5/train_landmarks.py', 'real_adaptation_v4/geometry.py',
                 'rpm_training_v1/infer_runtime.py', 'rpm_prototype/align.py', 'mixed_view_v5/calibration.json'):
        if name not in verified: raise ValueError('Incomplete runtime provenance: ' + name)
    return verified


def project(points, matrix):
    points = np.asarray(points, float)
    matrix = np.asarray(matrix, float)
    q = np.c_[points, np.ones(len(points))] @ matrix.T
    if not np.isfinite(q).all() or np.any(np.abs(q[:, 2]) <= np.finfo(float).eps):
        raise ValueError('nonfinite_or_zero_projective_denominator')
    result = q[:, :2] / q[:, 2:]
    if not np.isfinite(result).all(): raise ValueError('nonfinite_projection')
    return result


def ordered_quad(points):
    """Clockwise in image axes, start smallest x+y; semantic orientation unknown."""
    points = np.asarray(points, np.float64).reshape(4, 2)
    if not np.isfinite(points).all(): raise ValueError('nonfinite_rectangle')
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
    if abs(cv2.contourArea(points.astype(np.float32))) <= np.finfo(np.float32).eps:
        raise ValueError('degenerate_rectangle')
    return points


def lcd_region(probability, footprint, threshold):
    """Raw largest component. This is geometry construction, not a quality gate."""
    p = np.asarray(probability, float)
    valid = np.asarray(footprint, bool)
    if p.shape != (512, 512) or valid.shape != p.shape or not np.isfinite(p).all():
        return None, [], {'status': 'nonfinite_or_invalid_lcd_output', 'mask_pixels': 0}
    binary = ((p >= threshold) & valid).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1: return None, [], {'status': 'no_hard_component', 'mask_pixels': 0}
    chosen = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.uint8(labels == chosen) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = [c.reshape(-1, 2).astype(float) for c in contours if len(c) >= 3]
    return mask, shapes, {'status': 'raw_component_unfiltered', 'mask_pixels': int(np.count_nonzero(mask)),
                          'component_count': count - 1, 'pixel_threshold_for_shape_only': threshold,
                          'component_mean_probability': float(p[mask > 0].mean()),
                          'quality_gate_applied': False, 'semantic_lcd_identity_verified': False}


def fallback_quad(probability, footprint, mask):
    if mask is not None:
        y, x = np.nonzero(mask)
        if len(x) >= 3:
            try:
                quad = ordered_quad(cv2.boxPoints(cv2.minAreaRect(np.column_stack([x, y]).astype(np.float32))))
                return quad, 'hard_lcd_min_area_rectangle'
            except (ValueError, cv2.error): pass
    p = np.asarray(probability, np.float64)
    valid = np.asarray(footprint, bool)
    if p.shape != (512, 512) or not np.isfinite(p).all() or not valid.any():
        raise ValueError('invalid_soft_lcd_probability')
    y, x = np.nonzero(valid)
    weight = p[y, x]
    if np.any(weight < 0) or float(weight.max()) <= 0: raise ValueError('zero_or_invalid_probability_mass')
    weight = weight / weight.max()
    weight = weight / weight.sum()
    xy = np.column_stack([x, y]).astype(float)
    center = np.sum(xy * weight[:, None], axis=0)
    centered = xy - center
    covariance = (centered * weight[:, None]).T @ centered
    eigenvalues, axes = np.linalg.eigh(covariance)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= np.finfo(float).eps):
        raise ValueError('singular_soft_lcd_covariance')
    # Uniform rectangle covariance is side^2/12. This is a diagnostic shape
    # convention, not a calibrated bound or an inferred semantic corner label.
    half = np.sqrt(12 * eigenvalues) / 2
    corners = center + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half @ axes.T
    return ordered_quad(corners), 'soft_lcd_probability_moments_rectangle'


def raw_geometry(points, calibration):
    """All10 predicted semantic ticks, ordinary least-squares H; no quality gates."""
    result = {'speed_kmh': None, 'numerically_defined': False, 'reason': None,
              'method': 'all10_semantic_ticks_least_squares_homography_ungated',
              'quality_checked': False, 'availability_used': False, 'endpoint_clamped': False}
    try:
        points = np.asarray(points, float)
        if points.shape != (12, 2) or not np.isfinite(points).all(): raise ValueError('nonfinite_landmarks')
        ticks = np.asarray(calibration['ticks_xy'], float)
        h, _ = cv2.findHomography(points[2:], ticks, 0)
        if h is None or not np.isfinite(h).all() or np.linalg.matrix_rank(h) < 3:
            raise ValueError('singular_tick_least_squares_homography')
        mapped = project(points[:2], h)
        pivot, tip = mapped
        vector = tip - pivot
        if not np.isfinite(vector).all() or np.linalg.norm(vector) <= np.finfo(float).eps:
            raise ValueError('undefined_needle_direction')
        angles = np.unwrap(np.arctan2(ticks[:, 1] - pivot[1], ticks[:, 0] - pivot[0]))
        speeds = np.asarray(calibration['speeds'], float)
        # Sort angles only to define a diagnostic interpolation even when the
        # semantic order is wrong. No claim that the resulting speed is useful.
        order = np.argsort(angles)
        angles, speeds = angles[order], speeds[order]
        if np.any(np.diff(angles) <= np.finfo(float).eps): raise ValueError('nonunique_tick_angle')
        theta = math.atan2(vector[1], vector[0])
        theta += round(((angles[0] + angles[-1]) / 2 - theta) / (2 * math.pi)) * 2 * math.pi
        if theta < angles[0]: i = 0
        elif theta > angles[-1]: i = len(angles) - 2
        else: i = min(int(np.searchsorted(angles, theta, side='right')) - 1, len(angles) - 2)
        speed = speeds[i] + (theta - angles[i]) * (speeds[i + 1] - speeds[i]) / (angles[i + 1] - angles[i])
        if not math.isfinite(float(speed)): raise ValueError('nonfinite_speed')
        result.update(speed_kmh=float(speed), numerically_defined=True, reason='raw_ungated_estimate',
                      image_to_canonical_h=h.tolist(), rectified_pivot_xy=pivot.tolist(), rectified_tip_xy=tip.tolist(),
                      needle_angle_radians=float(theta), semantic_angle_order_sorted=bool(np.any(order != np.arange(10))),
                      extrapolated=bool(theta < angles[0] or theta > angles[-1]))
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        result['reason'] = str(error)
    return result


class RPMInputs:
    def __init__(self, reference, calibration, runtime_module):
        self.reference, self.calibration, self.runtime = reference, calibration, runtime_module
        self.local = threading.local()

    def aligner(self):
        if not hasattr(self.local, 'aligner'):
            self.local.aligner = self.runtime.registration_module.LCDAligner(self.reference)
        return self.local.aligner

    def prepare(self, frame, probability, footprint, matrix, mask):
        started = time.perf_counter()
        report = {'method': None, 'quality_gates_applied': False, 'placeholder': False,
                  'previous_frame_or_fixed_scene_roi_used': False, 'source_to_reference_h': None,
                  'corner_ordering_rule': 'clockwise image axes; start smallest x+y;180degree semantic orientation unresolved'}
        h = None
        try:
            aligner = self.aligner()
            keypoints, descriptors = aligner.sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
            if descriptors is None or len(keypoints) < 4: raise ValueError('fewer_than4_source_features')
            pairs = aligner.matcher.knnMatch(aligner.descriptors, descriptors, k=2)
            matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
            if len(matches) < 4: raise ValueError('fewer_than4_static_correspondences')
            src = np.float32([keypoints[m.trainIdx].pt for m in matches])
            dst = np.float32([aligner.keypoints[m.queryIdx].pt for m in matches])
            h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2., maxIters=4000, confidence=.999)
            if h is None or not np.isfinite(h).all() or np.linalg.matrix_rank(h) < 3:
                raise ValueError('singular_or_nonfinite_sift_homography')
            report.update(method='current_frame_sift_homography_quality_ungated', matches=len(matches),
                          inliers=int(inliers.sum()) if inliers is not None else None)
        except (ValueError, IndexError, np.linalg.LinAlgError, cv2.error) as error:
            report['sift_numeric_failure'] = str(error)
            h = None
        if h is None:
            try:
                quad, method = fallback_quad(probability, footprint, mask)
                source_quad = project(quad, np.linalg.inv(matrix))
                canonical = np.asarray(self.runtime.registration_module.LCD_QUAD, np.float32)
                h = cv2.getPerspectiveTransform(source_quad.astype(np.float32), canonical)
                if not np.isfinite(h).all() or np.linalg.matrix_rank(h) < 3:
                    raise ValueError('singular_fallback_mapping')
                report.update(method=method, fallback_quad_input_xy=quad.tolist(), fallback_quad_source_xy=source_quad.tolist())
            except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
                report.update(method='invalid_input_zero_placeholder', placeholder=True, numeric_failure=str(error))
                h = None
        if h is None:
            strip = np.zeros((256, 192), np.uint8)
        else:
            try:
                aligned = cv2.warpPerspective(frame, h, (478, 850), flags=cv2.INTER_LINEAR)
                strip = self.runtime.preprocess.preprocess_aligned(aligned, self.calibration)
                report['source_to_reference_h'] = h.tolist()
                report['lcd_quad_source_xy'] = project(self.runtime.registration_module.LCD_QUAD, np.linalg.inv(h)).tolist()
            except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
                report.update(method='invalid_input_zero_placeholder', placeholder=True, numeric_failure=str(error))
                strip = np.zeros((256, 192), np.uint8)
        report['cpu_seconds'] = time.perf_counter() - started
        return strip, report


class Networks:
    def __init__(self, models, torch):
        self.models, self.torch, self.timings = models, torch, []
        self.selected = {}
        self.mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]

    def input_tensor(self, name, images):
        tensor = self.torch.from_numpy(np.stack(images)).to('cuda', dtype=self.torch.float32).div_(255.)
        if name == 'rpm': return tensor[:, None]
        return (tensor[:, None].expand(-1, 3, -1, -1) - self.mean) / self.std

    def benchmark(self, requested):
        torch = self.torch
        started = time.perf_counter()
        reports = {}
        for name, model in self.models.items():
            entries = []
            for size in ([16, 32, 64, 128] if requested == 0 else [requested]):
                sample = output = None
                row = {'batch_size': size, 'status': 'pending', 'warmup_calls': 0, 'timed_calls': 0}
                try:
                    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    free_before, total = torch.cuda.mem_get_info()
                    shape = (size, 1, 256, 192) if name == 'rpm' else (size, 3, 512, 512)
                    sample = torch.zeros(shape, device='cuda', dtype=torch.float32)
                    wall, gpu = [], []
                    with torch.inference_mode():
                        output = model(sample); torch.cuda.synchronize(); row['warmup_calls'] = 1
                        output = None
                        for _ in range(3):
                            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start = time.perf_counter(); begin.record(); output = model(sample); end.record(); end.synchronize()
                            wall.append(time.perf_counter() - start); gpu.append(begin.elapsed_time(end) / 1000)
                            row['timed_calls'] += 1; output = None
                    free_after, _ = torch.cuda.mem_get_info()
                    headroom = max(1024 ** 3, int(total * .20))
                    row.update(status='eligible' if free_after >= headroom else 'insufficient_headroom',
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                               free_before_bytes=free_before, free_after_bytes=free_after, required_headroom_bytes=headroom,
                               total_memory_bytes=total, samples_per_forward_wall_second=size * 3 / sum(wall),
                               samples_per_forward_cuda_second=size * 3 / sum(gpu))
                except torch.cuda.OutOfMemoryError as error:
                    row.update(status='cuda_out_of_memory_skipped', error=str(error))
                finally:
                    del sample, output
                    gc.collect(); torch.cuda.empty_cache()
                entries.append(row)
                print(json.dumps({'benchmark_model': name, 'candidate': row}), flush=True)
            eligible = [row for row in entries if row['status'] == 'eligible']
            if not eligible: raise RuntimeError('No memory-safe batch candidate for ' + name)
            self.selected[name] = max(eligible, key=lambda row: row['samples_per_forward_wall_second'])['batch_size']
            reports[name] = entries
        return {'seconds': time.perf_counter() - started, 'requested_batch_size': requested,
                'selected_batch_sizes': dict(self.selected), 'models': reports,
                'excluded_from_main_calls_samples_and_rates': True, 'input': 'zero tensors only; outputs discarded',
                'required_headroom_fraction': .20, 'minimum_headroom_bytes': 1024 ** 3}

    def run(self, name, images, source_indices, *, record=True, batch_override=None, raw_logits=False):
        torch = self.torch
        results = []
        size = batch_override or self.selected[name]
        for start, stop in batches(len(images), size):
            torch.cuda.synchronize(); total_start = time.perf_counter()
            before = time.perf_counter(); tensor = self.input_tensor(name, images[start:stop]); torch.cuda.synchronize()
            prep = time.perf_counter() - before
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before = time.perf_counter(); begin.record()
            with torch.inference_mode(): output = self.models[name](tensor)
            end.record(); end.synchronize()
            wall, gpu = time.perf_counter() - before, begin.elapsed_time(end) / 1000
            before = time.perf_counter()
            if name == 'lcd':
                logits = output['lcd_logits'].detach().float()
                arrays = {'probability': logits.sigmoid().cpu().numpy()[:, 0]}
                if raw_logits: arrays['logits'] = logits.cpu().numpy()[:, 0]
                del logits
            elif name == 'speed':
                arrays = {key: output[key].detach().float().cpu().numpy() for key in ('points', 'visibility', 'speed')}
            else:
                arrays = {key: output[key].detach().float().cpu().numpy() for key in ('fill_fraction', 'readability_score')}
            output_s = time.perf_counter() - before
            results.extend([{key: value[index].copy() if hasattr(value[index], 'copy') else value[index]
                             for key, value in arrays.items()} for index in range(stop - start)])
            if record:
                self.timings.append({'model': name, 'source_indices': source_indices[start:stop], 'samples': stop - start,
                    'input_prepare_transfer_seconds': prep, 'forward_wall_seconds': wall, 'forward_cuda_seconds': gpu,
                    'output_transfer_seconds': output_s, 'model_pipeline_seconds': time.perf_counter() - total_start})
            del output, tensor, arrays
        return results


def put_text(image, text, position, scale=.43, color=(235, 235, 235)):
    cv2.putText(image, str(text), position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render(frame, row, source_index, timestamp):
    image = frame.copy()
    for contour in row['lcd']['contours_source_xy']:
        contour = np.asarray(contour, float)
        if np.isfinite(contour).all(): cv2.polylines(image, [np.rint(contour).astype(np.int32)], True, (255, 220, 40), 1, cv2.LINE_AA)
    points = np.asarray(row['speed']['points_source_xy'], float)
    if np.isfinite(points[:2]).all():
        valid, a, b = cv2.clipLine((0, 0, 848, 478), tuple(np.rint(points[0]).astype(int)), tuple(np.rint(points[1]).astype(int)))
        if valid: cv2.line(image, a, b, (45, 170, 255), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        if not np.isfinite(point).all(): continue
        outside = not (0 <= point[0] < 848 and 0 <= point[1] < 478)
        point = tuple(np.rint(np.clip(point, [2, 2], [845, 475])).astype(int))
        cv2.drawMarker(image, point, (45, 170, 255), cv2.MARKER_CROSS, 6, 1)
        label = ('P' if index == 0 else 'T' if index == 1 else str((index - 2) * 20)) + ('*' if outside else '')
        put_text(image, label, (min(point[0] + 4, 805), max(12, point[1] - 4)), .32, (45, 170, 255))
    board = np.full((598, 848, 3), 20, np.uint8)
    board[TOP:TOP + 478] = image
    speed = row['speed']['geometry']['speed_kmh']
    rpm = row['rpm']['raw_rpm_approx']
    speed_text = f'{speed:.1f}km/h' if speed is not None and math.isfinite(speed) else 'undefined'
    rpm_text = f'{rpm:.0f}' if rpm is not None and math.isfinite(rpm) else 'undefined'
    if row['rpm']['input_mapping']['placeholder']: rpm_text = 'invalid input (raw placeholder ' + rpm_text + ')'
    put_text(board, 'UNGATED DIAGNOSTIC | LCD9 + speed11 + RPM9 | NOT VALIDATED', (8, 18), .48, (75, 190, 255))
    put_text(board, f'Raw geometric speed: {speed_text} | Raw RPM: {rpm_text}', (8, 41), .49)
    put_text(board, 'RPM input (untrusted): ' + row['rpm']['input_mapping']['method'], (8, 62), .40)
    put_text(board, f'Source {source_index + 1}/{SOURCE_FRAMES} | {timestamp:.2f}s | playback30FPS | cyanLCD / orange needle+ticks', (8, 575), .40)
    put_text(board, 'Quality checks OFF. Speed input unmasked. RPM fallback orientation untrusted. No past-reading reuse.', (8, 593), .37)
    return board


def prepare_regions(frames, grays, boxes, lcd_outputs, threshold):
    regions = []
    for frame, gray, (_, matrix, footprint), prediction in zip(frames, grays, boxes, lcd_outputs):
        probability = prediction['probability']
        mask, contours, report = lcd_region(probability, footprint, threshold)
        source_contours = []
        for contour in contours:
            try: source_contours.append(project(contour, np.linalg.inv(matrix)).tolist())
            except ValueError: pass
        report.update(model_evaluated=True, probability_finite=bool(np.isfinite(probability).all()),
                      contours_source_xy=source_contours,
                      probability_min=float(probability.min()), probability_max=float(probability.max()))
        regions.append((mask, report))
    return regions


def process_chunk(frames, source_indices, timestamps, nets, speed_reader, rpm_inputs, pool, letterbox):
    started = time.perf_counter()
    start = time.perf_counter()
    boxes = [letterbox(frame) for frame in frames]
    if any(np.shape(box[2]) != (512, 512) for box in boxes):
        raise RuntimeError('Expected learned_lcd_v1 letterbox binary sensor mask, not V5 polygon footprint')
    grays = [box[0] for box in boxes]
    letterbox_s = time.perf_counter() - start
    lcd_outputs = nets.run('lcd', grays, source_indices)
    start = time.perf_counter()
    regions = prepare_regions(frames, grays, boxes, lcd_outputs, speed_reader.gates['pixel_threshold'])
    region_s = time.perf_counter() - start
    rpm_pool_started = time.perf_counter()
    futures = [pool.submit(rpm_inputs.prepare, frame, lcd['probability'], box[2], box[1], region[0])
               for frame, lcd, box, region in zip(frames, lcd_outputs, boxes, regions)]
    speed_outputs = nets.run('speed', grays, source_indices)
    start = time.perf_counter()
    speed_rows = []
    for result, box in zip(speed_outputs, boxes):
        points = np.asarray(result['points'], float) * 511
        source = np.full((12, 2), np.nan)
        finite = np.isfinite(points).all(axis=1)
        if finite.any():
            try: source[finite] = project(points[finite], np.linalg.inv(box[1]))
            except ValueError: pass
        legacy = float(result['speed'])
        speed_rows.append({'model_evaluated': True, 'model_input_lcd_masked': False,
                           'points_input_xy': points.tolist(), 'points_source_xy': source.tolist(),
                           'availability': result['visibility'].tolist(), 'availability_used_to_filter': False,
                           'geometry': raw_geometry(points, speed_reader.calibration),
                           'raw_legacy_head_kmh': legacy if math.isfinite(legacy) else None,
                           'legacy_head_scope': 'Inherited auxiliary head, frozen and not optimized for V4/V5 primary speed'})
    speed_post_s = time.perf_counter() - start
    start = time.perf_counter()
    rpm_prepared = [future.result() for future in futures]
    rpm_wait_s = time.perf_counter() - start
    rpm_pool_wall_s = time.perf_counter() - rpm_pool_started
    strips = [pair[0] for pair in rpm_prepared]
    rpm_outputs = nets.run('rpm', strips, source_indices)
    rows = []
    for source_index, timestamp, lcd, speed, rpm, prepared in zip(source_indices, timestamps, regions, speed_rows, rpm_outputs, rpm_prepared):
        q, readability = float(rpm['fill_fraction']), float(rpm['readability_score'])
        raw_rpm = q * 12000 if math.isfinite(q) else None
        rows.append({'source_frame': source_index, 'source_time_s': float(timestamp), 'diagnostic_mode': MODE,
                     'lcd': lcd[1], 'speed': speed,
                     'rpm': {'model_evaluated': True, 'raw_fill_fraction': q, 'raw_rpm_approx': raw_rpm,
                             'raw_readability_score': readability, 'quality_gate_applied': False,
                             'display_state_gate_applied': False, 'input_mapping': prepared[1],
                             'physical_rpm_claim': False, 'placeholder_raw_output': bool(prepared[1]['placeholder'])},
                     'all_outputs_unvalidated': True})
    timing = {'source_start': source_indices[0], 'source_stop_exclusive': source_indices[-1] + 1,
              'samples': len(frames), 'reader_wall_seconds': time.perf_counter() - started,
              'letterbox_seconds': letterbox_s, 'lcd_region_postprocess_seconds': region_s,
              'speed_geometry_postprocess_seconds': speed_post_s, 'rpm_cpu_wait_seconds': rpm_wait_s,
              'rpm_registration_pool_elapsed_wall_seconds': rpm_pool_wall_s,
              'rpm_cpu_work_seconds_sum': sum(pair[1]['cpu_seconds'] for pair in rpm_prepared),
              'rpm_cpu_work_overlaps_speed_gpu_and_geometry': True}
    return rows, timing, grays, strips


def compare_outputs(first, second, name):
    report = {'model': name, 'samples': len(first), 'fields': {}, 'passed': True,
              'pass_policy': 'All outputs must be finite; returned-output tolerance is blocking; internal LCD logits difference is informational.'}
    if len(first) != len(second): raise RuntimeError('Parity sample count differs')
    for key in first[0]:
        a = np.stack([row[key] for row in first])
        b = np.stack([row[key] for row in second])
        if a.shape != b.shape: raise RuntimeError('Parity output shapes differ')
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        same_finite = bool(np.array_equal(finite_a, finite_b))
        all_finite = bool(finite_a.all() and finite_b.all())
        # Explicitly declared FP32 numerical tolerances, not reading gates.
        atol = 1e-4 if key in ('logits', 'probability', 'speed') else 1e-5
        rtol = 1e-4
        passed = all_finite and same_finite and bool(np.allclose(a[finite_a], b[finite_b], atol=atol, rtol=rtol))
        difference = float(np.max(np.abs(a[finite_a] - b[finite_b]))) if same_finite and finite_a.any() else None
        informational = name == 'lcd' and key == 'logits'
        report['fields'][key] = {'max_absolute_difference': difference, 'absolute_tolerance': atol,
                                 'relative_tolerance': rtol, 'same_finite_locations': same_finite, 'passed': passed,
                                 'blocks_parity': not informational,
                                 'comparison_role': 'informational_internal_logits' if informational else 'blocking_returned_output'}
        report['fields'][key]['all_outputs_finite'] = all_finite
        # The LCD pipeline consumes sigmoid probabilities, not internal logits.
        # Retain the original logit comparison verbatim without loosening its
        # tolerance; a failed finite logit comparison does not fail returned-
        # output parity. Nonfinite logits still fail numeric integrity.
        report['passed'] &= all_finite if informational else passed
    return report


def parity_check(video, count, nets, speed_reader, rpm_inputs, pool, letterbox, destination):
    started = time.perf_counter()
    indices = np.linspace(0, SOURCE_FRAMES - 1, count, dtype=int).tolist()
    cap = cv2.VideoCapture(str(video)); frames = []
    try:
        for index in indices:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, index): raise RuntimeError('Parity video seek failed')
            okay, frame = cap.read()
            if not okay: raise RuntimeError('Parity frame decode failed')
            frames.append(frame)
    finally: cap.release()
    boxes = [letterbox(frame) for frame in frames]; grays = [box[0] for box in boxes]
    if any(np.shape(box[2]) != (512, 512) for box in boxes):
        raise RuntimeError('Expected learned_lcd_v1 letterbox binary sensor mask, not V5 polygon footprint')
    def check_model(name, images, raw_logits=False):
        one = nets.run(name, images, indices, record=False, batch_override=1, raw_logits=raw_logits)
        repeats = nets.selected[name] // count
        many = nets.run(name, images * repeats, indices * repeats, record=False, raw_logits=raw_logits)
        report = compare_outputs(one * repeats, many, name)
        report.update(distinct_review_frames=count, actual_selected_batch_size=nets.selected[name],
                      repeated_review_inputs_to_fill_selected_batch=True)
        return report, many[:count]
    lcd_report, lcd_many = check_model('lcd', grays, raw_logits=True)
    reports = [lcd_report]
    regions = prepare_regions(frames, grays, boxes, lcd_many, speed_reader.gates['pixel_threshold'])
    futures = [pool.submit(rpm_inputs.prepare, frame, out['probability'], box[2], box[1], region[0])
               for frame, out, box, region in zip(frames, lcd_many, boxes, regions)]
    speed_report, _ = check_model('speed', grays)
    reports.append(speed_report)
    prepared = [future.result() for future in futures]; strips = [value[0] for value in prepared]
    rpm_report, _ = check_model('rpm', strips)
    reports.append(rpm_report)
    report = {'status': 'passed' if all(row['passed'] for row in reports) else 'failed', 'source_indices': indices,
              'batch1_samples_per_model': count, 'selected_batch_samples_per_model': dict(nets.selected),
              'models': reports, 'seconds': time.perf_counter() - started,
              'extra_calls_per_model': {name: count + 1 for name in nets.selected},
              'extra_samples_per_model': {name: count + size for name, size in nets.selected.items()},
              'review_inputs_repeated_only_to_fill_parity_batch': True, 'excluded_from_main_counts_and_throughput': True,
              'comparison': 'Same prepared input tensors, original FP32 network batch1 versus selected batch. No old gated-reading parity is claimed.',
              'rpm_preparation_methods': dict(Counter(value[1]['method'] for value in prepared))}
    write_json(destination / 'parity_report.json', report)
    print(json.dumps({'raw_forward_parity': report['status'], 'models': reports}), flush=True)
    if report['status'] != 'passed': raise RuntimeError('Raw batch versus batch1 output parity failed; see parity_report.json')
    return report


def network_summary(nets):
    result = {}
    for name in nets.models:
        rows = [row for row in nets.timings if row['model'] == name]
        wall = sum(row['forward_wall_seconds'] for row in rows)
        gpu = sum(row['forward_cuda_seconds'] for row in rows)
        count = sum(row['samples'] for row in rows)
        result[name] = {'successful_forward_calls': len(rows), 'inference_samples': count,
                        'selected_batch_size': nets.selected[name], 'last_batch_samples': rows[-1]['samples'],
                        'forward_wall_seconds': wall, 'forward_cuda_seconds': gpu,
                        'samples_per_forward_wall_second': rate(count, wall),
                        'samples_per_forward_cuda_second': rate(count, gpu),
                        'input_prepare_transfer_seconds': sum(row['input_prepare_transfer_seconds'] for row in rows),
                        'output_transfer_seconds': sum(row['output_transfer_seconds'] for row in rows),
                        'model_pipeline_seconds': sum(row['model_pipeline_seconds'] for row in rows),
                        'dtype': 'float32', 'autocast': False, 'benchmark_and_parity_excluded': True}
    return result


def run_video(args, destination, nets, speed_reader, rpm_reader, rpm_inputs, pool, letterbox,
              probe_video, build_timeline, started, phase, benchmark, parity, source_hashes):
    start = time.perf_counter(); info = probe_video(args.video)
    phase['source_probe_seconds'] = time.perf_counter() - start
    if len(info['timestamps']) != SOURCE_FRAMES or (info['width'], info['height']) != (848, 478):
        raise ValueError('Unexpected source dimensions/count')
    indices, output_times = build_timeline(info['timestamps'], info['duration'], FPS)
    output_count = len(indices)
    output = destination / 'UNGATED_LCD_SPEED_RPM_150802_30fps.mp4'
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened(): raise RuntimeError('Source video failed to open')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', '848x598', '-r', '30', '-i', 'pipe:0', '-i', str(args.video), '-map', '0:v:0', '-map', '1:a?',
               '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20', '-threads', '2', '-pix_fmt', 'yuv420p',
               '-c:a', 'aac', '-b:a', '128k', '-af', 'asetpts=PTS-STARTPTS', '-t', f'{output_count / FPS:.9f}',
               '-movflags', '+faststart', str(output)]
    write_json(destination / 'encoder_command.json', command)
    log = (destination / 'encoder.log').open('xb'); encoder_started = time.perf_counter()
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
    selected = set(np.linspace(0, output_count - 1, 6, dtype=int).tolist())
    fields = ['source_frame', 'source_time_s', 'lcd_called', 'speed_called', 'rpm_called', 'raw_geometric_speed_kmh',
              'geometry_numeric_reason', 'raw_legacy_head_kmh', 'raw_rpm_approx', 'raw_fill_fraction', 'raw_readability_score',
              'rpm_input_method', 'rpm_input_placeholder']
    for name in KEYPOINT_NAMES: fields += [name + '_x', name + '_y', name + '_availability']
    counters, rpm_modes, geometry_reasons = Counter(), Counter(), Counter()
    chunk_timings = []; output_index = 0; last_source = None; next_timing = 0
    decode_s = render_s = pipe_s = artifact_s = 0.; loop_started = time.perf_counter(); completed = False
    block_size = max(nets.selected.values())
    try:
        with (destination / 'source_predictions.jsonl').open('x', encoding='utf8') as evidence, \
             (destination / 'source_predictions.csv').open('x', newline='', encoding='utf8') as csv_sink, \
             (destination / 'batch_timings.jsonl').open('x', encoding='utf8') as batch_sink, \
             (destination / 'chunk_timings.jsonl').open('x', encoding='utf8') as chunk_sink, \
             (destination / 'output_frame_mapping.csv').open('x', newline='', encoding='utf8') as map_sink:
            table = csv.DictWriter(csv_sink, fieldnames=fields); table.writeheader()
            mapping = csv.DictWriter(map_sink, fieldnames=['output_frame', 'time_s', 'source_frame', 'source_time_s', 'same_source_repeated']); mapping.writeheader()
            for begin, end in batches(SOURCE_FRAMES, block_size):
                start = time.perf_counter(); frames = []
                for source_index in range(begin, end):
                    okay, frame = capture.read()
                    if not okay or frame.shape[:2] != (478, 848): raise RuntimeError('Source decode failed at ' + str(source_index))
                    frames.append(frame)
                decode_s += time.perf_counter() - start
                source_indices = list(range(begin, end))
                rows, chunk, gray_inputs, rpm_strips = process_chunk(frames, source_indices, info['timestamps'][begin:end],
                    nets, speed_reader, rpm_inputs, pool, letterbox)
                chunk_timings.append(chunk)
                start = time.perf_counter()
                chunk_sink.write(json.dumps(chunk, allow_nan=False) + '\n')
                for timing in nets.timings[next_timing:]: batch_sink.write(json.dumps(timing, allow_nan=False) + '\n')
                next_timing = len(nets.timings); artifact_s += time.perf_counter() - start
                for offset, (frame, row) in enumerate(zip(frames, rows)):
                    source_index = begin + offset
                    counters['source_frames_processed'] += 1
                    for name in ('lcd', 'speed', 'rpm'): counters[name + '_model_samples'] += 1
                    counters['raw_speed_numerically_defined'] += int(row['speed']['geometry']['numerically_defined'])
                    counters['rpm_invalid_input_placeholder'] += int(row['rpm']['input_mapping']['placeholder'])
                    counters['raw_rpm_finite'] += int(row['rpm']['raw_rpm_approx'] is not None)
                    rpm_modes[row['rpm']['input_mapping']['method']] += 1
                    geometry_reasons[row['speed']['geometry']['reason']] += 1
                    flat = {'source_frame': source_index, 'source_time_s': row['source_time_s'],
                            'lcd_called': True, 'speed_called': True, 'rpm_called': True,
                            'raw_geometric_speed_kmh': row['speed']['geometry']['speed_kmh'],
                            'geometry_numeric_reason': row['speed']['geometry']['reason'],
                            'raw_legacy_head_kmh': row['speed']['raw_legacy_head_kmh'],
                            'raw_rpm_approx': row['rpm']['raw_rpm_approx'], 'raw_fill_fraction': row['rpm']['raw_fill_fraction'],
                            'raw_readability_score': row['rpm']['raw_readability_score'],
                            'rpm_input_method': row['rpm']['input_mapping']['method'],
                            'rpm_input_placeholder': row['rpm']['input_mapping']['placeholder']}
                    for index, name in enumerate(KEYPOINT_NAMES):
                        flat.update({name + '_x': row['speed']['points_source_xy'][index][0],
                                     name + '_y': row['speed']['points_source_xy'][index][1],
                                     name + '_availability': row['speed']['availability'][index]})
                    start = time.perf_counter()
                    evidence.write(json.dumps(safe_json(row), allow_nan=False) + '\n'); table.writerow(safe_json(flat))
                    artifact_s += time.perf_counter() - start
                    while output_index < output_count and int(indices[output_index]) == source_index:
                        start = time.perf_counter(); board = render(frame, row, source_index, float(output_times[output_index]))
                        render_s += time.perf_counter() - start
                        start = time.perf_counter(); encoder.stdin.write(memoryview(board).cast('B'))
                        pipe_s += time.perf_counter() - start
                        repeated = last_source == source_index; counters['repeated_output_frames'] += int(repeated)
                        start = time.perf_counter()
                        mapping.writerow({'output_frame': output_index, 'time_s': float(output_times[output_index]),
                                          'source_frame': source_index, 'source_time_s': row['source_time_s'], 'same_source_repeated': repeated})
                        if output_index in selected:
                            for name, image in [(f'preview_{output_index:05d}.jpg', board),
                                                (f'unmasked_lcd_and_speed_input_{output_index:05d}.png', gray_inputs[offset]),
                                                (f'rpm_input_{output_index:05d}.png', rpm_strips[offset])]:
                                if not cv2.imwrite(str(destination / name), image): raise RuntimeError('Preview write failed')
                        artifact_s += time.perf_counter() - start
                        output_index += 1; last_source = source_index
                if begin == 0 or end // 300 != begin // 300 or end == SOURCE_FRAMES:
                    for stream in (evidence, csv_sink, batch_sink, chunk_sink, map_sink): stream.flush()
                    reader_wall = sum(row['reader_wall_seconds'] for row in chunk_timings)
                    print(json.dumps({'source_frames_processed': end, 'total_source_frames': SOURCE_FRAMES,
                        'selected_batch_sizes': nets.selected, 'all_models_samples_each': end,
                        'reader_processing_fps': end / reader_wall,
                        'full_loop_processing_fps': end / (time.perf_counter() - loop_started),
                        'rpm_input_methods': dict(rpm_modes), 'raw_speed_defined': counters['raw_speed_numerically_defined']}), flush=True)
            if capture.read()[0] or output_index != output_count: raise RuntimeError('Source/output frame accounting mismatch')
        start = time.perf_counter(); encoder.stdin.close()
        if encoder.wait() != 0: raise RuntimeError('H264 encoding failed; see encoder.log')
        phase['encoder_finalization_wait_seconds'] = time.perf_counter() - start; completed = True
    finally:
        capture.release()
        if not completed and encoder.poll() is None: encoder.kill(); encoder.wait()
        log.close()
    phase.update(source_decode_seconds=decode_s, rendering_seconds=render_s, encoder_pipe_submit_wait_seconds=pipe_s,
                 evidence_and_preview_write_seconds=artifact_s, source_inference_render_encode_loop_seconds=time.perf_counter() - loop_started,
                 encoder_process_lifetime_seconds=time.perf_counter() - encoder_started)
    print('All three networks finished every source frame. Verifying full encoded video.', flush=True)
    start = time.perf_counter(); cap = cv2.VideoCapture(str(output)); decoded = 0; previews = []
    try:
        while True:
            okay, image = cap.read()
            if not okay: break
            if (image.shape[1], image.shape[0]) != DIMENSIONS: raise RuntimeError('Encoded dimensions mismatch')
            if decoded in selected:
                if not cv2.imwrite(str(destination / f'encoded_preview_{decoded:05d}.jpg'), image): raise RuntimeError('Encoded preview write failed')
                previews.append(cv2.resize(image, (424, 299), interpolation=cv2.INTER_AREA))
            decoded += 1
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally: cap.release()
    if decoded != output_count or abs(fps - FPS) > 1e-6 or abs(decoded / fps - info['duration']) > .5 / FPS + 1e-6:
        raise RuntimeError('Encoded frame count/FPS/duration mismatch')
    if len(previews) != 6 or not cv2.imwrite(str(destination / 'contact_sheet.jpg'), np.vstack([np.hstack(previews[:3]), np.hstack(previews[3:])])):
        raise RuntimeError('Contact sheet failed')
    phase['output_full_decode_verification_seconds'] = time.perf_counter() - start
    timing = network_summary(nets)
    for name, metrics in timing.items():
        sample_indices = [i for row in nets.timings if row['model'] == name for i in row['source_indices']]
        if sample_indices != list(range(SOURCE_FRAMES)) or metrics['successful_forward_calls'] != math.ceil(SOURCE_FRAMES / nets.selected[name]):
            raise RuntimeError('Missing/repeated model source samples: ' + name)
    phase['reader_wall_seconds'] = sum(row['reader_wall_seconds'] for row in chunk_timings)
    for key in ('letterbox_seconds', 'lcd_region_postprocess_seconds', 'speed_geometry_postprocess_seconds', 'rpm_cpu_wait_seconds',
                'rpm_registration_pool_elapsed_wall_seconds', 'rpm_cpu_work_seconds_sum'):
        phase[key] = sum(row[key] for row in chunk_timings)
    start = time.perf_counter(); digest = sha256(output); phase['output_hash_seconds'] = time.perf_counter() - start
    shutil.copy2(__file__, destination / 'evaluate_lcd_speed_rpm_batched.py')
    phase['evaluator_wall_seconds'] = time.perf_counter() - started
    summary = {'status': 'complete_and_fully_decoded', 'diagnostic_mode': MODE, 'input_filename': args.video.name,
        'input_sha256': VIDEO_SHA256, 'source_frames': SOURCE_FRAMES, 'source_frames_processed': counters['source_frames_processed'],
        'source_frames_skipped_by_models': 0, 'output': str(output), 'output_sha256': digest, 'output_bytes': output.stat().st_size,
        'width': 848, 'height': 598, 'output_frames': output_count, 'fps': fps, 'duration_s': decoded / fps, 'source_duration_s': info['duration'],
        'source_frames_not_selected_for_playback': SOURCE_FRAMES - len(set(indices.tolist())),
        'repeated_output_frames': counters['repeated_output_frames'], 'counts': dict(counters), 'rpm_input_methods': dict(rpm_modes),
        'raw_geometry_numeric_reasons': dict(geometry_reasons), 'network_timing': timing, 'batch_benchmark': benchmark,
        'parity_report': parity, 'phase_seconds': phase, 'rpm_cpu_workers': args.rpm_workers,
        'reader_processing_fps': SOURCE_FRAMES / phase['reader_wall_seconds'],
        'combined_network_only_processing_fps': SOURCE_FRAMES / sum(v['forward_wall_seconds'] for v in timing.values()),
        'decode_inference_render_encode_processing_fps': SOURCE_FRAMES / phase['source_inference_render_encode_loop_seconds'],
        'overall_evaluator_processing_fps': SOURCE_FRAMES / phase['evaluator_wall_seconds'],
        'timing_scope': 'All three models process every source frame. FP32 per-network CUDA/wall times exclude zero-tensor benchmark and16-frame batch-parity repeats. Reader wall includes required RPM mapping/strip, transfers, geometry and CPU waits; RPM worker work totals overlap GPU/otherCPU and are not additive wall time. Encoder lifetime overlaps processing; pipe/final wait is backpressure, not standalone encoding compute. Bootstrap/upload/download/Drive packaging excluded.',
        'batch_latency_warning': 'Offline throughput only. Larger batches require accumulating frames in a livefeed; processingFPS is distinct from30FPS playback.',
        'speed_reader_metadata': speed_reader.metadata, 'rpm_reader_identity': rpm_reader.identity,
        'runtime_source_hashes': source_hashes, 'code_commit': CODE_COMMIT, 'provenance_manifest_sha256': PROVENANCE_SHA256,
        'script_sha256': sha256(__file__), 'evaluation_location': 'Google Colab CUDA', 'training_performed': False,
        'reading_quality_checks_applied': False, 'original_frozen_pipeline_files_modified': False,
        'targeted_adapter_replaces_quality_gates': True, 'original_video_view_warped_or_cropped': False,
        'speed_input_lcd_masked': False, 'speed_input_distribution_shift': 'Unmaskedgray512 versusLCD128-masked V5training',
        'lcd_use': 'Rawoutline andRPMfallbackROI only; does not gate/mask speed',
        'speed_primary': 'Raw all10predictedsemantic-tick least-squares homography then needleangle. Noavailability orquality gates; angular sorting/extrapolation explicit; no rangeclamp.',
        'speed_auxiliary': 'Inherited legacy regression head is frozen/not optimized forV4/V5 primaryspeed; unvalidated JSON/CSV diagnostic only, not shown as a competing video speed.',
        'rpm_input_policy': 'Current-frame SIFT homography without original quality rejection; hardlargestLCDminAreaRect fallback; sensor-only softprobability moment rectangle fallback; zero placeholder only for nonfinite/singular mapping. Every fallback is untrusted and cornerorientation ambiguous.',
        'rpm_output_policy': 'Raw fraction×12000 andreadability logged on everyframe. No display-state/readability/range suppression. Placeholder outputs explicitly not physicalRPM.',
        'smoothing': False, 'previous_reading_fallback': False, 'ground_truth_accuracy': None, 'acceptance_is_accuracy': False,
        'source_audio_preserved_if_present': 'AAC transcode aligned to outputstart/duration',
        'playback_mapping': 'All13518sourceframes inferred once permodel. Nearest-timestamp30FPS output conversion mayomit/repeat playbackframes only, explicitly mapped.'}
    keys = ('source_frames_processed', 'output_frames', 'fps', 'network_timing', 'batch_benchmark', 'parity_report', 'phase_seconds',
            'reader_processing_fps', 'combined_network_only_processing_fps', 'decode_inference_render_encode_processing_fps',
            'overall_evaluator_processing_fps', 'timing_scope', 'batch_latency_warning')
    write_json(destination / 'timing_summary.json', {key: summary[key] for key in keys})
    write_json(destination / 'summary.json', summary)
    write_json(destination / 'completion.json', {'status': 'complete_and_fully_decoded', 'diagnostic_mode': MODE,
        'source_frames_processed': SOURCE_FRAMES, 'model_samples': {name: SOURCE_FRAMES for name in nets.models},
        'output_frames': decoded, 'output_sha256': digest, 'summary_sha256': sha256(destination / 'summary.json')})
    print(json.dumps({key: summary[key] for key in ('output', 'diagnostic_mode', 'source_frames_processed', 'output_frames', 'fps',
        'network_timing', 'reader_processing_fps', 'overall_evaluator_processing_fps', 'rpm_input_methods')}, indent=2), flush=True)


def main():
    started = time.perf_counter(); parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository-root', 'provenance-manifest', 'video', 'output-root', 'lcd-checkpoint', 'speed-checkpoint',
                 'speed-calibration', 'rpm-dataset', 'rpm-checkpoint', 'rpm-reference'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=0, help='0 benchmarks16/32/64/128 independently pernetwork')
    parser.add_argument('--rpm-workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--parity-frames', type=int, default=16)
    parser.add_argument('--diagnostic-mode', choices=['ungated'], required=True)
    args = parser.parse_args()
    if args.batch_size not in (0, 16, 32, 64, 128) or args.parity_frames != 16:
        raise ValueError('Use declared batchchoices and16parityframes')
    try: colab = importlib.util.find_spec('google.colab') is not None
    except (ImportError, ModuleNotFoundError, ValueError): colab = False
    import torch
    if not colab or not torch.cuda.is_available(): raise RuntimeError('All model/video execution requires GoogleColabCUDA')
    if cv2.__version__ != '4.14.0': raise ValueError('Expected protected OpenCV4.14.0 runtime')
    repo = args.repository_root.resolve(); phase = {}; start = time.perf_counter()
    hashes = verify_sources(repo, args.provenance_manifest)
    for path, expected in ((args.video, VIDEO_SHA256), (args.lcd_checkpoint, LCD_SHA256), (args.speed_checkpoint, SPEED_SHA256),
                           (args.rpm_checkpoint, RPM_SHA256), (args.speed_calibration, CALIBRATION_SHA256)):
        if sha256(path) != expected: raise ValueError('Unexpected frozen input/checkpoint: ' + str(path))
    phase['source_asset_verification_seconds'] = time.perf_counter() - start
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    sys.path.insert(0, str(repo))
    from learned_lcd_v1.inference import LearnedLCDSpeedReader, letterbox
    from rpm_training_v1 import infer_runtime
    from real_adaptation_v4.infer_video import probe_video, build_timeline
    destination = args.output_root.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    speed_reader = LearnedLCDSpeedReader(args.lcd_checkpoint, args.speed_checkpoint, args.speed_calibration, device='cuda')
    rpm_reader = infer_runtime.RPMInference(args.rpm_dataset, args.rpm_checkpoint, args.rpm_reference,
        device='cuda', expected_sha256=RPM_SHA256, expected_opencv_version='4.14.0')
    if speed_reader.metadata['lcd_best_epoch'] != 9 or rpm_reader.identity['best_epoch'] != 9:
        raise ValueError('Unexpected frozen model epochs')
    models = {'lcd': speed_reader.lcd_model, 'speed': speed_reader.speed_model, 'rpm': rpm_reader.model}
    for model in models.values(): model.eval()
    nets = Networks(models, torch)
    rpm_inputs = RPMInputs(args.rpm_reference, rpm_reader.calibration, infer_runtime)
    phase['model_loading_and_identity_seconds'] = time.perf_counter() - start
    benchmark = nets.benchmark(args.batch_size); write_json(destination / 'batch_benchmark.json', benchmark)
    phase['batch_benchmark_seconds'] = benchmark['seconds']
    with ThreadPoolExecutor(max_workers=args.rpm_workers, thread_name_prefix='rpm_cpu') as pool:
        parity = parity_check(args.video, args.parity_frames, nets, speed_reader, rpm_inputs, pool, letterbox, destination)
        phase['raw_batch_parity_seconds'] = parity['seconds']
        run_video(args, destination, nets, speed_reader, rpm_reader, rpm_inputs, pool, letterbox,
                  probe_video, build_timeline, started, phase, benchmark, parity, hashes)


if __name__ == '__main__': main()
