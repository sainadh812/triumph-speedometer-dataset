"""Bounded, paired Colab-only throughput diagnostic; no accuracy claim.

128 consecutive frames enter all three frozen CNNs in each mode. Only the RPM
positioning algorithm changes. The fast path never calls SIFT or uses a fixed
scene ROI. Its upright corner convention is an explicit clip-specific assumption.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

BASE_SHA256 = '0c5df9a13e5cd904d76339a74de8d84675965041ee166eacf94e933214947df3'
FRAME_START, FRAME_COUNT = 10810, 128
FIXED_BATCHES = {'lcd': 64, 'speed': 16, 'rpm': 128}
MODES = ('sift_baseline', 'lcd_positioning')
CORNER_RULE = 'Clockwise in image axes starting minimum x+y; assume upright nonmirrored LCD for this reviewed clip; no semantic corner detector.'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def load_base(path):
    if digest(path) != BASE_SHA256: raise ValueError('Unexpected base evaluator bytes')
    spec = importlib.util.spec_from_file_location('protected_raw_evaluator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quad_aspect(quad):
    q = np.asarray(quad, float)
    width = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    if min(width, height) <= np.finfo(float).eps: raise ValueError('degenerate_quad_sides')
    return width / height


def select_quad(probability, footprint, threshold, canonical, base):
    """Rank current-frame components using predeclared shape priors, not truth.

    Every mathematically usable component is eligible; the ranking is not a
    confidence gate. A selected background component remains a possible failure.
    No image-coordinate position prior, previous-frame state or SIFT is used.
    """
    p = np.asarray(probability, np.float64)
    valid = np.asarray(footprint, bool)
    if p.shape != (512, 512) or valid.shape != p.shape or not np.isfinite(p).all():
        raise ValueError('invalid_lcd_probability_or_sensor_mask')
    binary = np.uint8((p >= threshold) & valid)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    max_area = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    canonical_aspect = quad_aspect(canonical)
    candidates = []
    for label in range(1, count):
        component = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3 or cv2.contourArea(contour) <= 0: continue
        perimeter = cv2.arcLength(contour, True)
        quad, epsilon, method = None, None, None
        for fraction in (.01, .015, .02, .03, .04, .05, .06):
            approximation = cv2.approxPolyDP(contour, fraction * perimeter, True)
            if len(approximation) == 4 and cv2.isContourConvex(approximation):
                quad = base.ordered_quad(approximation.reshape(4, 2))
                epsilon, method = fraction, 'lcd_contour_quadrilateral'
                break
        try:
            if quad is None:
                quad = base.ordered_quad(cv2.boxPoints(cv2.minAreaRect(contour)))
                method = 'lcd_component_min_area_rectangle'
            aspect = quad_aspect(quad)
            shape_area = abs(cv2.contourArea(quad.astype(np.float32)))
            if shape_area <= 0: continue
            area = int(stats[label, cv2.CC_STAT_AREA])
            fill = min(float(cv2.contourArea(contour)) / shape_area, 1.)
            mean_probability = float(p[component > 0].mean())
            aspect_score = math.exp(-abs(math.log(aspect / canonical_aspect)))
            size_score = math.sqrt(area / max_area)
            score = fill * mean_probability * aspect_score * size_score
            candidates.append({'component_id': label, 'method': method, 'quad_input_xy': quad.tolist(),
                'contour_input_xy': contour.reshape(-1, 2).tolist(),
                'score': score, 'area_pixels': area, 'contour_quad_fill_ratio': fill,
                'mean_probability': mean_probability, 'aspect': aspect, 'aspect_score': aspect_score,
                'relative_area_sqrt': size_score, 'approximation_epsilon_fraction': epsilon})
        except (ValueError, cv2.error): continue
    if candidates:
        chosen = max(candidates, key=lambda item: (item['score'], item['area_pixels']))
        return np.asarray(chosen['quad_input_xy'], float), chosen['method'], {
            'chosen_component_id': chosen['component_id'], 'component_candidates': candidates,
            'largest_component_area_pixels': max_area, 'canonical_aspect_prior': canonical_aspect,
            'ranking_formula': 'contour_fill * mean_probability * exp(-abs(log(aspect/canonical_aspect))) * sqrt(area/largest_area)',
            'semantic_lcd_identity_verified': False}
    quad, method = base.fallback_quad(p, valid, None)
    return quad, method, {'component_candidates': [], 'semantic_lcd_identity_verified': False}


class FastRPMInputs:
    def __init__(self, calibration, runtime, threshold, base):
        self.calibration, self.runtime, self.threshold, self.base = calibration, runtime, threshold, base

    def prepare(self, frame, probability, footprint, matrix, unused_largest_mask):
        started = time.perf_counter()
        report = {'method': None, 'quality_gates_applied': False, 'placeholder': False,
            'previous_frame_or_fixed_scene_roi_used': False, 'source_to_reference_h': None,
            'corner_ordering_rule': CORNER_RULE, 'sift_called': False,
            'semantic_orientation_verified': False, 'alignment_untrusted': True}
        try:
            canonical = np.asarray(self.runtime.registration_module.LCD_QUAD, np.float32)
            quad, method, details = select_quad(probability, footprint, self.threshold, canonical, self.base)
            source_quad = self.base.project(quad, np.linalg.inv(matrix))
            h = cv2.getPerspectiveTransform(source_quad.astype(np.float32), canonical)
            if not np.isfinite(h).all() or np.linalg.matrix_rank(h) < 3:
                raise ValueError('singular_lcd_quad_mapping')
            aligned = cv2.warpPerspective(frame, h, (478, 850), flags=cv2.INTER_LINEAR)
            strip = self.runtime.preprocess.preprocess_aligned(aligned, self.calibration)
            report.update(details)
            for candidate in report.get('component_candidates', []):
                candidate['contour_source_xy'] = self.base.project(candidate['contour_input_xy'], np.linalg.inv(matrix)).tolist()
            report.update(method=method, source_to_reference_h=h.tolist(),
                lcd_quad_source_xy=source_quad.tolist(), quad_input_xy=quad.tolist())
        except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
            strip = np.zeros((256, 192), np.uint8)
            report.update(method='invalid_input_zero_placeholder', placeholder=True, numeric_failure=str(error))
        report['cpu_seconds'] = time.perf_counter() - started
        return strip, report


def read_frames(path):
    capture = cv2.VideoCapture(str(path))
    frames, times = [], []
    try:
        if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, FRAME_START):
            raise RuntimeError('Source open/seek failed')
        for index in range(FRAME_START, FRAME_START + FRAME_COUNT):
            okay, frame = capture.read()
            if not okay or frame.shape != (478, 848, 3): raise RuntimeError('Missing/unexpected bounded frame')
            if abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - (index + 1)) > .5:
                raise RuntimeError('Unexpected source frame position')
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(timestamp): raise RuntimeError('Nonfinite source timestamp')
            frames.append(frame); times.append(timestamp)
    finally: capture.release()
    return frames, times


def warm_workers(adapter, pool, workers):
    barrier = threading.Barrier(workers)
    def initialize():
        adapter.aligner()
        barrier.wait(timeout=60)
    for future in [pool.submit(initialize) for _ in range(workers)]: future.result()


def preview(base, frame, row, strip, mode):
    board = base.render(frame, row, row['source_frame'], row['source_time_s'])
    panel = np.zeros((board.shape[0] + 300, board.shape[1], 3), np.uint8)
    panel[:board.shape[0]] = board
    y = board.shape[0]
    cv2.putText(panel, mode + ' | actual RPM input (untrusted)', (12, y + 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    panel[y + 35:y + 291, 15:207] = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
    mapping = row['rpm']['input_mapping']
    for candidate in mapping.get('component_candidates', []):
        contour = np.asarray(candidate['contour_source_xy'], float)
        if len(contour) and np.isfinite(contour).all():
            points = np.rint(contour).astype(np.int32); points[:, 1] += base.TOP
            cv2.polylines(panel, [points], True, (180, 100, 180), 1)
    quad = np.asarray(mapping.get('lcd_quad_source_xy', []), float)
    if quad.shape == (4, 2) and np.isfinite(quad).all():
        points = np.rint(quad).astype(np.int32); points[:, 1] += base.TOP
        cv2.polylines(panel, [points], True, (0, 180, 255), 2)
    cv2.putText(panel, 'Orange quad = RPM mapping', (230, y + 65), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 180, 255), 1)
    cv2.putText(panel, 'Not a verified LCD / RPM measurement', (230, y + 100), cv2.FONT_HERSHEY_SIMPLEX, .55, (220, 220, 220), 1)
    return panel


def main():
    overall_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository-root', 'provenance-manifest', 'base-evaluator', 'video', 'output-root',
                 'lcd-checkpoint', 'speed-checkpoint', 'speed-calibration', 'rpm-dataset', 'rpm-checkpoint', 'rpm-reference'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    parser.add_argument('--frame-start', type=int, default=FRAME_START)
    parser.add_argument('--frame-count', type=int, default=FRAME_COUNT)
    for name, default in FIXED_BATCHES.items(): parser.add_argument('--' + name + '-batch-size', type=int, default=default)
    parser.add_argument('--rpm-workers', type=int, choices=[2], default=2)
    args = parser.parse_args()
    requested = {name: getattr(args, name + '_batch_size') for name in FIXED_BATCHES}
    if (args.frame_start, args.frame_count) != (FRAME_START, FRAME_COUNT) or requested != FIXED_BATCHES:
        raise ValueError('Use the declared paired128-frame scope and identical fixed batches')
    try: colab = importlib.util.find_spec('google.colab') is not None
    except (ValueError, ImportError, ModuleNotFoundError): colab = False
    import torch
    if not colab or not torch.cuda.is_available(): raise RuntimeError('Model/video execution is GoogleColabCUDA only')
    if cv2.__version__ != '4.14.0': raise ValueError('Expected protected OpenCV4.14.0')
    base = load_base(args.base_evaluator)
    repo = args.repository_root.resolve()
    protected = base.verify_sources(repo, args.provenance_manifest)
    for path, expected in ((args.video, base.VIDEO_SHA256), (args.lcd_checkpoint, base.LCD_SHA256),
        (args.speed_checkpoint, base.SPEED_SHA256), (args.rpm_checkpoint, base.RPM_SHA256),
        (args.speed_calibration, base.CALIBRATION_SHA256)):
        if digest(path) != expected: raise ValueError('Unexpected frozen asset: ' + str(path))
    destination = args.output_root.resolve(); destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    sys.path.insert(0, str(repo))
    from learned_lcd_v1.inference import LearnedLCDSpeedReader, letterbox
    from rpm_training_v1 import infer_runtime
    speed_reader = LearnedLCDSpeedReader(args.lcd_checkpoint, args.speed_checkpoint, args.speed_calibration, device='cuda')
    rpm_reader = infer_runtime.RPMInference(args.rpm_dataset, args.rpm_checkpoint, args.rpm_reference,
        device='cuda', expected_sha256=base.RPM_SHA256, expected_opencv_version='4.14.0')
    models = {'lcd': speed_reader.lcd_model, 'speed': speed_reader.speed_model, 'rpm': rpm_reader.model}
    for model in models.values(): model.eval()
    nets = base.Networks(models, torch); nets.selected = dict(FIXED_BATCHES)
    begin = time.perf_counter(); frames, timestamps = read_frames(args.video)
    decoding_seconds = time.perf_counter() - begin
    indices = list(range(FRAME_START, FRAME_START + FRAME_COUNT))
    begin = time.perf_counter()
    for name, batch in FIXED_BATCHES.items():
        shape = (256, 192) if name == 'rpm' else (512, 512)
        nets.run(name, [np.zeros(shape, np.uint8)] * batch, [-1] * batch, record=False)
    warmup_seconds = time.perf_counter() - begin
    baseline = base.RPMInputs(args.rpm_reference, rpm_reader.calibration, infer_runtime)
    fast = FastRPMInputs(rpm_reader.calibration, infer_runtime, speed_reader.gates['pixel_threshold'], base)
    results, saved, all_batches = {}, {}, []
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='rpm_pair') as pool:
        begin = time.perf_counter(); warm_workers(baseline, pool, 2)
        reference_initialization_seconds = time.perf_counter() - begin
        for mode, adapter in zip(MODES, (baseline, fast)):
            print('Starting ' + mode + ': 128frames, all3models, fixed batches ' + str(FIXED_BATCHES), flush=True)
            nets.timings = []
            rows, timing, grays, strips = base.process_chunk(frames, indices, timestamps, nets, speed_reader, adapter, pool, letterbox)
            network = base.network_summary(nets)
            if any(value['inference_samples'] != FRAME_COUNT for value in network.values()):
                raise RuntimeError('A model did not process every paired frame')
            for row in rows: row['benchmark_mode'] = mode
            all_batches.extend([dict(row, benchmark_mode=mode) for row in nets.timings])
            begin = time.perf_counter()
            # Measure the same simple overlay work on every frame in both modes.
            for frame, row in zip(frames, rows): base.render(frame, row, row['source_frame'], row['source_time_s'])
            rendering_seconds = time.perf_counter() - begin
            results[mode] = {'source_frames_processed': FRAME_COUNT, 'source_indices': indices,
                'network_timing': network, 'reader_wall_seconds': timing['reader_wall_seconds'],
                'reader_fps': base.rate(FRAME_COUNT, timing['reader_wall_seconds']),
                'rpm_preparation_pool_wall_seconds': timing['rpm_registration_pool_elapsed_wall_seconds'],
                'rpm_worker_seconds_sum': timing['rpm_cpu_work_seconds_sum'],
                'rendering_seconds': rendering_seconds,
                'reader_plus_render_fps': base.rate(FRAME_COUNT, timing['reader_wall_seconds'] + rendering_seconds),
                'method_counts': dict(Counter(row['rpm']['input_mapping']['method'] for row in rows)),
                'placeholder_frames': sum(row['rpm']['input_mapping']['placeholder'] for row in rows),
                'phase_seconds': timing}
            saved[mode] = (rows, strips)
            print(mode + ': ' + json.dumps(results[mode], allow_nan=False), flush=True)
    begin = time.perf_counter()
    with (destination / 'source_predictions.jsonl').open('w', encoding='utf8') as stream:
        for mode in MODES:
            for row in saved[mode][0]: stream.write(json.dumps(base.safe_json(row), allow_nan=False) + '\n')
    with (destination / 'predictions.csv').open('w', newline='', encoding='utf8') as stream:
        fields = ['mode', 'source_frame', 'source_time_s', 'speed_kmh_raw', 'rpm_approx_raw', 'rpm_readability_raw', 'rpm_mapping', 'placeholder']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for mode in MODES:
            for row in saved[mode][0]:
                writer.writerow(dict(zip(fields, [mode, row['source_frame'], row['source_time_s'],
                    row['speed']['geometry']['speed_kmh'], row['rpm']['raw_rpm_approx'], row['rpm']['raw_readability_score'],
                    row['rpm']['input_mapping']['method'], row['rpm']['input_mapping']['placeholder']])))
    with (destination / 'batch_timings.jsonl').open('w', encoding='utf8') as stream:
        for row in all_batches: stream.write(json.dumps(row, allow_nan=False) + '\n')
    panels, preview_names = [], []
    for offset in (0, 42, 84, 127):
        sides = []
        for mode in MODES:
            rows, strips = saved[mode]
            sides.append(preview(base, frames[offset], rows[offset], strips[offset], mode))
            cv2.imwrite(str(destination / f'{mode}_rpm_input_{indices[offset]}.png'), strips[offset])
        combined = np.concatenate(sides, axis=1)
        name = f'comparison_{indices[offset]}.jpg'
        if not cv2.imwrite(str(destination / name), combined): raise RuntimeError('Preview write failed')
        preview_names.append(name); panels.append(cv2.resize(combined, (848, 449), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(destination / 'contact_sheet.jpg'), np.concatenate(panels, axis=0))
    export_seconds = time.perf_counter() - begin
    summary = {'status': 'complete', 'benchmark_kind': 'paired_short_throughput_not_accuracy',
        'frame_start': FRAME_START, 'frame_count': FRAME_COUNT, 'source_indices': indices, 'mode_order': list(MODES),
        'modes': results, 'fixed_batch_sizes': FIXED_BATCHES, 'rpm_workers': 2,
        'speedup_reader_ratio': results['lcd_positioning']['reader_fps'] / results['sift_baseline']['reader_fps'],
        'input_sha256': base.VIDEO_SHA256, 'script_sha256': digest(__file__), 'base_evaluator_sha256': BASE_SHA256,
        'code_commit': base.CODE_COMMIT, 'protected_runtime_assets': protected,
        'checkpoints': {'lcd': base.LCD_SHA256, 'speed': base.SPEED_SHA256, 'rpm': base.RPM_SHA256},
        'speed_reader_metadata': speed_reader.metadata, 'rpm_reader_identity': rpm_reader.identity,
        'training_performed': False, 'original_frozen_pipeline_files_modified': False,
        'warmup_excluded': True, 'warmup_samples': FIXED_BATCHES,
        'network_warmup_seconds': warmup_seconds, 'reference_initialization_seconds': reference_initialization_seconds,
        'source_seek_decode_seconds': decoding_seconds, 'export_and_comparison_render_seconds': export_seconds,
        'overall_evaluator_seconds': time.perf_counter() - overall_started,
        'frame_selection': 'Predeclared short contiguous section10810..10937; prior visual evidence upright LCD; throughput demonstration, not representative accuracy sampling.',
        'timestamp_source': 'OpenCV CAP_PROP_POS_MSEC after each bounded sequential decode; no full-video probe.',
        'orientation_assumption': CORNER_RULE,
        'orientation_evidence': 'Previously saved nearby playback preview10810 (displayed source10813 one-based): upright nonmirrored display, printed12 above0. This is not automatic orientation recognition.',
        'fast_input_policy': 'All current-frame LCD components ranked by declared shape/aspect/probability/relative-size heuristic; contour quad preferred, minAreaRect else; sensor-only soft moments when no candidate. No SIFT or fixed/previous ROI.',
        'rpm_strip_preprocessing_unchanged': True, 'speed_input_fullframe_unmasked': True,
        'prediction_quality_gates_applied': False, 'all_outputs_unvalidated': True, 'ground_truth_accuracy': None,
        'previews': preview_names, 'no_video_encoded': True,
        'timing_scope': 'Reader wall includes all3CNN forward/input/output transfers, letterbox, LCD shapes, overlapping2worker RPM positioning and raw speed geometry. Worker seconds are summed work, not critical path; pool wall overlaps speed work. Reader FPS excludes source seek/decode, warmup, rendering/export. One fixed-order pair has cache/thermal bias and does not establish sustained full-video FPS, online latency, or alignment accuracy.'}
    base.write_json(destination / 'summary.json', summary)
    base.write_json(destination / 'timing_summary.json', {k: summary[k] for k in ('modes', 'speedup_reader_ratio', 'fixed_batch_sizes', 'timing_scope', 'warmup_excluded')})
    summary_path = destination / 'summary.json'
    base.write_json(destination / 'completion.json', {'status': 'complete', 'summary_sha256': digest(summary_path),
        'summary_bytes': summary_path.stat().st_size, 'script_sha256': digest(__file__), 'source_indices': indices,
        'main_model_samples_per_mode': FRAME_COUNT})
    print(json.dumps({'status': 'complete', 'frame_count': FRAME_COUNT, 'speedup_reader_ratio': summary['speedup_reader_ratio'],
        'reader_fps': {mode: results[mode]['reader_fps'] for mode in MODES}, 'output': str(destination)}, indent=2), flush=True)


if __name__ == '__main__': main()
