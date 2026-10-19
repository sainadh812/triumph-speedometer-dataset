"""Colab-only, three-model video tests with verified private Drive persistence.

Imports perform no model execution or network access. Source IDs stay in the
external notebook configuration; no credential values belong in this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import time
import uuid
import zipfile

from .colab_workflow import TASKS, _disconnect, _file_digest, sha256, write_json


def _verified_upload(store, path, parent, name=None):
    path = Path(path)
    digest, size = sha256(path), path.stat().st_size
    ref = store.upload_verified(path, parent, name)
    if _file_digest(ref) != digest or int(ref['size']) != size:
        raise ValueError('Persistent artifact verification differs from local bytes')
    return ref


def _download_verified(store, ref, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = ref.get('sha256') or _file_digest(ref)
    size = int(ref['size'])
    if not re.fullmatch('[0-9a-f]{64}', digest) or size <= 0:
        raise ValueError('Invalid source file digest or byte count')
    if not target.is_file() or target.stat().st_size != size or sha256(target) != digest:
        store.download_verified(ref['id'], target, digest, size)
    if target.stat().st_size != size or sha256(target) != digest:
        raise ValueError('Downloaded bytes differ from the approved reference')
    return target


def resolve_complete(store, prepared, run_ref):
    """Read a COMPLETE receipt; never create a training run or resume training."""
    if isinstance(run_ref, str):
        run_id, complete_id = run_ref, None
    else:
        run_id, complete_id = run_ref['run_id'], run_ref.get('complete_id')
    if not re.fullmatch('[A-Za-z0-9_-]{4,100}', run_id):
        raise ValueError('Invalid training run ID')
    runs = [row for row in store.list_files(prepared['drive_root_id'])
            if row['name'] == run_id and row.get('mimeType') == 'application/vnd.google-apps.folder']
    if len(runs) != 1:
        raise ValueError('Expected exactly one existing training run folder')
    items = store.list_files(runs[0]['id'])
    candidates = [row for row in items if row['name'].endswith('_COMPLETE.json')]
    if complete_id:
        candidates = [row for row in candidates if row['id'] == complete_id]
    if not candidates:
        raise ValueError('Training has no matching COMPLETE receipt; video testing did not start')
    if len({row['name'] for row in candidates}) != len(candidates):
        raise ValueError('Ambiguous COMPLETE receipt names')
    chosen = max(candidates, key=lambda row: row['name'])
    complete = store.read_json(chosen['id'])
    if (complete.get('status') != 'complete' or complete.get('run_id') != run_id or
            complete.get('drive_run_folder_id') != runs[0]['id'] or
            chosen['name'] != complete.get('attempt', '') + '_COMPLETE.json'):
        raise ValueError('COMPLETE receipt identity does not match the requested run')
    if set(complete.get('results', {})) != set(TASKS):
        raise ValueError('COMPLETE receipt lacks all three finished tasks')
    if any(complete['results'][task].get('status') != 'complete' for task in TASKS):
        raise ValueError('A model is not marked complete')
    if complete['archive']['id'] not in {row['id'] for row in items}:
        raise ValueError('Results archive is not in the completed run folder')
    return complete, chosen


def extract_models(archive, destination, complete, expected_identity=None):
    """Extract only selected weights, their SHA markers, and task reports."""
    destination = Path(destination)
    allowed = {f'{task}/{name}' for task in TASKS
               for name in ('best_model.pt', 'best_model.pt.sha256.json', 'report.json')}
    with zipfile.ZipFile(archive) as bundle:
        seen = set()
        for entry in bundle.infolist():
            name, path = entry.filename, PurePosixPath(entry.filename)
            if (name in seen or path.is_absolute() or '..' in path.parts or '\\' in name or
                    ':' in name or ((entry.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError('Unsafe or duplicate entry in results archive')
            seen.add(name)
            if name in allowed:
                limit = 2 * 1024 ** 3 if name.endswith('.pt') else 8 * 1024 ** 2
                if entry.file_size <= 0 or entry.file_size > limit:
                    raise ValueError('Unexpected selected artifact size')
        if not allowed.issubset(seen):
            raise ValueError('Results archive lacks a model, checksum marker, or report')
        for name in sorted(allowed):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as source, target.open('wb') as sink:
                shutil.copyfileobj(source, sink, 1024 * 1024)
    paths, provenance, common_identity = {}, {}, None
    for task in TASKS:
        directory = destination / task
        report = json.loads((directory / 'report.json').read_text(encoding='utf8'))
        marker = json.loads((directory / 'best_model.pt.sha256.json').read_text(encoding='utf8'))
        weights = directory / 'best_model.pt'
        identity = report.get('identity', {})
        if (report.get('status') != 'complete' or report.get('task') != task or
                identity.get('task') != task or identity != complete['results'][task].get('identity') or
                report.get('best_epoch') != complete['results'][task].get('best_epoch')):
            raise ValueError('Selected task report differs from COMPLETE receipt')
        if weights.stat().st_size != marker['bytes'] or sha256(weights) != marker['sha256']:
            raise ValueError('Selected model bytes fail their checksum marker')
        shared = {key: identity.get(key) for key in
                  ('architecture_id', 'preprocessing_id', 'dataset_manifest_sha256', 'code_sha256')}
        if common_identity is not None and common_identity != shared:
            raise ValueError('The selected three models do not share training provenance')
        common_identity = shared
        for key, value in (expected_identity or {}).items():
            if identity.get(key) != value:
                raise ValueError('Completed model does not match the configured training identity: ' + key)
        paths[task] = str(weights)
        provenance[task] = {'sha256': marker['sha256'], 'size': marker['bytes'],
                            'best_epoch': report['best_epoch'], 'identity': identity}
    return paths, provenance


def probe_video(path):
    """Read timestamps for every source frame, without neural inference."""
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
               '-show_streams', '-show_entries',
               'frame=best_effort_timestamp_time:stream=width,height,avg_frame_rate,r_frame_rate,duration,nb_frames',
               '-of', 'json', str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    document = json.loads(result.stdout)
    if len(document.get('streams', [])) != 1:
        raise ValueError('Expected a readable primary video stream')
    stream = document['streams'][0]
    def fraction(value):
        numerator, denominator = value.split('/')
        return float(numerator) / float(denominator) if float(denominator) else 0.
    fps = fraction(stream.get('avg_frame_rate', '0/0'))
    if not math.isfinite(fps) or not 0 < fps <= 1000:
        raise ValueError('Video frame rate is unavailable or unsupported')
    frames = document.get('frames', [])
    pts = [float(row['best_effort_timestamp_time']) for row in frames]
    if not pts or any(not math.isfinite(value) for value in pts) or any(b < a for a, b in zip(pts, pts[1:])):
        raise ValueError('Video has missing or non-monotonic frame timestamps')
    deltas = [b-a for a, b in zip(pts, pts[1:]) if b > a]
    period = sorted(deltas)[len(deltas)//2] if deltas else 1/fps
    duration = pts[-1] - pts[0] + period
    variable = bool(deltas and max(deltas) - min(deltas) > max(.001, .02 / fps))
    return {'fps': fps, 'timestamps': pts, 'duration_seconds': duration,
            'last_frame_period_seconds': period, 'variable_frame_rate': variable,
            'stream_width': int(stream['width']), 'stream_height': int(stream['height']),
            'reported_frame_count': stream.get('nb_frames'), 'decoded_timestamp_count': len(pts)}


def clip_indices(metadata, seconds):
    pts = metadata['timestamps']
    end = pts[-1] + metadata['last_frame_period_seconds']
    start = max(pts[0], end - seconds)
    indices = [index for index, timestamp in enumerate(pts) if timestamp >= start - 1e-9]
    if not indices:
        raise ValueError('Requested video window contains no frames')
    return indices[0], indices[-1] + 1, start, end


def _point(point):
    if point is None or len(point) != 2 or any(not math.isfinite(float(v)) for v in point):
        return None
    if any(abs(float(v)) >= 2 ** 30 for v in point):
        return None
    return tuple(int(round(float(v))) for v in point)


def draw_prediction(frame, prediction, source_time, source_name):
    """Full input frame with basic geometry and readings; no reader-input crops."""
    import cv2
    import numpy as np
    output = frame.copy()
    height, width = output.shape[:2]
    scale = max(.45, min(width, height) / 700.)
    thickness = max(1, round(scale * 2))
    def label(text, xy, color, size=1):
        font = scale * size
        cv2.putText(output, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, font,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(output, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, font,
                    color, thickness, cv2.LINE_AA)
    def polygon(points, color, close=True):
        values = [_point(p) for p in points or []]
        if len(values) >= 2 and all(v is not None for v in values):
            cv2.polylines(output, [np.asarray(values, np.int32)], close, color, thickness, cv2.LINE_AA)
    box = prediction.get('dashboard_box')
    if box and len(box) == 4 and _point(box[:2]) is not None and _point(box[2:]) is not None:
        cv2.rectangle(output, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])),
                      (0, 220, 255), thickness)
    polygon(prediction.get('lcd_polygon'), (255, 230, 0))
    speed = prediction.get('speed', {})
    points = {'pivot': speed.get('pivot'), 'tip': speed.get('tip')}
    points.update({'tick_' + str(name): point for name, point in speed.get('ticks', {}).items()})
    pivot, tip = _point(points.get('pivot')), _point(points.get('tip'))
    if pivot is not None and tip is not None:
        cv2.line(output, pivot, tip, (0, 80, 255), thickness + 1, cv2.LINE_AA)
    for name, point in points.items():
        xy = _point(point)
        if xy is None:
            continue
        cv2.circle(output, xy, max(2, round(scale * 3)), (50, 255, 80), -1, cv2.LINE_AA)
        if name.startswith('tick_'):
            label(name[5:], (xy[0] + 3, xy[1] - 3), (50, 255, 80), .6)
    rpm = prediction.get('rpm', {})
    polygon(prediction.get('rpm_roi_corners'), (255, 140, 0))
    for name, point in rpm.get('anchors', {}).items():
        xy = _point(point)
        if xy is not None:
            cv2.circle(output, xy, max(2, round(scale * 3)), (255, 200, 0), -1, cv2.LINE_AA)
            label('R' + str(name), (xy[0] + 3, xy[1] - 3), (255, 200, 0), .6)
    end = _point(rpm.get('bar_end'))
    if end is not None:
        cv2.drawMarker(output, end, (255, 0, 255), cv2.MARKER_CROSS,
                       max(8, round(scale * 12)), thickness + 1, cv2.LINE_AA)
    value = speed.get('value_kmh')
    speed_text = f'{value:.1f} km/h' if isinstance(value, (int, float)) and math.isfinite(value) else 'unavailable'
    value = rpm.get('value')
    rpm_text = f'{value:.0f}' if isinstance(value, (int, float)) and math.isfinite(value) else 'unavailable'
    line = max(24, round(31 * scale))
    label('Speed: ' + speed_text, (12, line), (255, 255, 255))
    label('RPM: ' + rpm_text, (12, line * 2 + 5), (0, 255, 255), 1.45)
    lcd_status = prediction.get('localizer', {}).get('lcd_status',
        prediction.get('lcd_status', 'detected' if prediction.get('lcd_polygon') else 'unavailable'))
    label(f'LCD: {lcd_status} | t={source_time:.2f}s',
          (12, height - 12), (255, 255, 255), .75)
    return output


def _encode_h264(source, target):
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
               '-map', '0:v:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
               '-movflags', '+faststart', str(target)]
    subprocess.run(command, check=True, capture_output=True)
    if not Path(target).is_file() or Path(target).stat().st_size <= 0:
        raise ValueError('H264 output was not created')


def _task_inference_flags(prediction):
    """Executed readers retain raw landmark visibility even if geometry rejects.

    An empty visibility dictionary is the predictor's explicit skipped-reader
    representation; unavailable numeric readings alone do not mean a skip.
    """
    return {'localizer': True,
            'speed': bool(prediction.get('speed', {}).get('point_visibility')),
            'rpm': bool(prediction.get('rpm', {}).get('point_visibility'))}


def process_video(source, output_dir, predictor, batch_size=16, last_seconds=120., label='video'):
    """Decode each selected frame once; batch prediction keeps original order."""
    import cv2
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    print(f'{label}: scanning source frame timestamps; prediction has not started.', flush=True)
    before = time.perf_counter()
    metadata = probe_video(source)
    initial_probe_seconds = time.perf_counter() - before
    first, stop, requested_start, clip_end = clip_indices(metadata, last_seconds)
    selected_count = stop - first
    # CFR output contains every selected frame. For VFR sources original PTS
    # stays in CSV/JSONL; playback cadence is explicitly documented as CFR.
    playback_duration = clip_end - metadata['timestamps'][first]
    output_fps = selected_count / playback_duration
    print(f'{label}: {selected_count} selected frames, source indices {first}..{stop-1}; '
          f'timestamp scan {initial_probe_seconds:.2f}s. Decoding source prefix before selected window.', flush=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError('Could not open source video')
    raw = output_dir / 'labeled_intermediate.mp4'
    final = output_dir / 'labeled.mp4'
    csv_path, jsonl_path = output_dir / 'predictions.csv', output_dir / 'predictions.jsonl'
    writer = None
    timers = {'initial_probe_seconds': initial_probe_seconds,
              'decode_seconds': 0., 'model_pipeline_seconds': 0., 'draw_seconds': 0.,
              'intermediate_encode_seconds': 0., 'prediction_log_seconds': 0.}
    decode_breakdown = {'prefix_decode_seconds': 0., 'selected_decode_seconds': 0.}
    processed, frame_index, batch, batch_indices = 0, 0, [], []
    batch_timings = []
    statuses = {'speed_available': 0, 'rpm_available': 0}
    task_frames = {'localizer': 0, 'speed': 0, 'rpm': 0, 'all_three': 0}
    output_shape = None
    with csv_path.open('w', newline='', encoding='utf8') as csv_file, jsonl_path.open('w', encoding='utf8') as log_file:
        columns = ['frame_index', 'clip_frame_index', 'source_time_seconds', 'speed_kmh',
                   'speed_status', 'rpm', 'rpm_status', 'lcd_status']
        table = csv.DictWriter(csv_file, fieldnames=columns)
        table.writeheader()
        def flush():
            nonlocal processed, batch, batch_indices, writer, output_shape
            if not batch:
                return
            before = time.perf_counter()
            predictions = predictor.predict_frames(batch)
            batch_wall = time.perf_counter() - before
            timers['model_pipeline_seconds'] += batch_wall
            batch_timings.append({'frames': len(batch), 'wall_seconds': batch_wall,
                                  'predictor_timing': getattr(predictor, 'last_batch_timing', None)})
            if len(predictions) != len(batch):
                raise ValueError('Predictor returned a different frame count')
            for frame, index, prediction in zip(batch, batch_indices, predictions):
                before = time.perf_counter()
                annotated = draw_prediction(frame, prediction, metadata['timestamps'][index], label)
                timers['draw_seconds'] += time.perf_counter() - before
                if output_shape is None:
                    output_shape = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), output_fps,
                                             (output_shape[1], output_shape[0]))
                    if not writer.isOpened():
                        raise ValueError('Could not create full-frame output video')
                if annotated.shape[:2] != output_shape:
                    raise ValueError('Variable decoded dimensions cannot preserve full-frame layout')
                before = time.perf_counter()
                writer.write(annotated)
                timers['intermediate_encode_seconds'] += time.perf_counter() - before
                before = time.perf_counter()
                speed, rpm = prediction.get('speed', {}), prediction.get('rpm', {})
                executed = _task_inference_flags(prediction)
                for task, ran in executed.items():
                    task_frames[task] += int(ran)
                task_frames['all_three'] += int(all(executed.values()))
                record = {'frame_index': index, 'clip_frame_index': processed,
                          'source_time_seconds': metadata['timestamps'][index],
                          'speed_kmh': speed.get('value_kmh'),
                          'speed_status': speed.get('reason') or ('available' if speed.get('value_kmh') is not None else 'unavailable'),
                          'rpm': rpm.get('value'),
                          'rpm_status': rpm.get('reason') or ('available' if rpm.get('value') is not None else 'unavailable'),
                          'lcd_status': prediction.get('localizer', {}).get('lcd_status',
                              prediction.get('lcd_status', 'detected' if prediction.get('lcd_polygon') else 'unavailable'))}
                table.writerow(record)
                log_file.write(json.dumps(dict(record, prediction=prediction), allow_nan=False) + '\n')
                statuses['speed_available'] += speed.get('value_kmh') is not None
                statuses['rpm_available'] += rpm.get('value') is not None
                processed += 1
                timers['prediction_log_seconds'] += time.perf_counter() - before
            csv_file.flush()
            log_file.flush()
            batch, batch_indices = [], []
            if processed % (batch_size * 10) == 0 or processed == selected_count:
                elapsed = time.perf_counter() - started
                pipeline_fps = processed / max(timers['model_pipeline_seconds'], 1e-12)
                print(f'{label}: {processed}/{selected_count} predicted frames; '
                      f'prediction_pipelineFPS={pipeline_fps:.2f} (conditional task coverage); '
                      f'elapsed={elapsed:.2f}s; export pending.', flush=True)
        try:
            while frame_index < stop:
                before = time.perf_counter()
                if frame_index < first:
                    ok, frame = capture.grab(), None
                else:
                    ok, frame = capture.read()
                decode_elapsed = time.perf_counter() - before
                timers['decode_seconds'] += decode_elapsed
                decode_breakdown['prefix_decode_seconds' if frame_index < first else 'selected_decode_seconds'] += decode_elapsed
                if not ok:
                    raise ValueError('Source decode ended before its verified timestamp sequence')
                if frame_index >= first:
                    batch.append(frame)
                    batch_indices.append(frame_index)
                    if len(batch) == batch_size:
                        flush()
                frame_index += 1
            flush()
        finally:
            capture.release()
            if writer is not None:
                writer.release()
    if processed != selected_count:
        raise ValueError('Not all frames in the requested final window were processed')
    print(f'{label}: predictions complete. Encoding full-frame H264 output now.', flush=True)
    before = time.perf_counter()
    _encode_h264(raw, final)
    timers['h264_encode_seconds'] = time.perf_counter() - before
    print(f'{label}: H264 encoding complete ({timers["h264_encode_seconds"]:.2f}s). Verifying output frame count.', flush=True)
    before = time.perf_counter()
    encoded = probe_video(final)
    if encoded['decoded_timestamp_count'] != processed:
        raise ValueError('Encoded output does not contain every predicted frame')
    timers['output_verification_seconds'] = time.perf_counter() - before
    timers['total_processing_seconds'] = time.perf_counter() - started
    report = {'status': 'complete', 'source_label': label, 'frames_processed': processed,
              'first_source_frame_index': first, 'stop_source_frame_index_exclusive': stop,
              'requested_start_seconds': requested_start, 'first_source_timestamp_seconds': metadata['timestamps'][first],
              'clip_end_seconds': clip_end, 'clip_duration_seconds': playback_duration,
              'requested_last_seconds': last_seconds, 'source_fps': metadata['fps'],
              'source_variable_frame_rate': metadata['variable_frame_rate'],
              'source_frame_count': len(metadata['timestamps']), 'output_playback_fps': output_fps,
              'output_frame_count': encoded['decoded_timestamp_count'], 'batch_size': batch_size,
              'original_decoded_height_width': list(output_shape),
              'encoded_height_width': [encoded['stream_height'], encoded['stream_width']],
              'timing': timers, 'model_pipeline_fps': processed / max(timers['model_pipeline_seconds'], 1e-12),
              'prediction_pipeline_fps': processed / max(timers['model_pipeline_seconds'], 1e-12),
              'decode_timing_breakdown': decode_breakdown,
              'decode_timing_note': 'Breakdown sums to timing.decode_seconds; it is not additional work and must not be added a second time.',
              'end_to_end_processing_fps': processed / max(timers['total_processing_seconds'], 1e-12),
              'availability': statuses, 'every_selected_frame_processed': True,
              'actual_task_inference_frames': task_frames,
              'actual_task_inference_coverage': {task: count / processed for task, count in task_frames.items()},
              'pipeline_work_scope': 'Conditional work: localizer runs every frame; speed requires a localized dashboard, RPM requires a valid RPM region. Pipeline FPS is not a benchmark with all three readers forced on every frame. Rejected numeric readings can still have executed their reader.',
              'batch_timings': batch_timings,
              'warmup_policy': 'No frames are discarded for warmup. First-batch startup costs are included. Models remain loaded across source videos.',
              'execution': getattr(predictor, 'execution_metadata', None),
              'timing_scope': 'Model pipeline includes preprocessing, prediction and geometry; CUDA execution is synchronized before predict_frames returns, and CPU execution is synchronous. Model construction and Drive IO are excluded from per-video processing time and included in run wall time.',
              'full_frame_preserved': True, 'audio_included': False,
              'playback_note': 'CFR playback preserves every selected frame; original timestamps are in CSV/JSONL. Playback FPS is not inference throughput.',
              'metric_scope': 'Visual development inspection on source recordings; no ground-truth speed/RPM accuracy is claimed.'}
    write_json(output_dir / 'report.json', report)
    # Intermediate is disposable only after frame-count verification succeeds.
    raw.unlink()
    return report


def _preserve_failure(store, output, folder_id, attempt, verified, failure):
    paths = [path for path in output.rglob('*') if path.is_file() and
             verified.get(str(path.resolve())) != sha256(path)]
    if not paths:
        return True
    try:
        rescue = output / 'failure_recovery.zip'
        with zipfile.ZipFile(rescue, 'w', zipfile.ZIP_STORED) as bundle:
            for path in paths:
                if path != rescue:
                    bundle.write(path, path.relative_to(output).as_posix())
        failure['recovery_archive'] = _verified_upload(store, rescue, folder_id,
                                                       attempt + '_failure_recovery.zip')
        write_json(output / 'FAILED.json', failure)
        _verified_upload(store, output / 'FAILED.json', folder_id, attempt + '_FAILED.json')
        return True
    except Exception:
        print('Failure artifacts are not verified in Drive. Runtime retained; attention required.', flush=True)
        return False


def run_video_tests(store, prepared, run_ref, config, auto_disconnect=True, download=False):
    """Test saved models, export outputs, then release only after durable saving.

    config: videos=[{id, sha256, size, name, label?}], last_seconds=120,
    batch_size=16, device='cuda' (explicit 'cpu' allowed in Colab only),
    cpu_threads=bounded host CPU count, expected_identity={}, code_commit=<pinned inference revision>.
    run_ref: training run ID or {run_id, complete_id} (preferred, exact receipt).
    """
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f') + '_' + uuid.uuid4().hex[:8]
    output = folder = None
    verified = {}
    started = time.perf_counter()
    saved_summary = None
    try:
        from .model import require_colab_device
        device = config.get('device', 'cuda')
        require_colab_device(device, inference_only=True)
        if device == 'cpu':
            import torch
            threads = config.get('cpu_threads', min(4, os.cpu_count() or 1))
            if type(threads) is not int or not 1 <= threads <= 32:
                raise ValueError('cpu_threads must be an integer from 1 to 32')
            torch.set_num_threads(threads)
        location = 'Google Colab ' + device.upper()
        print('Video tests: validating completed training receipt and private source references.', flush=True)
        batch_size, seconds = int(config.get('batch_size', 16)), float(config.get('last_seconds', 120))
        if not 1 <= batch_size <= 256 or not math.isfinite(seconds) or not 0 < seconds <= 120:
            raise ValueError('Use batch size 1..256 and a final window up to 120 seconds')
        videos = config.get('videos', [])
        if not videos or len(videos) > 10 or len({ref['id'] for ref in videos}) != len(videos):
            raise ValueError('Supply distinct verified video references')
        complete, receipt = resolve_complete(store, prepared, run_ref)
        workdir = Path(prepared['workdir']) / 'video_tests'
        cache = workdir / 'cache'
        print('Video tests: downloading/verifying selected model archive from Drive.', flush=True)
        archive = _download_verified(store, complete['archive'], cache / (_file_digest(complete['archive']) + '.zip'))
        expected = dict(config.get('expected_identity', {}))
        if prepared.get('dataset_manifest_sha256'):
            expected.setdefault('dataset_manifest_sha256', prepared['dataset_manifest_sha256'])
        paths, model_provenance = extract_models(archive, cache / _file_digest(complete['archive']), complete, expected)
        source_paths = []
        for index, ref in enumerate(videos, 1):
            print(f'Video tests: downloading/verifying source {index}/{len(videos)} from Drive.', flush=True)
            source_paths.append(_download_verified(store, ref, cache / ('source_' + ref['sha256'] + '.mp4')))
        folder = store.ensure_folder('video_test_' + attempt, complete['drive_run_folder_id'])
        output = workdir / attempt
        output.mkdir(parents=True, exist_ok=False)
        from .inference import DashboardPredictor
        print('Video tests: loading the three verified models on ' + location + '.', flush=True)
        predictor = DashboardPredictor(paths, batch_size=batch_size, device=device)
        print('Video tests: models loaded; beginning frame processing.', flush=True)
        reports = []
        def save(path, name=None):
            ref = _verified_upload(store, path, folder['id'], name)
            verified[str(Path(path).resolve())] = sha256(path)
            return ref
        for index, (ref, source) in enumerate(zip(videos, source_paths)):
            label = str(ref.get('label', f'video_{index + 1:02d}'))
            directory = output / f'video_{index + 1:02d}'
            report = process_video(source, directory, predictor, batch_size, seconds, label)
            report['source_reference'] = {key: ref[key] for key in ('id', 'sha256', 'size', 'name')}
            report['training_complete_id'] = receipt['id']
            report['execution'] = getattr(predictor, 'execution_metadata', None)
            write_json(directory / 'report.json', report)
            artifacts = {}
            for name in ('labeled.mp4', 'predictions.csv', 'predictions.jsonl', 'report.json'):
                print(f'{label}: uploading/verifying {name} in Drive.', flush=True)
                artifacts[name] = save(directory / name, directory.name + '_' + name)
            reports.append({'report': report, 'artifacts': artifacts})
        provenance = {'training_run_id': complete['run_id'], 'training_attempt': complete['attempt'],
                      'training_complete_id': receipt['id'], 'training_archive': complete['archive'],
                      'model_provenance': model_provenance, 'inference_code_commit': config.get('code_commit'),
                      'inference_thresholds': getattr(predictor, 'thresholds', None),
                      'predictor_metadata': getattr(predictor, 'metadata', None),
                      'batch_size': batch_size, 'last_seconds': seconds,
                      'source_references': [{key: ref[key] for key in ('id', 'sha256', 'size', 'name')} for ref in videos],
                      'execution': getattr(predictor, 'execution_metadata', None),
                      'no_old_cpu_matching': True, 'location': location}
        write_json(output / 'provenance.json', provenance)
        print('Video tests: saving provenance and packaging the combined results archive.', flush=True)
        save(output / 'provenance.json')
        result_archive = output / 'video_results.zip'
        with zipfile.ZipFile(result_archive, 'w', zipfile.ZIP_STORED) as bundle:
            for path in sorted(output.rglob('*')):
                if path.is_file() and path != result_archive:
                    bundle.write(path, path.relative_to(output).as_posix())
        print('Video tests: uploading/verifying the combined results archive in Drive.', flush=True)
        saved_archive = save(result_archive)
        saved_summary = {'status': 'complete', 'attempt': attempt, 'training_run_id': complete['run_id'],
                         'training_attempt': complete['attempt'], 'training_complete_id': receipt['id'],
                         'training_archive': complete['archive'], 'model_provenance': model_provenance,
                         'inference_code_commit': config.get('code_commit'), 'videos': reports,
                         'results_archive': saved_archive, 'drive_output_folder_id': folder['id'],
                         'total_wall_seconds_including_download_save': time.perf_counter() - started,
                         'execution': getattr(predictor, 'execution_metadata', None),
                         'all_result_files_verified_in_drive': True, 'location': location}
        write_json(output / 'COMPLETE.json', saved_summary)
        completion_ref = save(output / 'COMPLETE.json')
        saved_summary['completion_receipt'] = completion_ref
        print('Video tests saved and verified in Drive:', folder['id'], flush=True)
        for item in reports:
            report = item['report']
            print(f'{report["source_label"]}: {report["frames_processed"]} frames; '
                  f'model pipeline {report["model_pipeline_fps"]:.2f} FPS; '
                  f'end-to-end {report["end_to_end_processing_fps"]:.2f} FPS; '
                  f'{report["timing"]["total_processing_seconds"]:.2f} s processing.', flush=True)
    except BaseException as error:
        failure = {'status': 'failed', 'attempt': attempt, 'exception_type': type(error).__name__}
        preserved = output is None or not any(path.is_file() for path in output.rglob('*'))
        if output is not None and folder is not None:
            preserved = _preserve_failure(store, output, folder['id'], attempt, verified, failure)
        if auto_disconnect and preserved:
            print('Required artifacts are preserved. Releasing the runtime.', flush=True)
            _disconnect()
        raise
    if download:
        try:
            from google.colab import files
            files.download(str(result_archive))
            print('Browser download requested; durable Drive copy is already verified.', flush=True)
        except Exception:
            print('Browser download could not start. Download the verified Drive results archive.', flush=True)
    if auto_disconnect:
        print('All video outputs are verified in Drive. Releasing the runtime.', flush=True)
        _disconnect()
    return saved_summary
