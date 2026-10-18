"""Colab CUDA only: reusable FAST, UNGATED LCD/speed/RPM video diagnostic.

Every selected source frame enters all three frozen CNNs once. RPM uses the
unchanged contour-positioning experiment; there is no SIFT, training, quality
suppression, smoothing or previous-reading fallback. Raw readings are unvalidated.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import importlib.util
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
BASE_SHA = '0c5df9a13e5cd904d76339a74de8d84675965041ee166eacf94e933214947df3'
FAST_SHA = '68c299ada88ff433fea7e57ac379823ee9d6848d61f0605556642ecc3c5957e0'
BATCHES = {'lcd': 64, 'speed': 16, 'rpm': 128}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def protected_module(name, filename, expected):
    path = HERE / filename
    if sha256(path) != expected: raise ValueError('Changed protected experiment helper: ' + filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def positive_rate(value):
    try:
        rate = Fraction(str(value))
        return rate if 0 < float(rate) <= 1000 else None
    except (ValueError, ZeroDivisionError, TypeError): return None


def probe(path):
    metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], text=True))
    streams = [s for s in metadata['streams'] if s['codec_type'] == 'video']
    if not streams: raise ValueError('No video stream')
    stream = streams[0]
    sar_text = stream.get('sample_aspect_ratio', '1:1')
    if sar_text not in (None, 'N/A', '0:1') and Fraction(str(sar_text).replace(':', '/')) != 1:
        raise ValueError('Non-square source pixels are unsupported; export square-pixel video before uploading')
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_entries', 'frame=best_effort_timestamp_time,pkt_duration_time,duration_time', '-of', 'json', str(path)], text=True))
    frames = data.get('frames', [])
    timestamps = np.asarray([float(f['best_effort_timestamp_time']) for f in frames], float)
    if not len(timestamps) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Source needs finite, strictly increasing decoded video timestamps')
    first_pts = float(timestamps[0]); timestamps -= first_pts
    rate = positive_rate(stream.get('avg_frame_rate')) or positive_rate(stream.get('r_frame_rate'))
    if rate is None:
        rate = Fraction(str(1 / np.median(np.diff(timestamps)) if len(timestamps) > 1 else 30)).limit_denominator(100000)
    last_duration = float(frames[-1].get('duration_time', frames[-1].get('pkt_duration_time', 1 / float(rate))))
    if not math.isfinite(last_duration) or last_duration <= 0: last_duration = 1 / float(rate)
    duration = float(stream.get('duration', timestamps[-1] + last_duration))
    if not math.isfinite(duration) or duration <= timestamps[-1]: duration = float(timestamps[-1] + last_duration)
    rotation = float(stream.get('tags', {}).get('rotate', 0))
    for item in stream.get('side_data_list', []):
        if 'rotation' in item: rotation = float(item['rotation'])
    if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > 1e-4:
        raise ValueError('Unsupported non-90-degree display-matrix rotation; keep orientation metadata at0/90/180/270')
    width, height = int(stream['width']), int(stream['height'])
    if round(rotation / 90) % 2: width, height = height, width
    audio_streams = [s for s in metadata['streams'] if s['codec_type'] == 'audio']
    audio_start = float(audio_streams[0].get('start_time', 'nan')) if audio_streams else None
    return {'timestamps': timestamps, 'duration': duration, 'source_first_pts': first_pts,
        'source_average_fps': float(rate), 'source_average_fps_fraction': str(rate),
        'encoded_width': int(stream['width']), 'encoded_height': int(stream['height']),
        'display_width': width, 'display_height': height, 'metadata_rotation_degrees': rotation,
        'sample_aspect_ratio': sar_text, 'has_audio': bool(audio_streams),
        'audio_start_time': audio_start if audio_start is not None and math.isfinite(audio_start) else None,
        'audio_video_starts_aligned': bool(audio_start is not None and math.isfinite(audio_start) and abs(audio_start - first_pts) < .001),
        'vfr_detected': bool(len(timestamps) > 2 and np.ptp(np.diff(timestamps)) > max(1e-4, .01 / float(rate)))}


def select_timeline(info, start_seconds, max_seconds, output_fps):
    if not math.isfinite(start_seconds) or not math.isfinite(max_seconds) or min(start_seconds, max_seconds) < 0:
        raise ValueError('Time limits must be finite and nonnegative; max_seconds0 means whole remainder')
    times = np.asarray(info['timestamps'], float)
    requested_end = min(info['duration'], start_seconds + max_seconds) if max_seconds else info['duration']
    indices = np.flatnonzero((times >= start_seconds) & (times < requested_end))
    if not len(indices): raise ValueError('Requested interval contains no source frame')
    selected = times[indices]; actual_start = float(selected[0]); duration = float(requested_end - actual_start)
    rate = positive_rate(output_fps) if output_fps else positive_rate(info['source_average_fps_fraction'])
    if rate is None: raise ValueError('Invalid output FPS')
    relative = selected - selected[0]
    output_count = max(1, int(math.floor(duration * float(rate) + .5)))
    output_times = np.arange(output_count, dtype=float) / float(rate)
    right = np.searchsorted(relative, output_times).clip(0, len(relative) - 1)
    left = (right - 1).clip(0, len(relative) - 1)
    mapping = np.where(np.abs(relative[left] - output_times) <= np.abs(relative[right] - output_times), left, right)
    return {'indices': indices, 'timestamps': selected, 'relative_timestamps': relative,
        'actual_start_seconds': actual_start, 'requested_end_seconds': requested_end, 'duration': duration,
        'output_fps': float(rate), 'output_fps_fraction': str(rate), 'output_mapping': mapping,
        'output_times': output_times, 'output_frames': output_count}


def layout(width, height):
    scale = min(1., 960 / width, 720 / height)
    sw, sh = max(1, round(width * scale)), max(1, round(height * scale))
    left_width = max(460, sw)
    return {'width': int(2 * math.ceil((left_width + 512 + 24) / 2)), 'height': 1024,
        'source_display_width': sw, 'source_display_height': sh, 'source_left': (left_width - sw) // 2,
        'source_top': 150, 'source_scale_x': sw / width, 'source_scale_y': sh / height,
        'right_left': left_width + 24, 'source_panel_width': left_width}


def line_text(board, text, xy, scale=.55, color=(235, 235, 235)):
    cv2.putText(board, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def format_raw(value, digits=0):
    return f'{value:.{digits}f}' if value is not None and math.isfinite(value) else 'undefined'


def render(frame, row, gray, strip, board_layout, processing_fps):
    spec = board_layout
    board = np.full((spec['height'], spec['width'], 3), 18, np.uint8)
    source = cv2.resize(frame, (spec['source_display_width'], spec['source_display_height']), interpolation=cv2.INTER_AREA)
    sx, sy = spec['source_scale_x'], spec['source_scale_y']
    def points(xy): return np.rint(np.asarray(xy, float) * [sx, sy]).astype(np.int32)
    mapping = row['rpm']['input_mapping']
    contours = list(row['lcd'].get('contours_source_xy', []))
    contours += [item['contour_source_xy'] for item in mapping.get('component_candidates', [])]
    for contour in contours:
        xy = np.asarray(contour, float)
        if xy.ndim == 2 and len(xy) and np.isfinite(xy).all(): cv2.polylines(source, [points(xy)], True, (255, 220, 40), 1)
    quad = np.asarray(mapping.get('lcd_quad_source_xy', []), float)
    if quad.shape == (4, 2) and np.isfinite(quad).all(): cv2.polylines(source, [points(quad)], True, (0, 255, 140), 2)
    landmarks = np.asarray(row['speed']['points_source_xy'], float)
    if np.isfinite(landmarks[:2]).all():
        q = points(landmarks[:2]); okay, a, b = cv2.clipLine((0, 0, source.shape[1], source.shape[0]), tuple(q[0]), tuple(q[1]))
        if okay: cv2.line(source, a, b, (0, 140, 255), 2, cv2.LINE_AA)
    for index, xy in enumerate(landmarks):
        if not np.isfinite(xy).all(): continue
        q = points([xy])[0]
        if not (0 <= q[0] < source.shape[1] and 0 <= q[1] < source.shape[0]): continue
        name = 'P' if index == 0 else 'TIP' if index == 1 else str((index - 2) * 20)
        cv2.circle(source, tuple(q), 3 if index < 2 else 2, (0, 140, 255), -1)
        line_text(source, name, (min(int(q[0]) + 4, source.shape[1] - 34), max(12, int(q[1]) - 4)), .36, (0, 140, 255))
    y, x = spec['source_top'], spec['source_left']
    board[y:y + source.shape[0], x:x + source.shape[1]] = source
    right = spec['right_left']
    board[150:662, right:right + 512] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    board[710:966, right:right + 192] = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
    rpm = format_raw(row['rpm']['raw_rpm_approx'])
    if mapping['placeholder']: rpm = 'INVALID INPUT'
    line_text(board, 'Raw RPM ' + rpm, (12, 49), 1.4, (0, 255, 180))
    line_text(board, 'Raw speed: ' + format_raw(row['speed']['geometry']['speed_kmh'], 1) + ' km/h', (12, 84), .72)
    line_text(board, 'UNGATED DIAGNOSTIC | unvalidated outputs | no reading-quality checks', (12, 112), .55, (90, 190, 255))
    line_text(board, f"Frame {row['source_frame'] + 1} | {row['source_time_s']:.2f}s | {processing_fps:.1f} FPS", (12, 136), .51)
    line_text(board, 'Actual LCD + speed input: gray512 (unmasked)', (right, 136), .53)
    line_text(board, 'Actual RPM input: 192x256', (right, 695), .57)
    line_text(board, 'Green: RPM mapping', (right + 208, 746), .48, (0, 255, 140))
    line_text(board, 'Cyan: LCD candidates', (right + 208, 774), .48, (255, 220, 40))
    line_text(board, 'Orange: P pivot / TIP', (right + 208, 802), .48, (0, 140, 255))
    line_text(board, 'needle + numbered ticks', (right + 208, 830), .46, (0, 140, 255))
    line_text(board, 'RPM scale provisional', (right + 208, 870), .47)
    line_text(board, 'Corner orientation heuristic', (right + 208, 898), .43)
    line_text(board, 'not arbitrary-angle verified', (right + 208, 926), .43)
    line_text(board, 'RPM mapping: ' + mapping['method'], (12, 988), .50)
    line_text(board, 'Raw outputs. Full frame shown. Speed input unmasked; trained masked. No smoothing.', (12, 1013), .48)
    return board


def read_exact(stream, size):
    result = bytearray()
    while len(result) < size:
        block = stream.read(size - len(result))
        if not block: raise EOFError('Video decoder ended before all timestamp-indexed frames were read')
        result.extend(block)
    return result


def run(args, base, fast, nets, speed_reader, rpm_model_identity, rpm_calibration, rpm_runtime, letterbox, phases, overall_started):
    dest = args.output_root.resolve()
    started = time.perf_counter(); info = probe(args.video)
    timeline = select_timeline(info, args.start_seconds, args.max_seconds, args.output_fps)
    phases['source_probe_seconds'] = time.perf_counter() - started
    width, height = info['display_width'], info['display_height']
    spec = layout(width, height)
    indices = timeline['indices']; count = len(indices)
    chunk_size = min(args.chunk_size, max(1, (256 * 1024 * 1024) // (width * height * 3)))
    output = dest / 'annotated_fast_lcd_speed_rpm.mp4'
    decoder_command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(args.video), '-map', '0:v:0',
        '-vf', f'select=between(n\\,{int(indices[0])}\\,{int(indices[-1])})', '-frames:v', str(count),
        '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
    encoder_command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f"{spec['width']}x{spec['height']}", '-r', timeline['output_fps_fraction'], '-i', 'pipe:0']
    audio = args.audio == 'auto' and info['has_audio'] and info['audio_video_starts_aligned']
    audio_reason = 'aligned_track_AAC_transcode' if audio else ('disabled_by_user' if args.audio == 'none' else 'no_audio_track' if not info['has_audio'] else 'nonaligned_or_unknown_stream_start_omitted')
    if audio: encoder_command += ['-ss', str(timeline['actual_start_seconds']), '-i', str(args.video)]
    encoder_command += ['-map', '0:v:0']
    if audio: encoder_command += ['-map', '1:a:0?', '-c:a', 'aac', '-b:a', '128k', '-af', 'asetpts=PTS-STARTPTS']
    else: encoder_command += ['-an']
    encoder_command += ['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20', '-threads', '2', '-pix_fmt', 'yuv420p',
        '-t', str(timeline['output_frames'] / timeline['output_fps']), '-movflags', '+faststart', str(output)]
    base.write_json(dest / 'commands.json', {'decoder': decoder_command, 'encoder': encoder_command})
    decoder_log = (dest / 'decoder.log').open('xb'); encoder_log = (dest / 'encoder.log').open('xb')
    decoder = subprocess.Popen(decoder_command, stdout=subprocess.PIPE, stderr=decoder_log)
    encoder = subprocess.Popen(encoder_command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encoder_log)
    repeats = np.bincount(timeline['output_mapping'], minlength=count)
    preview_indices = set(np.linspace(0, count - 1, min(6, count), dtype=int).tolist())
    adapter = fast.FastRPMInputs(rpm_calibration, rpm_runtime, speed_reader.gates['pixel_threshold'], base)
    durations = Counter(); methods = Counter(); output_frames = 0; processed = 0; all_chunk_timings = []
    csv_fields = ['source_frame', 'source_time_s', 'speed_kmh_raw', 'legacy_speed_head_unvalidated', 'rpm_approx_raw',
        'fill_fraction_raw', 'readability_raw', 'rpm_mapping', 'placeholder', 'playback_repetitions']
    loop_started = time.perf_counter()
    try:
        with (dest / 'source_predictions.jsonl').open('w', encoding='utf8') as predictions, (dest / 'predictions.csv').open('w', newline='', encoding='utf8') as csv_file, ThreadPoolExecutor(max_workers=2, thread_name_prefix='fast_lcd_geometry') as pool:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields); writer.writeheader()
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                begin = time.perf_counter()
                frames = [np.frombuffer(read_exact(decoder.stdout, width * height * 3), np.uint8).reshape(height, width, 3) for _ in range(start, stop)]
                durations['decode_wait_seconds'] += time.perf_counter() - begin
                source_indices = indices[start:stop].astype(int).tolist()
                rows, timing, grays, strips = base.process_chunk(frames, source_indices, timeline['timestamps'][start:stop].tolist(), nets, speed_reader, adapter, pool, letterbox)
                all_chunk_timings.append(timing); durations['reader_wall_seconds'] += timing['reader_wall_seconds']
                for offset, (frame, row, gray, strip) in enumerate(zip(frames, rows, grays, strips)):
                    local = start + offset; row['playback_repetitions'] = int(repeats[local])
                    row['input_display_orientation'] = 'ffmpeg metadata autorotation; full frame, no model-input ROI for speed/LCD'
                    begin = time.perf_counter()
                    board = render(frame, row, gray, strip, spec, stop / durations['reader_wall_seconds'])
                    durations['render_seconds'] += time.perf_counter() - begin
                    begin = time.perf_counter()
                    for _ in range(int(repeats[local])):
                        encoder.stdin.write(board.tobytes()); output_frames += 1
                    durations['encoder_pipe_seconds'] += time.perf_counter() - begin
                    begin = time.perf_counter()
                    if local in preview_indices:
                        if not cv2.imwrite(str(dest / f'preview_{row["source_frame"]:07d}.jpg'), board): raise RuntimeError('Preview write failed')
                        cv2.imwrite(str(dest / f'speed_input_{row["source_frame"]:07d}.png'), gray)
                        cv2.imwrite(str(dest / f'rpm_input_{row["source_frame"]:07d}.png'), strip)
                    predictions.write(json.dumps(base.safe_json(row), allow_nan=False) + '\n')
                    writer.writerow(dict(zip(csv_fields, [row['source_frame'], row['source_time_s'], row['speed']['geometry']['speed_kmh'],
                        row['speed']['raw_legacy_head_kmh'], row['rpm']['raw_rpm_approx'], row['rpm']['raw_fill_fraction'],
                        row['rpm']['raw_readability_score'], row['rpm']['input_mapping']['method'], row['rpm']['input_mapping']['placeholder'], int(repeats[local]) ])))
                    methods[row['rpm']['input_mapping']['method']] += 1; processed += 1
                    durations['prediction_and_preview_io_seconds'] += time.perf_counter() - begin
                if processed == count or start == 0 or processed // 512 != start // 512:
                    print(json.dumps({'frames_processed': processed, 'source_frames_selected': count,
                        'reader_fps': processed / durations['reader_wall_seconds'],
                        'full_loop_fps': processed / (time.perf_counter() - loop_started)}, allow_nan=False), flush=True)
            encoder.stdin.close(); decoder.stdout.close()
            begin = time.perf_counter()
            if decoder.wait(timeout=60) != 0 or encoder.wait(timeout=120) != 0:
                raise RuntimeError('Decoder/encoder failed; see logs')
            durations['encoder_decoder_finish_seconds'] = time.perf_counter() - begin
    finally:
        for process in (decoder, encoder):
            if process.poll() is None: process.kill(); process.wait()
        decoder_log.close(); encoder_log.close()
    phases['decode_inference_render_encode_seconds'] = time.perf_counter() - loop_started
    phases.update(dict(durations))
    if processed != count or output_frames != timeline['output_frames']: raise RuntimeError('Source/output frame accounting differs')
    network = base.network_summary(nets)
    if any(row['inference_samples'] != count for row in network.values()): raise RuntimeError('Every model must see every selected source frame exactly once')
    begin = time.perf_counter()
    decoded = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,codec_name,avg_frame_rate,nb_read_frames,duration', '-of', 'json', str(output)], text=True))['streams'][0]
    if (int(decoded['nb_read_frames']), int(decoded['width']), int(decoded['height']), decoded['codec_name']) != (output_frames, spec['width'], spec['height'], 'h264'):
        raise RuntimeError('Final H264 decode/frame/dimension verification failed')
    if not math.isclose(float(Fraction(decoded['avg_frame_rate'])), timeline['output_fps'], rel_tol=1e-6): raise RuntimeError('Final playback FPS mismatch')
    phases['output_decode_verification_seconds'] = time.perf_counter() - begin
    with (dest / 'batch_timings.jsonl').open('w', encoding='utf8') as stream:
        for row in nets.timings: stream.write(json.dumps(row, allow_nan=False) + '\n')
    with (dest / 'playback_mapping.jsonl').open('w', encoding='utf8') as stream:
        for index, source in enumerate(timeline['output_mapping']):
            stream.write(json.dumps({'output_frame': index, 'output_time_s': float(timeline['output_times'][index]),
                'source_frame': int(indices[source]), 'source_time_s': float(timeline['timestamps'][source])}) + '\n')
    base.write_json(dest / 'chunk_timings.json', all_chunk_timings)
    source_info = {key: value for key, value in info.items() if key != 'timestamps'}
    begin = time.perf_counter()
    output_digest = sha256(output)
    input_digest = getattr(args, 'verified_input_sha256', None) or sha256(args.video)
    phases['artifact_hash_seconds'] = time.perf_counter() - begin
    summary = {'status': 'complete', 'diagnostic_mode': 'UNGATED_FAST_LCD_POSITIONING', 'training_performed': False,
        'original_frozen_pipeline_files_modified': False, 'sift_calls': 0,
        'output': str(output), 'output_sha256': output_digest, 'output_bytes': output.stat().st_size,
        'output_width': spec['width'], 'output_height': spec['height'], 'output_frames': output_frames,
        'output_fps': timeline['output_fps'], 'output_fps_fraction': timeline['output_fps_fraction'],
        'source_frames_total': len(info['timestamps']), 'source_frames_processed': processed,
        'source_first_frame': int(indices[0]), 'source_last_frame': int(indices[-1]), 'source_metadata': source_info,
        'requested_start_seconds': args.start_seconds, 'requested_max_seconds': args.max_seconds,
        'effective_start_seconds': timeline['actual_start_seconds'], 'selected_duration_seconds': timeline['duration'],
        'output_duration_seconds': output_frames / timeline['output_fps'], 'audio_preserved': audio,
        'audio_policy': 'First track preserved only when source video/audio starts match within1ms, then trimmed at selected video-relative start and AAC-transcoded. Nonaligned/unknown offsets omitted.',
        'audio_reason': audio_reason,
        'playback_mapping_policy': 'Nearest source timestamp at constant outputFPS. Everyselectedsourceframe inferred once; playback only may omit/repeat.',
        'source_frames_omitted_from_playback': int(np.count_nonzero(repeats == 0)),
        'source_frames_repeated_in_playback': int(np.count_nonzero(repeats > 1)),
        'extra_playback_repetitions': int(np.maximum(repeats - 1, 0).sum()),
        'maximum_model_batch_sizes': BATCHES, 'requested_chunk_size': args.chunk_size, 'effective_chunk_size': chunk_size,
        'network_timing': network, 'reader_processing_fps': processed / durations['reader_wall_seconds'],
        'decode_inference_render_encode_fps': processed / phases['decode_inference_render_encode_seconds'],
        'phase_seconds': phases, 'elapsed_seconds': time.perf_counter() - overall_started,
        'input_sha256': input_digest, 'input_filename': args.video.name, 'script_sha256': sha256(__file__),
        'protected_base_sha256': BASE_SHA, 'protected_fast_positioning_sha256': FAST_SHA, 'code_commit': base.CODE_COMMIT,
        'speed_reader_metadata': speed_reader.metadata, 'rpm_reader_identity': rpm_model_identity,
        'rpm_input_methods': dict(methods), 'speed_input_lcd_masked': False, 'no_prediction_quality_gates': True,
        'all_outputs_unvalidated': True, 'ground_truth_accuracy': None,
        'original_source_aspect_preserved': True, 'display_layout': spec,
        'scope': 'Fast mapping is exactly the prior contour-ranking diagnostic. Full-frame LCD/speed input gray512; speed trainedmasked. RPMusesunchangedcanonicalstrip, provisionalscale, heuristicuprightcorners; arbitraryorientationnotvalidated. Numericalplaceholder outputneverphysicalRPM.',
        'timing_scope': 'Maincountsexclude warmup. Reader includes all3CNN transfers/forwards, letterbox, rawgeometry, overlapping2worker LCDquad mapping. Rendering/encoding/probe/verification separate. Pipeline FPS is throughput, not perframe online latency. GPUbatch dimensions may reduce at chunk tails or high native resolutions.'}
    base.write_json(dest / 'summary.json', summary)
    base.write_json(dest / 'timing_summary.json', {key: summary[key] for key in ('source_frames_processed', 'output_frames', 'output_fps', 'network_timing', 'reader_processing_fps', 'decode_inference_render_encode_fps', 'phase_seconds', 'elapsed_seconds', 'timing_scope')})
    base.write_json(dest / 'completion.json', {'status': 'complete_and_fully_decoded', 'summary_sha256': sha256(dest / 'summary.json'),
        'output_sha256': summary['output_sha256'], 'output_bytes': summary['output_bytes'], 'source_frames_processed': processed,
        'model_samples': {name: processed for name in BATCHES}, 'output_frames': output_frames, 'script_sha256': sha256(__file__)})
    print(json.dumps({key: summary[key] for key in ('status', 'output', 'source_frames_processed', 'output_frames', 'output_fps', 'reader_processing_fps', 'decode_inference_render_encode_fps', 'elapsed_seconds')}, indent=2), flush=True)


def main():
    started = time.perf_counter(); parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository-root', 'provenance-manifest', 'video', 'output-root', 'lcd-checkpoint', 'speed-checkpoint',
                 'speed-calibration', 'rpm-dataset', 'rpm-checkpoint', 'rpm-reference'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-video-sha256')
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    parser.add_argument('--start-seconds', type=float, default=0)
    parser.add_argument('--max-seconds', type=float, default=0)
    parser.add_argument('--output-fps', type=float, default=0)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--audio', choices=['auto', 'none'], default='auto')
    args = parser.parse_args()
    if args.chunk_size < 1 or args.chunk_size > 128: raise ValueError('Chunk size must be1..128')
    try: colab = importlib.util.find_spec('google.colab') is not None
    except (ValueError, ImportError, ModuleNotFoundError): colab = False
    import torch
    if not colab or not torch.cuda.is_available(): raise RuntimeError('All model/video execution requires GoogleColabCUDA')
    if cv2.__version__ != '4.14.0': raise ValueError('Expected OpenCV4.14.0 for frozen preprocessing')
    base = protected_module('uploaded_base', 'evaluate_lcd_speed_rpm_batched.py', BASE_SHA)
    fast = protected_module('uploaded_fast', 'benchmark_lcd_positioning.py', FAST_SHA)
    begin = time.perf_counter(); repo = args.repository_root.resolve()
    base.verify_sources(repo, args.provenance_manifest)
    for path, expected in ((args.lcd_checkpoint, base.LCD_SHA256), (args.speed_checkpoint, base.SPEED_SHA256),
        (args.rpm_checkpoint, base.RPM_SHA256), (args.speed_calibration, base.CALIBRATION_SHA256)):
        if sha256(path) != expected: raise ValueError('Unexpected frozen model/calibration: ' + str(path))
    args.verified_input_sha256 = sha256(args.video)
    if args.expected_video_sha256 and args.verified_input_sha256 != args.expected_video_sha256: raise ValueError('Uploaded video SHA differs from job')
    phases = {'asset_verification_seconds': time.perf_counter() - begin}
    args.output_root.resolve().mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1); sys.path.insert(0, str(repo))
    from learned_lcd_v1.inference import LearnedLCDSpeedReader, letterbox
    from rpm_training_v1 import infer_runtime
    begin = time.perf_counter()
    reader = LearnedLCDSpeedReader(args.lcd_checkpoint, args.speed_checkpoint, args.speed_calibration, device='cuda')
    rpm_model, _, rpm_calibration, rpm_identity = infer_runtime.load_verified_model(args.rpm_dataset, args.rpm_checkpoint,
        device='cuda', expected_sha256=base.RPM_SHA256)
    if sha256(args.rpm_reference) != rpm_calibration['reference_sha256']: raise ValueError('RPM reference identity mismatch')
    rpm_identity['sift_initialized_or_called'] = False
    models = {'lcd': reader.lcd_model, 'speed': reader.speed_model, 'rpm': rpm_model}
    for model in models.values(): model.eval()
    nets = base.Networks(models, torch); nets.selected = dict(BATCHES)
    phases['model_load_seconds'] = time.perf_counter() - begin
    begin = time.perf_counter()
    for name, batch in BATCHES.items():
        shape = (256, 192) if name == 'rpm' else (512, 512)
        nets.run(name, [np.zeros(shape, np.uint8)] * batch, [-1] * batch, record=False)
    phases['excluded_warmup_seconds'] = time.perf_counter() - begin
    run(args, base, fast, nets, reader, rpm_identity, rpm_calibration, infer_runtime, letterbox, phases, started)


if __name__ == '__main__': main()
