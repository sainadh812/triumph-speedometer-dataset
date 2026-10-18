"""Colab-only 30FPS video inspection of learned LCD + frozen V5 speed."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from learned_lcd_v1.inference import LearnedLCDSpeedReader, sha256, SPEED_CHECKPOINT_SHA256
from real_adaptation_v4.infer_video import probe_video, build_timeline

FPS = 30
BOARD = (1280, 960)
# Immutable reported results from the completed Colab V5 test archive. These
# values are for AFTER-THE-RUN comparison only and never enter reader.predict.
PREVIOUS_V5_COVERAGE = {
    'f6aa7bcd1b4dc2f5e93ad6a91cf2e865f2f05391485710cb716d568116dc6828': {'accepted': 602, 'output_frames': 602},
    '6125503654f3d2a06ac67765bfc4f5af670f027335a501d50e2a88412b8ffaf9': {'accepted': 340, 'output_frames': 540},
}
PREVIOUS_V5_INFERENCE_SHA256 = 'a3db8e418d79caf205bfb70b2b89ac682bdf4c0d28822fa3d7a05c98396eeadb'


def require_colab():
    try: available = importlib.util.find_spec('google.colab') is not None
    except ModuleNotFoundError: available = False
    if not available:
        raise RuntimeError('All model/video evaluation is authorized in Google Colab only')


def json_safe(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()
                if key not in ('model_input', 'candidate_masked_input', 'lcd_probability', 'lcd_mask')}
    if isinstance(value, (list, tuple)): return [json_safe(item) for item in value]
    return value


def text(image, label, x, y, scale=.6, color=(225, 232, 240), thickness=1):
    cv2.putText(image, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def wrapped(image, label, x, y, width=45):
    line = ''
    for word in str(label).replace('_', ' ').split():
        if len(line)+len(word)+1 > width:
            text(image, line, x, y, .45); y += 20; line = word
        else: line = (line+' '+word).strip()
    if line: text(image, line, x, y, .45)


def display_label(video, index):
    """Presentation only: filenames are never supplied to either model."""
    match = re.search(r'(\d{1,2})\.(\d{2})\.(\d{2})\s*(AM|PM)', Path(video).stem, re.I)
    if match:
        hour, minute, second = int(match[1]), int(match[2]), int(match[3])
        if 1 <= hour <= 12 and minute < 60 and second < 60:
            hour = hour % 12 + (12 if match[4].upper() == 'PM' else 0)
            return f'learned_lcd_{hour:02d}{minute:02d}{second:02d}'
    return f'learned_lcd_clip{index}'


def render(frame, result, index, total, timestamp, metadata, label):
    board = np.full((BOARD[1], BOARD[0], 3), (29, 24, 20), np.uint8)
    source = frame.copy()
    color = (110, 240, 155) if result['accepted'] else (65, 180, 250)
    points = result.get('points_source_xy')
    if points is not None:
        points = np.asarray(points, float)
        if points.shape == (12, 2) and np.isfinite(points).all() and np.max(np.abs(points)) < 10000:
            integer = np.rint(points).astype(int)
            cv2.line(source, tuple(integer[0]), tuple(integer[1]), color, 2, cv2.LINE_AA)
            for j, point in enumerate(integer):
                available = result['availability'][j] >= .5
                point_color = color if available else (150, 150, 150)
                cv2.circle(source, tuple(point), 3, point_color, 1, cv2.LINE_AA)
                if j >= 2: text(source, str((j-2)*20), point[0]+4, point[1]-4, .34, point_color)
    for contour in result.get('lcd_contours_source_xy', []):
        cv2.polylines(source, [np.rint(contour).astype(np.int32)], True, (255, 225, 30), 2, cv2.LINE_AA)
    height, width = source.shape[:2]
    scale = min(760/width, 835/height)
    nw, nh = round(width*scale), round(height*scale)
    x, y = 20+(760-nw)//2, 100+(835-nh)//2
    board[y:y+nh, x:x+nw] = cv2.resize(source, (nw, nh), interpolation=cv2.INTER_AREA)
    text(board, 'TRIUMPH | Learned LCD + frozen speed', 20, 35, .85, thickness=2)
    text(board, 'Full original frame | no image alignment or crop', 20, 65, .57)
    text(board, f"LCD epoch {metadata.get('lcd_best_epoch')} | frozen V5 speed epoch 11", 20, 87, .46)
    text(board, label, 815, 43, .52)
    text(board, 'INDICATED SPEED', 815, 105, .65)
    value = f"{result['speed_kmh']:.1f} km/h" if result['accepted'] else 'Unavailable'
    text(board, value, 815, 153, 1.05, color, 2)
    wrapped(board, 'Checks passed; not proof of accuracy' if result['accepted'] else result['reason'], 815, 190)
    wrapped(board, 'Stage: '+str(result.get('failure_stage') or 'accepted'), 815, 247)
    detection = result.get('lcd_detection') or {}
    score, coverage = detection.get('component_mean_probability'), detection.get('sensor_coverage_fraction')
    boundary = detection.get('boundary_uncertain_fraction')
    text(board, 'LCD score: '+(f'{score:.3f}' if score is not None else '--')+' (uncalibrated)', 815, 290, .46)
    text(board, 'Sensor mask coverage: '+(f'{100*coverage:.1f}%' if coverage is not None else '--'), 815, 315, .46)
    text(board, 'Uncertain boundary: '+(f'{100*boundary:.1f}%' if boundary is not None else '--'), 815, 340, .46)
    model_input = result.get('model_input')
    probability = result.get('lcd_probability')
    if model_input is not None:
        inset = cv2.cvtColor(model_input, cv2.COLOR_GRAY2BGR)
        inset_label = 'Actual speed input: gray512 / LCD128'
    elif result.get('candidate_masked_input') is not None:
        inset = cv2.cvtColor(result['candidate_masked_input'], cv2.COLOR_GRAY2BGR)
        inset_label = 'Proposed masked input; speed disabled'
    elif probability is not None:
        inset = cv2.applyColorMap(np.rint(np.clip(probability, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
        inset_label = 'LCD probability map; speed model skipped'
    else:
        inset = np.zeros((512, 512, 3), np.uint8); inset_label = 'No valid detector output'
    board[375:775, 830:1230] = cv2.resize(inset, (400, 400), interpolation=cv2.INTER_AREA)
    text(board, inset_label, 815, 803, .44)
    text(board, f'Frame {index+1}/{total} | {timestamp:.2f}s | 30FPS playback', 815, 836, .47)
    text(board, 'Cyan: LCD candidate | Green/amber: speed points', 815, 866, .42)
    text(board, 'No SIFT, previous reading, or assumed zero.', 815, 896, .44)
    text(board, 'A mask score cannot guarantee needle preservation.', 815, 925, .42)
    text(board, 'Indicated needle speed; ignition sweeps are not road speed.', 20, 950, .43)
    return board


def run_video(video, destination, reader, label):
    require_colab()
    video, destination = Path(video).resolve(), Path(destination).resolve()
    if destination.exists(): raise FileExistsError('Preserving existing evaluation: '+str(destination))
    destination.mkdir(parents=True)
    print('Probing source:', video.name, flush=True)
    info = probe_video(video)
    indices, times = build_timeline(info['timestamps'], info['duration'], FPS)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError('Cannot decode uploaded video')
    intermediate, output = destination/'annotated_intermediate.avi', destination/(label+'_30fps.mp4')
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*'MJPG'), FPS, BOARD)
    if not writer.isOpened(): raise RuntimeError('Cannot create inspection video')
    selected = set(np.linspace(0, len(indices)-1, 6, dtype=int).tolist())
    rows, previews, cursor, last = [], [], -1, None
    start = time.perf_counter()
    try:
        with (destination/'predictions.jsonl').open('x', encoding='utf-8') as sink:
            for index, wanted in enumerate(indices):
                fresh = False
                while cursor < wanted:
                    okay, frame = cap.read()
                    if not okay: raise RuntimeError('Source decode ended early')
                    cursor += 1; fresh = True
                if fresh: last = reader.predict(frame, include_model_input=True)
                if last is None: raise RuntimeError('Missing current source frame')
                row = {'output_frame': index, 'source_frame': int(wanted), 'time_s': float(times[index]),
                       'source_time_s': float(info['timestamps'][wanted]),
                       'same_source_frame_repeated': not fresh, 'prediction': json_safe(last)}
                sink.write(json.dumps(row, allow_nan=False)+'\n'); rows.append(row)
                board = render(frame, last, index, len(indices), float(times[index]), reader.metadata, label)
                writer.write(board)
                if index in selected:
                    cv2.imwrite(str(destination/f'preview_{index:04d}.jpg'), board)
                    for key in ('model_input', 'candidate_masked_input', 'lcd_mask'):
                        if last.get(key) is not None: cv2.imwrite(str(destination/f'{key}_{index:04d}.png'), last[key])
                    previews.append(cv2.resize(board, (640, 480), interpolation=cv2.INTER_AREA))
                if index == 0 or (index+1) % 60 == 0 or index+1 == len(indices):
                    sink.flush()
                    print(json.dumps({'clip': label, 'frames': index+1, 'total': len(indices),
                                      'elapsed_s': round(time.perf_counter()-start, 1)}), flush=True)
            while cursor < len(info['timestamps'])-1:
                if not cap.read()[0]: raise RuntimeError('Unexpected missing source frame')
                cursor += 1
            if cap.read()[0]: raise RuntimeError('Source has more frames than its probed timeline')
    finally:
        cap.release(); writer.release()
    processing_seconds = time.perf_counter()-start
    print('Encoding H264 and preserving source audio:', label, flush=True)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-i', str(intermediate), '-i', str(video),
                    '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                    '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-af', 'asetpts=PTS-STARTPTS',
                    '-t', f'{len(rows)/FPS:.9f}', '-movflags', '+faststart', str(output)], check=True)
    check, decoded = cv2.VideoCapture(str(output)), 0
    while True:
        okay, picture = check.read()
        if not okay: break
        if decoded in selected: cv2.imwrite(str(destination/f'encoded_preview_{decoded:04d}.jpg'), picture)
        decoded += 1
    fps = check.get(cv2.CAP_PROP_FPS); check.release()
    if decoded != len(rows) or abs(fps-FPS) > 1e-6: raise RuntimeError('Encoded frame count/FPS mismatch')
    if abs(decoded/fps-info['duration']) > .5/FPS+1e-6: raise RuntimeError('Encoded duration mismatch')
    if len(previews) == 6:
        cv2.imwrite(str(destination/'contact_sheet.jpg'), np.vstack([np.hstack(previews[:3]), np.hstack(previews[3:])]))
    with (destination/'predictions.csv').open('x', newline='', encoding='utf-8') as sink:
        fields = ['output_frame', 'source_frame', 'time_s', 'speed_kmh', 'accepted', 'failure_stage', 'reason',
                  'lcd_score', 'lcd_sensor_coverage_fraction', 'lcd_boundary_uncertain_fraction']
        writer_csv = csv.DictWriter(sink, fieldnames=fields); writer_csv.writeheader()
        for row in rows:
            prediction = row['prediction']; lcd = prediction.get('lcd_detection') or {}
            writer_csv.writerow({**{key: row[key] for key in fields[:3]},
                                 **{key: prediction[key] for key in fields[3:7]},
                                 'lcd_score': lcd.get('component_mean_probability'),
                                 'lcd_sensor_coverage_fraction': lcd.get('sensor_coverage_fraction'),
                                 'lcd_boundary_uncertain_fraction': lcd.get('boundary_uncertain_fraction')})
    input_sha = sha256(video)
    accepted = sum(row['prediction']['accepted'] for row in rows)
    previous = PREVIOUS_V5_COVERAGE.get(input_sha)
    comparison = None
    if previous is not None and previous['output_frames'] == decoded:
        comparison = {**previous, 'speed_checkpoint_sha256': SPEED_CHECKPOINT_SHA256,
                      'inference_code_sha256': PREVIOUS_V5_INFERENCE_SHA256,
                      'pipeline': 'V5_SIFT_visible_sensor_clipping',
                      'source': 'Completed Colab V5 test archive/video_evaluation/after/clip*/summary.json',
                      'accepted_frame_change': accepted-previous['accepted'],
                      'acceptance_change_percentage_points': 100*(accepted-previous['accepted'])/decoded,
                      'is_accuracy_comparison': False}
    summary = {'status': 'complete_and_fully_decoded', 'input_filename': video.name, 'input_sha256': input_sha,
               'output': str(output), 'output_sha256': sha256(output), 'source_frames': len(info['timestamps']),
               'output_frames': decoded, 'fps': fps, 'duration_s': decoded/fps, 'source_duration_s': info['duration'],
               'distinct_source_frames_attempted': len(set(indices.tolist())),
               'distinct_source_frames_lcd_evaluated': sum(not r['same_source_frame_repeated'] and r['prediction']['lcd_model_evaluated'] for r in rows),
               'distinct_source_frames_speed_evaluated': sum(not r['same_source_frame_repeated'] and r['prediction']['model_evaluated'] for r in rows),
               'processing_seconds': processing_seconds, 'offline_processing_fps': len(rows)/processing_seconds,
               'accepted': accepted, 'accepted_coverage': accepted/decoded,
               'lcd_gate_passed': sum(bool((r['prediction'].get('lcd_detection') or {}).get('accepted')) for r in rows),
               'failure_stages': dict(Counter(r['prediction']['failure_stage'] or 'accepted' for r in rows)),
               'reasons': dict(Counter(r['prediction']['reason'] for r in rows)),
               'reader_metadata': json_safe(reader.metadata), 'previous_v5_coverage_comparison': comparison,
               'ground_truth_accuracy': None, 'lcd_iou_without_annotation': None, 'acceptance_is_accuracy': False,
               'training_performed': False, 'evaluation_location': 'Google Colab', 'smoothing': False,
               'source_audio_preserved_if_present': 'AAC transcode aligned to start/video duration',
               'sampling': '30FPS nearest source timestamp; identical-source repeats share the same prediction',
               'script_sha256': sha256(__file__)}
    (destination/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    shutil.copy2(__file__, destination/'evaluate_videos.py')
    print(json.dumps({key: summary[key] for key in ('output', 'output_frames', 'accepted', 'failure_stages', 'previous_v5_coverage_comparison')}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', action='append', type=Path, required=True)
    parser.add_argument('--lcd-checkpoint', type=Path, required=True)
    parser.add_argument('--speed-checkpoint', type=Path, required=True)
    parser.add_argument('--speed-calibration', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'auto'], default='cuda')
    args = parser.parse_args(); require_colab()
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    print('Loading learned LCD and verified frozen V5 epoch11 speed checkpoints...', flush=True)
    reader = LearnedLCDSpeedReader(args.lcd_checkpoint, args.speed_checkpoint, args.speed_calibration, args.device)
    print('Models and frozen speed identity verified. Starting full-frame video evaluation.', flush=True)
    summaries = []
    for index, video in enumerate(args.video, 1):
        summaries.append(run_video(video, args.output_root/f'clip{index}', reader, display_label(video, index)))
    (args.output_root/'evaluation_summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__': main()
