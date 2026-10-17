"""Timestamp-preserving 30 FPS video inspection with verified V4 best weights.

The current image is stabilized by FixedViewPreprocessor before its fixed LCD
mask is applied. Joint predicted needle/tick landmarks determine speed. Source
labels, synthetic transforms and old V3 camera gates never enter inference.
30 FPS denotes the video sampling/playback grid, not measured live throughput.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from real_adaptation_v4.geometry import LANDMARK_NAMES, load_calibration, read_speed
from real_adaptation_v4.stabilize_mask import FixedViewPreprocessor, prepare_frame, project
from real_adaptation_v4.train_landmarks import load_model, INFERENCE_KIND, AVAILABILITY_SEMANTICS
from training_pilot.train_core import preprocess_image

FPS = 30
BOARD_SIZE = (1024, 720)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _safe_asset(root, relative):
    if not isinstance(relative, str):
        raise ValueError('Missing dataset asset path')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Missing or out-of-root dataset asset: '+relative)
    return path


def verify_geometry_runtime(payload):
    """OpenCV RANSAC behavior must match the version used for model evaluation.

    Identical predicted landmarks produced different acceptance decisions in
    OpenCV4.6 and4.14. This is a mandatory inference guard, not a warning. NumPy
    versions are also recorded; the isolated runtime pins the checkpoint pair.
    """
    identity = payload.get('identity', {})
    expected = identity.get('opencv_version')
    if not isinstance(expected, str) or not expected:
        raise ValueError('Checkpoint has no recorded OpenCV version; geometry runtime cannot be verified')
    if cv2.__version__ != expected:
        raise ValueError(f'OpenCV runtime mismatch: checkpoint={expected}, current={cv2.__version__}. '
                         'Use the recorded OpenCV version before processing video.')
    return {'opencv_version': cv2.__version__, 'checkpoint_opencv_version': expected,
            'opencv_version_matches_checkpoint': True,
            'numpy_version': np.__version__, 'checkpoint_numpy_version': identity.get('numpy_version'),
            'numpy_version_matches_checkpoint': np.__version__ == identity.get('numpy_version')}


def load_inference_assets(dataset, payload):
    """Load only hash-protected static configuration, never per-image labels."""
    runtime = verify_geometry_runtime(payload)
    root = Path(dataset).resolve()
    manifest_path = root / 'manifest.json'
    identity = payload['identity']
    expected_dataset = identity['dataset']
    if sha256(manifest_path) != expected_dataset['manifest_sha256']:
        raise ValueError('Dataset manifest differs from the checkpoint training dataset')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    hashes = manifest.get('sha256', manifest.get('files'))
    if not isinstance(hashes, dict):
        raise ValueError('Dataset manifest has no asset hashes')
    for relative, expected in identity['implementation_hashes'].items():
        if sha256(_safe_asset(PROJECT, relative)) != expected:
            raise ValueError('Inference implementation differs from trained checkpoint: '+relative)
    if payload.get('keypoint_names') != list(LANDMARK_NAMES):
        raise ValueError('Checkpoint landmark ordering differs')
    if payload.get('landmark_availability_semantics') != AVAILABILITY_SEMANTICS:
        raise ValueError('Checkpoint must use amodal landmark availability')

    def protected(relative):
        path = _safe_asset(root, relative)
        expected = hashes.get(relative)
        expected = expected.get('sha256') if isinstance(expected, dict) else expected
        if not expected or sha256(path) != expected:
            raise ValueError('Unprotected/corrupt inference asset: '+relative)
        return path

    config = json.loads(protected('preprocessing_config.json').read_text(encoding='utf-8'))
    if config != manifest['preprocessing'] or config != expected_dataset['preprocessing']:
        raise ValueError('Preprocessing configuration differs from training')
    calibration_path = protected('calibration.json')
    if sha256(calibration_path) != expected_dataset['calibration_sha256']:
        raise ValueError('Calibration differs from training')
    reference_path = protected(config['reference_image_path'])
    fixed_mask_path = protected(config['fixed_mask_path'])
    if (sha256(reference_path) != config['reference_sha256']
            or sha256(fixed_mask_path) != config['fixed_mask_sha256']
            or sha256(calibration_path) != config['canonical_calibration_sha256']
            or sha256(PROJECT / 'real_adaptation_v4/stabilize_mask.py') != config['stabilize_mask_script_sha256']):
        raise ValueError('Preprocessing dependency hash mismatch')
    if config.get('lcd_mask_value') != 128 or config.get('lcd_mask_applied') is not True:
        raise ValueError('Expected approved LCD128 preprocessing')
    calibration = load_calibration(calibration_path)
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    processor = FixedViewPreprocessor(reference, config['lcd_polygon_reference_xy'],
                                     calibration['ticks_xy'], calibration['pivot_xy'], mask_value=128)
    saved_mask = cv2.imread(str(fixed_mask_path), cv2.IMREAD_GRAYSCALE)
    if saved_mask is None or not np.array_equal(saved_mask, processor.fixed_mask):
        raise ValueError('Reconstructed fixed LCD mask differs from training')
    return processor, calibration, {'manifest_sha256': sha256(manifest_path),
        'calibration_sha256': sha256(calibration_path), 'reference_sha256': sha256(reference_path),
        'preprocessing': config, 'verified_implementation_hashes': identity['implementation_hashes'],
        'geometry_runtime': runtime}


def probe_video(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,duration,nb_frames:frame=best_effort_timestamp_time,pkt_duration_time',
        '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    if len(data.get('streams', [])) != 1 or not data.get('frames'):
        raise ValueError('Expected one decodable selected video stream')
    stream = data['streams'][0]
    timestamps = np.asarray([float(frame['best_effort_timestamp_time']) for frame in data['frames']])
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Video timestamps must be finite and strictly increasing')
    timestamps -= timestamps[0]
    duration = float(stream.get('duration', 'nan'))
    if not np.isfinite(duration) or duration <= timestamps[-1]:
        final = float(data['frames'][-1].get('pkt_duration_time', 1/FPS))
        duration = float(timestamps[-1] + final)
    if int(stream.get('nb_frames', len(timestamps))) != len(timestamps):
        raise ValueError('ffprobe frame count differs from decoded timestamps')
    return {'timestamps': timestamps, 'duration': duration,
            'width': int(stream['width']), 'height': int(stream['height'])}


def build_timeline(timestamps, duration, fps=FPS):
    """Nearest timestamp sampling; duration error is at most half an output frame."""
    timestamps = np.asarray(timestamps, dtype=float)
    if (timestamps.ndim != 1 or not len(timestamps) or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0) or timestamps[0] != 0
            or not np.isfinite(duration) or duration <= timestamps[-1] or fps <= 0):
        raise ValueError('Invalid source timeline')
    count = max(1, int(np.floor(duration*fps+.5)))
    output_times = np.arange(count, dtype=float)/fps
    right = np.searchsorted(timestamps, output_times).clip(0, len(timestamps)-1)
    left = (right-1).clip(0, len(timestamps)-1)
    nearest = np.where(np.abs(timestamps[left]-output_times) <= np.abs(timestamps[right]-output_times), left, right)
    return nearest.astype(int), output_times


def _source_color(frame):
    height, width = frame.shape[:2]
    side = max(height, width)
    top, left = (side-height)//2, (side-width)//2
    square = cv2.copyMakeBorder(frame, top, side-height-top, left, side-width-left, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(square, (512, 512), interpolation=cv2.INTER_LINEAR)


def predict_items(model, processor, items, calibration, device='cpu'):
    """Batch only correctly stabilized/masked current frames; failures stay rejected."""
    valid = []
    for index, item in enumerate(items):
        masked, registration = processor.process(item['gray'], apply_mask=True)
        item['masked'] = masked
        item['registration'] = registration
        item['points'] = None
        item['availability'] = None
        item['source_points'] = None
        item['reading'] = {'accepted': False, 'speed_kmh': None,
            'reason': 'preprocessing_rejected:'+str(registration['reason']),
            'raw_indicated_speed_kmh': None, 'endpoint_clamped': False}
        if masked is not None and registration['accepted']:
            if not registration['lcd_mask_applied'] or not np.all(masked[processor.fixed_mask > 0] == 128):
                raise ValueError('LCD preprocessing invariant failed')
            valid.append(index)
    if not valid:
        return 0
    batch = torch.stack([preprocess_image(Image.fromarray(items[i]['masked']), 512) for i in valid]).to(device)
    with torch.inference_mode():
        output = model(batch)
    points = output['points'].float().cpu().numpy()*511
    availability = output['visibility'].float().cpu().numpy()
    if points.shape != (len(valid), 12, 2) or availability.shape != (len(valid), 12):
        raise ValueError('Unexpected joint model output shape')
    for offset, index in enumerate(valid):
        item = items[index]
        p, a = points[offset], availability[offset]
        if not np.isfinite(p).all() or not np.isfinite(a).all():
            item['reading']['reason'] = 'nonfinite_model_output'
            continue
        item['points'], item['availability'] = p, a
        item['reading'] = read_speed(p, visibility=a, calibration=calibration)
        # The stabilization inverse is used only for display on the original.
        # Geometry above receives predicted landmarks and static calibration.
        h = np.asarray(item['registration']['image_to_reference_h'])
        item['source_points'] = project(p, np.linalg.inv(h))
    return len(valid)


def _text(image, label, xy, scale=.5, color=(230, 230, 230), thickness=1):
    cv2.putText(image, label, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render(item, epoch, total):
    board = np.full((BOARD_SIZE[1], BOARD_SIZE[0], 3), (29, 24, 20), np.uint8)
    original = item['color'].copy()
    reading = item['reading']
    if item['source_points'] is not None:
        points = item['source_points']
        if np.isfinite(points).all() and np.max(np.abs(points)) < 10000:
            p, t = [tuple(np.rint(v).astype(int)) for v in points[:2]]
            cv2.line(original, p, t, (255, 235, 0), 2, cv2.LINE_AA)
            cv2.circle(original, p, 5, (255, 235, 0), 2, cv2.LINE_AA)
            cv2.circle(original, t, 4, (255, 235, 0), -1, cv2.LINE_AA)
            for speed, xy, available in zip(range(0, 181, 20), points[2:], item['availability'][2:]):
                point = tuple(np.rint(xy).astype(int))
                color = (40, 195, 255) if available >= .5 else (130, 130, 130)
                cv2.circle(original, point, 3, color, 1, cv2.LINE_AA)
                _text(original, str(speed), (point[0]+4, point[1]-4), .32, color)
    board[112:624, :512] = original
    if item['masked'] is not None:
        board[112:624, 512:] = cv2.cvtColor(item['masked'], cv2.COLOR_GRAY2BGR)
    else:
        _text(board, 'Alignment failed: no model input', (537, 350), .58, (60, 180, 255))
    color = (140, 255, 160) if reading['accepted'] else (60, 180, 255)
    label = f"{reading['speed_kmh']:.1f} km/h" if reading['accepted'] else 'READING REJECTED'
    _text(board, label, (15, 44), 1.06, color, 2)
    _text(board, 'Indicated speed from predicted needle + predicted ticks', (15, 70), .47)
    _text(board, 'Original frame + model landmarks', (15, 101), .55)
    _text(board, f'V4 best epoch {epoch} | no speed smoothing', (530, 29), .56)
    _text(board, f"Frame {item['output_frame_index']+1}/{total} | {item['output_timestamp_seconds']:.2f}s | 30 FPS", (530, 56), .54)
    _text(board, 'Actual model input: stabilized, LCD hidden', (530, 101), .53)
    _text(board, 'CYAN: predicted pivot/needle | AMBER: predicted semantic ticks | GRAY: low availability', (14, 646), .46)
    reason = reading['reason'].replace('_', ' ')
    if reading['accepted']:
        reason = 'Reading accepted' + ('; endpoint tolerance applied' if reading['endpoint_clamped'] else '')
    _text(board, reason[:112], (14, 668), .47, color)
    _text(board, 'Same development recording: inspection only, not an independent accuracy test.', (14, 691), .46)
    _text(board, '30 FPS is sampling/playback rate; processing throughput is reported separately.', (14, 711), .43)
    return board


def row_from_item(item):
    reading = item['reading']
    return {key: item[key] for key in ('output_frame_index', 'source_frame_index', 'source_timestamp_seconds',
                                     'output_timestamp_seconds')} | {
        'primary_speed_kmh': reading['speed_kmh'], 'geometry_accepted': bool(reading['accepted']),
        'geometry_reason': reading['reason'], 'raw_indicated_speed_kmh': reading['raw_indicated_speed_kmh'],
        'endpoint_clamped': bool(reading['endpoint_clamped']),
        'predicted_points_aligned_xy': item['points'].tolist() if item['points'] is not None else None,
        'predicted_points_source512_xy': item['source_points'].tolist() if item['source_points'] is not None else None,
        'predicted_amodal_availability': item['availability'].tolist() if item['availability'] is not None else None,
        'stabilization': item['registration'], 'geometry': reading}


def run(args):
    output = Path(args.output).resolve()
    diagnostics = Path(args.run_dir).resolve() if args.run_dir else output.with_name(output.stem+'_diagnostics')
    if output.exists() or diagnostics.exists():
        raise FileExistsError('Existing video/results are preserved; choose unused output paths')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(2)
    model, payload = load_model(args.checkpoint, str(device))
    if payload.get('checkpoint_kind') != INFERENCE_KIND:
        raise ValueError('Use the exported V4 best_model.pt with its completion marker')
    processor, calibration, asset_report = load_inference_assets(args.dataset, payload)
    probe = probe_video(args.video)
    source_indices, output_times = build_timeline(probe['timestamps'], probe['duration'])
    diagnostics.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), diagnostics / 'inference_script.py')
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError('Cannot open source video')
    intermediate = diagnostics / 'annotated_intermediate.avi'
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*'MJPG'), FPS, BOARD_SIZE)
    if not writer.isOpened():
        raise ValueError('Cannot open video writer')
    rows, batches, previews = [], [], []
    selected = set(np.linspace(0, len(source_indices)-1, 6, dtype=int).tolist())
    source_cursor, source_frame, inferred_count = -1, None, 0
    started = time.perf_counter()
    try:
        with (diagnostics / 'predictions.jsonl').open('x', encoding='utf-8') as sink:
            for start in range(0, len(source_indices), args.batch_size):
                items = []
                for index in range(start, min(start+args.batch_size, len(source_indices))):
                    wanted = int(source_indices[index])
                    while source_cursor < wanted:
                        okay, source_frame = capture.read()
                        if not okay:
                            raise ValueError(f'Decode stopped before source frame {wanted}')
                        source_cursor += 1
                    items.append({'output_frame_index': index, 'source_frame_index': wanted,
                        'source_timestamp_seconds': float(probe['timestamps'][wanted]),
                        'output_timestamp_seconds': float(output_times[index]),
                        'gray': prepare_frame(source_frame), 'color': _source_color(source_frame)})
                batch_started = time.perf_counter()
                inferred = predict_items(model, processor, items, calibration, str(device))
                inferred_count += inferred
                batches.append({'first_output_frame': start, 'output_frames': len(items),
                    'model_inputs': inferred, 'preprocess_and_inference_seconds': time.perf_counter()-batch_started})
                for item in items:
                    row = row_from_item(item)
                    rows.append(row)
                    sink.write(json.dumps(row, allow_nan=False)+'\n')
                    board = render(item, payload['best_epoch'], len(source_indices))
                    writer.write(board)
                    if item['output_frame_index'] in selected:
                        i = item['output_frame_index']
                        cv2.imwrite(str(diagnostics / f'preview_{i:04d}.jpg'), board)
                        if item['masked'] is not None:
                            cv2.imwrite(str(diagnostics / f'model_input_{i:04d}.png'), item['masked'])
                        previews.append(cv2.resize(board, (512, 360), interpolation=cv2.INTER_AREA))
                sink.flush()
                if start//60 != (start+len(items))//60 or len(rows) == len(source_indices):
                    print(f'Processed {len(rows)}/{len(source_indices)} timeline frames; model inputs {inferred_count}; elapsed {time.perf_counter()-started:.1f}s', flush=True)
            while source_cursor < len(probe['timestamps'])-1:
                okay, _ = capture.read()
                if not okay:
                    raise ValueError('Source decoder ended before probed frame count')
                source_cursor += 1
            if capture.read()[0]:
                raise ValueError('Source has more decoded frames than ffprobe reported')
    finally:
        capture.release()
        writer.release()
    processing_seconds = time.perf_counter()-started
    fields = ['output_frame_index', 'source_frame_index', 'source_timestamp_seconds', 'output_timestamp_seconds',
              'primary_speed_kmh', 'geometry_accepted', 'geometry_reason', 'raw_indicated_speed_kmh', 'endpoint_clamped']
    with (diagnostics / 'predictions.csv').open('x', newline='', encoding='utf-8') as sink:
        csv_writer = csv.DictWriter(sink, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows({key: row[key] for key in fields} for row in rows)
    if len(previews) == 6:
        cv2.imwrite(str(diagnostics / 'video_contact_sheet.jpg'), np.vstack([np.hstack(previews[:3]), np.hstack(previews[3:])]))
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-i', str(intermediate),
        '-i', str(args.video), '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264', '-preset', 'medium',
        '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart', str(output)], check=True)
    encoded = cv2.VideoCapture(str(output))
    frame_count = 0
    while True:
        okay, frame = encoded.read()
        if not okay:
            break
        if frame_count in selected:
            cv2.imwrite(str(diagnostics / f'encoded_preview_{frame_count:04d}.jpg'), frame)
        frame_count += 1
    output_fps = float(encoded.get(cv2.CAP_PROP_FPS))
    encoded.release()
    if frame_count != len(rows) or abs(output_fps-FPS) > 1e-6:
        raise ValueError('Encoded frame count or frame rate verification failed')
    output_duration = frame_count/output_fps
    if abs(output_duration-probe['duration']) > .5/FPS+1e-6:
        raise ValueError('Output duration differs by more than half a frame')
    unique_sources = set(source_indices.tolist())
    accepted = [row for row in rows if row['geometry_accepted']]
    summary = {'status': 'completed_and_decoded', 'input_video': str(Path(args.video).resolve()),
        'input_video_sha256': sha256(args.video), 'output_video': str(output), 'output_video_sha256': sha256(output),
        'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': sha256(args.checkpoint),
        'checkpoint_best_epoch': payload['best_epoch'], 'verified_inference_assets': asset_report,
        'inference_script_sha256': sha256(diagnostics / 'inference_script.py'),
        'source_frame_count': len(probe['timestamps']), 'decoded_source_frames': source_cursor+1,
        'output_frame_count': len(rows), 'decoded_output_frames': frame_count, 'model_inputs': inferred_count,
        'distinct_source_frames_used': len(unique_sources), 'duplicate_timeline_frames': len(rows)-len(unique_sources),
        'source_frames_not_selected': sorted(set(range(len(probe['timestamps'])))-unique_sources),
        'source_video_duration_seconds': probe['duration'], 'output_video_duration_seconds': output_duration,
        'duration_difference_seconds': output_duration-probe['duration'], 'output_fps': output_fps,
        'sampling': '30FPS nearest source presentation timestamp; every selected valid frame gets inference',
        'source_size': [probe['width'], probe['height']], 'model_input_size': [512, 512], 'output_size': list(BOARD_SIZE),
        'device': str(device), 'batch_size': args.batch_size, 'processing_seconds': processing_seconds,
        'offline_output_frames_per_second': len(rows)/processing_seconds,
        'accepted_readings': len(accepted), 'rejected_readings': len(rows)-len(accepted),
        'reading_coverage_percent': 100*len(accepted)/len(rows),
        'rejection_reasons': dict(Counter(row['geometry_reason'] for row in rows if not row['geometry_accepted'])),
        'endpoint_clamps': sum(row['endpoint_clamped'] for row in rows),
        'speed_smoothing': False, 'stale_mask_or_speed_fallback': False, 'training_performed': False,
        'uses_target_labels_or_transforms': False, 'original_audio_streams_copied_if_present': True,
        'inference_batches': batches,
        'limitation': 'Same development recording; acceptance is geometric consistency, not measured accuracy. '
                      '30FPS is the sampling/playback grid, not a live-throughput claim.'}
    _json(diagnostics / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in ('verified_inference_assets', 'inference_batches')}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    run(args)


if __name__ == '__main__':
    main()
