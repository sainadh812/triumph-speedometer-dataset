"""Compare V4 video predictions with eight approximate visual needle references.

This is a same-video development diagnostic, not an independent accuracy test.
Repeated 30 FPS timeline frames count once, chosen by first output occurrence.
No labels influence model inference, checkpoint selection or speed decoding.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _point(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value)):
        raise ValueError('Expected two finite coordinates')
    return [float(v) for v in value]


def _points(row):
    values = row.get('predicted_points_aligned_xy')
    if values is None:
        return None
    if len(values) != 12:
        raise ValueError('Expected twelve predictions in stabilized512 coordinates')
    return [_point(point) for point in values]


def _distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def _mean(values):
    return sum(values)/len(values) if values else None


def compare(reference, predictions, inference_summary):
    if inference_summary.get('status') != 'completed_and_decoded':
        raise ValueError('Require completed and verified video inference')
    expected = inference_summary['verified_inference_assets']['calibration_sha256']
    if reference['calibration_sha256'] != expected:
        raise ValueError('Reference and model inference use different calibration')
    if inference_summary.get('model_input_size') != [512, 512]:
        raise ValueError('Reference coordinates require512px stabilized inputs')
    video_name = str(inference_summary['input_video']).replace('\\', '/').rsplit('/', 1)[-1]
    if reference['source_video'] != video_name:
        raise ValueError('Manual references belong to a different video')
    if (inference_summary['output_frame_count'] != len(predictions)
            or inference_summary['decoded_output_frames'] != len(predictions)):
        raise ValueError('Prediction log is incomplete relative to encoded output')
    if [row['output_frame_index'] for row in predictions] != list(range(len(predictions))):
        raise ValueError('Prediction output indices must be contiguous and ordered')
    by_source = defaultdict(list)
    for row in predictions:
        index = row['source_frame_index']
        if type(index) is not int or index < 0:
            raise ValueError('Invalid source frame index')
        _points(row)
        speed = row['primary_speed_kmh']
        if row['geometry_accepted']:
            if speed is None or not math.isfinite(speed) or not 0 <= speed <= 180:
                raise ValueError('Accepted prediction must have a finite calibrated speed')
        elif speed is not None:
            raise ValueError('Rejected prediction must not contain a reported speed')
        by_source[index].append(row)
    duplicate_checks = []
    for source, group in sorted(by_source.items()):
        if len(group) < 2:
            continue
        first = group[0]
        first_points = _points(first)
        point_differences, speed_differences = [], []
        missingness_agrees = True
        for row in group[1:]:
            if abs(row['source_timestamp_seconds']-first['source_timestamp_seconds']) > 1e-6:
                raise ValueError('Repeated source frame has inconsistent timestamps')
            points = _points(row)
            missingness_agrees &= (points is None) == (first_points is None)
            if points is not None and first_points is not None:
                point_differences.extend(_distance(a, b) for a, b in zip(points[:2], first_points[:2]))
            if first['primary_speed_kmh'] is not None and row['primary_speed_kmh'] is not None:
                speed_differences.append(abs(first['primary_speed_kmh']-row['primary_speed_kmh']))
        same_decision = all(row['geometry_accepted'] == first['geometry_accepted']
                            and row['geometry_reason'] == first['geometry_reason'] for row in group)
        duplicate_checks.append({'source_frame_index': source,
            'output_frame_indices': [row['output_frame_index'] for row in group],
            'selected_output_frame_index': first['output_frame_index'],
            'same_geometry_decision': same_decision,
            'same_keypoint_availability': bool(missingness_agrees),
            'max_duplicate_needle_difference_px': max(point_differences, default=0.),
            'max_duplicate_speed_difference_kmh': max(speed_differences, default=0.),
            'note': 'First occurrence retained; repeats never increase evaluation sample count.'})
    rows, pivot_errors, tip_errors, speed_differences = [], [], [], []
    seen = set()
    for manual in reference['frames']:
        source = manual['frame_index']
        if source in seen:
            raise ValueError('Duplicate manual reference frame')
        seen.add(source)
        pivot, tip = _point(manual['pivot_xy']), _point(manual['tip_xy'])
        visual_speed = float(manual['approximate_indicated_speed_kmh'])
        group = by_source.get(source, [])
        result = {'source_frame_index': source, 'matched': bool(group),
            'manual_pivot_xy': pivot, 'manual_tip_xy': tip,
            'manual_pivot_visual_uncertainty_px': manual['pivot_visual_uncertainty_px'],
            'manual_tip_visual_uncertainty_px': manual['tip_visual_uncertainty_px'],
            'manual_uncertainty_kind': manual['uncertainty_kind'],
            'approximate_visual_speed_kmh': visual_speed,
            'output_occurrences': len(group), 'selected_output_frame_index': None,
            'predicted_pivot_xy': None, 'predicted_tip_xy': None,
            'pivot_difference_from_visual_px': None, 'tip_difference_from_visual_px': None,
            'geometry_accepted': False, 'predicted_speed_kmh': None,
            'signed_speed_difference_from_visual_kmh': None,
            'absolute_speed_difference_from_visual_kmh': None,
            'geometry_reason': 'source_frame_not_present_in_predictions'}
        if group:
            chosen = group[0]
            points = _points(chosen)
            result.update(selected_output_frame_index=chosen['output_frame_index'],
                source_timestamp_seconds=chosen['source_timestamp_seconds'],
                geometry_accepted=chosen['geometry_accepted'], geometry_reason=chosen['geometry_reason'],
                predicted_speed_kmh=chosen['primary_speed_kmh'])
            if points is not None:
                pe, te = _distance(points[0], pivot), _distance(points[1], tip)
                pivot_errors.append(pe)
                tip_errors.append(te)
                result.update(predicted_pivot_xy=points[0], predicted_tip_xy=points[1],
                    pivot_difference_from_visual_px=pe, tip_difference_from_visual_px=te)
            if chosen['geometry_accepted']:
                delta = chosen['primary_speed_kmh']-visual_speed
                speed_differences.append(delta)
                result.update(signed_speed_difference_from_visual_kmh=delta,
                              absolute_speed_difference_from_visual_kmh=abs(delta))
        rows.append(result)
    accepted = sum(row['geometry_accepted'] for row in rows)
    return {'schema': 'v4_manual_visual_comparison_v1',
        'purpose': 'Same-video development comparison with approximate visual labels; not independent accuracy.',
        'calibration_sha256': reference['calibration_sha256'],
        'checkpoint_sha256': inference_summary.get('checkpoint_sha256'),
        'checkpoint_best_epoch': inference_summary.get('checkpoint_best_epoch'),
        'input_video_sha256': inference_summary.get('input_video_sha256'),
        'reference_frame_count': len(rows), 'matched_reference_frames': sum(row['matched'] for row in rows),
        'needle_coordinate_comparisons': len(pivot_errors),
        'accepted_speed_comparisons': accepted,
        'rejected_matched_reference_frames': sum(row['matched'] and not row['geometry_accepted'] for row in rows),
        'missing_reference_frames': [row['source_frame_index'] for row in rows if not row['matched']],
        'mean_pivot_difference_from_visual_px': _mean(pivot_errors),
        'mean_tip_difference_from_visual_px': _mean(tip_errors),
        'mean_absolute_speed_difference_from_visual_kmh': _mean([abs(x) for x in speed_differences]),
        'mean_signed_speed_difference_from_visual_kmh': _mean(speed_differences),
        'rejection_reasons': dict(Counter(row['geometry_reason'] for row in rows if row['matched'] and not row['geometry_accepted'])),
        'deduplication': {'policy': 'Keep first output occurrence for each source_frame_index, without selecting by acceptance or error.',
            'total_prediction_rows': len(predictions), 'distinct_source_frames': len(by_source),
            'repeated_timeline_rows_removed': len(predictions)-len(by_source), 'duplicate_checks': duplicate_checks},
        'manual_uncertainty_note': 'Pivot uncertainty is about2.5px; tip uncertainty1.5–2px, judged visually. These are not statistical error bounds. Frozen tick calibration is also approximate.',
        'speed_uncertainty_note': reference['speed_uncertainty_note'],
        'caveats': ['Only eight manually inspected frames from the same development recording.',
                    'Pixel differences include annotation, stabilization, calibration and model error.',
                    'Accepted-speed differences exclude rejected readings; rejection counts remain explicit.',
                    'No accuracy percentage or claimed generalization is derived from these approximate references.'],
        'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True,
                        help='Explicit JSON visual reference labels; private source-frame annotations are not published.')
    parser.add_argument('--inference-summary', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summary_path = args.inference_summary or args.predictions.parent / 'summary.json'
    output = args.output or args.predictions.parent / 'visual_reference_comparison.json'
    csv_path = output.with_suffix('.csv')
    if output.exists() or csv_path.exists():
        raise FileExistsError('Existing comparison is preserved; choose an unused output path')
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding='utf-8').splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    report = compare(reference, predictions, summary)
    report['provenance'] = {'reference_path': str(args.reference.resolve()), 'reference_sha256': sha256(args.reference),
        'predictions_path': str(args.predictions.resolve()), 'predictions_sha256': sha256(args.predictions),
        'inference_summary_path': str(summary_path.resolve()), 'inference_summary_sha256': sha256(summary_path)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    fields = ['source_frame_index', 'matched', 'selected_output_frame_index', 'output_occurrences',
              'manual_pivot_visual_uncertainty_px', 'manual_tip_visual_uncertainty_px',
              'pivot_difference_from_visual_px', 'tip_difference_from_visual_px',
              'approximate_visual_speed_kmh', 'geometry_accepted', 'predicted_speed_kmh',
              'signed_speed_difference_from_visual_kmh', 'absolute_speed_difference_from_visual_kmh', 'geometry_reason']
    with csv_path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in report['rows'])
    print(json.dumps({key: value for key, value in report.items() if key not in ('rows', 'provenance', 'deduplication')}, indent=2))
    print(f'Comparison: {output.resolve()}\nCSV: {csv_path.resolve()}')


if __name__ == '__main__':
    main()
