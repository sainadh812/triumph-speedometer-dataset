"""Software tests for manual-reference comparison, not inference results."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from real_adaptation_v4.compare_real_reference import compare


def fixture():
    reference = {'calibration_sha256': 'calibration', 'source_video': 'test.mp4',
        'speed_uncertainty_note': 'Approximate visual labels.', 'frames': []}
    for index in range(2):
        reference['frames'].append({'frame_index': index, 'pivot_xy': [10, 10], 'tip_xy': [10, 30],
            'pivot_visual_uncertainty_px': 2.5, 'tip_visual_uncertainty_px': 2.,
            'uncertainty_kind': 'subjective', 'approximate_indicated_speed_kmh': 50.})
    points = [[13, 14], [10, 32]] + [[20, 20]]*10
    predictions = [
        {'output_frame_index': 0, 'source_frame_index': 0, 'source_timestamp_seconds': 0.,
         'geometry_accepted': True, 'geometry_reason': 'accepted', 'primary_speed_kmh': 52.,
         'predicted_points_aligned_xy': deepcopy(points)},
        {'output_frame_index': 1, 'source_frame_index': 0, 'source_timestamp_seconds': 0.,
         'geometry_accepted': True, 'geometry_reason': 'accepted', 'primary_speed_kmh': 52.,
         'predicted_points_aligned_xy': deepcopy(points)},
        {'output_frame_index': 2, 'source_frame_index': 1, 'source_timestamp_seconds': .06,
         'geometry_accepted': False, 'geometry_reason': 'preprocessing_rejected:missing_features',
         'primary_speed_kmh': None, 'predicted_points_aligned_xy': None},
    ]
    summary = {'status': 'completed_and_decoded', 'verified_inference_assets': {'calibration_sha256': 'calibration'},
               'model_input_size': [512, 512], 'input_video': 'C:/videos/test.mp4',
               'output_frame_count': 3, 'decoded_output_frames': 3}
    return reference, predictions, summary


class ComparisonTests(unittest.TestCase):
    def test_duplicate_not_double_counted_and_rejection_kept(self):
        report = compare(*fixture())
        self.assertEqual(report['matched_reference_frames'], 2)
        self.assertEqual(report['needle_coordinate_comparisons'], 1)
        self.assertEqual(report['accepted_speed_comparisons'], 1)
        self.assertEqual(report['rejected_matched_reference_frames'], 1)
        self.assertEqual(report['mean_pivot_difference_from_visual_px'], 5.)
        self.assertEqual(report['mean_tip_difference_from_visual_px'], 2.)
        self.assertEqual(report['mean_absolute_speed_difference_from_visual_kmh'], 2.)
        self.assertEqual(report['deduplication']['repeated_timeline_rows_removed'], 1)
        self.assertIsNone(report['rows'][1]['predicted_speed_kmh'])

    def test_duplicate_disagreement_is_reported_without_picking_better(self):
        reference, predictions, summary = fixture()
        predictions[1].update(geometry_accepted=False, geometry_reason='rejected', primary_speed_kmh=None)
        report = compare(reference, predictions, summary)
        self.assertFalse(report['deduplication']['duplicate_checks'][0]['same_geometry_decision'])
        self.assertEqual(report['rows'][0]['selected_output_frame_index'], 0)
        self.assertEqual(report['rows'][0]['predicted_speed_kmh'], 52.)

    def test_different_calibration_or_video_fails(self):
        reference, predictions, summary = fixture()
        reference['calibration_sha256'] = 'other'
        with self.assertRaises(ValueError):
            compare(reference, predictions, summary)
        reference, predictions, summary = fixture()
        summary['input_video'] = 'different.mp4'
        with self.assertRaises(ValueError):
            compare(reference, predictions, summary)

    def test_incomplete_prediction_log_fails(self):
        reference, predictions, summary = fixture()
        with self.assertRaises(ValueError):
            compare(reference, predictions[:-1], summary)

    def test_no_speed_average_when_all_rejected(self):
        reference, predictions, summary = fixture()
        for row in predictions:
            row.update(geometry_accepted=False, geometry_reason='rejected', primary_speed_kmh=None)
        report = compare(reference, predictions, summary)
        self.assertIsNone(report['mean_absolute_speed_difference_from_visual_kmh'])
        self.assertEqual(report['accepted_speed_comparisons'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
