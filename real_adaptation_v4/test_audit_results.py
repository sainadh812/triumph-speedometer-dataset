"""Adversarial fixture checks for independent result accounting."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from real_adaptation_v4 import audit_results as audit
from real_adaptation_v4.geometry import read_speed


def fixture():
    pivot = np.array([250., 250.])
    angles = np.linspace(math.pi / 2, math.pi * 7 / 4, 10)
    ticks = pivot + 140 * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    calibration = {'pivot_xy': pivot.tolist(), 'ticks': [
        {'speed_kmh': speed, 'xy': point.tolist()} for speed, point in zip(range(0, 181, 20), ticks)]}
    target = np.vstack((pivot, ticks[4], ticks))
    records, predictions = [], []
    for index in range(3):
        points = target.copy()
        if index == 1:
            points[1] = ticks[5]
        if index == 2:
            points[2:] = points[2]
        reading = read_speed(points, visibility=np.ones(12), calibration=calibration)
        record = {'sample_id': f'fixture_{index}', 'points_xy': target.tolist(),
                  'visibility': [True] * 12, 'speed_kmh': 80., 'speed_valid': True}
        records.append(record)
        predictions.append({'sample_id': record['sample_id'], 'target_speed_kmh': 80.,
            'predicted_points_xy': points.tolist(), 'predicted_visibility': [1.] * 12,
            'point_errors_px': np.linalg.norm(points - target, axis=1).tolist(),
            'speed_error_kmh': abs(reading['speed_kmh'] - 80.) if reading['accepted'] else None,
            'reading': reading})
    errors = [row['speed_error_kmh'] for row in predictions if row['reading']['accepted']]
    point_errors = np.array([row['point_errors_px'] for row in predictions])
    primary = {'accepted': 2, 'rejected': 1, 'coverage_percent': 200/3, 'eligible_count': 3,
        'within_2_kmh_all_eligible_inputs_percent': 100/3, 'accepted_only': audit.error_summary(errors),
        'rejection_reasons': {predictions[2]['reading']['reason']: 1}}
    metrics = {'count': 3, 'combined_loss': 1., 'needle_mean_error_px': point_errors[:, :2].mean(),
        'needle_pivot_mean_error_px': point_errors[:, 0].mean(), 'needle_tip_mean_error_px': point_errors[:, 1].mean(),
        'needle_tip_p95_error_px': np.percentile(point_errors[:, 1], 95),
        'per_landmark_mean_error_px': {name: point_errors[:, index].mean() for index, name in enumerate(audit.KEYPOINT_NAMES)},
        'tick_mean_error_px': point_errors[:, 2:].mean(), 'tick_p95_error_px': np.percentile(point_errors[:, 2:], 95),
        'primary_speed': primary, 'uses_target_transform_for_speed': False, 'uses_target_ticks_for_speed': False}
    metrics['selection_score'] = metrics['needle_mean_error_px'] + metrics['tick_mean_error_px'] + 20 * (1 - 1/3)
    return records, predictions, metrics, calibration


class AuditTests(unittest.TestCase):
    def test_recomputes_rejections_and_distinct_accuracy_denominators(self):
        records, predictions, metrics, calibration = fixture()
        result = audit.audit_prediction_accounting(predictions, records, metrics, calibration, 20.)
        self.assertEqual(result['accepted'], 2)
        self.assertEqual(result['rejected'], 1)
        self.assertAlmostEqual(result['all_input_within_2_kmh_percent'], 100/3)
        self.assertAlmostEqual(metrics['primary_speed']['accepted_only']['within_2_kmh_percent'], 50.)

    def test_rejects_accepted_only_accuracy_misrepresented_as_all_input(self):
        records, predictions, metrics, calibration = fixture()
        metrics['primary_speed']['within_2_kmh_all_eligible_inputs_percent'] = 50.
        with self.assertRaises(ValueError):
            audit.audit_prediction_accounting(predictions, records, metrics, calibration, 20.)

    def test_rejects_fabricated_speed_that_disagrees_with_predicted_geometry(self):
        records, predictions, metrics, calibration = fixture()
        predictions[1]['reading']['speed_kmh'] = 80.
        predictions[1]['speed_error_kmh'] = 0.
        with self.assertRaisesRegex(ValueError, 'Replayed indicated speed'):
            audit.audit_prediction_accounting(predictions, records, metrics, calibration, 20.)

    def test_rejects_missing_predictions_and_changed_landmark_errors(self):
        records, predictions, metrics, calibration = fixture()
        with self.assertRaisesRegex(ValueError, 'count differs'):
            audit.audit_prediction_accounting(predictions[:-1], records, metrics, calibration, 20.)
        predictions[0]['point_errors_px'][0] = 3.
        with self.assertRaisesRegex(ValueError, 'landmark errors differ'):
            audit.audit_prediction_accounting(predictions, records, metrics, calibration, 20.)

    def test_compact_archive_rejects_path_traversal_before_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                for name in audit.REQUIRED:
                    bundle.writestr(name, b'fixture')
                bundle.writestr('../escape.txt', b'bad')
            with self.assertRaisesRegex(ValueError, 'unexpected members'):
                audit.extract_compact_zip(archive, root / 'out')
            self.assertFalse((root / 'escape.txt').exists())
            self.assertFalse((root / 'out').exists())

    def test_json_nonfinite_values_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text('{"metric":NaN}')
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                audit.read_json(path)

    def test_geometry_runtime_mismatch_fails_before_replay(self):
        with self.assertRaisesRegex(ValueError, 'OpenCV runtime differs'):
            audit.check_geometry_runtime({'opencv_version': '0.0.0', 'numpy_version': np.__version__})
        result = audit.check_geometry_runtime({'opencv_version': audit.cv2.__version__, 'numpy_version': np.__version__})
        self.assertTrue(result['opencv_matches'])
        self.assertTrue(result['numpy_matches'])


if __name__ == '__main__':
    unittest.main()
