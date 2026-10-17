"""Meaningful geometry checks independent of the neural model and dataset."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from real_adaptation_v4.geometry import read_speed


# Small explicit calibration fixture keeps tests independent of private images,
# extracted datasets and workstation paths. These are the reviewed V3 tick
# coordinates; other calibration layouts are covered separately below.
CAL = {
    'pivot_xy': [211., 225.],
    'speeds': list(range(0, 181, 20)),
    'ticks_xy': [[211., 332.], [171.5, 324.5], [135.5, 303.], [108., 268.5],
                [98.5, 227.5], [105.5, 184.5], [125.5, 146.], [160., 120.5],
                [203.5, 109.], [245., 113.]],
}


def landmarks(speed=73., pivot=None):
    pivot = np.asarray(CAL['pivot_xy'] if pivot is None else pivot, float)
    ticks = np.asarray(CAL['ticks_xy'], float)
    angles = np.unwrap(np.arctan2(ticks[:, 1]-pivot[1], ticks[:, 0]-pivot[0]))
    if speed < 0:
        theta = angles[0]+speed/20*(angles[1]-angles[0])
    elif speed > 180:
        theta = angles[-1]+(speed-180)/20*(angles[-1]-angles[-2])
    else:
        theta = np.interp(speed, CAL['speeds'], angles)
    tip = pivot+115*np.array([math.cos(theta), math.sin(theta)])
    return np.vstack([pivot, tip, ticks])


def transform(points, matrix):
    q = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    return q[:, :2]/q[:, 2, None]


class GeometryTests(unittest.TestCase):
    def assert_speed(self, points, speed, **kwargs):
        value = read_speed(points, calibration=CAL, **kwargs)
        self.assertTrue(value['accepted'], value)
        self.assertAlmostEqual(value['speed_kmh'], speed, delta=.002)
        self.assertFalse(value['uses_ground_truth_transform'])
        return value

    def test_whole_range_piecewise_interpolation(self):
        for speed in [0, 1, 20, 39.7, 60, 73, 100, 140, 179, 180]:
            with self.subTest(speed=speed):
                self.assert_speed(landmarks(speed), speed)

    def test_translation_rotation_scale_perspective_not_fixed_camera(self):
        matrix = np.array([[-.83, -.35, 610], [.31, -.79, 630], [.0008, -.0004, 1.]])
        points = transform(landmarks(108.3), matrix)
        self.assertGreater(np.linalg.norm(points[0]-landmarks()[0]), 100)
        self.assert_speed(points, 108.3)

    def test_predicted_pivot_changes_reference_angles(self):
        self.assert_speed(landmarks(58.5, pivot=[219, 219]), 58.5)

    def test_accepts_a_different_explicit_calibration_layout(self):
        shift = np.array([80., 17.])
        calibration = {'pivot_xy': (np.asarray(CAL['pivot_xy'])+shift).tolist(),
                       'ticks_xy': (np.asarray(CAL['ticks_xy'])+shift).tolist(),
                       'speeds': CAL['speeds']}
        result = read_speed(landmarks(105)+shift, calibration=calibration)
        self.assertTrue(result['accepted'], result)
        self.assertAlmostEqual(result['speed_kmh'], 105, delta=.002)

    def test_isolated_bad_ticks_discarded_with_redundant_support(self):
        points = landmarks(94.2)
        points[4] += [55, -50]
        points[7] += [-65, 40]
        result = self.assert_speed(points, 94.2)
        self.assertEqual(result['inlier_count'], 8)
        self.assertEqual(set(result['outlier_tick_names']), {'tick_40', 'tick_100'})

    def test_many_bad_ticks_rejected(self):
        points = landmarks()
        points[[3, 5, 8]] += [[60, -50], [-70, 90], [90, 40]]
        result = read_speed(points, calibration=CAL)
        self.assertFalse(result['accepted'], result)
        self.assertEqual(result['reason'], 'too_many_inconsistent_tick_predictions')

    def test_missing_ticks_supported_but_short_arc_rejected(self):
        visible = np.ones(12)
        visible[[3, 5, 7, 9]] = 0
        points = landmarks(120)
        points[visible == 0] = np.nan
        self.assert_speed(points, 120, visibility=visible)
        visible = np.zeros(12)
        visible[:2] = 1
        visible[4:10] = 1
        result = read_speed(landmarks(), visible, CAL, limits={'require_endpoint_ticks': False})
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'insufficient_arc_support')

    def test_nonfinite_visible_point_and_short_needle_rejected(self):
        points = landmarks()
        points[1] = np.nan
        self.assertEqual(read_speed(points, calibration=CAL)['reason'], 'nonfinite_available_landmark')
        points = landmarks()
        points[1] = points[0]+[1, 1]
        self.assertEqual(read_speed(points, calibration=CAL)['reason'], 'implausible_needle_length')

    def test_missing_needle_rejected(self):
        visible = np.ones(12)
        visible[1] = .4
        self.assertEqual(read_speed(landmarks(), visible, CAL)['reason'], 'needle_landmark_unavailable')

    def test_collinear_and_duplicate_ticks_rejected(self):
        points = landmarks()
        points[2:] = np.c_[np.arange(10)*10+100, np.ones(10)*120]
        self.assertEqual(read_speed(points, calibration=CAL)['reason'], 'degenerate_or_poorly_spread_tick_points')
        points = landmarks()
        points[3] = points[2]
        self.assertEqual(read_speed(points, calibration=CAL)['reason'], 'duplicate_or_unresolved_tick_points')

    def test_reflection_explicitly_opt_in(self):
        mirrored = transform(landmarks(83), [[-1, 0, 511], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(read_speed(mirrored, calibration=CAL)['reason'], 'reflected_or_reversed_semantic_ticks')
        self.assert_speed(mirrored, 83, limits={'allow_reflection': True})

    def test_wrong_pivot_rejected(self):
        points = landmarks()
        points[0] += [35, 0]
        self.assertEqual(read_speed(points, calibration=CAL)['reason'], 'pivot_inconsistent_with_tick_calibration')

    def test_coherent_cyclic_tick_mislabel_does_not_redefine_scale(self):
        for shift in [-2, -1, 1, 2]:
            points = landmarks(73)
            points[2:] = np.roll(points[2:], shift, axis=0)
            result = read_speed(points, calibration=CAL)
            self.assertFalse(result['accepted'], result)
        points = landmarks(73)
        points[2:] = points[2:][::-1]
        self.assertEqual(read_speed(points, calibration=CAL)['reason'], 'reflected_or_reversed_semantic_ticks')

    def test_semantic_endpoints_required_even_with_nine_visible_ticks(self):
        visible = np.ones(12)
        visible[2] = 0
        result = read_speed(landmarks(), visibility=visible, calibration=CAL)
        self.assertEqual(result['reason'], 'semantic_endpoint_ticks_unavailable')

    def test_endpoint_tolerance_retains_raw_and_rejects_beyond(self):
        for raw, expected in [(-1.8, 0), (181.8, 180)]:
            result = self.assert_speed(landmarks(raw), expected)
            self.assertTrue(result['endpoint_clamped'])
            self.assertAlmostEqual(result['raw_indicated_speed_kmh'], raw, delta=.002)
        for raw in [-2.3, 182.3]:
            result = read_speed(landmarks(raw), calibration=CAL)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['reason'], 'needle_outside_calibrated_arc')

    def test_noisy_landmarks_have_bounded_error(self):
        rng = np.random.default_rng(42)
        for speed in [10., 58., 110., 170.]:
            points = landmarks(speed)
            points[2:] += rng.normal(0, .35, (10, 2))
            result = read_speed(points, calibration=CAL)
            self.assertTrue(result['accepted'], result)
            self.assertLess(abs(result['speed_kmh']-speed), 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
