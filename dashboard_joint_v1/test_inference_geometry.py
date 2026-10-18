"""Pure geometry tests. Never construct or execute a neural model locally."""
import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

try:
    from . import inference as inf
    from .preprocess import (rpm_matrix, rpm_roi_corners, rpm_from_roi_matrix,
                             dashboard_matrix, warp_input)
    from .metrics import heatmap_to_points
except ImportError:
    import inference as inf
    from preprocess import (rpm_matrix, rpm_roi_corners, rpm_from_roi_matrix,
                            dashboard_matrix, warp_input)
    from metrics import heatmap_to_points


def reference_needle(speed):
    ref = inf.SPEED_REFERENCE
    pivot, ticks = np.asarray(ref['pivot']), np.asarray(ref['ticks'])
    angles = np.unwrap(np.arctan2((ticks-pivot)[:, 1], (ticks-pivot)[:, 0]))
    angle = np.interp(speed, ref['speeds'], angles)
    tip = pivot+160*np.array([np.cos(angle), np.sin(angle)])
    return np.vstack([pivot, tip, ticks])


class GeometryTests(unittest.TestCase):
    def test_exact_tick_knots(self):
        for speed in range(0, 181, 20):
            result = inf.read_speed(reference_needle(speed), np.ones(12))
            self.assertIsNone(result['reason'])
            self.assertAlmostEqual(result['value_kmh'], speed, places=4)

    def test_perspective_rotation_mirroring_invariance(self):
        base = reference_needle(58.)
        for angle in (0, 30, -110, 180):
            c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            for reflect in (-1, 1):
                matrix = np.array([[reflect*c, -s, 600.], [reflect*s, c, 700.], [.00015, -.00008, 1.]])
                result = inf.read_speed(inf.project_points(base, matrix), np.ones(12))
                self.assertIsNone(result['reason'], result)
                self.assertAlmostEqual(result['value_kmh'], 58., places=3)

    def test_outlier_tick_is_reported_not_used(self):
        points = reference_needle(75.)
        points[5] += [85, -80]
        points[9] += [-100, 130]
        result = inf.read_speed(points, np.ones(12))
        self.assertIsNone(result['reason'], result)
        self.assertAlmostEqual(result['value_kmh'], 75., places=3)
        self.assertEqual(result['diagnostics']['inlier_tick_count'], 8)

    def test_missing_needle_and_few_ticks(self):
        points = reference_needle(10.)
        visibility = np.ones(12)
        visibility[1] = .2
        self.assertEqual(inf.read_speed(points, visibility)['reason'], 'needle_pivot_or_tip_not_visible')
        visibility[:] = 0
        visibility[:5] = 1
        self.assertEqual(inf.read_speed(points, visibility)['reason'], 'fewer_than_four_visible_speed_ticks')

    def test_degenerate_ticks_and_outside_arc(self):
        points = reference_needle(20.)
        points[2:] = np.c_[np.arange(10)*20, np.ones(10)*100]
        self.assertEqual(inf.read_speed(points, np.ones(12))['reason'], 'degenerate_speed_tick_geometry')
        points = reference_needle(0.)
        direction = points[1]-points[0]
        c, s = np.cos(-.3), np.sin(-.3)
        points[1] = points[0]+np.array([[c, -s], [s, c]])@direction
        self.assertEqual(inf.read_speed(points, np.ones(12))['reason'], 'needle_outside_calibrated_speed_arc')

    def test_rpm_interpolation_and_missing_endpoint(self):
        anchors = np.array([[120, 350], [125, 300], [128, 250], [125, 200], [117, 150], [103, 100], [85, 50]], float)
        endpoint = .75*anchors[2]+.25*anchors[3]
        points = np.vstack([anchors, endpoint])
        result = inf.read_rpm(points, np.ones(8))
        self.assertAlmostEqual(result['value'], 4500., places=4)
        self.assertEqual(result['diagnostics']['interval_rpm'], [4000, 6000])
        visibility = np.ones(8)
        visibility[-1] = .1
        self.assertEqual(inf.read_rpm(points, visibility)['reason'], 'rpm_fill_endpoint_not_visible')

    def test_rpm_all_exact_anchor_knots(self):
        anchors = np.c_[100+np.sin(np.arange(7))*8, 350-np.arange(7)*50.]
        for index in range(7):
            result = inf.read_rpm(np.vstack([anchors, anchors[index]]), np.ones(8))
            self.assertEqual(result['reason'], None)
            self.assertAlmostEqual(result['value'], index*2000., places=4)

    def test_rpm_distant_endpoint_or_missing_interval(self):
        anchors = np.c_[np.ones(7)*100, 350-np.arange(7)*50.]
        points = np.vstack([anchors, [200, 250]])
        self.assertEqual(inf.read_rpm(points, np.ones(8))['reason'], 'rpm_endpoint_too_far_from_scale')
        visibility = np.array([1, 0, 0, 0, 0, 0, 1, 1])
        self.assertEqual(inf.read_rpm(points, visibility)['reason'], 'rpm_scale_missing_usable_anchor_intervals')
        points[-1] = [100, 40]
        self.assertEqual(inf.read_rpm(points, np.ones(8))['reason'], 'rpm_endpoint_outside_visible_anchor_interval')

    def test_direct_roi_matches_training_matrix(self):
        lcd = np.array([[400, 160], [790, 210], [730, 500], [350, 440]], np.float32)
        first = rpm_matrix(lcd)
        second = rpm_from_roi_matrix(rpm_roi_corners(lcd))
        points = np.array([[410, 220], [430, 300], [490, 400], [470, 200]], float)
        np.testing.assert_allclose(inf.project_points(points, first), inf.project_points(points, second), atol=1e-4)

    def test_model_point_maps_back_to_original(self):
        source = np.array([[125, 180], [475, 180], [475, 490], [125, 490]], float)
        matrix = dashboard_matrix(source, (720, 1280, 3))
        np.testing.assert_allclose(inf.project_points(inf.project_points(source, matrix), np.linalg.inv(matrix)), source, atol=1e-6)

    def test_mask_component_is_inside_dashboard_and_not_bounding_rectangle(self):
        probabilities = np.zeros((512, 512), float)
        probabilities[0:100, 0:100] = 1  # Largest false LCD, outside dashboard.
        cv2.fillPoly(probabilities, [np.array([[210, 220], [250, 230], [245, 260], [205, 250]], np.int32)], 1.)
        mask, polygon = inf.select_lcd_mask(probabilities, [150, 150, 350, 350])
        self.assertGreater(len(polygon), 2)
        self.assertEqual(mask[50, 50], 0)
        self.assertEqual(mask[220, 250], 0)  # Bounding rectangle would erase this pixel.
        self.assertEqual(mask[240, 230], 255)

    def test_semantic_roi_rejects_crossing_and_missing(self):
        q = np.array([[20, 20], [120, 20], [120, 200], [20, 200]], float)
        np.testing.assert_array_equal(inf.validated_roi(q, np.ones(4)), q)
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            inf.validated_roi(q[[0, 2, 1, 3]], np.ones(4))
        with self.assertRaisesRegex(ValueError, 'not_visible'):
            inf.validated_roi(q, [1, 1, 0, 1])

    def test_heatmap_pixel_centers_match_training(self):
        random = np.random.default_rng(31).normal(size=(3, 8, 96, 64))
        np.testing.assert_array_equal(inf.decode_heatmaps(random, (384, 256)), heatmap_to_points(random, (384, 256)))

    def test_horizon_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, 'horizon'):
            inf.project_points([[1, 1]], [[1, 0, 0], [0, 1, 0], [1, 0, -1]])

    def test_local_model_constructor_rejected_before_torch_or_weights(self):
        with mock.patch.object(inf.importlib.util, 'find_spec', return_value=None), mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'restricted to Google Colab'):
                inf.DashboardPredictor({'localizer': 'does-not-exist.pt'})
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
