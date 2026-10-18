"""Label-contract regressions, using geometry only (no model execution)."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np

from build_dataset import annotation_targets, rpm_crop_geometry, make_dataset
from preprocess import map_labels, rpm_matrix, rpm_roi_corners


class AnnotationVisibilityTests(unittest.TestCase):
    def test_unreviewed_registration_cannot_produce_training_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'source_inventory.json').write_text(json.dumps({'sources':[{'id':'proposal'}]}))
            (root/'annotations_day.json').write_text(json.dumps({'sources':[
                {'id':'proposal','annotation_status':'UNREVIEWED_REGISTRATION_PROPOSAL_NOT_FOR_TRAINING'}]}))
            (root/'annotations_night.json').write_text(json.dumps({'sources':[]}))
            with self.assertRaisesRegex(ValueError, 'Unreviewed annotation proposal'):
                make_dataset(root,root/'output')
            self.assertFalse((root/'output').exists())

    def setUp(self):
        self.row = {'lcd_corners': [[100, 100], [500, 100], [500, 300], [100, 300]],
                    'rpm_points': {}}

    def test_unreadable_numbers_do_not_mean_absent_crop(self):
        points, visible, known = annotation_targets(self.row, 'localizer', (480, 640))
        self.assertEqual(visible[4:], [False]*4)
        self.assertEqual(known[4:], [False]*4)

    def test_reviewed_crop_without_readable_numbers_is_positive(self):
        self.row.update(rpm_roi_corners=[[120, 90], [300, 90], [300, 310], [120, 310]],
                        rpm_roi_review={'reviewed': True, 'reason': 'Bounds inspected in original frame'})
        points, visible, known = annotation_targets(self.row, 'localizer', (480, 640))
        self.assertTrue(all(visible[4:]))
        self.assertTrue(all(known[4:]))
        np.testing.assert_allclose(points[4:], self.row['rpm_roi_corners'])

    def test_unreviewed_geometry_is_rejected(self):
        self.row['rpm_roi_corners'] = self.row['lcd_corners']
        with self.assertRaisesRegex(ValueError, 'requires a review'):
            annotation_targets(self.row, 'localizer', (480, 640))

    def test_unknown_stays_unknown_after_transform(self):
        points, visible, known = annotation_targets(self.row, 'localizer', (480, 640))
        _, mapped_visible = map_labels(points, visible, np.eye(3), (640, 480))
        self.assertFalse(mapped_visible[4:].any())
        self.assertFalse(any(known[4:]))

    def test_reviewed_out_of_frame_corner_is_known_negative(self):
        self.row.update(rpm_roi_corners=[[-10, 90], [300, 90], [300, 310], [-10, 310]],
                        rpm_roi_review={'reviewed': True, 'reason': 'Reviewed context extends outside frame'})
        _, visible, known = annotation_targets(self.row, 'localizer', (480, 640))
        self.assertEqual(visible[4:], [False, True, True, False])
        self.assertTrue(all(known[4:]))

    def test_unknown_landmark_cannot_be_visible(self):
        self.row['rpm_points'] = {'bar_end': {'xy': [200, 200], 'visible': True,
                                             'visibility_known': False}}
        with self.assertRaisesRegex(ValueError, 'known visibility'):
            annotation_targets(self.row, 'rpm', (480, 640))

    def test_direct_roi_crop_does_not_need_off_frame_lcd_corners(self):
        full_quad=self.row['lcd_corners']
        roi=rpm_roi_corners(full_quad)
        self.row.update(lcd_corners=[full_quad[0],None,None,full_quad[3]],
                        rpm_roi_corners=roi.tolist(),
                        rpm_roi_review={'reviewed':True,'reason':'Visible strip reviewed independently'})
        geometry,transform=rpm_crop_geometry(self.row)
        actual=transform(geometry);expected=rpm_matrix(full_quad)
        np.testing.assert_allclose(actual/actual[2,2],expected/expected[2,2],atol=1e-4)


if __name__ == '__main__':
    unittest.main()
