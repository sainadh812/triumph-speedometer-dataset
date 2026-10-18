import unittest
import numpy as np
from preprocess import project,rpm_matrix,rpm_roi_corners,rpm_from_roi_matrix,map_labels,letterbox_matrix

class GeometryTests(unittest.TestCase):
    def test_rpm_roi_warp_matches_full_lcd_warp(self):
        quad=np.array([[120,60],[610,160],[570,390],[80,290]],np.float32)
        roi=rpm_roi_corners(quad)
        points=np.array([[180,100],[220,200],[160,280]],np.float32)
        np.testing.assert_allclose(project(points,rpm_matrix(quad)),project(points,rpm_from_roi_matrix(roi)),atol=1e-4)

    def test_roi_stays_usable_with_far_lcd_corner_clipped(self):
        quad=np.array([[200,150],[740,200],[700,450],[160,400]],np.float32)
        roi=rpm_roi_corners(quad)
        self.assertTrue((roi[:,0]<640).all())
        self.assertGreater(quad[1,0],640)
        np.testing.assert_allclose(project(roi,rpm_from_roi_matrix(roi)),
            project([[64,-16],[360,-16],[360,336],[64,336]],letterbox_matrix([64,-16,360,336],(256,384))),atol=1e-4)

    def test_unavailable_and_border_labels_remain_consistent(self):
        pts,vis=map_labels([[511,511],[511.2,20],[50,50]],[True,True,False],np.eye(3),(512,512))
        self.assertEqual(vis.tolist(),[True,False,False])
        np.testing.assert_array_equal(pts[1:],0)

if __name__=='__main__':unittest.main()
