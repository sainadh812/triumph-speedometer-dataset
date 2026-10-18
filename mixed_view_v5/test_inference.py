"""Synthetic mask/coordinate unit checks only. No video or model inference."""
import unittest
import cv2
import numpy as np
from mixed_view_v5.inference import (DualReferenceMasker,apply_projected_mask,letterbox,
    mask_polygon,mask_iou,project,projected_mask_safety,compare_reference_masks)


class MaskTests(unittest.TestCase):
    def test_partial_lcd_is_clipped_without_masking_sensor_padding(self):
        pivot=np.array([210,260.]);angles=np.linspace(np.pi/2,3*np.pi/2,10)
        ticks=pivot+110*np.c_[np.cos(angles),np.sin(angles)]
        polygon=np.array([[350,120],[460,120],[460,220],[350,220]],float)
        footprint=[[95.4,0],[400.25,0],[400.25,511],[95.4,511]]
        mask,report=projected_mask_safety(polygon,ticks,pivot,footprint)
        self.assertTrue(report['accepted']);self.assertTrue(report['partial_visibility'])
        self.assertEqual(report['reason'],'safe_partial_visible_lcd')
        self.assertAlmostEqual(report['visible_area_px2'],5025.,places=3)
        self.assertAlmostEqual(report['visible_fraction'],5025./11000,places=5)
        self.assertFalse(np.any(mask[:,401:]));self.assertFalse(np.any(mask[:,:96]))
        image=np.arange(512*512,dtype=np.uint8).reshape(512,512)
        out=apply_projected_mask(image,mask)
        self.assertTrue(np.array_equal(out[mask==0],image[mask==0]))
        self.assertEqual(out[160,390],128)
        self.assertTrue(np.array_equal(out[:,401:],image[:,401:]))

    def test_near_edge_reference_can_move_beyond_sensor(self):
        pivot=np.array([210,260.]);angles=np.linspace(np.pi/2,3*np.pi/2,10)
        ticks=pivot+110*np.c_[np.cos(angles),np.sin(angles)]
        polygon=np.array([[350,120],[417,120],[417,220],[350,220]],float)
        shift=np.array([8,0])
        mask,report=projected_mask_safety(polygon+shift,ticks+shift,pivot+shift,
                                         [[95,0],[416,0],[416,511],[95,511]])
        self.assertIsNotNone(mask);self.assertTrue(report['accepted'])
        self.assertTrue(report['partial_visibility']);self.assertFalse(np.any(mask[:,417:]))

    def test_lcd_sliver_or_fully_offscreen_remains_rejected(self):
        pivot=np.array([210,260.]);angles=np.linspace(np.pi/2,3*np.pi/2,10)
        ticks=pivot+110*np.c_[np.cos(angles),np.sin(angles)]
        polygon=[[350,120],[460,120],[460,220],[350,220]]
        for right in (352,340):
            mask,report=projected_mask_safety(polygon,ticks,pivot,
                                             [[95,0],[right,0],[right,511],[95,511]])
            self.assertIsNone(mask);self.assertFalse(report['accepted'])
            self.assertEqual(report['reason'],'insufficient_visible_lcd_area')

    def test_clipping_does_not_bypass_analog_protection(self):
        pivot=np.array([210,260.]);angles=np.linspace(np.pi/2,3*np.pi/2,10)
        ticks=pivot+110*np.c_[np.cos(angles),np.sin(angles)]
        mask,report=projected_mask_safety([[190,190],[460,190],[460,310],[190,310]],ticks,pivot,
                                         [[95,0],[400,0],[400,511],[95,511]])
        self.assertIsNone(mask);self.assertTrue(report['partial_visibility'])
        self.assertEqual(report['reason'],'projected_lcd_overlaps_protected_needle_or_ticks')

    def test_fully_visible_mask_is_unchanged(self):
        pivot=np.array([210,260.]);angles=np.linspace(np.pi/2,3*np.pi/2,10)
        ticks=pivot+110*np.c_[np.cos(angles),np.sin(angles)]
        polygon=[[350,120],[460,120],[460,220],[350,220]]
        mask,report=projected_mask_safety(polygon,ticks,pivot,[[0,0],[511,0],[511,511],[0,511]])
        self.assertTrue(report['accepted']);self.assertFalse(report['partial_visibility'])
        self.assertEqual(report['visible_fraction'],1.)
        self.assertTrue(np.array_equal(mask,mask_polygon(polygon)))

    def test_partial_reference_coverage_requires_semantic_agreement(self):
        small=mask_polygon([[350,120],[390,120],[390,220],[350,220]])
        full=mask_polygon([[350,120],[460,120],[460,220],[350,220]])
        points=np.c_[np.linspace(80,250,11),np.linspace(140,340,11)]
        report={'protected_points_input_xy':points.tolist(),'mask_safety':{'accepted':True}}
        compatible=compare_reference_masks(small,report,full,report)
        self.assertTrue(compatible['compatible'])
        self.assertEqual(compatible['reason'],'compatible_partial_reference_mask')
        shifted={**report,'protected_points_input_xy':(points+[8,0]).tolist()}
        rejected=compare_reference_masks(small,report,full,shifted)
        self.assertFalse(rejected['compatible'])
        self.assertGreater(rejected['semantic_point_p90_disagreement_px'],4)
        unsafe={**report,'mask_safety':{'accepted':False}}
        self.assertFalse(compare_reference_masks(small,report,full,unsafe)['compatible'])

    def test_partial_mask_selects_fuller_safe_reference(self):
        points=np.c_[np.linspace(80,250,11),np.linspace(140,340,11)].tolist()
        class Fake:
            def __init__(self,ident,right,inliers):self.id=ident;self.right=right;self.inliers=inliers
            def propose(self,*args):
                poly=[[350,120],[self.right,120],[self.right,220],[350,220]]
                return mask_polygon(poly),{'accepted':True,'reference_id':self.id,'inliers':self.inliers,'rmse_px':.2,
                    'lcd_polygon_input_xy':poly,'reference_to_input_h':np.eye(3).tolist(),
                    'protected_points_input_xy':points,'mask_safety':{'accepted':True}}
        masker=DualReferenceMasker([Fake('clipped',390,30),Fake('full',460,20)])
        image=np.zeros((512,512),np.uint8)
        out,mask,report=masker.process(image,[[0,0],[511,0],[511,511],[0,511]])
        self.assertTrue(report['accepted']);self.assertEqual(report['selected_reference_id'],'full')
        self.assertEqual(report['reason'],'compatible_partial_reference_mask')
        self.assertEqual(int(out[160,440]),128)

    def test_video_labels_are_only_presentation(self):
        from mixed_view_v5.evaluate_videos import display_label
        baseline='frozen_V4_weights_with_V5_preprocessing_baseline'
        self.assertEqual(display_label('WhatsApp Video 2026-10-17 at 5.43.59 PM.mp4',1,baseline),'before_174359')
        self.assertEqual(display_label('WhatsApp Video 2026-10-17 at 6.13.23 PM.mp4',2,'trained_V5'),'after_181323')
        self.assertEqual(display_label('other.mp4',2,'trained_V5'),'after_clip2')

    def test_letterbox_preserves_full_orientation(self):
        frame=np.zeros((320,180),np.uint8);frame[40:60,30:50]=220
        out,matrix,footprint=letterbox(frame)
        expected=cv2.resize(cv2.copyMakeBorder(frame,0,0,70,70,cv2.BORDER_CONSTANT,value=0),(512,512),interpolation=cv2.INTER_LINEAR)
        self.assertTrue(np.array_equal(out,expected))
        points=np.array([[0,0],[179,319],[40,50]],float)
        self.assertTrue(np.allclose(project(project(points,matrix),np.linalg.inv(matrix)),points))
        self.assertEqual(footprint.shape,(4,2))

    def test_mask_changes_only_polygon_pixels(self):
        rng=np.random.default_rng(2);image=rng.integers(0,256,(512,512),dtype=np.uint8)
        mask=mask_polygon([[350,120],[460,120],[460,220],[350,220]])
        output=apply_projected_mask(image,mask)
        self.assertTrue(np.array_equal(output[mask==0],image[mask==0]))
        self.assertTrue(np.all(output[mask>0]==128))

    def test_projected_geometry_and_safety(self):
        pivot=np.array([210,260.]);angles=np.linspace(np.pi/2,3*np.pi/2,10)
        ticks=pivot+110*np.c_[np.cos(angles),np.sin(angles)]
        polygon=np.array([[350,120],[460,120],[460,220],[350,220]],float)
        h=np.vstack([cv2.getRotationMatrix2D((256,256),-8,.9),[0,0,1]])
        footprint=[[0,0],[511,0],[511,511],[0,511]]
        mask,result=projected_mask_safety(project(polygon,h),project(ticks,h),project([pivot],h)[0],footprint)
        self.assertTrue(result['accepted']);self.assertIsNotNone(mask)
        mask,bad=projected_mask_safety([[100,100],[400,100],[400,400],[100,400]],ticks,pivot,footprint)
        self.assertIsNone(mask);self.assertEqual(bad['reason'],'projected_lcd_overlaps_protected_needle_or_ticks')

    def test_horizon_is_rejected(self):
        with self.assertRaises(ValueError):project([[0,0],[1,1]],[[1,0,0],[0,1,0],[1,0,0]])

    def test_conflicting_reference_masks_are_rejected(self):
        class Fake:
            def __init__(self,ident,poly):self.id=ident;self.poly=poly
            def propose(self,*args):
                return mask_polygon(self.poly),{'accepted':True,'reference_id':self.id,'inliers':20,'rmse_px':.2,
                    'lcd_polygon_input_xy':self.poly,'reference_to_input_h':np.eye(3).tolist()}
        masker=DualReferenceMasker([Fake('a',[[330,120],[440,120],[440,220],[330,220]]),
                                   Fake('b',[[200,120],[300,120],[300,220],[200,220]])])
        out,mask,report=masker.process(np.zeros((512,512),np.uint8),[[0,0],[511,0],[511,511],[0,511]])
        self.assertIsNone(out);self.assertIsNone(mask)
        self.assertEqual(report['reason'],'ambiguous_reference_mask_projection')
        self.assertFalse(report['input_image_warped'])

    def test_agreeing_references_keep_image_pose(self):
        class Fake:
            def __init__(self,ident):self.id=ident
            def propose(self,*args):
                poly=[[350,120],[460,120],[460,220],[350,220]]
                return mask_polygon(poly),{'accepted':True,'reference_id':self.id,'inliers':20,'rmse_px':.2,
                    'lcd_polygon_input_xy':poly,'reference_to_input_h':np.eye(3).tolist()}
        masker=DualReferenceMasker([Fake('a'),Fake('b')])
        image=np.arange(512*512,dtype=np.uint8).reshape(512,512)
        out,mask,report=masker.process(image,[[0,0],[511,0],[511,511],[0,511]])
        self.assertTrue(report['accepted']);self.assertTrue(np.array_equal(out[mask==0],image[mask==0]))
        self.assertEqual(mask_iou(mask,mask),1.)


if __name__=='__main__':unittest.main()
