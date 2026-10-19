"""CPU-only crop geometry and dataset-byte regression tests; no neural code."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from build_dataset import make_dataset, rpm_training_crop, rpm_context_retained
from preprocess import POINT_NAMES, project, rpm_matrix, rpm_roi_corners, rpm_from_roi_matrix, map_labels


QUAD=np.array([[180,120],[420,120],[420,330],[180,330]],np.float32)
POINTS=np.array([[230,130+i*27] for i in range(7)]+[[246,242]],np.float32)


def fixture(root):
    root=Path(root);root.mkdir()
    sources=[];annotations=[]
    y,x=np.mgrid[:512,:512]
    image=np.repeat(((x+y)%256).astype(np.uint8)[:,:,None],3,axis=2)
    cv2.imwrite(str(root/'frame.png'),image)
    for index,split in enumerate(('train','val','test')):
        name=split+'_frame'
        sources.append({'id':name,'image':'frame.png','split':split,'lighting':'day','time_seconds':index*100.})
        speed=[[120,250],[80,200]]+[[70+i*5,150+i*16] for i in range(10)]
        annotations.append({'id':name,'annotation_status':'reviewed','present':1,
            'dashboard_polygon':[[40,60],[460,60],[460,470],[40,470]],
            'lcd_corners':QUAD.tolist(),'lcd_polygon':QUAD.tolist(),
            'rpm_roi_corners':rpm_roi_corners(QUAD).tolist(),
            'rpm_roi_review':{'reviewed':True,'reason':'Controlled synthetic geometry fixture'},
            'speed_points':{name:{'xy':xy,'visible':True} for name,xy in zip(POINT_NAMES['speed'],speed)},
            'rpm_points':{name:{'xy':xy,'visible':True} for name,xy in zip(POINT_NAMES['rpm'],POINTS.tolist())}})
    for name,value in [('source_inventory.json',{'sources':sources}),
                        ('annotations_day.json',{'sources':annotations}),('annotations_night.json',{'sources':[]})]:
        (root/name).write_text(json.dumps(value))
    return root


def rows(root,task):return json.loads((root/task/'samples.json').read_text())['samples']


class RpmJitterTests(unittest.TestCase):
    def test_modest_jitter_is_bounded_and_preserves_reviewed_scale_context(self):
        roi=rpm_roi_corners(QUAD);width=np.linalg.norm(roi[1]-roi[0]);nonzero=0
        for seed in range(100):
            matrix,protected,meta=rpm_training_crop(roi,rpm_from_roi_matrix,POINTS,[True]*8,
                                                  np.random.default_rng(seed),.025)
            offsets=np.asarray(meta['corner_offsets_source_pixels'])
            self.assertLessEqual(np.abs(offsets).max(),.05*width+1e-6)
            self.assertTrue(rpm_context_retained(protected,matrix))
            _,visible=map_labels(POINTS,[True]*8,matrix,(256,384))
            self.assertTrue(visible.all())
            nonzero+=int(np.any(offsets))
        self.assertGreater(nonzero,95)

    def test_old_lcd_geometry_and_reviewed_roi_get_same_inference_type_error(self):
        first,_,_=rpm_training_crop(QUAD,rpm_matrix,POINTS,[True]*8,np.random.default_rng(12),.025)
        second,_,_=rpm_training_crop(rpm_roi_corners(QUAD),rpm_from_roi_matrix,POINTS,[True]*8,
                                    np.random.default_rng(12),.025)
        np.testing.assert_allclose(project(POINTS,first),project(POINTS,second),atol=1e-3)

    def test_impossible_jitter_has_bounded_safe_fallback(self):
        class ShiftRight:
            def normal(self,*args):return np.ones((4,2))*100
        roi=rpm_roi_corners(QUAD)
        edge_points=project([[2,50],[10,150],[20,250]],np.linalg.inv(rpm_from_roi_matrix(roi)))
        matrix,protected,meta=rpm_training_crop(roi,rpm_from_roi_matrix,edge_points,[True]*3,
                                              ShiftRight(),.05,max_attempts=4)
        self.assertTrue(meta['fallback_to_baseline'])
        self.assertEqual(meta['attempts'],4)
        np.testing.assert_allclose(matrix,rpm_from_roi_matrix(roi))

    def test_opt_in_leaves_all_validation_test_rows_and_images_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=fixture(root/'source')
            default,opt=root/'default',root/'opt'
            make_dataset(source,default,train_variants=8,val_variants=3)
            manifest=make_dataset(source,opt,train_variants=8,val_variants=3,rpm_train_corner_jitter_fraction=.025)
            self.assertEqual(manifest['augmentation_config']['rpm_train_corner_jitter_fraction'],.025)
            for task in ('localizer','speed','rpm'):
                before={row['id']:row for row in rows(default,task)}
                after={row['id']:row for row in rows(opt,task)}
                for key,row in before.items():
                    if task=='rpm' and row['split']=='train' and not key.endswith('_000'):continue
                    self.assertEqual(row,after[key])
                    self.assertEqual((default/task/row['image']).read_bytes(),(opt/task/row['image']).read_bytes())
                    if task=='localizer':
                        self.assertEqual((default/task/row['lcd_mask']).read_bytes(),(opt/task/row['lcd_mask']).read_bytes())
            enhanced=[row for row in rows(opt,'rpm') if row['split']=='train' and not row['id'].endswith('_000')]
            self.assertGreater(len(enhanced),0)
            for row in enhanced:
                self.assertTrue(all(row['visible']))
                self.assertEqual(row['augmentation']['geometry']['rpm_detector_corner_jitter']['mode'],
                                 'train_only_learned_roi_error')

    def test_explicit_none_is_legacy_and_invalid_fraction_refuses_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=fixture(root/'source')
            default,explicit=root/'default',root/'explicit'
            make_dataset(source,default,train_variants=3,val_variants=2)
            make_dataset(source,explicit,train_variants=3,val_variants=2,rpm_train_corner_jitter_fraction=None)
            for task in ('localizer','speed','rpm'):
                self.assertEqual(rows(default,task),rows(explicit,task))
                for row in rows(default,task):
                    self.assertEqual((default/task/row['image']).read_bytes(),(explicit/task/row['image']).read_bytes())
            for bad in (-.1,.051,float('nan'),True,'0.025'):
                with self.assertRaises(ValueError):
                    make_dataset(source,root/'invalid',rpm_train_corner_jitter_fraction=bad)
                self.assertFalse((root/'invalid').exists())


if __name__=='__main__':unittest.main()
