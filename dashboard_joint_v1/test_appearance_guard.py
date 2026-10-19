"""CPU-only evidence preservation and holdout-byte regressions; no neural code."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from build_dataset import appearance_evidence_guard,photometric_for_sample,make_dataset
from test_rpm_crop_jitter import fixture,rows


def pattern():
    im=np.full((96,96),100,np.uint8)
    cv2.line(im,(10,48),(86,48),225,3)
    cv2.rectangle(im,(44,35),(52,61),35,2)
    return im


def destructive(img,rng,variant):
    # Consume a draw to prove the guard neither retries nor changes random state.
    draw=float(rng.random())
    if variant%4==0:return img,{'identity':True}
    return np.full_like(img,255),{'fixture':'saturated','draw':draw}


class AppearanceGuardTests(unittest.TestCase):
    def test_white_saturation_and_black_collapse_restore_exact_preappearance(self):
        original=pattern()
        for value in (0,255,128):
            candidate=np.full_like(original,value)
            result,meta=appearance_evidence_guard(original,candidate,[[48,48]],[True],'rpm')
            self.assertTrue(meta['fallback_to_preappearance'])
            np.testing.assert_array_equal(result,original)
            self.assertIn('structural_contrast_collapse',meta['failures'][0]['reasons'])

    def test_noise_on_erased_signal_does_not_fake_structural_contrast(self):
        original=pattern();rng=np.random.default_rng(5)
        noise=np.clip(128+rng.normal(0,7,original.shape),0,255).astype(np.uint8)
        _,meta=appearance_evidence_guard(original,noise,[[48,48]],[True],'rpm')
        self.assertTrue(meta['fallback_to_preappearance'])

    def test_mild_noise_brightness_and_ordinary_blur_are_retained(self):
        original=pattern();rng=np.random.default_rng(8)
        candidate=np.clip(cv2.GaussianBlur(original,(3,3),.6).astype(float)*.95+4+rng.normal(0,1.5,original.shape),0,255).astype(np.uint8)
        result,meta=appearance_evidence_guard(original,candidate,[[48,48]],[True],'speed')
        self.assertFalse(meta['fallback_to_preappearance']);np.testing.assert_array_equal(result,candidate)

    def test_original_glare_is_not_removed_or_reinterpreted(self):
        original=np.full((96,96),255,np.uint8)
        result,meta=appearance_evidence_guard(original,original.copy(),[[48,48]],[True],'rpm')
        self.assertFalse(meta['fallback_to_preappearance']);np.testing.assert_array_equal(result,original)

    def test_unknown_or_invisible_landmarks_do_not_drive_guard(self):
        original=pattern();candidate=np.full_like(original,255)
        result,meta=appearance_evidence_guard(original,candidate,[[48,48]],[False],'rpm')
        self.assertEqual(meta['checked_patches'],0);self.assertFalse(meta['fallback_to_preappearance'])
        np.testing.assert_array_equal(result,candidate)

    def test_needle_middle_evidence_checked_when_endpoints_untouched(self):
        original=pattern();candidate=original.copy();candidate[35:62,30:66]=255
        _,meta=appearance_evidence_guard(original,candidate,[[10,48],[86,48]],[True,True],'speed')
        self.assertTrue(meta['fallback_to_preappearance'])
        self.assertTrue(any(f['sample'].startswith('needle_') for f in meta['failures']))
        self.assertFalse(any(f['sample'].startswith('point_') for f in meta['failures']))

    def test_virtual_roi_points_are_not_treated_as_physical_marks(self):
        original=pattern();candidate=np.full_like(original,255)
        pts=[[1,1]]*4+[[48,48]]*4
        _,meta=appearance_evidence_guard(original,candidate,pts,[False]*4+[True]*4,'localizer')
        self.assertEqual(meta['checked_patches'],0);self.assertFalse(meta['fallback_to_preappearance'])
        _,physical=appearance_evidence_guard(original,candidate,[[48,48]]*8,[True]*4+[False]*4,'localizer')
        self.assertTrue(physical['fallback_to_preappearance'])

    def test_opt_in_training_only_preserves_rng_and_input_arrays(self):
        original=pattern();snapshot=original.copy()
        for split in ('train','val','test'):
            r0=np.random.default_rng(12);r1=np.random.default_rng(12)
            with patch('build_dataset.photometric',destructive):
                plain,pmeta=photometric_for_sample(original,r0,1,split=split,task='rpm',points=[[48,48]],visible=[True])
                guarded,gmeta=photometric_for_sample(original,r1,1,split=split,task='rpm',points=[[48,48]],visible=[True],enabled=True)
            self.assertEqual(r0.random(),r1.random())
            np.testing.assert_array_equal(original,snapshot)
            if split=='train':
                np.testing.assert_array_equal(guarded,original);self.assertTrue(gmeta['identity'])
                self.assertEqual(gmeta['evidence_guard']['rejected_appearance'],pmeta)
            else:
                np.testing.assert_array_equal(plain,guarded);self.assertEqual(pmeta,gmeta)

    def test_identity_variants_are_not_changed_or_tagged(self):
        original=pattern()
        for variant in (0,4,16):
            out,meta=photometric_for_sample(original,np.random.default_rng(3),variant,split='train',task='rpm',points=[[48,48]],visible=[True],enabled=True)
            np.testing.assert_array_equal(out,original);self.assertEqual(meta,{'identity':True})

    def test_completed_fixture_holdouts_geometry_masks_and_visibility_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=fixture(root/'source');plain=root/'plain';guarded=root/'guarded'
            with patch('build_dataset.photometric',destructive):
                make_dataset(source,plain,train_variants=3,val_variants=3)
                manifest=make_dataset(source,guarded,train_variants=3,val_variants=3,train_appearance_guard=True)
            self.assertTrue(manifest['augmentation_config']['train_appearance_guard'])
            fallback_count=0
            for task in ('localizer','speed','rpm'):
                before={r['id']:r for r in rows(plain,task)};after={r['id']:r for r in rows(guarded,task)}
                self.assertEqual(set(before),set(after))
                for ident,a in before.items():
                    b=after[ident]
                    for key in ('points','visible','visibility_known','transform','source_id','split'):
                        if key in a:self.assertEqual(a[key],b[key])
                    if task=='localizer':self.assertEqual((plain/task/a['lcd_mask']).read_bytes(),(guarded/task/b['lcd_mask']).read_bytes())
                    if a['split']!='train' or a.get('negative') or ident.endswith('_000'):
                        self.assertEqual(a,b);self.assertEqual((plain/task/a['image']).read_bytes(),(guarded/task/b['image']).read_bytes())
                    elif b['augmentation']['appearance'].get('evidence_guard',{}).get('fallback_to_preappearance'):
                        fallback_count+=1
            self.assertGreater(fallback_count,0)
            qa=json.loads((guarded/'qa.json').read_text())['training_appearance_guard']
            self.assertTrue(qa['enabled']);self.assertGreater(sum(v for k,v in qa['counts'].items() if k.endswith('_fallback')),0)

    def test_synthetic_rpm_branch_uses_same_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=fixture(root/'source')
            for filename in ('source_inventory.json','annotations_day.json'):
                obj=json.loads((source/filename).read_text());obj['sources'][0]['id']='night_00';(source/filename).write_text(json.dumps(obj))
            original=np.tile(pattern(),(4,3))[:384,:256]
            points=[[80,40+i*40] for i in range(7)]+[[80,240]]
            examples=[dict(image=original,points=points,visible=[True]*8,source_id='night_00',split='train',synthetic=True) for _ in range(2)]
            with patch('synthetic_rpm.generate_examples',return_value=iter(examples)),patch('build_dataset.photometric',destructive):
                make_dataset(source,root/'out',train_variants=1,val_variants=1,synthetic_rpm_count=2,train_appearance_guard=True)
            generated=next(r for r in rows(root/'out','rpm') if r['id']=='night_00_rpm_render_001')
            self.assertTrue(generated['augmentation']['appearance']['evidence_guard']['fallback_to_preappearance'])
            self.assertTrue(generated['synthetic']);self.assertTrue(all(generated['visible']))


if __name__=='__main__':unittest.main()
