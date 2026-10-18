"""Pure metrics/data/checkpoint tests; never execute a model forward or optimizer."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn

from learned_lcd_v1 import checkpoints as cp
from learned_lcd_v1 import evaluate as ev
from learned_lcd_v1 import model as architecture
from learned_lcd_v1 import train as trainer

cv2.setNumThreads(1)


class MetricTests(unittest.TestCase):
    def masks(self):
        target=np.zeros((512,512),np.uint8);target[100:200,300:450]=255
        valid=np.ones_like(target)*255
        return target,valid

    def test_perfect_and_empty_predictions_have_honest_accounting(self):
        target,valid=self.masks()
        perfect=ev.mask_metrics(target,target,valid)
        self.assertEqual(perfect['iou'],1.);self.assertEqual(perfect['boundary_f1'],1.)
        self.assertEqual(perfect['boundary_p95_error_px'],0.)
        empty=ev.mask_metrics(np.zeros_like(target),target,valid)
        self.assertEqual(empty['dice'],0.);self.assertEqual(empty['recall'],0.)
        self.assertIsNone(empty['precision']);self.assertEqual(empty['boundary_f1'],0.)
        self.assertIsNone(empty['protected_analog_false_mask_fraction'])

    def test_sensor_clipped_edge_ignored_and_analog_intrusion_measured(self):
        target,valid=self.masks();valid[:,400:]=0;target[:,400:]=0
        result=ev.mask_metrics(target,target,valid)
        self.assertEqual(result['boundary_f1'],1.)
        # The sensor cut at x399 is excluded, not treated as a physical LCD edge.
        self.assertLess(result['target_boundary_pixels'],2*(100+100)-4)
        analog=np.zeros_like(target);analog[50:60,50:60]=255
        prediction=target.copy();prediction[50:55,50:60]=255
        result=ev.mask_metrics(prediction,target,valid,analog)
        self.assertEqual(result['protected_analog_false_mask_pixels'],50)
        self.assertEqual(result['protected_analog_false_mask_fraction'],.5)

    def gate_rows(self,all_reject=False):
        rows=[]
        for index in range(10):
            candidates={}
            for threshold in ev.THRESHOLDS:
                # Threshold .8 is the only high-precision/high-coverage choice.
                accepted=not all_reject and (index<9 if threshold==.8 else index<7 if threshold==.9 else True)
                precision=.99 if threshold>=.8 else .96
                candidates[f'{threshold:.2f}']={'gate':{'accepted':accepted,'reason':'ok' if accepted else 'rejected'},
                    'metrics':{'target_pixels':100,'precision':precision if accepted else None,'recall':.95 if accepted else 0.}}
            rows.append({'raw':{'target_pixels':100},'candidates':candidates})
        return rows

    def test_validation_selection_requires_precision_and_coverage(self):
        selection=ev.select_gate(self.gate_rows())
        self.assertTrue(selection['ready_for_integration'])
        self.assertEqual(selection['gate_config']['pixel_threshold'],.8)
        rejected=ev.select_gate(self.gate_rows(all_reject=True))
        self.assertFalse(rejected['ready_for_integration'])
        self.assertEqual(rejected['selected_validation_result']['all_positive_macro_recall_rejections_zero'],0.)

    def test_precise_small_lcd_fragments_cannot_qualify_for_readiness(self):
        rows=self.gate_rows()
        for row in rows:
            for candidate in row['candidates'].values():
                if candidate['gate']['accepted']:
                    candidate['metrics'].update(precision=.999,recall=.1)
        selection=ev.select_gate(rows)
        self.assertFalse(selection['ready_for_integration'])
        self.assertTrue(all(not candidate['eligible'] for candidate in selection['candidates']))
        self.assertEqual(selection['policy']['minimum_accepted_macro_recall'],.90)

    def test_eligible_epoch_outranks_better_raw_dice_but_ineligible_epoch(self):
        target,valid=self.masks();metric=ev.mask_metrics(target,target,valid)
        row={'raw':metric,'source_group':'group','rotation_bucket':'0','appearance_group':'day',
             'augmentation_group':'original','candidates':{'0.70':{'gate':{'accepted':True,'reason':'ok'},'metrics':metric}}}
        gate={'gate_config':{'pixel_threshold':.7},'ready_for_integration':False}
        ineligible=ev.summarize_evaluation([row],gate,0.,.5)
        worse=deepcopy(row);worse['raw']['dice']=.7
        eligible=ev.summarize_evaluation([worse],{**gate,'ready_for_integration':True},0.,.5)
        self.assertLess(eligible['selection_score'],ineligible['selection_score'])
        fixed=ev.summarize_evaluation([row],{'gate_config':{'pixel_threshold':.7}},0.,.5)
        self.assertIsNone(fixed['selection_score'])


class LossAndStageTests(unittest.TestCase):
    def test_loss_ignores_padding_and_missing_protected_labels(self):
        logits=torch.zeros(1,1,8,8);target=torch.zeros_like(logits);target[:,:,2:5,2:5]=1
        valid=torch.ones_like(logits);valid[:,:,:,0]=0
        first,parts=architecture.segmentation_loss(logits,target,valid)
        changed=logits.clone();changed[:,:,:,0]=100
        second,_=architecture.segmentation_loss(changed,target,valid)
        self.assertEqual(float(first),float(second));self.assertEqual(parts['analog_false_mask_loss'],0.)
        with self.assertRaises(ValueError):architecture.segmentation_loss(logits,target,torch.zeros_like(valid))

    def test_analog_loss_increases_only_when_reviewed_region_available(self):
        logits=torch.full((1,1,8,8),3.);target=torch.zeros_like(logits);target[:,:,5:,5:]=1
        valid=torch.ones_like(logits);protected=torch.zeros_like(logits);protected[:,:,:3,:3]=1
        baseline,_=architecture.segmentation_loss(logits,target,valid,protected,torch.tensor([False]))
        penalized,_=architecture.segmentation_loss(logits,target,valid,protected,torch.tensor([True]))
        self.assertGreater(float(penalized),float(baseline))

    def test_warmup_freezes_encoder_bn_and_fresh_can_unfreeze_all(self):
        fake=nn.Module();fake.features=nn.Sequential(*[nn.Sequential(nn.Linear(2,2),nn.BatchNorm1d(2)) for _ in range(6)])
        fake.head=nn.Linear(2,1)
        architecture.configure_stage(fake,1,2)
        self.assertTrue(all(not p.requires_grad for p in fake.features.parameters()))
        self.assertTrue(all(not layer.training for layer in fake.features))
        self.assertTrue(all(p.requires_grad for p in fake.head.parameters()))
        architecture.configure_stage(fake,3,2,fresh=False)
        self.assertFalse(fake.features[0].training);self.assertTrue(fake.features[-1].training)
        architecture.configure_stage(fake,3,2,fresh=True)
        self.assertTrue(all(p.requires_grad for p in fake.features.parameters()))

    def test_no_local_execution_or_silent_cpu_fallback(self):
        with patch.object(torch.cuda,'is_available',return_value=False),patch.object(trainer,'verify_dataset') as verify:
            with self.assertRaisesRegex(RuntimeError,'Colab CUDA'):
                trainer.train(trainer.TrainConfig('unused','unused'),mode='fresh')
            verify.assert_not_called()


class AmpStepTests(unittest.TestCase):
    def fixtures(self,enabled=True):
        model=Mock();model.parameters.return_value=[]
        optimizer=Mock();scaler=Mock()
        scaler.is_enabled.return_value=enabled;scaler.get_scale.return_value=65536.
        scaler.get_backoff_factor.return_value=.5
        scaler.state_dict.return_value={'scale':32768.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000,'_growth_tracker':1999}
        return model,optimizer,scaler

    def test_nonfinite_amp_gradient_skips_and_backs_off_without_optimizer_step(self):
        model,optimizer,scaler=self.fixtures()
        with patch.object(torch.nn.utils,'clip_grad_norm_',return_value=torch.tensor(float('inf'))):
            self.assertFalse(trainer.optimizer_attempt(model,optimizer,scaler,5.))
        scaler.unscale_.assert_called_once_with(optimizer)
        scaler.step.assert_not_called();optimizer.step.assert_not_called()
        optimizer.zero_grad.assert_called_once_with(set_to_none=True)
        scaler.update.assert_called_once_with(new_scale=32768.)
        scaler.load_state_dict.assert_called_once_with({'scale':32768.,'growth_factor':2.,'backoff_factor':.5,'growth_interval':2000,'_growth_tracker':0})

    def test_non_amp_failure_and_finite_amp_update_paths(self):
        model,optimizer,scaler=self.fixtures(enabled=False)
        with patch.object(torch.nn.utils,'clip_grad_norm_',return_value=torch.tensor(float('nan'))):
            with self.assertRaisesRegex(ValueError,'without AMP'):trainer.optimizer_attempt(model,optimizer,scaler,5.)
        scaler.step.assert_not_called();scaler.update.assert_not_called();optimizer.step.assert_not_called()
        model,optimizer,scaler=self.fixtures()
        with patch.object(torch.nn.utils,'clip_grad_norm_',return_value=torch.tensor(2.)):
            self.assertTrue(trainer.optimizer_attempt(model,optimizer,scaler,5.))
        scaler.step.assert_called_once_with(optimizer);scaler.update.assert_called_once_with()
        optimizer.step.assert_not_called() # mock scaler never executes any optimizer

    def test_bounded_skips_reset_on_success_and_zero_update_epoch_fails(self):
        progress={key:0 for key in ('optimizer_updates','optimized_training_images','amp_skipped_batches','consecutive_amp_skips','max_consecutive_amp_skips_seen')}
        for _ in range(7):trainer.record_optimizer_attempt(progress,False,12,8)
        trainer.record_optimizer_attempt(progress,True,12,8)
        self.assertEqual(progress['consecutive_amp_skips'],0)
        self.assertEqual(progress['optimized_training_images'],12)
        for _ in range(7):trainer.record_optimizer_attempt(progress,False,12,8)
        with self.assertRaisesRegex(ValueError,'8 consecutive'):trainer.record_optimizer_attempt(progress,False,12,8)
        self.assertEqual(progress['amp_skipped_batches'],15)
        with self.assertRaisesRegex(ValueError,'zero optimizer'):trainer.require_epoch_updates(0)
        trainer.require_epoch_updates(1)


def checkpoint_payload(epoch):
    scores=[3.,.2,.4][:epoch+1];best=min(range(len(scores)),key=lambda i:scores[i])
    rng=torch.Generator().manual_seed(6).get_state()
    return {'checkpoint_kind':cp.KIND,'format_version':cp.FORMAT_VERSION,'identity':{'dataset':'fixture'},
        'model_state_dict':{'w':torch.tensor([float(epoch)])},
        'optimizer_state_dict':{'state':{0:{'step':torch.tensor(float(epoch))}} if epoch else {},'param_groups':[]},
        'scaler_state_dict':{},'scheduler_state_dict':None,'rng_state':cp.rng_state(),
        'sampler_state':{'epoch':epoch},'loader_generator_states':{'train':rng,'val':rng},'epoch':epoch,
        'history':[{'epoch':i,'optimizer_updates':int(i>0),'optimized_training_images':12*int(i>0),'amp_skipped_batches':0,
                    'validation':{'selection_score':value}} for i,value in enumerate(scores)],
        'optimization_progress':{'optimizer_updates':epoch,'optimized_training_images':12*epoch,'amp_skipped_batches':0,
                                 'consecutive_amp_skips':0,'max_consecutive_amp_skips_seen':0},
        'best_epoch':best,'best_score':scores[best],
        'best_checkpoint':{'path':'previous_best.pt','sha256':'0'*64,'epoch':best} if best<epoch else None,
        'bad_epochs':0,'config':{},'initialization':{'mode':'fresh'}}


class ResumeTests(unittest.TestCase):
    def test_epoch0_best_latest_explicit_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for epoch in range(3):cp.save(checkpoint_payload(epoch),root/f'epoch_{epoch:05d}_fixture.pt')
            identity={'dataset':'fixture'}
            best,_=cp.resolve(root,'resume_best',identity)
            latest,_=cp.resolve(root,'resume_latest',identity)
            self.assertEqual(best['epoch'],1);self.assertEqual(latest['epoch'],2)
            explicit,_=cp.resolve(root,'resume_explicit',identity,best['path'])
            restored,_=cp.load(explicit['path'],identity)
            self.assertEqual(restored['optimizer_state_dict']['state'][0]['step'].item(),1.)
            self.assertEqual(cp.resolve(root,'fresh',identity),(None,[]))
            with self.assertRaises(ValueError):cp.resolve(root,'fresh',identity,best['path'])
            with self.assertRaisesRegex(ValueError,'no silent fresh'):
                cp.resolve(root,'resume_best',{'dataset':'other'})
            path=Path(best['path']);path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError):cp.resolve(root,'resume_explicit',identity,path)

    def test_full_state_history_and_sampler_required(self):
        good=checkpoint_payload(2);cp.validate(good)
        for field in ('rng_state','scaler_state_dict','sampler_state','optimizer_state_dict','optimization_progress'):
            broken=deepcopy(good);del broken[field]
            with self.subTest(field=field),self.assertRaises(ValueError):cp.validate(broken)
        broken=deepcopy(good);broken['best_epoch']=2
        with self.assertRaises(ValueError):cp.validate(broken)
        broken=deepcopy(good);broken['optimization_progress']['amp_skipped_batches']=1
        with self.assertRaisesRegex(ValueError,'differs from epoch history'):cp.validate(broken)
        broken=deepcopy(good);broken['history'][1]['optimizer_updates']=0
        with self.assertRaisesRegex(ValueError,'zero optimizer updates'):cp.validate(broken)

    def test_sampler_resume_same_order_and_incompatible_rows_rejected(self):
        rows=[{'id':str(i)} for i in range(20)]
        a=trainer.EpochShuffleSampler(rows,42);a.set_epoch(3)
        b=trainer.EpochShuffleSampler(rows,42);b.load_state_dict(a.state_dict())
        self.assertEqual(list(a),list(b));a.set_epoch(4);self.assertNotEqual(list(a),list(b))
        c=trainer.EpochShuffleSampler(list(reversed(rows)),42)
        with self.assertRaises(ValueError):c.load_state_dict(b.state_dict())


class DatasetTests(unittest.TestCase):
    def fixture(self,root):
        lcd=np.zeros((512,512),np.uint8);lcd[100:200,300:450]=255
        valid=np.ones_like(lcd)*255
        Image.fromarray(lcd).save(root/'lcd.png');Image.fromarray(valid).save(root/'valid.png')
        sources=[];rows=[]
        for i,split in enumerate(('train','train','val','val','test','test')):
            source={'source_id':f's{i}','source_group':'recordingA' if split!='test' else 'recordingB','split':split,
                    'annotation_status':'manually_reviewed','model_predictions_used_for_labels':False}
            sources.append(source)
            image=np.full_like(lcd,i+1);image[lcd>0]=180+i
            Image.fromarray(image).save(root/f'image{i}.png')
            rows.append({'id':str(i),**{k:source[k] for k in ('source_id','source_group','split')},
                'image':f'image{i}.png','lcd_mask':'lcd.png','valid_mask':'valid.png','protected_mask':None,
                'rotation_deg':0,'is_augmented':False})
        (root/'sources.json').write_text(json.dumps(sources),encoding='utf-8')
        (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        self.rehash(root)
        return sources,rows

    def rehash(self,root):
        hashes={p.name:cp.sha256(p) for p in root.iterdir() if p.name!='manifest.json'}
        manifest={'preprocessing_id':architecture.PREPROCESSING_ID,'input_size':512,
                  'samples_file':'samples.jsonl','source_file':'sources.json','sha256':hashes}
        (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')

    def test_temporal_val_allowed_but_test_recording_leak_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sources,rows=self.fixture(root)
            partitions,identity=trainer.verify_dataset(root,progress=False)
            self.assertEqual(identity['sample_counts'],{'train':2,'val':2,'test':2})
            self.assertEqual(identity['train_validation_recording_overlap'],['recordingA'])
            sources[4]['source_group']='recordingA';rows[4]['source_group']='recordingA'
            (root/'sources.json').write_text(json.dumps(sources),encoding='utf-8')
            (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            self.rehash(root)
            with self.assertRaisesRegex(ValueError,'test recording overlaps'):trainer.verify_dataset(root,progress=False)

    def test_cross_split_source_family_and_corrupt_target_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sources,rows=self.fixture(root)
            rows[2]['source_id']=rows[0]['source_id']
            (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8');self.rehash(root)
            with self.assertRaisesRegex(ValueError,'provenance mismatch'):trainer.verify_dataset(root,progress=False)
            self.fixture(root)
            (root/'lcd.png').write_bytes(b'invalid')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):trainer.verify_dataset(root,progress=False)


if __name__=='__main__':unittest.main()
