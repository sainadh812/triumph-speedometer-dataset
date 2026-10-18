"""Static/unit checks only: no model forward pass, optimization or video inference."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from PIL import Image

from mixed_view_v5 import train_landmarks as trainer


def records():
    return [{'sample_id': f'{view}_{roll}_{index}', 'source_view': view, 'roll_bucket': roll}
            for view, roll in trainer.STRATA for index in range(2 if view == trainer.VIEWS[0] else 8)]


def metric_rows():
    rows = []
    for view, roll in trainer.STRATA:
        for index in range(1 if view == trainer.VIEWS[0] else 4):
            accepted = view == trainer.VIEWS[0]
            rows.append({'source_view': view, 'roll_bucket': roll, 'point_errors_px': [1.] * 12,
                         'target_availability': [True] * 12, 'needle_angle_error_input_deg': 2.,
                         'speed_valid': True, 'speed_error_kmh': 1. if accepted else None,
                         'reading': {'accepted': accepted, 'reason': 'ok' if accepted else 'bad_arc'}})
    return rows


def payload(epoch, scores=(10., 3., 5.)):
    metrics = [{'selection_score': value, 'worst_view_selection_score': value+1} for value in scores[:epoch+1]]
    best = min(range(len(metrics)), key=lambda i: metrics[i]['selection_score'])
    states = {'w': torch.tensor([float(epoch)])}
    rng = torch.Generator().manual_seed(13).get_state()
    return {'format_version': trainer.FORMAT_VERSION, 'checkpoint_kind': trainer.CHECKPOINT_KIND,
            'identity': {'dataset': {'manifest_sha256': 'fixture'}, 'architecture': trainer.ARCHITECTURE_ID,
                         'preprocessing': trainer.PREPROCESSING_ID},
            'model_state_dict': states, 'optimizer_state_dict': {'state': {0: {'step': torch.tensor(float(epoch))}} if epoch else {}, 'param_groups': []},
            'scaler_state_dict': {}, 'scheduler_state_dict': None, 'rng_state': trainer.durable._rng_state(),
            'loader_generator_states': {'train': rng, 'evaluation': rng}, 'sampler_state': {'epoch': epoch},
            'training_state': {'epoch': epoch},
            'history': [{'epoch': index, 'validation_metrics': metrics[index]} for index in range(1, epoch+1)],
            'before_metrics': metrics[0], 'current_metrics': metrics[-1],
            'best_state': {'epoch': best, 'selection_loss': metrics[best]['selection_score'], 'model_state_dict': {'w': torch.tensor([float(best)])}},
            'worst_view_best_state': {'epoch': best, 'selection_loss': metrics[best]['worst_view_selection_score']},
            'initialization': {'mode': 'fresh'}, 'config': {}}


class BalancedSamplerTests(unittest.TestCase):
    def test_every_batch_and_epoch_balances_six_strata(self):
        data = records()
        sampler = trainer.BalancedStrataBatchSampler(data, 12, epoch_samples=60)
        sampler.set_epoch(3)
        batches = list(sampler)
        self.assertEqual(len(batches), 5)
        for batch in batches:
            self.assertEqual(Counter((data[i]['source_view'], data[i]['roll_bucket']) for i in batch), Counter({key: 2 for key in trainer.STRATA}))
        self.assertEqual(len([index for batch in batches for index in batch]), 60)

    def test_epoch_boundary_resume_is_deterministic(self):
        data = records()
        first = trainer.BalancedStrataBatchSampler(data, 6, 60)
        first.set_epoch(4)
        restored = trainer.BalancedStrataBatchSampler(data, 6, 60)
        restored.load_state_dict(first.state_dict())
        self.assertEqual(list(first), list(restored))
        first.set_epoch(5)
        self.assertNotEqual(list(first), list(restored))
        incompatible = trainer.BalancedStrataBatchSampler(list(reversed(data)), 6, 60)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            incompatible.load_state_dict(restored.state_dict())

    def test_reject_missing_stratum_or_nonbalanced_batch(self):
        with self.assertRaises(ValueError):
            trainer.BalancedStrataBatchSampler(records(), 8)
        with self.assertRaises(ValueError):
            trainer.BalancedStrataBatchSampler(records()[2:], 12)


class MetricTests(unittest.TestCase):
    def test_macro_is_equal_weight_and_rejections_are_failures(self):
        result = trainer.grouped_metrics(metric_rows())
        self.assertAlmostEqual(result['selection_score'], 12.)
        self.assertAlmostEqual(result['overall_pooled_selection_score'], 18.)
        self.assertAlmostEqual(result['worst_view_selection_score'], 22.)
        self.assertEqual(result['primary_speed']['accepted'], 3)
        self.assertEqual(result['primary_speed']['rejected'], 12)
        self.assertEqual(result['primary_speed']['within_2_kmh_all_eligible_inputs_percent'], 20.)
        self.assertEqual(result['primary_speed']['rejection_reasons'], {'bad_arc': 12})

    def test_circular_angle_wrap_and_degenerate_vector(self):
        a = np.deg2rad(179.)
        b = np.deg2rad(-179.)
        self.assertAlmostEqual(trainer._angle_error([[0,0], [np.cos(a), np.sin(a)]], [[0,0], [np.cos(b), np.sin(b)]], [True, True]), 2.)
        self.assertIsNone(trainer._angle_error([[0,0],[0,0]], [[0,0],[1,0]], [True,True]))


class TransferStageTests(unittest.TestCase):
    def toy(self):
        model = nn.Module()
        model.features = nn.Sequential(*[nn.Sequential(nn.Linear(2,2), nn.BatchNorm1d(2)) for _ in range(6)])
        model.heatmap_head = nn.Linear(2,2)
        model.tick_head = nn.Linear(2,10)
        model.visibility_head = nn.Linear(2,12)
        model.speed_head = nn.Linear(2,1)
        return model

    def test_warmup_freezes_encoder_batchnorm_but_both_heads_learn(self):
        model = self.toy()
        config = trainer.TrainConfig('unused', 'unused')
        trainer.configure_stage(model, 1, config)
        self.assertTrue(all(not layer.training for layer in model.features))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.features.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.tick_head.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.heatmap_head.parameters()))
        trainer.configure_stage(model, 2, config)
        self.assertFalse(model.features[0].training)
        self.assertTrue(model.features[-1].training)
        self.assertFalse(model.speed_head.training)

    def test_all_transfer_tensors_preserved_and_unverified_file_rejected(self):
        destination, source = self.toy(), self.toy()
        digest = next(iter(trainer.SOURCE_V4_SHA256))
        with patch.object(trainer, '_sha', return_value=digest), patch.object(trainer.v4, 'load_model', return_value=(source, {'best_epoch':17})):
            metadata = trainer.load_transfer_checkpoint(destination, 'unused')
        self.assertFalse(metadata['optimizer_restored'])
        for name, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, destination.state_dict()[name]))
        with patch.object(trainer, '_sha', return_value='wrong'):
            with self.assertRaises(ValueError):
                trainer.load_transfer_checkpoint(destination, 'unused')

    def test_colab_guard_prevents_any_local_data_or_model_execution(self):
        with patch.object(trainer.torch.cuda, 'is_available', return_value=False), patch.object(trainer, 'verify_dataset') as verify:
            with self.assertRaisesRegex(RuntimeError, 'no local fallback'):
                trainer.train(trainer.TrainConfig('unused', 'unused'), mode='fresh')
            verify.assert_not_called()


class CheckpointTests(unittest.TestCase):
    def test_epoch_zero_and_best_latest_keep_distinct_full_states(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for epoch in range(3):
                trainer.save_checkpoint(payload(epoch), root/f'epoch_{epoch:05d}_fixture.pt')
            identity = payload(0)['identity']
            best, _ = trainer.select_resume_checkpoint(root, identity, 'resume_best')
            latest, _ = trainer.select_resume_checkpoint(root, identity, 'resume_latest')
            self.assertEqual(best['epoch'], 1)
            self.assertEqual(latest['epoch'], 2)
            restored, _ = trainer.durable._read_complete(best['path'])
            self.assertEqual(restored['optimizer_state_dict']['state'][0]['step'].item(), 1.)
            self.assertTrue(torch.equal(restored['model_state_dict']['w'], torch.tensor([1.])))
            with self.assertRaisesRegex(ValueError, 'refusing silent fresh'):
                trainer.select_resume_checkpoint(root, {'incompatible': True}, 'resume_best')

    def test_corrupt_and_incomplete_saves_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'epoch_00000_fixture.pt'
            trainer.save_checkpoint(payload(0), path)
            path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError, 'No verified compatible'):
                trainer.discover_checkpoints(root)
            path.with_suffix('.complete.json').unlink()
            with self.assertRaises(ValueError):
                trainer.discover_checkpoints(root)

    def test_payload_requires_optimizer_rng_sampler_and_correct_best(self):
        valid = payload(2)
        trainer.validate_payload(valid)
        for mutation in ('optimizer', 'sampler', 'best', 'current', 'rng'):
            broken = deepcopy(valid)
            if mutation == 'optimizer': broken['optimizer_state_dict']['state'] = {}
            elif mutation == 'sampler': broken['sampler_state']['epoch'] = 1
            elif mutation == 'best': broken['best_state']['epoch'] = 2
            elif mutation == 'current': broken['current_metrics'] = {'selection_score': 999.}
            else: del broken['rng_state']['torch']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                trainer.validate_payload(broken)


class DatasetIntegrityTests(unittest.TestCase):
    def fixture(self, root):
        def write(name, value):
            (root/name).write_text(json.dumps(value), encoding='utf-8')
        ticks = [{'speed_kmh': speed, 'xy': [256+100*np.cos(angle), 256+100*np.sin(angle)]}
                 for speed, angle in zip(range(0,181,20), np.linspace(.5,4.8,10))]
        write('calibration.json', {'pivot_xy':[256,256], 'ticks':ticks})
        config = {'preprocessing_id':trainer.PREPROCESSING_ID, 'color':'grayscale', 'input_size':[512,512],
                  'lcd_mask_applied':True, 'lcd_mask_value':128, 'camera_stabilization':'none',
                  'input_geometry':'full_frame_square_letterbox_then512_no_crop_no_canonical_warp',
                  'mask_placement':'image_only_multi_reference_polygon_projection',
                  'landmark_supervision':trainer.AVAILABILITY_SEMANTICS}
        write('preprocessing_config.json', config)
        mask = np.zeros((512,512), np.uint8)
        mask[470:491,470:491] = 255
        Image.fromarray(mask).save(root/'mask.png')
        samples = []
        for split in ('train','validation'):
            for view, roll in trainer.STRATA:
                index = len(samples)
                pixels = np.full((512,512), index+1, np.uint8)
                pixels[mask == 255] = 128
                Image.fromarray(pixels).save(root/f'image_{index}.png')
                samples.append({'sample_id':str(index), 'image_path':f'image_{index}.png', 'lcd_mask_path':'mask.png',
                    'source_view':view, 'roll_bucket':roll, 'split':split, 'parent_split':split,
                    'parent_sample_id':str(index), 'source_group':f'{split}_{view}', 'visibility':[True]*12,
                    'points_xy':[[256,256],[356,256]]+[tick['xy'] for tick in ticks], 'speed_kmh':60., 'speed_valid':True})
        write('mask_references.json', {'references':[{'id':view, 'reference_image':'image_0.png', 'calibration':'calibration.json',
              'lcd_polygon_xy':[[470,470],[490,470],[490,490],[470,490]]} for view in trainer.VIEWS]})
        write('annotations.json', {'keypoint_names':trainer.KEYPOINT_NAMES, 'samples':samples})
        self.rehash(root, config)
        return samples, config

    def rehash(self, root, config):
        hashes = {path.name:trainer._sha(path) for path in root.iterdir() if path.name != 'manifest.json'}
        (root/'manifest.json').write_text(json.dumps({'sha256':hashes, 'preprocessing':config}), encoding='utf-8')

    def test_valid_fixture_and_parent_split_leak_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            samples, config = self.fixture(root)
            partitions, _ = trainer.verify_dataset(root, progress=False)
            self.assertEqual([len(partitions[name]) for name in ('train','validation')], [6,6])
            samples[6]['parent_sample_id'] = samples[0]['parent_sample_id']
            (root/'annotations.json').write_text(json.dumps({'keypoint_names':trainer.KEYPOINT_NAMES, 'samples':samples}), encoding='utf-8')
            self.rehash(root, config)
            with self.assertRaisesRegex(ValueError, 'parent leakage'):
                trainer.verify_dataset(root, progress=False)

    def test_mask_tamper_is_rejected_even_if_manifest_rehashed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, config = self.fixture(root)
            with Image.open(root/'image_0.png') as image:
                pixels = np.asarray(image).copy()
            pixels[480,480] = 255
            Image.fromarray(pixels).save(root/'image_0.png')
            self.rehash(root, config)
            with self.assertRaisesRegex(ValueError, 'exactly128'):
                trainer.verify_dataset(root, progress=False)


if __name__ == '__main__':
    unittest.main()
