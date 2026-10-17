"""Small stochastic software tests. They do not measure speedometer accuracy."""
from dataclasses import replace
from pathlib import Path
import copy
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from ring_training_v2 import train as trainer


class TinyModel(nn.Module):
    def __init__(self, pretrained=False, freeze_needle=False):
        super().__init__()
        assert not pretrained
        self.features = nn.Linear(3, 4)
        self.shared = nn.Linear(4, 4)
        self.tick_head = nn.Linear(4, 1)
        self.needle = nn.Linear(3, 1)
        self.dormant = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(.25)
        self.freeze_needle = freeze_needle

    def configure_stage(self, stage, *, unfreeze_all_encoder=False):
        self.unfreeze_all_encoder = unfreeze_all_encoder
        for name, parameter in self.named_parameters():
            group = parameter_group(name)
            enabled = group == 'new' or stage == 'finetune'
            if group == 'needle':
                enabled = not self.freeze_needle
            parameter.requires_grad_(enabled and group is not None)

    def forward(self, image):
        hidden = torch.tanh(self.shared(torch.tanh(self.features(image))))
        return {'speed': (self.tick_head(self.dropout(hidden)) + self.needle(image)).squeeze(-1)}


def parameter_group(name):
    return {'features': 'backbone', 'shared': 'shared', 'tick_head': 'new',
            'needle': 'needle', 'dormant': None}[name.split('.')[0]]


class TinyDataset(Dataset):
    def __init__(self, data_dir, records=None, input_size=512):
        self.records, self.root = records, Path(data_dir)
        assert all(row['dataset_split'] != 'test' for row in records), 'Test data must never be loaded by trainer'

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        value = float(self.records[index]['value'])
        jitter = (random.random() + np.random.random() + float(torch.rand(()))) * .005
        return {'image': torch.tensor([value, value * .25, 1.]) + jitter,
                'speed': torch.tensor(value * .6 + .4), 'index': index}


def tiny_loss(output, batch, loss_config=None):
    loss = (output['speed'] - batch['speed']).square().mean()
    return loss, {'speed_loss': float(loss.detach())}


def metric_fixture(score, base=None):
    return {**(base or {'combined_loss': score, 'samples': 2, 'components': {'speed_loss': score}}),
            'selection_score': score, 'visible_tick_mean_error_px': score,
            'visible_tick_p95_error_px': score, 'direct_mae_kmh': score,
            'arc_iou': .1, 'visibility_balanced_accuracy': .5, 'geometry_coverage_percent': 0,
            'geometry_within_2_all_inputs_percent': 0, 'test_samples_evaluated': 0}


def tiny_evaluation(model, loader, device, loss_config, geometry_penalty_px=20):
    base = trainer.evaluate_loss(model, loader, device, loss_config)
    return metric_fixture(base['combined_loss'] + geometry_penalty_px, base)


def fake_migration(model, ring_path, needle_path):
    ring = torch.load(ring_path, weights_only=True)
    needle = torch.load(needle_path, weights_only=True)
    state = model.state_dict()
    state.update(ring['weights'])
    state.update(needle['weights'])
    model.load_state_dict(state)
    model.freeze_needle = True
    return {'mode': 'two_source_migration', 'source_ring_sha256': trainer._sha(ring_path),
            'source_legacy_sha256': trainer._sha(needle_path), 'source_ring_epoch': ring['epoch'],
            'source_legacy_epoch': needle['epoch'], 'optimizer_restored': False}


def assert_tree_equal(test, first, second):
    if isinstance(first, torch.Tensor):
        test.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_tree_equal(test, first[key], second[key])
    elif isinstance(first, (list, tuple)):
        test.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(first, second)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix='v2_')
        self.root = Path(self.temporary.name)
        self.data = self.root / 'd'
        self.data.mkdir()
        (self.data / 'labels.json').write_text('{"fixture":1}')
        self.records = [{'sample_id': f's{i}', 'dataset_split': 'train' if i < 8 else ('validation' if i < 10 else 'test'),
                         'value': i / 12, 'source_pose_group': f'g{i}',
                         'tick_landmarks': [{'heatmap_loss_valid': j < 5} for j in range(6)]}
                        for i in range(12)]
        def verify(data_dir):
            return copy.deepcopy(self.records), {'count': 12}
        self.patches = [patch.object(trainer.ring_model, 'AugmentedRingModel', TinyModel),
                        patch.object(trainer.ring_model, 'training_loss', tiny_loss),
                        patch.object(trainer.ring_model, 'parameter_group', parameter_group),
                        patch.object(trainer.ring_model, 'load_transfer_checkpoints', fake_migration),
                        patch.object(trainer.ring_data, 'RingDataset', TinyDataset),
                        patch.object(trainer.ring_data, 'verify_dataset', verify),
                        patch.object(trainer, 'evaluate_validation', tiny_evaluation)]
        for p in self.patches:
            p.start()
        self.config = trainer.TrainConfig(str(self.data), str(self.root / 'runs'), batch_size=4,
                                         head_epochs=1, finetune_epochs=1, device='cpu', amp=False,
                                         head_lr=.02, finetune_lr=.01, backbone_lr=.005, needle_lr=.01)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temporary.cleanup()

    def payload(self, report):
        return torch.load(report['latest_checkpoint_path'], map_location='cpu', weights_only=True)

    def sources(self):
        source = TinyModel()
        ring_path, needle_path = self.root / 'ring.pt', self.root / 'needle.pt'
        ring = {name: torch.full_like(value, .3) for name, value in source.state_dict().items()
                if parameter_group(name) in {'backbone', 'shared'}}
        needle = {name: torch.full_like(value, .2) for name, value in source.state_dict().items()
                  if parameter_group(name) == 'needle'}
        torch.save({'weights': ring, 'epoch': 33}, ring_path)
        torch.save({'weights': needle, 'epoch': 19}, needle_path)
        return ring_path, needle_path, ring, needle

    def test_exact_resume_across_stage_boundary_preserves_all_stochastic_state(self):
        uninterrupted = trainer.train(replace(self.config, output_dir=str(self.root / 'full')), mode='fresh')
        interrupted = trainer.train(self.config, mode='fresh', stop_after_epochs=1)
        initial = torch.load(interrupted['initial_checkpoint_path'], weights_only=True)
        middle = self.payload(interrupted)
        self.assertTrue(torch.equal(initial['model_state_dict']['features.weight'], middle['model_state_dict']['features.weight']))
        self.assertFalse(torch.equal(initial['model_state_dict']['tick_head.weight'], middle['model_state_dict']['tick_head.weight']))
        self.assertFalse(torch.equal(initial['model_state_dict']['needle.weight'], middle['model_state_dict']['needle.weight']))
        resumed = trainer.train(self.config, mode='resume_latest', search_root=self.root / 'runs')
        self.assertEqual(resumed['epochs_this_call'], 1)
        self.assertTrue(resumed['resume_metadata']['optimizer_restored'])
        self.assertEqual(resumed['source_baseline_checkpoint_path'], interrupted['initial_checkpoint_path'])
        full, actual = self.payload(uninterrupted), self.payload(resumed)
        for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict',
                    'rng_state', 'loader_generator_states', 'training_state', 'history', 'best_state', 'before_metrics'):
            assert_tree_equal(self, full[key], actual[key])
        rates = {group['name']: group['lr'] for group in actual['optimizer_state_dict']['param_groups']}
        self.assertEqual(rates, {'backbone': .005, 'shared': .01, 'new': .001, 'needle': .01})
        self.assertEqual(len([p for g in actual['optimizer_state_dict']['param_groups'] for p in g['params']]), 8)
        hashes = {p: trainer._sha(p) for p in (self.root / 'runs').rglob('epoch_*.pt')}
        completed = trainer.train(self.config, mode='resume_latest', search_root=self.root / 'runs')
        self.assertEqual(completed['epochs_this_call'], 0)
        self.assertEqual(completed['target_epoch'], 2)
        self.assertNotEqual(completed['run_dir'], resumed['run_dir'])
        for path, checksum in hashes.items():
            self.assertEqual(trainer._sha(path), checksum)

    def test_two_source_migration_starts_zero_and_frozen_needle_survives_resume(self):
        ring_path, needle_path, ring, needle = self.sources()
        report = trainer.train(self.config, mode='transfer', ring_checkpoint=ring_path,
                               needle_checkpoint=needle_path, stop_after_epochs=0)
        initial = self.payload(report)
        self.assertEqual(initial['training_state']['epoch'], 0)
        self.assertEqual(initial['optimizer_state_dict']['state'], {})
        self.assertEqual(initial['initialization']['source_ring_epoch'], 33)
        self.assertEqual(initial['initialization']['source_legacy_epoch'], 19)
        self.assertTrue(initial['initialization']['freeze_needle'])
        for name, value in {**ring, **needle}.items():
            self.assertTrue(torch.equal(initial['model_state_dict'][name], value))
        resumed = trainer.train(self.config, mode='resume_latest', search_root=self.root / 'runs')
        final = self.payload(resumed)
        for name, value in needle.items():
            self.assertTrue(torch.equal(final['model_state_dict'][name], value))
        loaded, saved = trainer.load_model(resumed['best_model_path'])
        self.assertTrue(loaded.freeze_needle)
        self.assertEqual(saved['checkpoint_kind'], 'augmented_ring_inference_weights_v2')

    def test_fresh_ignores_prior_files_and_learns_random_needle(self):
        corrupt = self.root / 'old'
        corrupt.mkdir()
        (corrupt / 'epoch_00001.complete.json').write_text('bad')
        report = trainer.train(self.config, mode='fresh', ring_checkpoint=corrupt / 'absent.pt',
                               needle_checkpoint=corrupt / 'missing.pt', search_root=corrupt, stop_after_epochs=0)
        self.assertIsNone(report['resume_metadata']['source_checkpoint_path'])
        self.assertFalse(report['initialization']['freeze_needle'])
        self.assertTrue(report['initialization']['unfreeze_all_encoder'])

    def test_resume_best_restores_entire_selected_state(self):
        scores = iter([4., 1., 3.])
        with patch.object(trainer, 'evaluate_validation', lambda *args: metric_fixture(next(scores))):
            first = trainer.train(self.config, mode='fresh')
        self.assertEqual(first['best_epoch'], 1)
        best = trainer.train(self.config, mode='resume_best', search_root=self.root / 'runs', stop_after_epochs=0)
        self.assertEqual(best['completed_epoch'], 1)
        self.assertEqual(best['target_epoch'], 2)
        source = torch.load(first['best_checkpoint_path'], weights_only=True)
        selected = self.payload(best)
        for key in ('model_state_dict', 'optimizer_state_dict', 'rng_state', 'loader_generator_states', 'history'):
            assert_tree_equal(self, source[key], selected[key])

    def test_corrupt_latest_recovers_verified_previous_but_never_fresh(self):
        first = trainer.train(self.config, mode='fresh')
        latest = Path(first['latest_checkpoint_path'])
        latest.write_bytes(latest.read_bytes() + b'corrupt')
        (latest.parent / 'epoch_99999_unpublished.pt').write_bytes(b'partial')
        with self.assertWarnsRegex(RuntimeWarning, 'Skipped 2'):
            recovered = trainer.train(self.config, mode='resume_latest', search_root=self.root / 'runs', stop_after_epochs=0)
        self.assertEqual(recovered['completed_epoch'], 1)
        self.assertEqual(len(recovered['resume_metadata']['recovery_diagnostics']), 2)
        empty = self.root / 'empty'
        empty.mkdir()
        (empty / 'epoch_00001_dead.pt').write_bytes(b'unpublished')
        with self.assertRaisesRegex(ValueError, 'No verified'):
            trainer.train(self.config, mode='resume_latest', search_root=empty)

    def test_changed_dataset_or_budget_is_rejected(self):
        trainer.train(self.config, mode='fresh', stop_after_epochs=0)
        for changed in (replace(self.config, batch_size=2), replace(self.config, finetune_epochs=2)):
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                trainer.train(changed, mode='resume_latest', search_root=self.root / 'runs')
        (self.data / 'labels.json').write_text('{"fixture":2}')
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            trainer.train(self.config, mode='resume_latest', search_root=self.root / 'runs')

    def test_visibility_weight_uses_training_labels_only(self):
        for row in self.records[8:]:
            row['tick_landmarks'] = [{'heatmap_loss_valid': False} for _ in range(6)]
        report = trainer.train(self.config, mode='fresh', stop_after_epochs=0)
        self.assertEqual(report['identity']['loss_config']['visibility_negative_weight'], 5.)
        self.assertEqual(report['identity']['visibility_weight_basis']['train_visible'], 40)
        self.assertEqual(report['identity']['visibility_weight_basis']['train_hidden'], 8)
        self.assertEqual(report['training_samples'], 8)
        self.assertEqual(report['evaluation_samples'], 2)
        self.assertEqual(report['test_samples_evaluated'], 0)

    def test_test_group_leakage_rejected_before_training(self):
        self.records[-1]['source_pose_group'] = self.records[0]['source_pose_group']
        with self.assertRaisesRegex(ValueError, 'leakage'):
            trainer.train(self.config, mode='fresh')


class ValidationSelectionTests(unittest.TestCase):
    def test_rejections_count_in_all_input_geometry_penalty(self):
        counts = {'visible_detected': 5, 'visible_missed': 5, 'occluded_detected': 4, 'occluded_called_visible': 1}
        metrics = {'landmarks': {'visibility_confusion_counts': counts, 'visible_tick_mean_error_px': 2.,
                                'visible_tick_p95_error_px': 4., 'needle_pivot_mean_error_px': 1., 'needle_tip_mean_error_px': 3.},
                   'arc': {'valid_pixel_iou': .2}, 'direct_speed': {'mae_kmh': 2.},
                   'geometric_speed': {'coverage_percent': 50.}}
        rows = [{'geometry_accepted': True, 'geometry_error_kmh': 1.},
                {'geometry_accepted': False, 'geometry_error_kmh': None}]
        with patch.object(trainer, 'evaluate_loss', return_value={'combined_loss': 3.}), \
             patch.object(trainer.ring_eval, 'evaluate', return_value=(metrics, rows)):
            result = trainer.evaluate_validation(None, None, 'cpu', None)
        self.assertEqual(result['selection_score'], 12.)
        self.assertEqual(result['geometry_within_2_all_inputs_percent'], 50.)
        self.assertAlmostEqual(result['visibility_balanced_accuracy'], .65)


if __name__ == '__main__':
    unittest.main(verbosity=2)
