"""Bounded CPU checks: negative targets, split integrity, durable exact resume."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rpm_training_v1 import train_rpm as tr


def fixture(root):
    root.mkdir(parents=True)
    rows, hashes = [], {}
    for split, count in [('train', 8), ('val', 4)]:
        for index in range(count):
            readable = index % 4 != 0
            q = (index + 1) / (count + 2) if readable else None
            rng = np.random.default_rng(index + (100 if split == 'train' else 200))
            pixels = rng.integers(170, 190, size=tr.INPUT_HW, dtype=np.uint8)
            if readable:
                pixels[round((1 - q) * 240):248, 70:100] = 60
            relative = f'{split}_{index}.png'
            Image.fromarray(pixels).save(root / relative)
            hashes[relative] = tr.sha256(root / relative)
            rows.append({'id': f'{split}_{index}', 'image': relative, 'split': split,
                         'readable': readable, 'fill_fraction': q,
                         'edge_xy': [85, round((1 - q) * 240)] if readable else None,
                         'indicated_rpm_approx': q * tr.RPM_SCALE if readable else None,
                         'label_origin': 'unit_test_render_geometry', 'source_group': f'{split}_family_{index}',
                         'augmentation': {'fixture': True}})
    (root / 'labels.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    hashes['labels.jsonl'] = tr.sha256(root / 'labels.jsonl')
    (root / 'preprocessing.json').write_text(json.dumps({'input_hw': list(tr.INPUT_HW)}))
    hashes['preprocessing.json'] = tr.sha256(root / 'preprocessing.json')
    (root / 'manifest.json').write_text(json.dumps({'sha256': hashes, 'preprocessing': 'fixture grayscale strip'}))
    return rows


def rewrite_labels(root, rows):
    (root / 'labels.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['sha256']['labels.jsonl'] = tr.sha256(root / 'labels.jsonl')
    (root / 'manifest.json').write_text(json.dumps(manifest))


def assert_tree_equal(test, left, right):
    if torch.is_tensor(left):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, np.ndarray):
        test.assertTrue(np.array_equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class RPMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_all_negative_loss_has_no_numeric_supervision(self):
        fractions = torch.tensor([0.2, 0.8], requires_grad=True)
        logits = torch.tensor([0.3, -0.7], requires_grad=True)
        result = tr.loss_terms({'fill_fraction': fractions, 'readable_logit': logits},
                              {'fill_fraction': torch.tensor([float('nan'), float('nan')]),
                               'readable': torch.zeros(2)})
        self.assertEqual(result['numeric_count'], 0)
        self.assertTrue(torch.isfinite(result['loss']))
        result['loss'].backward()
        self.assertTrue(torch.equal(fractions.grad, torch.zeros(2)))
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_mixed_loss_ignores_negative_numeric_value(self):
        fractions = torch.tensor([0.3, 0.8], requires_grad=True)
        result = tr.loss_terms({'fill_fraction': fractions, 'readable_logit': torch.zeros(2, requires_grad=True)},
                              {'fill_fraction': torch.tensor([0.5, float('nan')]),
                               'readable': torch.tensor([1., 0.])})
        result['loss'].backward()
        self.assertNotEqual(float(fractions.grad[0]), 0.)
        self.assertEqual(float(fractions.grad[1]), 0.)

    def test_split_collision_and_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'data'
            rows = fixture(root)
            tr.verify_dataset(root)
            rows[-1]['source_group'] = rows[0]['source_group']
            rewrite_labels(root, rows)
            with self.assertRaisesRegex(ValueError, 'Source-group collision'):
                tr.verify_dataset(root)
            rows[-1]['source_group'] = 'val_fixed'
            rewrite_labels(root, rows)
            with (root / rows[0]['image']).open('ab') as stream:
                stream.write(b'alteration')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                tr.verify_dataset(root)

    def test_cross_split_pixel_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'data'
            rows = fixture(root)
            target = root / rows[-1]['image']
            target.write_bytes((root / rows[0]['image']).read_bytes())
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest['sha256'][rows[-1]['image']] = tr.sha256(target)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Identical image pixels'):
                tr.verify_dataset(root)

    def test_negative_label_contract_and_no_all_invalid_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'data'
            rows = fixture(root)
            rows[0]['fill_fraction'] = 0.
            rewrite_labels(root, rows)
            with self.assertRaisesRegex(ValueError, 'null numeric'):
                tr.verify_dataset(root)
            rows[0]['fill_fraction'] = None
            for row in rows:
                if row['split'] == 'val':
                    row.update(readable=False, fill_fraction=None, edge_xy=None, indicated_rpm_approx=None)
            rewrite_labels(root, rows)
            with self.assertRaisesRegex(ValueError, 'Both splits'):
                tr.verify_dataset(root)

    def test_diagnostics_and_predicted_targets_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'data'
            rows = fixture(root)
            rows[1]['training_eligible'] = False
            rewrite_labels(root, rows)
            with self.assertRaisesRegex(ValueError, 'Diagnostic/ineligible'):
                tr.verify_dataset(root)
            rows[1]['training_eligible'] = True
            rows[1]['renderer'] = {'numeric_target_from_existing_predictions': True}
            rewrite_labels(root, rows)
            with self.assertRaisesRegex(ValueError, 'Existing model predictions'):
                tr.verify_dataset(root)

    def test_reject_everything_still_scores_all_readable_targets(self):
        class RejectAll(torch.nn.Module):
            def forward(self, image):
                n = len(image)
                return {'fill_fraction': torch.full((n,), 0.9), 'readability_score': torch.full((n,), 0.001)}
        batch = {'image': torch.zeros(3, 1, *tr.INPUT_HW), 'fill_fraction': torch.tensor([0.2, 0.6, 0.]),
                 'readable': torch.tensor([1., 1., 0.]), 'id': ['a', 'b', 'c']}
        metrics, _ = tr.evaluate(RejectAll(), [batch], 'cpu')
        self.assertEqual(metrics['numeric_output_coverage'], 0.)
        self.assertAlmostEqual(metrics['fill_fraction_mae'], .5, places=6)
        self.assertGreater(metrics['selection_score'], .5)
        self.assertIsNone(metrics['accepted_readable_approx_rpm_mae'])

    def test_best_is_earlier_than_latest_and_resume_preserves_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture(base / 'data')
            config = tr.TrainConfig(str(base / 'data'), str(base / 'out'), epochs=2, batch_size=4,
                                    patience=0, device='cpu', amp=False)
            original_evaluate = tr.evaluate
            def evaluation_sequence(scores):
                scores = iter(scores)
                def evaluate(*args, **kwargs):
                    metrics, rows = original_evaluate(*args, **kwargs)
                    metrics['selection_score'] = next(scores)
                    return metrics, rows
                return evaluate
            with patch.object(tr, 'evaluate', side_effect=evaluation_sequence([.4, .2, .3, .2])):
                first = tr.train(config)
            self.assertEqual(first['best_epoch'], 1)
            self.assertEqual(first['latest_checkpoint']['epoch'], 2)
            with patch.object(tr, 'evaluate', side_effect=evaluation_sequence([.35, .2])):
                continued = tr.train(replace(config, epochs=3, resume='latest'))
            self.assertEqual(continued['best_checkpoint']['path'], first['best_checkpoint']['path'])
            self.assertEqual(continued['best_epoch'], 1)
            self.assertEqual(continued['latest_checkpoint']['epoch'], 3)
            with patch.object(tr, 'evaluate', side_effect=evaluation_sequence([.1, .1])):
                branch = tr.train(replace(config, resume='best'))
            self.assertEqual(branch['best_epoch'], 2)
            self.assertNotEqual(branch['run_dir'], first['run_dir'])
            self.assertTrue(Path(first['latest_checkpoint']['path']).exists())

    def test_latest_exact_resume_best_explicit_fresh_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data = base / 'data'
            fixture(data)
            config = tr.TrainConfig(str(data), str(base / 'uninterrupted'), epochs=2,
                                    batch_size=4, patience=0, device='cpu', amp=False)
            uninterrupted = tr.train(config)
            split_config = replace(config, output=str(base / 'resumed'))
            first = tr.train(split_config, stop_after_epochs=1)
            resumed = tr.train(replace(split_config, resume='latest'))
            self.assertNotEqual(first['run_dir'], resumed['run_dir'])
            full = tr.load_checkpoint(uninterrupted['latest_checkpoint']['path'])
            replay = tr.load_checkpoint(resumed['latest_checkpoint']['path'])
            for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict',
                        'rng_state', 'loader_generator_states', 'history'):
                assert_tree_equal(self, full[key], replay[key])
            best_path = tr.resolve_checkpoint(split_config.output, 'best', identity=replay['identity'])
            best_payload = tr.load_checkpoint(best_path)
            expected_best = min(replay['history'], key=lambda row: row['validation']['selection_score'])
            self.assertEqual(best_payload['epoch'], expected_best['epoch'])
            best_resumed = tr.train(replace(split_config, resume='best', epochs=best_payload['epoch'] + 1))
            init = json.loads((Path(best_resumed['run_dir']) / 'initialization.json').read_text())
            self.assertEqual(init['start_epoch'], best_payload['epoch'])
            self.assertTrue(init['optimizer_restored'])
            self.assertFalse(init['random_initialization'])
            explicit = tr.train(replace(split_config, output=str(base / 'explicit'), resume='explicit',
                                        checkpoint=first['latest_checkpoint']['path']))
            explicit_payload = tr.load_checkpoint(explicit['latest_checkpoint']['path'])
            assert_tree_equal(self, full['model_state_dict'], explicit_payload['model_state_dict'])
            fresh_again = tr.train(replace(split_config, epochs=1, report_path=str(base / 'known_report.json')))
            self.assertNotEqual(first['run_dir'], fresh_again['run_dir'])
            fresh_initial = tr.load_checkpoint(Path(fresh_again['run_dir']) / 'checkpoints/epoch_0000.pt')
            old_initial = tr.load_checkpoint(Path(first['run_dir']) / 'checkpoints/epoch_0000.pt')
            assert_tree_equal(self, fresh_initial['model_state_dict'], old_initial['model_state_dict'])
            self.assertEqual(fresh_initial['optimizer_state_dict']['state'], {})
            self.assertTrue((base / 'known_report.json').is_file())
            self.assertTrue((Path(fresh_again['run_dir']) / 'validation_predictions.json').is_file())
            self.assertEqual(json.loads((Path(split_config.output) / 'latest_run.json').read_text())['run_dir'], fresh_again['run_dir'])
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                tr.train(replace(split_config, resume='latest', epochs=3, batch_size=2))
            with self.assertRaises(FileNotFoundError):
                tr.resolve_checkpoint(base / 'missing', 'latest')
            with self.assertRaisesRegex(ValueError, 'Fresh random'):
                tr.resolve_checkpoint(base, 'fresh', first['latest_checkpoint']['path'])
            inference = Path(fresh_again['best_model'])
            with self.assertRaisesRegex(ValueError, 'marker or checksum'):
                tr.load_checkpoint(inference)
            corrupt = Path(first['latest_checkpoint']['path'])
            with corrupt.open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                tr.load_checkpoint(corrupt)


if __name__ == '__main__':
    unittest.main()
