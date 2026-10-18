"""Offline numerical/data/identity checks. Does not import torch or run models."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from dataset import POINT_NAMES, normalize_gray, sha256, validate_dataset, validate_points
from metrics import MetricAccumulator, binary_scores, box_iou, heatmap_to_points
from state import CHECKPOINT_KIND, FORMAT_VERSION, identity_digest, should_stop, validate_checkpoint


def fixture(root):
    files, tasks = [], {}
    for task, names in POINT_NAMES.items():
        directory = root / task
        directory.mkdir()
        rows = []
        for number, split in enumerate(('train', 'val', 'test')):
            image = np.full((64, 32), 128, np.uint8)
            image_name = f'{split}.png'
            cv2.imwrite(str(directory / image_name), image)
            files.append({'path': task + '/' + image_name, 'sha256': sha256(directory / image_name)})
            row = {'id': task + str(number), 'source_id': 'source' + str(number), 'split': split,
                   'image': image_name, 'height': 64, 'width': 32, 'points': [[10, 20]] * len(names),
                   'visible': [True] * len(names)}
            if task == 'localizer':
                mask = np.zeros((64, 32), np.uint8)
                mask[10:30, 2:20] = 255
                mask_name = f'mask_{split}.png'
                cv2.imwrite(str(directory / mask_name), mask)
                files.append({'path': task + '/' + mask_name, 'sha256': sha256(directory / mask_name)})
                row.update(lcd_mask=mask_name, dashboard_box=[1, 2, 30, 61], present=1)
            rows.append(row)
        document = {'point_names': names, 'input_size': [32, 64], 'samples': rows}
        (directory / 'samples.json').write_text(json.dumps(document))
        files.append({'path': task + '/samples.json', 'sha256': sha256(directory / 'samples.json')})
        tasks[task] = {'input_size': [32, 64], 'point_names': names, 'counts': {'train': 1, 'val': 1, 'test': 1}}
    manifest = {'schema_version': 1, 'files': files, 'tasks': tasks}
    (root / 'dataset_manifest.json').write_text(json.dumps(manifest))
    return manifest


def refresh_manifest(root, manifest):
    for row in manifest['files']:
        row['sha256'] = sha256(root / row['path'])
    (root / 'dataset_manifest.json').write_text(json.dumps(manifest))


class DatasetTests(unittest.TestCase):
    def test_valid_manifest_uses_explicit_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            result = validate_dataset(root)
            self.assertEqual(result['tasks']['rpm']['input_hw'], (64, 32))

    def test_tampered_image_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            (root / 'speed/train.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                validate_dataset(root)

    def test_cross_task_source_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            path = root / 'rpm/samples.json'
            document = json.loads(path.read_text())
            document['samples'][1]['source_id'] = 'source0'
            path.write_text(json.dumps(document))
            refresh_manifest(root, manifest)
            with self.assertRaisesRegex(ValueError, 'cross dataset splits'):
                validate_dataset(root)

    def test_declared_input_dimension_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            manifest['tasks']['rpm']['input_size'] = [64, 32]
            refresh_manifest(root, manifest)
            with self.assertRaisesRegex(ValueError, 'input_size'):
                validate_dataset(root)

    def test_true_negative_and_partial_corners_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            path = root / 'localizer/samples.json'
            document = json.loads(path.read_text())
            row = document['samples'][0]
            count = len(POINT_NAMES['localizer'])
            row.update(present=0, points=[[0, 0]] * count, visible=[False] * count,
                       dashboard_box=[0, 0, 0, 0])
            cv2.imwrite(str(root / 'localizer' / row['lcd_mask']), np.zeros((64, 32), np.uint8))
            document['samples'][1]['points'][2] = [0, 0]
            document['samples'][1]['visible'][2] = False
            path.write_text(json.dumps(document))
            refresh_manifest(root, manifest)
            validate_dataset(root)

    def test_unavailable_coordinate_must_be_zero(self):
        with self.assertRaisesRegex(ValueError, 'Unavailable'):
            validate_points({'points': [[9, 3]], 'visible': [False]}, 1, (64, 32))

    def test_rpm_roi_can_remain_visible_when_full_lcd_corners_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            path = root / 'localizer/samples.json'
            document = json.loads(path.read_text())
            row = document['samples'][0]
            row['points'][:4] = [[0, 0]] * 4
            row['visible'][:4] = [False] * 4
            path.write_text(json.dumps(document))
            refresh_manifest(root, manifest)
            result = validate_dataset(root)
            self.assertEqual(result['tasks']['localizer']['point_names'][4:],
                             ['rpm_roi_tl', 'rpm_roi_tr', 'rpm_roi_br', 'rpm_roi_bl'])
            self.assertEqual(result['tasks']['localizer']['rows'][0]['visible'],
                             [False] * 4 + [True] * 4)

    def test_semantic_point_order_must_match_model_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root)
            path = root / 'localizer/samples.json'
            document = json.loads(path.read_text())
            names = document['point_names']
            names[0], names[4] = names[4], names[0]
            manifest['tasks']['localizer']['point_names'] = names
            path.write_text(json.dumps(document))
            refresh_manifest(root, manifest)
            with self.assertRaisesRegex(ValueError, 'model contract'):
                validate_dataset(root)

    def test_normalization_is_three_identical_gray_channels_before_imagenet(self):
        output = normalize_gray(np.full((2, 3), 128, np.uint8))
        reconstructed = output * np.asarray([.229, .224, .225])[:, None, None] + np.asarray([.485, .456, .406])[:, None, None]
        np.testing.assert_allclose(reconstructed, 128 / 255., atol=1e-7)


class MetricTests(unittest.TestCase):
    def test_heatmap_mapping_uses_pixel_centers(self):
        values = np.zeros((1, 1, 2, 4))
        values[0, 0, 1, 2] = 9
        np.testing.assert_allclose(heatmap_to_points(values, (8, 16)), [[[9.5, 5.5]]])

    def test_invisible_points_do_not_affect_error(self):
        accumulator = MetricAccumulator(['first', 'second'])
        accumulator.add(np.array([[[3, 4], [999, 999]]]), np.zeros((1, 2, 2)), [[True, False]], [[.9, .1]])
        result = accumulator.result()
        self.assertEqual(result['visible_point_error_px']['mean'], 5.)
        self.assertIsNone(result['per_landmark_error_px']['second']['mean'])

    def test_empty_masks_scored_as_negatives_not_perfect_positive_iou(self):
        accumulator = MetricAccumulator(['corner'])
        accumulator.add(np.zeros((1, 1, 2)), np.zeros((1, 1, 2)), [[False]], [[.1]],
                        presence=[.1], present=[0], masks=np.ones((1, 1, 2, 2)),
                        mask_targets=np.zeros((1, 1, 2, 2)), boxes=[[0, 0, 0, 0]], box_targets=[[0, 0, 0, 0]])
        result = accumulator.result()
        self.assertEqual(result['lcd_false_positive_fraction_on_empty_targets']['mean'], 1.)
        self.assertIsNone(result['lcd_iou_on_nonempty_targets']['mean'])

    def test_box_iou_known_case(self):
        np.testing.assert_allclose(box_iou([[0, 0, 2, 2]], [[1, 0, 3, 2]]), [1 / 3])


class CheckpointTests(unittest.TestCase):
    def payload(self):
        identity, config = {'dataset': 'abc', 'architecture': 'v1'}, {'epochs': 40}
        payload = {'checkpoint_kind': CHECKPOINT_KIND, 'format_version': FORMAT_VERSION,
                   'identity': identity, 'identity_sha256': identity_digest(identity), 'config': config,
                   'epoch': 2, 'model_state_dict': {}, 'optimizer_state_dict': {}, 'scheduler_state_dict': {},
                   'scaler_state_dict': {}, 'rng_state': dict.fromkeys(('python', 'numpy', 'torch_cpu', 'torch_cuda')),
                   'loader_generator_states': dict.fromkeys(('train', 'val', 'test')),
                   'history': [{'epoch': 1}, {'epoch': 2}], 'best_state_dict': {'fake': 1}, 'best_epoch': 1,
                   'best_score': .3, 'epochs_without_improvement': 1, 'training_complete': False,
                   'initialization': {'old_heads': False}}
        return payload, identity, config

    def test_full_checkpoint_identity_passes(self):
        validate_checkpoint(*self.payload())

    def test_changed_dataset_and_architecture_rejected(self):
        payload, identity, config = self.payload()
        for key in ('dataset', 'architecture'):
            changed = dict(identity, **{key: 'different'})
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                validate_checkpoint(payload, changed, config)

    def test_missing_optimizer_or_rng_rejected(self):
        payload, identity, config = self.payload()
        del payload['optimizer_state_dict']
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            validate_checkpoint(payload, identity, config)
        payload, identity, config = self.payload()
        del payload['rng_state']['torch_cuda']
        with self.assertRaisesRegex(ValueError, 'RNG'):
            validate_checkpoint(payload, identity, config)

    def test_changed_schedule_rejected(self):
        payload, identity, config = self.payload()
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            validate_checkpoint(payload, identity, {'epochs': 50})

    def test_early_stop_respects_minimum_epochs(self):
        config = {'epochs': 40, 'min_epochs': 10, 'patience': 8}
        self.assertFalse(should_stop(9, 9, config))
        self.assertTrue(should_stop(10, 8, config))
        self.assertTrue(should_stop(40, 0, config))


if __name__ == '__main__':
    unittest.main()
