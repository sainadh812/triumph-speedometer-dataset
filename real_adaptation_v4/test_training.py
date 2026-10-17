"""CPU checks for transfer preservation, training gradients and durable resume."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from real_adaptation_v3 import train_needle as v3
from real_adaptation_v4 import train_landmarks as v4

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'real_adaptation_v3/trained_results/best_model.pt'


class TinyLandmarkModel(nn.Module):
    """Small stochastic network exercises the real trainer/checkpoint machinery."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
        self.tick_head = nn.Conv2d(4, 12, 1)
        self.visibility_head = nn.Linear(4, 12)
        self.speed_head = nn.Linear(4, 1)
        self.dropout = nn.Dropout(.1)
    def forward(self, image):
        value = self.features(torch.nn.functional.adaptive_avg_pool2d(image, (8, 8)))
        value = self.dropout(value)
        logits = self.tick_head(value)
        visibility_logits = self.visibility_head(value.mean((-2, -1)))
        return {'heatmap_logits': logits, 'points': v4.legacy.heatmap_coordinates(logits),
                'visibility_logits': visibility_logits, 'visibility': visibility_logits.sigmoid()}


def make_dataset(root):
    root.mkdir(parents=True)
    pivot = [211., 225.]
    ticks = [[211.,332.],[171.5,324.5],[135.5,303.],[108.,268.5],[98.5,227.5],
             [105.5,184.5],[125.5,146.],[160.,120.5],[203.5,109.],[247.,114.5]]
    calibration = {'pivot_xy': pivot, 'ticks': [{'speed_kmh': speed, 'xy': xy} for speed, xy in zip(range(0,181,20), ticks)]}
    (root / 'calibration.json').write_text(json.dumps(calibration))
    points = [calibration['pivot_xy'], calibration['ticks'][4]['xy']] + [row['xy'] for row in calibration['ticks']]
    rows = []
    rng = np.random.default_rng(49)
    mask = np.zeros((512, 512), np.uint8)
    mask[100:251, 300:501] = 255
    Image.fromarray(mask).save(root / 'lcd_mask.png')
    for index, split in enumerate(('train', 'train', 'validation')):
        relative = f'image_{index}.png'
        pixels = rng.integers(0, 255, (512, 512), dtype=np.uint8)
        pixels[mask > 0] = 128
        Image.fromarray(pixels).save(root / relative)
        rows.append({'sample_id': f'sample_{index}', 'image_path': relative, 'split': split,
                     'lcd_mask_path': 'lcd_mask.png',
                     'points_xy': points, 'visibility': [True] * 12, 'speed_kmh': 80., 'speed_valid': True,
                     'source_group': split + '_group', 'source_frame': index})
    (root / 'annotations.json').write_text(json.dumps({'schema_version': 1, 'keypoint_names': v4.KEYPOINT_NAMES, 'samples': rows}))
    Image.open(root / 'image_0.png').save(root / 'reference.png')
    preprocessing = {'color': 'grayscale', 'input_size': [512, 512],
                    'camera_stabilization': 'image_only_day_reference_SIFT_static_scale_RANSAC',
                    'lcd_mask_applied': True, 'lcd_mask_value': 128,
                    'lcd_polygon_reference_xy': [[300, 100], [500, 100], [500, 250], [300, 250]],
                    'mask_before_geometric_augmentation': True,
                    'fixed_mask_path': 'lcd_mask.png', 'fixed_mask_sha256': v4._sha(root / 'lcd_mask.png'),
                    'reference_image_path': 'reference.png', 'reference_sha256': v4._sha(root / 'reference.png'),
                    'canonical_calibration_sha256': v4._sha(root / 'calibration.json'),
                    'stabilize_mask_script_sha256': v4._sha(ROOT / 'real_adaptation_v4/stabilize_mask.py')}
    (root / 'preprocessing_config.json').write_text(json.dumps(preprocessing))
    manifest = {'sha256': {path.name: v4._sha(path) for path in root.iterdir()}, 'preprocessing': preprocessing,
                'landmark_supervision': v4.AVAILABILITY_SEMANTICS}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @unittest.skipUnless(SOURCE.is_file(), 'Private verified V3 checkpoint is not available')
    def test_transfer_preserves_needle_predictions_exactly(self):
        torch.manual_seed(5)
        model = v4.LandmarkModel()
        v4.load_transfer_checkpoint(model, SOURCE)
        source, _ = v3.load_model(SOURCE)
        model.eval()
        image = torch.rand(1, 3, 512, 512)
        with torch.no_grad():
            expected, actual = source(image), model(image)
        torch.testing.assert_close(actual['points'][:, :2], expected['points'], rtol=0, atol=0)
        torch.testing.assert_close(actual['heatmap_logits'][:, :2], expected['heatmap_logits'], rtol=0, atol=0)
        torch.testing.assert_close(actual['speed'], expected['speed'], rtol=0, atol=0)

    def test_warmup_and_joint_gradients(self):
        model, config = v4.LandmarkModel(), v4.TrainConfig('unused', 'unused')
        batch = {'image': torch.rand(1, 3, 512, 512), 'points': torch.rand(1, 12, 2), 'visibility': torch.ones(1, 12)}
        v4.configure_stage(model, 1, config)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not name.startswith(('tick_', 'visibility_head.'))}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loss, _ = v4.training_loss(model(batch['image']), batch, config)
        loss.backward()
        self.assertGreater(float(model.tick_head.weight.grad.abs().sum()), 0)
        self.assertIsNone(model.heatmap_head.weight.grad)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in before:
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        optimizer.zero_grad(set_to_none=True)
        v4.configure_stage(model, 3, config)
        loss, _ = v4.training_loss(model(batch['image']), batch, config)
        loss.backward()
        self.assertGreater(float(model.heatmap_head.weight.grad.abs().sum()), 0)
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in model.features[-1].parameters() if parameter.grad is not None), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.features[0].parameters()))

    def test_resume_matches_uninterrupted_and_rejects_corruption(self):
        # The small model isolates full-state save/resume determinism from costly
        # MobileNet compute. The preceding test covers the real model gradients.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_dataset(root / 'dataset')
            config = v4.TrainConfig(str(dataset), str(root / 'full'), epochs=2, warmup_epochs=0, batch_size=2, amp=False, device='cpu')
            with patch.object(v4, 'LandmarkModel', TinyLandmarkModel):
                full = v4.train(config, mode='fresh')
                interrupted_config = deepcopy(config)
                interrupted_config.output_dir = str(root / 'split')
                first = v4.train(interrupted_config, mode='fresh', stop_after_epochs=1)
                resumed = v4.train(interrupted_config, mode='resume_latest', search_root=root / 'split')
                full_state, _ = v3._read_complete(full['latest_checkpoint_path'])
                resumed_state, _ = v3._read_complete(resumed['latest_checkpoint_path'])
                self.assertEqual(full_state['history'], resumed_state['history'])
                for key, tensor in full_state['model_state_dict'].items():
                    torch.testing.assert_close(tensor, resumed_state['model_state_dict'][key], rtol=0, atol=0)
                self.assertTrue(resumed['resume_metadata']['full_state_restored'])
                self.assertEqual(resumed['resume_metadata']['source_epoch'], 1)
                latest = Path(resumed['latest_checkpoint_path'])
                latest.write_bytes(latest.read_bytes()[:-100])
                with self.assertWarns(RuntimeWarning):
                    candidates, rejected = v4.discover_checkpoints(root / 'split', resumed_state['identity'])
                self.assertTrue(rejected)
                self.assertEqual(max(row['epoch'] for row in candidates), 1)
                with self.assertRaises(ValueError):
                    v4.discover_checkpoints(root / 'split', {'wrong': 'identity'})
                # Resume best must use that epoch's optimizer state, not mix
                # best weights with the last epoch optimizer state.
                selected = min(candidates, key=lambda row: (row['selection_loss'], -row['created_at_ns']))
                best = v4.train(interrupted_config, mode='resume_best', search_root=root / 'split', stop_after_epochs=0)
                self.assertEqual(best['resume_metadata']['source_checkpoint_path'], selected['path'])
                self.assertTrue(best['resume_metadata']['optimizer_restored'])

    def test_manifest_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_dataset(Path(directory) / 'dataset')
            v4.verify_dataset(root)
            image = root / 'image_0.png'
            image.write_bytes(image.read_bytes() + b'tampered')
            with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                v4.verify_dataset(root)

    def test_incorrect_mask_pixels_rejected_even_with_valid_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_dataset(Path(directory) / 'dataset')
            image = root / 'image_0.png'
            pixels = np.asarray(Image.open(image)).copy()
            pixels[150, 350] = 255
            Image.fromarray(pixels).save(image)
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest['sha256'][image.name] = v4._sha(image)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'not exactly128'):
                v4.verify_dataset(root)

    def test_mask_over_needle_rejected_even_when_pixels_and_hashes_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_dataset(Path(directory) / 'dataset')
            mask_path = root / 'lcd_mask.png'
            mask = np.asarray(Image.open(mask_path)).copy()
            mask[220:230, 205:215] = 255
            Image.fromarray(mask).save(mask_path)
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest['sha256'][mask_path.name] = v4._sha(mask_path)
            for path in root.glob('image_*.png'):
                pixels = np.asarray(Image.open(path)).copy()
                pixels[mask > 0] = 128
                Image.fromarray(pixels).save(path)
                manifest['sha256'][path.name] = v4._sha(path)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                v4.verify_dataset(root)


if __name__ == '__main__':
    unittest.main()
