"""Bounded CPU software checks; these tests do not measure model accuracy."""
from dataclasses import replace
from pathlib import Path
import copy
import hashlib
import json
import random
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from real_adaptation_v3 import notebook_helpers as setup
from real_adaptation_v3 import train_needle as trainer
from real_adaptation_v3 import needle_data


class TinyNeedle(nn.Module):
    """Small stochastic network, retaining the trainer's actual loss contract."""
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            raise AssertionError("Tests must never download weights")
        self.features = nn.Sequential(*(nn.Linear(4, 4) for _ in range(6)))
        self.dropout = nn.Dropout(.25)
        self.heatmap_head = nn.Linear(4, 2 * 8 * 8)
        self.speed_head = nn.Linear(4, 1)

    def forward(self, image):
        latent = self.dropout(torch.tanh(self.features(image)))
        logits = self.heatmap_head(latent).reshape(-1, 2, 8, 8)
        points = trainer.legacy.heatmap_coordinates(logits)
        return dict(heatmap_logits=logits, points=points,
                    speed=self.speed_head(latent).squeeze(1).sigmoid() * 180)


class TinyDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        # All three RNGs plus loader shuffling and model dropout must resume.
        perturbation = random.random() + float(np.random.random()) + float(torch.rand(()))
        return dict(image=torch.tensor([.1, .3, .7, .9]) + index * .03 + perturbation * .01,
                    points=torch.tensor(row['points_xy']) / 511,
                    speed=torch.tensor(row['speed_kmh'] if row['speed_valid'] else 0.),
                    speed_valid=torch.tensor(row['speed_valid']), index=index)


def fixture_calibration():
    speeds = np.arange(0, 181, 20, dtype=float)
    angles = np.deg2rad(90 + speeds * 1.5)
    pivot = np.array([211., 225.])
    ticks = pivot + 110 * np.c_[np.cos(angles), np.sin(angles)]
    return dict(pivot_xy=pivot.tolist(), speeds=speeds.tolist(), angles=angles.tolist(), ticks_xy=ticks.tolist())


def tiny_records(split, count):
    calibration = fixture_calibration()
    return [dict(sample_id=f'{split}:{i}', points_xy=[[211., 225.], [108., 268.5]],
                 speed_kmh=60., speed_valid=True, split=split, kind='synthetic', source_frame=i,
                 registration=dict(accepted=True, image_to_reference_h=np.eye(3).tolist()),
                 calibration=calibration, label_status='software_fixture') for i in range(count)]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix='nv3_')
        self.root = Path(self.temporary.name)
        self.partitions = dict(train=tiny_records('train', 4), validation=tiny_records('validation', 2), real_diagnostic=[])
        self.integrity = dict(dataset_sha='software_fixture', counts=dict(train=4, validation=2, real_diagnostic=0))
        self.patches = [patch.object(trainer, 'SpeedometerModel', TinyNeedle),
                        patch.object(trainer, 'NeedleDataset', TinyDataset),
                        patch.object(trainer, 'verify_dataset', return_value=(self.partitions, self.integrity)),
                        patch.object(trainer, 'plot_history', return_value=None)]
        for item in self.patches:
            item.start()
        self.config = trainer.NeedleTrainConfig(data_dir='fixture', output_dir=str(self.root / 'runs'),
                                                epochs=3, batch_size=2, device='cpu', amp=False,
                                                decoder_lr=.003, encoder_lr=.001)

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def state(self, report):
        return trainer._read_complete(report['latest_checkpoint_path'])[0]

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_exact_resume_restores_every_training_state_and_fixed_target(self):
        uninterrupted = trainer.train(replace(self.config, output_dir=str(self.root / 'all')), mode='fresh')
        first = trainer.train(self.config, mode='fresh', stop_after_epochs=1)
        before_hash = setup.sha256(first['latest_checkpoint_path'])
        random.seed(987); np.random.seed(655); torch.manual_seed(42)
        resumed = trainer.train(self.config, mode='resume_latest', search_root=self.root / 'runs')
        a, b = self.state(uninterrupted), self.state(resumed)
        for key in ('model_state_dict', 'optimizer_state_dict', 'scaler_state_dict', 'rng_state',
                    'loader_generator_states', 'history', 'training_state'):
            with self.subTest(state=key):
                self.assert_nested_equal(a[key], b[key])
        self.assertEqual(setup.sha256(first['latest_checkpoint_path']), before_hash)
        self.assertNotEqual(first['run_dir'], resumed['run_dir'])
        self.assertEqual(resumed['target_epoch'], 3)
        self.assertEqual(resumed['epochs_completed_this_call'], 2)
        completed = trainer.train(self.config, mode='resume_latest', search_root=self.root / 'runs')
        self.assertEqual(completed['epochs_completed_this_call'], 0)
        self.assertEqual(completed['target_epoch'], 3)
        with self.assertRaisesRegex(ValueError, 'compatible|Compatible|verified'):
            trainer.train(replace(self.config, epochs=4), mode='resume_latest', search_root=self.root / 'runs')

    def test_transfer_is_exact_weights_new_optimizer_epoch_zero_then_fresh_ignores_it(self):
        torch.manual_seed(898)
        source_model = TinyNeedle()
        source = self.root / 'source34.pt'
        payload = dict(format_version=1, checkpoint_kind='augmented_ring_inference_weights_v2',
                       architecture=trainer.SOURCE_ARCHITECTURE_ID, preprocessing=trainer.PREPROCESSING_ID,
                       config=dict(input_size=512), best_epoch=34,
                       model_state_dict={'needle_branch.' + k: v for k, v in source_model.state_dict().items()})
        torch.save(payload, source)
        with patch.object(trainer, 'APPROVED_SOURCE_SHA256', setup.sha256(source)):
            result = trainer.train(self.config, mode='transfer', source_checkpoint=source, stop_after_epochs=0)
        saved = self.state(result)
        self.assert_nested_equal(saved['model_state_dict'], source_model.state_dict())
        self.assertEqual(saved['optimizer_state_dict']['state'], {})
        self.assertEqual(saved['training_state']['epoch'], 0)
        self.assertEqual(saved['training_state']['target_epoch'], 3)
        self.assertEqual(result['initialization']['source_epoch'], 34)
        self.assertTrue(result['initialization']['frozen_needle_policy_removed'])
        fresh = trainer.train(self.config, mode='fresh', source_checkpoint=self.root / 'absent.pt',
                              search_root=self.root / 'runs', stop_after_epochs=0)
        self.assertTrue(fresh['initialization']['full_encoder'])
        self.assertEqual(fresh['initialization']['mode'], 'fresh')
        self.assertFalse(torch.equal(self.state(fresh)['model_state_dict']['features.0.weight'],
                                     source_model.state_dict()['features.0.weight']))
        loaded, _ = trainer.load_model(result['best_model_path'])
        self.assert_nested_equal(loaded.state_dict(), source_model.state_dict())

    def test_resume_best_uses_selected_full_state_including_optimizer(self):
        actual_evaluate = trainer.evaluate
        scores = iter([2., 0., 3., 4., 0.])  # Baseline, epochs1â€“3, selected-best export.
        def scored_evaluate(*args, **kwargs):
            metrics, rows = actual_evaluate(*args, **kwargs)
            metrics['selection_score'] = next(scores)
            return metrics, rows
        with patch.object(trainer, 'evaluate', side_effect=scored_evaluate):
            original = trainer.train(self.config, mode='fresh')
        candidates = trainer.discover_checkpoints(self.root / 'runs')
        selected = min(candidates, key=lambda row: (row['selection_loss'], -row['created_at_ns']))
        self.assertEqual(selected['epoch'], 1)
        expected, _ = trainer._read_complete(selected['path'])
        resumed = trainer.train(self.config, mode='resume_best', search_root=self.root / 'runs', stop_after_epochs=0)
        actual = self.state(resumed)
        self.assertEqual(resumed['resume_metadata']['source_epoch'], selected['epoch'])
        self.assertEqual(resumed['resume_metadata']['source_checkpoint_path'], selected['path'])
        self.assert_nested_equal(actual['optimizer_state_dict'], expected['optimizer_state_dict'])
        self.assertEqual(resumed['target_epoch'], original['target_epoch'])

    def test_private_diagnostic_baseline_survives_resume_and_ignores_uncertain_speed(self):
        diagnostic = tiny_records('real_diagnostic', 2)
        for row in diagnostic:
            row.update(kind='real', confidence='medium')
        diagnostic[1].update(confidence='low', speed_valid=False, speed_kmh=None)
        self.partitions['real_diagnostic'] = diagnostic
        first = trainer.train(self.config, mode='fresh', stop_after_epochs=1)
        self.assertIsNone(first['after_real_metrics'])
        baseline = first['before_real_metrics']
        self.assertEqual(baseline['keypoint_scored_count'], 1)
        self.assertEqual(baseline['primary_speed']['speed_target_eligible_count'], 1)
        self.assertFalse(baseline['used_for_model_selection'])
        resumed = trainer.train(self.config, mode='resume_latest', search_root=self.root / 'runs')
        self.assert_nested_equal(resumed['before_real_metrics'], baseline)
        self.assertEqual(resumed['after_real_metrics']['count'], 2)
        a = json.loads((Path(first['run_dir']) / 'before_real_diagnostic_predictions.json').read_text())
        b = json.loads((Path(resumed['run_dir']) / 'before_real_diagnostic_predictions.json').read_text())
        self.assertEqual(a, b)

    def test_nonfinite_or_incomplete_payload_is_not_a_complete_checkpoint(self):
        result = trainer.train(self.config, mode='fresh', stop_after_epochs=0)
        payload = self.state(result)
        broken = copy.deepcopy(payload)
        broken['model_state_dict']['features.0.weight'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            trainer._validate_payload(broken)
        broken = copy.deepcopy(payload)
        broken['loader_generator_states']['train'] = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with self.assertRaises((ValueError, RuntimeError)):
            trainer._validate_payload(broken)
        broken = copy.deepcopy(payload)
        del broken['before_real_metrics']
        with self.assertRaises(ValueError):
            trainer._validate_payload(broken)

    def test_corrupt_newest_recovers_older_but_never_silently_fresh(self):
        result = trainer.train(self.config, mode='fresh', stop_after_epochs=1)
        latest = Path(result['latest_checkpoint_path'])
        latest.write_bytes(b'interrupted copy')
        with self.assertWarns(RuntimeWarning):
            resumed = trainer.train(self.config, mode='resume_latest', search_root=self.root / 'runs', stop_after_epochs=0)
        self.assertEqual(resumed['resume_metadata']['source_epoch'], 0)
        self.assertTrue(resumed['resume_metadata']['recovery_diagnostics'])
        for path in (self.root / 'runs').rglob('epoch_*.pt'):
            path.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            trainer.train(self.config, mode='resume_latest', search_root=self.root / 'runs')
        with self.assertRaises((FileNotFoundError, ValueError)):
            trainer.train(self.config, mode='resume_latest', search_root=self.root / 'missing')

    def test_invalid_speed_label_has_zero_speed_gradient(self):
        speed = torch.tensor([40., 80.], requires_grad=True)
        output = dict(speed=speed, points=torch.full((2, 2, 2), .5, requires_grad=True),
                      heatmap_logits=torch.zeros(2, 2, 8, 8, requires_grad=True))
        batch = dict(speed=torch.tensor([60., 0.]), speed_valid=torch.tensor([True, False]),
                     points=torch.full((2, 2, 2), .4))
        loss, _ = trainer.training_loss(output, batch, self.config)
        loss.backward()
        self.assertNotEqual(float(speed.grad[0]), 0)
        self.assertEqual(float(speed.grad[1]), 0)


class NeedleGeometryTests(unittest.TestCase):
    def test_endpoint_tolerance_is_bounded_and_clamping_is_explicit(self):
        calibration = fixture_calibration()
        pivot = np.asarray(calibration['pivot_xy'])
        registration = dict(accepted=True, image_to_reference_h=np.eye(3).tolist())
        # These analytic tick angles span1.5 degrees per km/h. Extrapolated
        # speeds are known without calling the reader's interpolation code.
        for raw_speed, expected in [(-1.5, 0.), (181.5, 180.), (0., 0.), (58., 58.), (180., 180.)]:
            with self.subTest(raw_speed=raw_speed):
                angle = np.deg2rad(90 + raw_speed * 1.5)
                tip = pivot + 105 * np.array([np.cos(angle), np.sin(angle)])
                reading = needle_data.read_speed([pivot, tip], registration, calibration)
                self.assertTrue(reading['accepted'], reading)
                self.assertAlmostEqual(reading['speed_kmh'], expected, places=8)
                self.assertAlmostEqual(reading['raw_indicated_speed_kmh'], raw_speed, places=8)
                self.assertEqual(reading['endpoint_clamped'], raw_speed < 0 or raw_speed > 180)
                self.assertEqual(reading['endpoint_tolerance_kmh'], 2.)
        for raw_speed in (-2.1, 182.1):
            with self.subTest(rejected_raw_speed=raw_speed):
                angle = np.deg2rad(90 + raw_speed * 1.5)
                tip = pivot + 105 * np.array([np.cos(angle), np.sin(angle)])
                reading = needle_data.read_speed([pivot, tip], registration, calibration)
                self.assertFalse(reading['accepted'])
                self.assertIsNone(reading['speed_kmh'])
                self.assertAlmostEqual(reading['raw_indicated_speed_kmh'], raw_speed, places=8)

    def test_shifted_pivot_and_projective_image_mapping_use_observed_pivot(self):
        calibration = fixture_calibration()
        pivot = np.array(calibration['pivot_xy']) + [5.1, 2.04]
        ticks = np.asarray(calibration['ticks_xy'])
        angles = np.unwrap(np.arctan2(ticks[:, 1] - pivot[1], ticks[:, 0] - pivot[0]))
        angle = np.interp(58., calibration['speeds'], angles)
        tip = pivot + 105 * np.array([np.cos(angle), np.sin(angle)])
        h = np.array([[.96, -.08, 20.], [.07, .98, -6.], [.00012, -.00008, 1.]])
        forward = np.c_[np.array([pivot, tip]), np.ones(2)] @ h.T
        image_points = forward[:, :2] / forward[:, 2:]
        reading = needle_data.read_speed(image_points, dict(accepted=True, image_to_reference_h=np.linalg.inv(h)), calibration)
        self.assertTrue(reading['accepted'], reading)
        self.assertAlmostEqual(reading['speed_kmh'], 58., places=8)
        no_estimate = needle_data.registration_from_record(dict(reference_to_image_h=h.tolist(),
                                                                image_registration=dict(accepted=False)))
        self.assertFalse(needle_data.read_speed(image_points, no_estimate, calibration)['accepted'])


class DatasetContractTests(unittest.TestCase):
    def test_lcd_and_coordinate_corruption_and_source_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='nv3_data_') as temporary:
            root = Path(temporary)
            calibration = fixture_calibration()
            raw_calibration = dict(pivot_xy=calibration['pivot_xy'],
                                   ticks=[dict(speed_kmh=s, xy=p) for s, p in zip(calibration['speeds'], calibration['ticks_xy'])])
            (root / 'calibration.json').write_text(json.dumps(raw_calibration))
            image = np.zeros((512, 512), dtype=np.uint8)
            image[100:200, 300:400] = 128
            lcd = np.zeros_like(image); lcd[100:200, 300:400] = 255
            needle = np.zeros_like(image); needle[225, 101:212] = 255
            Image.fromarray(image).save(root / 'sample.png')
            Image.fromarray(image).save(root / 'masked_reference.png')
            Image.fromarray(lcd).save(root / 'lcd.png')
            Image.fromarray(needle).save(root / 'needle.png')
            points = dict(pivot=[211., 225.], tip=[101., 225.])
            samples = [dict(id=f'fixture{i}', split='train' if i < 400 else 'validation',
                            source_frame=0 if i < 200 else 180 if i < 400 else 204,
                            image='sample.png', image_sha256=setup.sha256(root / 'sample.png'),
                            lcd_mask='lcd.png', needle_mask='needle.png', speed_kmh=60.,
                            needle_position_px=points,
                            needle_position_normalized={k:(np.asarray(v)/511).tolist() for k,v in points.items()})
                       for i in range(500)]
            def write(rows):
                (root / 'labels.json').write_text(json.dumps(dict(samples=rows)))
            write(samples)
            partitions, integrity = needle_data.verify_dataset(root)
            self.assertEqual(integrity['counts'], dict(train=400, validation=100, real_diagnostic=0))
            self.assertTrue(set(r['source_frame'] for r in partitions['train']).isdisjoint(
                r['source_frame'] for r in partitions['validation']))
            broken = copy.deepcopy(samples)
            broken[0]['needle_position_normalized']['tip'][0] += .02
            write(broken)
            with self.assertRaisesRegex(ValueError, 'normalized'):
                needle_data.verify_dataset(root)
            write(samples)
            needle[0, 0] = 127
            Image.fromarray(needle).save(root / 'needle.png')
            with self.assertRaisesRegex(ValueError, 'Binary'):
                needle_data.verify_dataset(root)
            needle[0, 0] = 0
            Image.fromarray(needle).save(root / 'needle.png')
            image[100, 300] = 129
            Image.fromarray(image).save(root / 'sample.png')
            for row in samples:
                row['image_sha256'] = setup.sha256(root / 'sample.png')
            write(samples)
            with self.assertRaisesRegex(ValueError, 'LCD'):
                needle_data.verify_dataset(root)
            image[100, 300] = 128
            Image.fromarray(image).save(root / 'sample.png')
            for row in samples:
                row['image_sha256'] = setup.sha256(root / 'sample.png')
            samples[0]['split'], samples[-1]['split'] = samples[-1]['split'], samples[0]['split']
            write(samples)
            with self.assertRaisesRegex(ValueError, 'source-plate split'):
                needle_data.verify_dataset(root)


class NotebookSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nv3_")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def archive(self, name, members):
        path = self.root / name
        with zipfile.ZipFile(path, "w") as bundle:
            for member, content in members:
                if isinstance(member, str) and "\\" in member:
                    # ZipInfo normalizes the host separator on Windows. Preserve
                    # the deliberately unsafe on-disk name for this fixture.
                    unsafe = zipfile.ZipInfo(member.replace("\\", "/"))
                    unsafe.filename = member
                    member = unsafe
                bundle.writestr(member, content)
        return path, setup.sha256(path)

    def test_zip_rejects_escape_symlink_duplicate_and_changed_cache(self):
        good, digest = self.archive("good.zip", [("dataset/labels.json", "{}"), ("dataset/image.png", "pixels")])
        destination = self.root / "verified"
        self.assertEqual(setup.safe_extract(good, destination, digest, "dataset"), destination / "dataset")
        self.assertEqual(setup.safe_extract(good, destination, digest, "dataset"), destination / "dataset")
        (destination / "dataset/image.png").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "changed"):
            setup.safe_extract(good, destination, digest, "dataset")
        for n, member in enumerate(("../escape", "dataset/../../escape", "C:/escape", "dataset\\escape", "dataset/CON.txt")):
            with self.subTest(member=member):
                archive, checksum = self.archive(f"unsafe{n}.zip", [(member, "x")])
                with self.assertRaises(ValueError):
                    setup.safe_extract(archive, self.root / f"out{n}", checksum, "dataset")
        link = zipfile.ZipInfo("dataset/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive, checksum = self.archive("link.zip", [(link, "../../escape")])
        with self.assertRaises(ValueError):
            setup.safe_extract(archive, self.root / "linkout", checksum, "dataset")
        archive, checksum = self.archive("duplicate.zip", [("dataset/Image", "x"), ("dataset/image", "y")])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            setup.safe_extract(archive, self.root / "dupout", checksum, "dataset")

    def test_saved_target_and_fresh_isolation_and_no_resume_fallback(self):
        pins = dict(code_commit="a" * 40, public_archive_sha256="b" * 64,
                    private_archive_sha256="c" * 64, runtime={"torch": "fixture"})
        config = dict(data_dir="old_data", real_data_dir="old_real", epochs=25, batch_size=8, device="cuda")
        experiment, saved, _ = setup.prepare_experiment(self.root, "pilot", "transfer", config, **pins)
        sentinel = experiment / "old_result.txt"
        sentinel.write_text("keep")
        resumed_dir, resumed, _ = setup.prepare_experiment(self.root, "pilot", "resume_best", dict(config, epochs=99, data_dir="new_data"), **pins)
        self.assertEqual(resumed_dir, experiment)
        self.assertEqual(resumed["epochs"], 25)
        self.assertEqual(resumed["data_dir"], "new_data")
        fresh_dir, fresh, _ = setup.prepare_experiment(self.root, "pilot", "fresh", config, **pins)
        self.assertNotEqual(fresh_dir, experiment)
        self.assertEqual(sentinel.read_text(), "keep")
        with self.assertRaises(FileExistsError):
            setup.prepare_experiment(self.root, "pilot", "transfer", config, **pins)
        with self.assertRaises(FileNotFoundError):
            setup.prepare_experiment(self.root, "missing", "resume_latest", config, **pins)
        with self.assertRaises(ValueError):
            setup.prepare_experiment(self.root, "pilot", "resume_latest", config, **dict(pins, code_commit="d" * 40))
        with self.assertRaises(ValueError):
            setup.prepare_experiment(self.root, "bad", "transfer", dict(config, epochs=0), **pins)
        self.assertFalse((self.root / "experiments/bad").exists())

    def test_source_discovery_is_narrow_and_hash_verified(self):
        data = b"approved checkpoint fixture"
        digest = hashlib.sha256(data).hexdigest()
        outside = self.root / "experiments/unrelated/runs/run/best_model.pt"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(data)
        with self.assertRaises(FileNotFoundError):
            setup.find_source_checkpoint(self.root, digest)
        approved = self.root / "experiments/speedometer_arc_v2_augmented/runs/run/best_model.pt"
        approved.parent.mkdir(parents=True)
        approved.write_bytes(b"wrong file")
        with self.assertRaises(FileNotFoundError):
            setup.find_source_checkpoint(self.root, digest)
        approved.write_bytes(data)
        self.assertEqual(setup.find_source_checkpoint(self.root, digest), approved)
        self.assertEqual(setup.find_source_checkpoint(self.root, digest, str(outside)), outside)
        with self.assertRaises(ValueError):
            setup.find_source_checkpoint(self.root, "f" * 64, str(outside))


if __name__ == "__main__":
    unittest.main(verbosity=2)
