"""Pure policies and mocked loader/evaluation integration; no torch or NN import."""
import ast
import contextlib
import copy
from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from metrics import MetricAccumulator, heatmap_to_points
from sampling import training_sampling_plan, batch_lighting_groups


def isolate(**bindings):
    wanted = {'TrainConfig', '_make_loaders', '_slice_batch', 'evaluate',
              '_accumulate_evaluation_components', '_finalize_evaluation_losses'}
    tree = ast.parse(Path(__file__).with_name('train.py').read_text(encoding='utf8'))
    code = ast.Module(body=[node for node in tree.body if getattr(node, 'name', None) in wanted], type_ignores=[])
    fake_torch = SimpleNamespace(no_grad=lambda: lambda fn: fn, autocast=lambda *a, **k: contextlib.nullcontext(),
                                 is_tensor=lambda value: isinstance(value, np.ndarray), isfinite=np.isfinite)
    space = {'dataclass': dataclass, 'math': math, 'np': np, 'torch': fake_torch,
             'training_sampling_plan': training_sampling_plan, 'batch_lighting_groups': batch_lighting_groups,
             'MetricAccumulator': MetricAccumulator, 'heatmap_to_points': heatmap_to_points,
             '_to_device': lambda raw, device: raw, **bindings}
    exec(compile(code, 'train.py:isolated-sampling-and-evaluation', 'exec'), space)
    return space


def rows():
    return [{'id': f'{source}_{i}', 'source_id': source, 'lighting': lighting, 'split': 'train'}
            for source, lighting, count in [('d_a', 'day', 8), ('d_b', 'day', 2),
                                           ('n_a', 'night', 4), ('n_b', 'night', 1)] for i in range(count)]


class SamplingPolicyTests(unittest.TestCase):
    def test_augmentation_count_does_not_dominate_source_or_lighting(self):
        data = rows()
        original = copy.deepcopy(data)
        weights, plan = training_sampling_plan(data, .7)
        self.assertAlmostEqual(sum(weights), 1.)
        self.assertEqual(plan['epoch_samples'], len(data))
        self.assertTrue(plan['replacement'])
        for light, mass in [('day', .7), ('night', .3)]:
            self.assertAlmostEqual(sum(w for row, w in zip(data, weights) if row['lighting'] == light), mass)
        for source, mass in [('d_a', .35), ('d_b', .35), ('n_a', .15), ('n_b', .15)]:
            self.assertAlmostEqual(sum(w for row, w in zip(data, weights) if row['source_id'] == source), mass)
        self.assertEqual(data, original)

    def test_default_none_preserves_legacy_shuffle(self):
        weights, plan = training_sampling_plan([{'id': 'legacy'}])
        self.assertIsNone(weights)
        self.assertFalse(plan['replacement'])
        self.assertEqual(plan['row_counts'], {'unknown': 1})
        self.assertIsNone(isolate()['TrainConfig']().checked().daylight_fraction)

    def test_fraction_validation_and_required_groups(self):
        cls = isolate()['TrainConfig']
        for invalid in [False, True, 0, 1, -.1, 1.1, math.nan, math.inf, '0.7']:
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError): training_sampling_plan(rows(), invalid)
                with self.assertRaises(ValueError): cls(daylight_fraction=invalid).checked()
        for invalid_rows in [[], rows()[:2], rows() + [{'source_id': 'x'}],
                             [{'source_id': 'same', 'lighting': 'day'}, {'source_id': 'same', 'lighting': 'night'}],
                             [{'lighting': 'day'}, {'source_id': 'n', 'lighting': 'night'}]]:
            with self.assertRaises(ValueError): training_sampling_plan(invalid_rows, .7)
        self.assertEqual(cls(daylight_fraction=.65).checked().daylight_fraction, .65)

    def test_group_indices_never_infer_missing_lighting(self):
        self.assertEqual(batch_lighting_groups(['day', 'night', None, 'daylight'], 4),
                         {'day': [0], 'night': [1], 'unknown': [2, 3]})
        self.assertEqual(batch_lighting_groups(None, 2), {'unknown': [0, 1]})
        with self.assertRaises(ValueError): batch_lighting_groups(['day'], 2)


class FakeDataset:
    def __init__(self, validated, task, split):
        self.rows = [row for row in validated[task] if row['split'] == split]
    def __len__(self): return len(self.rows)


class FakeSampler:
    def __init__(self, weights, num_samples, replacement, generator):
        self.weights, self.num_samples, self.replacement, self.generator = weights, num_samples, replacement, generator
    def __iter__(self):
        return iter(self.generator.choice(len(self.weights), size=self.num_samples,
                                         replace=self.replacement, p=self.weights).tolist())


class FakeLoader:
    def __init__(self, dataset, **kwargs): self.dataset, self.kwargs = dataset, kwargs
    def __iter__(self):
        generator, sampler = self.kwargs['generator'], self.kwargs['sampler']
        generator.integers(0, 2**31)  # Simulate DataLoader's generator-backed worker base seed.
        indices = list(sampler) if sampler is not None else list(range(len(self.dataset)))
        if self.kwargs['shuffle']: generator.shuffle(indices)
        return iter(self.dataset.rows[index]['id'] for index in indices)


class LoaderIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.space = isolate(TaskDataset=FakeDataset, WeightedRandomSampler=FakeSampler, DataLoader=FakeLoader)
        self.config = self.space['TrainConfig'](daylight_fraction=.7)
        self.data = {'rpm': rows() + [dict(row, split=split, id=split + row['id'])
                                    for split in ('val', 'test') for row in rows()[:2]]}
    @staticmethod
    def generators(): return {split: np.random.default_rng(812 + i) for i, split in enumerate(('train', 'val', 'test'))}

    def test_only_training_is_weighted_and_shared_generator_is_used(self):
        generators = self.generators()
        loaders, plan = self.space['_make_loaders'](self.data, 'rpm', self.config, generators)
        sampler = loaders['train'].kwargs['sampler']
        self.assertIs(sampler.generator, generators['train'])
        self.assertEqual(sampler.num_samples, len(rows()))
        self.assertFalse(loaders['train'].kwargs['shuffle'])
        for split in ('val', 'test'):
            self.assertIsNone(loaders[split].kwargs['sampler'])
            self.assertFalse(loaders[split].kwargs['shuffle'])
            self.assertEqual(list(loaders[split]), [row['id'] for row in self.data['rpm'] if row['split'] == split])
        self.assertEqual(len(list(loaders['train'])), len(rows()))

    def test_saved_generator_restores_next_epoch_draws(self):
        generators = self.generators()
        loaders, _ = self.space['_make_loaders'](self.data, 'rpm', self.config, generators)
        list(loaders['train'])
        checkpoint_states = {split: copy.deepcopy(gen.bit_generator.state) for split, gen in generators.items()}
        expected = list(loaders['train'])
        restored = self.generators()
        new_loaders, _ = self.space['_make_loaders'](self.data, 'rpm', self.config, restored)
        for split, state in checkpoint_states.items(): restored[split].bit_generator.state = state
        self.assertEqual(list(new_loaders['train']), expected)
        # Existing checkpoint code must still save and restore this exact mapping.
        source = Path(__file__).with_name('train.py').read_text(encoding='utf8')
        self.assertIn("generators[split].set_state(state)", source)
        self.assertIn("{key: gen.get_state() for key, gen in generators.items()}", source)

    def test_none_keeps_original_no_replacement_shuffle(self):
        config = self.space['TrainConfig']()
        loaders, _ = self.space['_make_loaders'](self.data, 'rpm', config, self.generators())
        self.assertIsNone(loaders['train'].kwargs['sampler'])
        self.assertTrue(loaders['train'].kwargs['shuffle'])
        self.assertEqual(set(list(loaders['train'])), {row['id'] for row in rows()})


class ArrayTensor(np.ndarray):
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return np.asarray(self)
    def numel(self): return self.size
    def sigmoid(self): return tensor(1. / (1. + np.exp(-self)))


def tensor(values): return np.asarray(values, dtype=float).view(ArrayTensor)


def batch(values, lighting):
    n = len(values)
    return {'id': [str(v) for v in values], 'source_id': [str(v) for v in values],
            'lighting': lighting, 'image': tensor(values), 'points': tensor(np.full((n, 2, 2), 1.5)),
            'visible': tensor([[1, 0]] * n), 'visibility_known': tensor([[1, 1]] * n)}


class MockModel:
    task, input_hw = 'speed', (4, 4)
    def __init__(self): self.calls = 0
    def eval(self): pass
    def __call__(self, images):
        self.calls += 1
        return {'heatmap_logits': tensor(np.zeros((len(images), 2, 1, 1))),
                'visibility_logits': tensor([[1, -1]] * len(images))}


class EvaluationIntegrationTests(unittest.TestCase):
    def test_lighting_metrics_losses_from_one_forward_per_batch(self):
        calls = []
        def fake_loss(output, current, hw, sigma):
            calls.append(len(current['id']))
            mean = float(current['image'].mean())
            visibility = mean * 2
            return mean + .25 * visibility, {'total': mean + .25 * visibility, 'heatmap': mean, 'visibility': visibility}
        space = isolate(training_loss=fake_loss)
        model = MockModel()
        # Mixed batch, followed by a pure night batch. Unknown remains in overall metrics only.
        loader = [batch([1, 3, 5], ['day', 'night', 'unknown']), batch([7], ['night'])]
        result = space['evaluate'](model, loader, 'unused', space['TrainConfig'](amp=False), ['pivot', 'tip'])
        self.assertEqual(model.calls, 2)
        self.assertEqual(calls, [3, 1, 1, 1])
        self.assertEqual(result['samples'], 4)
        self.assertEqual(result['by_lighting']['day']['samples'], 1)
        self.assertEqual(result['by_lighting']['night']['samples'], 2)
        self.assertEqual(result['lighting_unknown_samples_excluded_from_groups'], 1)
        self.assertAlmostEqual(result['loss']['total'], 6.)
        self.assertAlmostEqual(result['selection_score'], 6.)
        self.assertAlmostEqual(result['by_lighting']['day']['loss']['total'], 1.5)
        self.assertAlmostEqual(result['by_lighting']['night']['loss']['total'], 7.5)
        availability = result['by_lighting']['night']['predicted_landmark_availability_at_0_5']
        self.assertEqual(availability['pivot'], {'count': 2, 'fraction': 1.})
        self.assertEqual(availability['tip'], {'count': 0, 'fraction': 0.})

    def test_legacy_unknown_and_empty_group_have_honest_counts(self):
        def fake_loss(output, current, hw, sigma):
            return 1.25, {'total': 1.25, 'visibility': 1., 'heatmap': 1.}
        space = isolate(training_loss=fake_loss)
        raw = batch([1], None)
        del raw['lighting']
        result = space['evaluate'](MockModel(), [raw], 'unused', space['TrainConfig'](amp=False), ['pivot', 'tip'])
        self.assertEqual(result['samples'], 1)
        self.assertEqual(result['lighting_unknown_samples_excluded_from_groups'], 1)
        for light in ('day', 'night'):
            self.assertEqual(result['by_lighting'][light]['samples'], 0)
            self.assertIsNone(result['by_lighting'][light]['loss'])
            self.assertIsNone(result['by_lighting'][light]['predicted_landmark_availability_at_0_5']['pivot']['fraction'])


if __name__ == '__main__': unittest.main()
