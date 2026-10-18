"""CPU data/metric and AST-isolated tensor tests; no neural model imports."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from dataset import TaskDataset, validate_dataset, validate_points, validate_visibility_known
from metrics import MetricAccumulator
from test_pure_contract import fixture, refresh_manifest


def schema2_fixture(root):
    manifest = fixture(root)
    manifest['schema_version'] = 2
    for task in ('localizer', 'speed', 'rpm'):
        path = root / task / 'samples.json'
        document = json.loads(path.read_text())
        for row in document['samples']:
            row['visibility_known'] = [True] * len(row['visible'])
        path.write_text(json.dumps(document))
    refresh_manifest(root, manifest)
    return manifest


class Tensor(np.ndarray):
    """Minimal NumPy tensor for executing the actual isolated BCE helper."""
    def __new__(cls, value): return np.asarray(value, dtype=np.float64).view(cls)
    def float(self): return self
    def sum(self, *args, **kwargs): return Tensor(np.asarray(self).sum(*args, **kwargs))
    def clamp_min(self, minimum): return Tensor(np.maximum(np.asarray(self), minimum))


def isolated_functions():
    functions = {}
    selected_names = {'visibility_supervision_loss', '_accumulate_evaluation_components', '_finalize_evaluation_losses'}
    def bce(logits, targets, reduction):
        if reduction != 'none':
            raise AssertionError('Expected per-point visibility losses')
        x, y = np.asarray(logits), np.asarray(targets)
        return Tensor(np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x))))
    functions.update(torch=SimpleNamespace(ones_like=lambda value: Tensor(np.ones_like(value)), any=np.any),
                     F=SimpleNamespace(binary_cross_entropy_with_logits=bce))
    for filename in ('model.py', 'train.py'):
        tree = ast.parse(Path(__file__).with_name(filename).read_text())
        selected = ast.Module(body=[node for node in tree.body if getattr(node, 'name', None) in selected_names], type_ignores=[])
        exec(compile(selected, filename + ':isolated-visibility-policy', 'exec'), functions)
    return functions


class VisibilityDatasetTests(unittest.TestCase):
    def test_old_archives_default_all_known_in_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            validated = validate_dataset(root)
            row = TaskDataset(validated, 'speed', 'train')[0]
            np.testing.assert_array_equal(row['visibility_known'], np.ones(12, np.float32))

    def test_new_archive_unknown_visibility_passes_to_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = schema2_fixture(root)
            path = root / 'rpm/samples.json'
            document = json.loads(path.read_text())
            row = document['samples'][0]
            row['visibility_known'] = [True] * 7 + [False]
            row['visible'][7], row['points'][7] = False, [0, 0]
            path.write_text(json.dumps(document))
            refresh_manifest(root, manifest)
            validated = validate_dataset(root)
            sample = TaskDataset(validated, 'rpm', 'train')[0]
            self.assertEqual(sample['visibility_known'].tolist(), [1.] * 7 + [0.])
            self.assertEqual(sample['visible'][-1], 0.)
            np.testing.assert_array_equal(sample['points'][-1], [0., 0.])

    def test_schema2_requires_visibility_known_on_every_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = schema2_fixture(root)
            path = root / 'rpm/samples.json'
            document = json.loads(path.read_text())
            del document['samples'][-1]['visibility_known']
            path.write_text(json.dumps(document))
            refresh_manifest(root, manifest)
            with self.assertRaisesRegex(ValueError, 'Schema 2 requires visibility_known'):
                validate_dataset(root)

    def test_missing_schema_version_retains_legacy_all_known_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = fixture(root)
            manifest.pop('schema_version', None)
            refresh_manifest(root, manifest)
            validated = validate_dataset(root)
            np.testing.assert_array_equal(TaskDataset(validated, 'rpm', 'train')[0]['visibility_known'],
                                          np.ones(8, np.float32))

    def test_unsupported_or_malformed_schema_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = fixture(root)
            for version in (0, 3, '2', None, True):
                with self.subTest(schema_version=version):
                    manifest['schema_version'] = version
                    refresh_manifest(root, manifest)
                    with self.assertRaisesRegex(ValueError, 'Unsupported dataset schema_version'):
                        validate_dataset(root)

    def test_visible_implies_known(self):
        with self.assertRaisesRegex(ValueError, 'known visibility'):
            validate_points({'points': [[4, 4]], 'visible': [True], 'visibility_known': [False]}, 1, (32, 32))

    def test_known_array_must_be_boolean_and_correct_length(self):
        for value in ([True, False], ['False'], [0], [None]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Boolean'):
                validate_visibility_known({'visibility_known': value}, 1, [False])


class VisibilityLossTests(unittest.TestCase):
    def setUp(self): self.loss = isolated_functions()['visibility_supervision_loss']

    def test_unknown_logits_do_not_change_loss_or_finite_difference_gradient(self):
        visible, known = Tensor([[1, 0, 0]]), Tensor([[1, 1, 0]])
        first = self.loss(Tensor([[2, -1, -40]]), visible, known)
        second = self.loss(Tensor([[2, -1, 40]]), visible, known)
        expected = (np.log1p(np.exp(-2)) + np.log1p(np.exp(-1))) / 2
        self.assertAlmostEqual(float(first), expected)
        self.assertAlmostEqual(float(second), expected)
        epsilon = 1e-5
        plus = self.loss(Tensor([[2, -1, epsilon]]), visible, known)
        minus = self.loss(Tensor([[2, -1, -epsilon]]), visible, known)
        self.assertEqual(float((plus-minus)/(2*epsilon)), 0.)
        plus_known = self.loss(Tensor([[2, -1+epsilon, 0]]), visible, known)
        minus_known = self.loss(Tensor([[2, -1-epsilon, 0]]), visible, known)
        self.assertGreater(float((plus_known-minus_known)/(2*epsilon)), 0.)

    def test_old_default_is_same_as_explicit_all_known(self):
        logits, visible = Tensor([[1, 0, -2]]), Tensor([[1, 0, 0]])
        self.assertAlmostEqual(float(self.loss(logits, visible)),
                               float(self.loss(logits, visible, Tensor([[1, 1, 1]]))))

    def test_all_unknown_has_zero_loss_without_division_by_zero(self):
        value = self.loss(Tensor([[500, -500]]), Tensor([[0, 0]]), Tensor([[0, 0]]))
        self.assertEqual(float(value), 0.)
        self.assertTrue(np.isfinite(value))

    def test_invalid_supervision_rejected_in_loss(self):
        for visible, known in (([[1]], [[0]]), ([[0]], [[.5]]), ([[0]], [[1, 1]])):
            with self.subTest(visible=visible, known=known), self.assertRaises(ValueError):
                self.loss(Tensor([[0]]), Tensor(visible), Tensor(known))

    def test_unknown_only_batch_does_not_dilute_validation_visibility_loss(self):
        functions = isolated_functions()
        accumulate, finalize = functions['_accumulate_evaluation_components'], functions['_finalize_evaluation_losses']
        sums, weights = {}, {}
        accumulate(sums, weights, {'heatmap': 1., 'visibility': 2., 'total': 1.5}, 1, 2)
        accumulate(sums, weights, {'heatmap': 1., 'visibility': 0., 'total': 1.}, 1, 0)
        result = finalize(sums, weights)
        self.assertEqual(result, {'heatmap': 1., 'visibility': 2., 'total': 1.5})

    def test_validation_matches_old_sample_average_when_all_known(self):
        functions = isolated_functions()
        sums, weights = {}, {}
        functions['_accumulate_evaluation_components'](sums, weights, {'heatmap': 1., 'visibility': 2., 'total': 1.5}, 2, 8)
        functions['_accumulate_evaluation_components'](sums, weights, {'heatmap': 2., 'visibility': 4., 'total': 3.}, 1, 4)
        result = functions['_finalize_evaluation_losses'](sums, weights)
        self.assertAlmostEqual(result['total'], (2 * 1.5 + 3.) / 3)
        self.assertAlmostEqual(result['visibility'], (2 * 2. + 4.) / 3)


class VisibilityMetricTests(unittest.TestCase):
    def test_unknown_high_probability_is_excluded_not_false_positive(self):
        metric = MetricAccumulator(['positive', 'negative', 'unknown'])
        metric.add(np.zeros((1, 3, 2)), np.zeros((1, 3, 2)), [[True, False, False]],
                   [[.9, .1, .99]], visibility_known=[[1., 1., 0.]])
        result = metric.result()
        self.assertEqual(result['visibility_known_point_count'], 2)
        self.assertEqual(result['visibility_unknown_point_count_excluded'], 1)
        self.assertEqual(result['visibility_at_0_5']['true_positive'], 1)
        self.assertEqual(result['visibility_at_0_5']['true_negative'], 1)
        self.assertEqual(result['visibility_at_0_5']['false_positive'], 0)
        self.assertEqual(result['visibility_at_0_5']['accuracy'], 1.)

    def test_all_unknown_has_no_visibility_accuracy_claim(self):
        metric = MetricAccumulator(['unknown'])
        metric.add(np.zeros((1, 1, 2)), np.zeros((1, 1, 2)), [[False]], [[.9]], visibility_known=[[False]])
        result = metric.result()
        self.assertEqual(result['visibility_known_point_count'], 0)
        self.assertIsNone(result['visibility_at_0_5']['accuracy'])
        self.assertEqual(result['visible_point_error_px']['count'], 0)

    def test_metric_rejects_visible_unknown_and_preserves_old_default(self):
        metric = MetricAccumulator(['one'])
        with self.assertRaisesRegex(ValueError, 'known visibility'):
            metric.add(np.zeros((1, 1, 2)), np.zeros((1, 1, 2)), [[True]], [[.9]], visibility_known=[[False]])
        metric.add(np.zeros((1, 1, 2)), np.zeros((1, 1, 2)), [[False]], [[.9]])
        result = metric.result()
        self.assertEqual(result['visibility_at_0_5']['false_positive'], 1)
        self.assertEqual(result['visibility_unknown_point_count_excluded'], 0)


if __name__ == '__main__':
    unittest.main()
