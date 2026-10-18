"""AST-isolated AMP control-flow tests with NumPy fakes; never imports torch."""
import ast
import contextlib
from dataclasses import dataclass, replace
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


def load_policy():
    source = ast.parse(Path(__file__).with_name('train.py').read_text(encoding='utf8'))
    names = {'TrainConfig', '_next_amp_scale', '_restore_step_state', '_checked_training_step'}
    selected = ast.Module(body=[node for node in source.body if getattr(node, 'name', None) in names], type_ignores=[])
    space = {'dataclass': dataclass, 'math': math}
    exec(compile(selected, 'train.py:isolated-numerical-policy', 'exec'), space)
    return space


class FakeBuffer:
    def __init__(self, value=0): self.value = value
    def detach(self): return self
    def clone(self): return FakeBuffer(self.value)
    def copy_(self, other): self.value = other.value


class FakeModel:
    input_hw = (32, 32)
    def __init__(self, scenarios, rng):
        self.scenarios, self.rng, self.calls, self.weights = scenarios, rng, 0, 0
        self.buffer = FakeBuffer()
        self.parameter = SimpleNamespace(grad=None)
        self.gradient_dtype = np.float32
    def parameters(self): return iter([self.parameter])
    def named_buffers(self): return iter([('batchnorm_counter', self.buffer)])
    def __call__(self, image):
        self.calls += 1
        self.buffer.value += 1
        self.rng['counter'] += 1
        return self


class FakeLoss:
    def __init__(self, model, value=1.): self.model, self.value, self.scale = model, value, 1.
    def backward(self):
        scenario = self.model.scenarios[min(self.model.calls - 1, len(self.model.scenarios) - 1)]
        if scenario is not None:
            self.model.parameter.grad = np.array([scenario * self.scale], dtype=self.model.gradient_dtype)
    def detach(self): return self
    def __float__(self): return self.value


class FakeOptimizer:
    def __init__(self, model): self.model, self.updates = model, 0
    def zero_grad(self, set_to_none): self.model.parameter.grad = None
    def step(self):
        assert np.isfinite(self.model.parameter.grad).all(), 'Invalid gradient reached optimizer!'
        self.model.weights += 1
        self.updates += 1


class FakeScaler:
    def __init__(self, model, scale=256.): self.model, self.current, self.found_inf = model, scale, False
    def get_scale(self): return self.current
    def scale(self, loss): loss.scale = self.current; return loss
    def unscale_(self, optimizer):
        gradient = self.model.parameter.grad
        self.found_inf = gradient is not None and not np.isfinite(gradient).all()
        if gradient is not None: gradient /= self.current
    def step(self, optimizer):
        if not self.found_inf: optimizer.step()
    def update(self):
        if self.found_inf: self.current *= .5
        self.found_inf = False


def environment(scenarios, loss_value=1., corrupt_component=False, clip_failure=False, clip_corruption=False):
    space, rng, events = load_policy(), {'counter': 0}, []
    model = FakeModel(scenarios, rng)
    optimizer, scaler = FakeOptimizer(model), FakeScaler(model)
    def isfinite(value): return np.isfinite(value.value if isinstance(value, FakeLoss) else value)
    def clip(parameters, maximum, error_if_nonfinite):
        assert error_if_nonfinite is True
        if clip_failure: raise RuntimeError('Nonfinite total norm')
        norm = np.sqrt(sum(float(np.sum(parameter.grad ** 2)) for parameter in parameters))
        for parameter in parameters:
            if clip_corruption: parameter.grad[:] = np.nan
            else: parameter.grad *= min(1., maximum / (norm + 1e-6))
        return np.float32(norm)
    fake_torch = SimpleNamespace(float32=np.dtype('float32'), isfinite=isfinite, stack=np.stack,
        autocast=lambda *a, **kw: contextlib.nullcontext(), no_grad=contextlib.nullcontext,
        nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=clip)))
    def losses(output, batch, hw, sigma):
        loss = FakeLoss(model, loss_value)
        return loss, {'total': loss, 'part': np.float32(np.nan if corrupt_component else 1.)}
    space.update(torch=fake_torch, training_loss=losses,
                 _rng_state=lambda: dict(rng), _restore_rng=lambda state: rng.update(state),
                 emit=lambda event, callback: events.append(event))
    return space, model, optimizer, scaler, rng, events


class NumericalPolicyTests(unittest.TestCase):
    def run_step(self, setup, config=None):
        space, model, optimizer, scaler, rng, events = setup
        config = config or space['TrainConfig']().checked()
        return space['_checked_training_step'](model, optimizer, scaler, {'image': object()}, config)

    def test_overflow_replays_batch_restores_rng_and_buffers_and_updates_once(self):
        setup = environment([np.inf, np.nan, 2.])
        _, _, diagnostics = self.run_step(setup)
        _, model, optimizer, scaler, rng, events = setup
        self.assertEqual((model.calls, optimizer.updates, model.weights), (3, 1, 1))
        self.assertEqual((model.buffer.value, rng['counter']), (1, 1))
        self.assertEqual(scaler.get_scale(), 64.)
        self.assertEqual(diagnostics['overflow_retries'], 2)
        self.assertEqual([event['new_loss_scale'] for event in events], [128., 64.])

    def test_persistent_overflow_aborts_without_update_or_buffer_mutation(self):
        setup = environment([np.inf])
        with self.assertRaisesRegex(RuntimeError, 'Persistent'):
            self.run_step(setup)
        _, model, optimizer, scaler, rng, events = setup
        self.assertEqual((model.calls, optimizer.updates, model.weights), (9, 0, 0))
        self.assertEqual((model.buffer.value, rng['counter']), (0, 0))
        self.assertEqual(scaler.get_scale(), 1.)
        self.assertEqual(len(events), 8)

    def test_retry_bound_is_enforced_before_minimum_scale(self):
        setup = environment([np.inf])
        config = replace(setup[0]['TrainConfig'](), amp_max_retries=2).checked()
        with self.assertRaisesRegex(RuntimeError, 'Persistent'):
            self.run_step(setup, config)
        self.assertEqual((setup[1].calls, setup[2].updates), (3, 0))

    def test_nonfinite_loss_and_hidden_component_fail_without_backward_or_update(self):
        for kwargs in ({'loss_value': np.inf}, {'corrupt_component': True}):
            with self.subTest(kwargs=kwargs):
                setup = environment([1.], **kwargs)
                with self.assertRaisesRegex(RuntimeError, 'loss/component'):
                    self.run_step(setup)
                self.assertEqual((setup[1].calls, setup[2].updates, setup[1].buffer.value), (1, 0, 0))
                self.assertFalse(setup[5])

    def test_no_amp_cannot_retry_nonfinite_gradients(self):
        setup = environment([np.inf])
        config = replace(setup[0]['TrainConfig'](), amp=False).checked()
        with self.assertRaisesRegex(RuntimeError, 'usable AMP'):
            self.run_step(setup, config)
        self.assertEqual((setup[1].calls, setup[2].updates), (1, 0))

    def test_gradient_dtype_and_missing_gradients_are_rejected(self):
        setup = environment([1.])
        setup[1].gradient_dtype = np.float64
        with self.assertRaisesRegex(RuntimeError, 'FP32'):
            self.run_step(setup)
        empty = environment([None])
        with self.assertRaisesRegex(RuntimeError, 'FP32'):
            self.run_step(empty)
        self.assertEqual(setup[2].updates + empty[2].updates, 0)

    def test_norm_failure_or_clip_corruption_never_updates(self):
        for kwargs in ({'clip_failure': True}, {'clip_corruption': True}):
            with self.subTest(kwargs=kwargs):
                setup = environment([2.], **kwargs)
                with self.assertRaises(RuntimeError):
                    self.run_step(setup)
                self.assertEqual((setup[2].updates, setup[1].buffer.value), (0, 0))

    def test_invalid_config_rejected(self):
        cls = load_policy()['TrainConfig']
        for kwargs in ({'amp_initial_scale': float('inf')}, {'amp_min_scale': 0.},
                       {'amp_min_scale': 512.}, {'amp_max_retries': 99}, {'amp_max_retries': 2.5},
                       {'amp': 'yes'}, {'weight_decay': float('nan')}, {'min_improvement': float('nan')},
                       {'batch_size': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cls(**kwargs).checked()


if __name__ == '__main__':
    unittest.main()
