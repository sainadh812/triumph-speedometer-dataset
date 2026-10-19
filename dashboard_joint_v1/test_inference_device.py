"""Device policy/control-flow tests using AST and NumPy fakes, never neural execution."""
import ast
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

try:
    from . import inference as inf
except ImportError:
    import inference as inf


class NoCuda:
    def __getattr__(self, name):
        raise AssertionError('CPU path touched CUDA: ' + name)


class FakeTensor:
    """Only array arithmetic/shape operations needed to exercise the dispatcher."""
    def __init__(self, value):
        self.value = np.asarray(value)
    @property
    def shape(self):
        return self.value.shape
    def to(self, device):
        return self
    def float(self):
        return FakeTensor(self.value.astype(np.float32))
    def div_(self, value):
        self.value /= value
        return self
    def reshape(self, *shape):
        return FakeTensor(self.value.reshape(*shape))
    def __getitem__(self, key):
        return FakeTensor(self.value[key])
    def expand(self, *shape):
        return FakeTensor(np.broadcast_to(self.value, [a if b == -1 else b for a, b in zip(self.shape, shape)]))
    def flatten(self, dimension):
        return FakeTensor(self.value.reshape(*self.shape[:dimension], -1))
    def argmax(self, axis):
        return FakeTensor(self.value.argmax(axis))
    def clamp(self, low, high):
        return FakeTensor(np.clip(self.value, low, high))
    def cpu(self):
        return self
    def numpy(self):
        return self.value
    def __add__(self, other):
        return FakeTensor(self.value + getattr(other, 'value', other))
    def __sub__(self, other):
        return FakeTensor(self.value - getattr(other, 'value', other))
    def __mul__(self, other):
        return FakeTensor(self.value * getattr(other, 'value', other))
    def __truediv__(self, other):
        return FakeTensor(self.value / getattr(other, 'value', other))
    def __floordiv__(self, other):
        return FakeTensor(self.value // getattr(other, 'value', other))
    def __mod__(self, other):
        return FakeTensor(self.value % getattr(other, 'value', other))


def fake_torch(cuda=None):
    value = ModuleType('torch')
    value.cuda = NoCuda() if cuda is None else cuda
    value.device = mock.Mock(side_effect=lambda name: SimpleNamespace(type=name))
    value.is_grad_enabled = mock.Mock(return_value=False)
    value.is_inference_mode_enabled = mock.Mock(return_value=True)
    value.inference_mode = mock.Mock(side_effect=nullcontext)
    value.autocast = mock.Mock(side_effect=nullcontext)
    value.float16 = 'float16'
    value.get_num_threads = mock.Mock(return_value=4)
    value.set_num_threads = mock.Mock()
    value.tensor = lambda data, **kwargs: FakeTensor(data)
    value.from_numpy = lambda data: FakeTensor(data)
    value.stack = lambda data, axis: FakeTensor(np.stack([x.value for x in data], axis))
    return value


def isolated_policy(torch):
    tree = ast.parse(Path(inf.__file__).with_name('model.py').read_text(encoding='utf-8'))
    wanted = {'require_colab_device', 'require_colab_cuda', '_validate_execution'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {'torch': torch, 'importlib': importlib, 'os': os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'model_policy_only', 'exec'), namespace)
    return namespace


def predictor_dispatcher(torch, device='cpu', model=None):
    predictor = object.__new__(inf.DashboardPredictor)
    predictor.torch, predictor.device = torch, SimpleNamespace(type=device)
    predictor._mean = FakeTensor(np.array([.485, .456, .406]).reshape(1, 3, 1, 1))
    predictor._std = FakeTensor(np.array([.229, .224, .225]).reshape(1, 3, 1, 1))
    predictor.effective_batch_size = {'speed': 2}
    def arrays_only(tensor):
        batch = tensor.shape[0]
        logits = np.broadcast_to(np.arange(4).reshape(1, 1, 2, 2), (batch, 12, 2, 2))
        return {'heatmap_logits': FakeTensor(logits), 'visibility_logits': FakeTensor(np.zeros((batch, 12)))}
    predictor.models = {'speed': arrays_only if model is None else model}
    return predictor


class DevicePolicyTests(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.spec = mock.patch.object(importlib.util, 'find_spec', return_value=object())
        self.spec.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.spec.stop)

    def test_cpu_requires_explicit_inference_and_never_checks_cuda(self):
        torch = fake_torch()
        policy = isolated_policy(torch)
        self.assertEqual(policy['require_colab_device']('cpu', inference_only=True).type, 'cpu')
        with self.assertRaisesRegex(RuntimeError, 'training requires CUDA'):
            policy['require_colab_device']('cpu')

    def test_cuda_default_and_training_guard_never_fall_back(self):
        torch = fake_torch(SimpleNamespace(is_available=mock.Mock(return_value=False)))
        policy = isolated_policy(torch)
        for function in (policy['require_colab_device'], policy['require_colab_cuda']):
            with self.assertRaisesRegex(RuntimeError, 'no CPU/local model fallback'):
                function()
        torch.device.assert_not_called()

    def test_local_cpu_and_cuda_rejected_before_device_access(self):
        torch = fake_torch()
        policy = isolated_policy(torch)
        with mock.patch.object(importlib.util, 'find_spec', return_value=None):
            for device in ('cpu', 'cuda'):
                with self.assertRaisesRegex(RuntimeError, 'restricted to Google Colab'):
                    policy['require_colab_device'](device, inference_only=True)
                with self.assertRaisesRegex(RuntimeError, 'restricted to Google Colab'):
                    inf.DashboardPredictor({}, device=device)
        torch.device.assert_not_called()

    def test_unknown_device_rejected(self):
        with self.assertRaises(ValueError):
            isolated_policy(fake_torch())['require_colab_device']('auto', inference_only=True)

    def test_cpu_forward_policy_requires_eval_and_inference_mode(self):
        torch = fake_torch()
        validate = isolated_policy(torch)['_validate_execution']
        images = SimpleNamespace(device=SimpleNamespace(type='cpu'), ndim=4, shape=(2, 3, 384, 256))
        validate(images, (384, 256), 'cpu', True, False)
        for training, grad, inference in ((True, False, True), (False, True, True), (False, False, False)):
            torch.is_grad_enabled.return_value = grad
            torch.is_inference_mode_enabled.return_value = inference
            with self.assertRaisesRegex(RuntimeError, 'eval.*inference_mode'):
                validate(images, (384, 256), 'cpu', True, training)
        with self.assertRaisesRegex(RuntimeError, 'CPU training'):
            validate(images, (384, 256), 'cpu', False, False)

    def test_cuda_training_policy_preserved_and_device_shape_checked(self):
        torch = fake_torch()
        torch.is_grad_enabled.return_value = True
        torch.is_inference_mode_enabled.return_value = False
        validate = isolated_policy(torch)['_validate_execution']
        image = SimpleNamespace(device=SimpleNamespace(type='cuda'), ndim=4, shape=(2, 3, 512, 512))
        validate(image, (512, 512), 'cuda', False, True)
        for device, hw in (('cpu', (512, 512)), ('cuda', (384, 256))):
            with self.assertRaises(ValueError):
                validate(image, hw, device, False, True)

    def test_model_constructor_checks_runtime_before_allocation(self):
        tree = ast.parse(Path(inf.__file__).with_name('model.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'InstrumentModel')
        constructor = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        isolated = ast.ClassDef(name='InstrumentModel', bases=[ast.Name(id='Base', ctx=ast.Load())], keywords=[], body=[constructor], decorator_list=[])
        guarded = mock.Mock(side_effect=RuntimeError('runtime rejected'))
        namespace = {'Base': object, 'require_colab_device': guarded}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[])), 'constructor_policy', 'exec'), namespace)
        with self.assertRaisesRegex(RuntimeError, 'runtime rejected'):
            namespace['InstrumentModel']('speed', (512, 512), device='cpu', inference_only=True)
        guarded.assert_called_once_with('cpu', inference_only=True)


class DispatcherTests(unittest.TestCase):
    def test_cpu_prediction_decode_has_no_cuda_or_autocast_calls(self):
        torch = fake_torch()
        predictor = predictor_dispatcher(torch)
        rows = predictor._run('speed', [np.ones((8, 8), np.uint8)] * 3)
        predictor._synchronize()
        self.assertEqual(len(rows), 3)
        np.testing.assert_array_equal(rows[0]['points'], np.full((12, 2), 383.5))
        np.testing.assert_array_equal(rows[0]['visibility'], np.full(12, .5))
        torch.autocast.assert_not_called()
        self.assertEqual(torch.inference_mode.call_count, 2)

    def test_cpu_exception_propagates_without_cuda_recovery(self):
        def failed(_):
            raise MemoryError('CPU memory failure')
        predictor = predictor_dispatcher(fake_torch(), model=failed)
        with self.assertRaisesRegex(MemoryError, 'CPU memory failure'):
            predictor._run('speed', [np.ones((8, 8), np.uint8)])
        self.assertEqual(predictor.effective_batch_size['speed'], 2)

    def test_cuda_uses_autocast_and_synchronization(self):
        cuda = SimpleNamespace(synchronize=mock.Mock())
        torch = fake_torch(cuda)
        torch.autocast = mock.Mock(side_effect=lambda **kwargs: nullcontext())
        predictor = predictor_dispatcher(torch, 'cuda')
        rows = predictor._run('speed', [np.ones((8, 8), np.uint8)])
        predictor._synchronize()
        self.assertEqual(len(rows), 1)
        torch.autocast.assert_called_once_with(device_type='cuda', dtype='float16')
        cuda.synchronize.assert_called_once_with()

    def test_cuda_oom_halves_batch_without_cpu_fallback_or_frame_loss(self):
        class FakeOom(RuntimeError):
            pass
        cuda = SimpleNamespace(OutOfMemoryError=FakeOom, empty_cache=mock.Mock())
        torch = fake_torch(cuda)
        torch.autocast = mock.Mock(side_effect=lambda **kwargs: nullcontext())
        predictor = predictor_dispatcher(torch, 'cuda')
        success = predictor.models['speed']
        calls = []
        def failed_once(tensor):
            calls.append(tensor.shape[0])
            if len(calls) == 1:
                raise FakeOom('synthetic OOM')
            return success(tensor)
        predictor.models['speed'] = failed_once
        rows = predictor._run('speed', [np.ones((8, 8), np.uint8)] * 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(calls, [2, 1, 1, 1])
        self.assertEqual(predictor.device.type, 'cuda')
        self.assertEqual(predictor.effective_batch_size['speed'], 1)
        cuda.empty_cache.assert_called_once_with()


class PredictorConstructionTests(unittest.TestCase):
    def test_cpu_constructor_metadata_checkpoint_validation_and_threads(self):
        torch = fake_torch()
        module = ModuleType('model')
        module.ARCHITECTURE_ID = 'mock-compatible-architecture'
        module.require_colab_device = isolated_policy(torch)['require_colab_device']
        instances = []
        def fake_model(*args, **kwargs):
            model = mock.Mock()
            model.to.return_value = model
            instances.append(model)
            return model
        module.InstrumentModel = mock.Mock(side_effect=fake_model)
        def checkpoint(path, **kwargs):
            task = str(path)
            return {'checkpoint_kind': 'dashboard_joint_v1_inference', 'format_version': 1,
                    'identity': {'task': task, 'architecture_id': module.ARCHITECTURE_ID,
                                 'preprocessing_id': inf.PREPROCESSING_ID, 'point_names': inf.POINT_NAMES[task],
                                 'input_hw': list(inf.SIZES[task][::-1]), 'dataset_manifest_sha256': 'same-manifest'},
                    'model_state_dict': {'sentinel': 'unchanged'}, 'best_epoch': 7}
        torch.load = mock.Mock(side_effect=checkpoint)
        module_name = inf.__package__ + '.model' if inf.__package__ else 'model'
        with mock.patch.dict(sys.modules, {'torch': torch, module_name: module}), \
                mock.patch.object(importlib.util, 'find_spec', return_value=object()):
            predictor = inf.DashboardPredictor({task: task for task in inf.SIZES}, device='cpu', batch_size=4)
        self.assertEqual(predictor.execution_metadata['device'], 'cpu')
        self.assertEqual(predictor.execution_metadata['precision'], 'float32')
        self.assertEqual(predictor.execution_metadata['torch_num_threads'], 4)
        self.assertEqual(predictor.thresholds, inf.DEFAULT_THRESHOLDS)
        torch.set_num_threads.assert_not_called()
        for model in instances:
            model.float.assert_called_once_with()
            model.eval.assert_called_once_with()
            model.load_state_dict.assert_called_once_with({'sentinel': 'unchanged'}, strict=True)
        for call in module.InstrumentModel.call_args_list:
            self.assertEqual(call.kwargs['device'], 'cpu')
            self.assertTrue(call.kwargs['inference_only'])
            self.assertFalse(call.kwargs['pretrained'])
        self.assertEqual(torch.load.call_count, 3)
        for call in torch.load.call_args_list:
            self.assertEqual(call.kwargs, {'map_location': 'cpu', 'weights_only': True})


if __name__ == '__main__':
    unittest.main()
