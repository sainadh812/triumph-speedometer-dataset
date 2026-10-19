"""Execute real training control flow with Python fakes; no torch/model imports."""
import ast
import copy
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import pickle
import tempfile
import time
from types import SimpleNamespace
import unittest

from state import (CHECKPOINT_KIND, FORMAT_VERSION, RESUME_MODES, identity_digest,
                   selected_validation, should_stop, validate_checkpoint)


class Value:
    def __init__(self, value): self.value = value
    def detach(self): return self
    def cpu(self): return self
    def clone(self): return Value(self.value)
    def __float__(self): return float(self.value)


class Model:
    def __init__(self): self.state = {'weight': Value(10), 'batchnorm': Value(7)}
    def state_dict(self): return self.state
    def named_parameters(self): return [('features.weight', object()), ('head.weight', object())]
    def load_state_dict(self, state, strict): self.state = copy.deepcopy(state)


class Stateful:
    def __init__(self, **kwargs): self.steps = 0; self.param_groups = [{'lr': .01}, {'lr': .001}]
    def state_dict(self): return {'steps': self.steps}
    def load_state_dict(self, state): self.steps = state['steps']
    def step(self): self.steps += 1
    def get_scale(self): return 256.


class Generator:
    def manual_seed(self, value): self.value = value; return self
    def get_state(self): return self.value
    def set_state(self, value): self.value = value


def environment(scores):
    source = ast.parse(Path(__file__).with_name('train.py').read_text(encoding='utf8'))
    names = {'TrainConfig', '_cpu_state', '_resume_path', 'train_task'}
    module = ast.Module(body=[node for node in source.body if getattr(node, 'name', None) in names], type_ignores=[])
    events, saved, callbacks, instances = [], {}, [], {}
    space = dict(dataclass=dataclass, asdict=asdict, math=math, Path=Path, json=json, time=time, gc=gc,
                 CHECKPOINT_KIND=CHECKPOINT_KIND, FORMAT_VERSION=FORMAT_VERSION, RESUME_MODES=RESUME_MODES,
                 identity_digest=identity_digest, validate_checkpoint=validate_checkpoint,
                 selected_validation=selected_validation, should_stop=should_stop)
    exec(compile(module, 'train.py:isolated-control-flow', 'exec'), space)
    def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def write_json(path, value):
        Path(path).write_text(json.dumps(value, allow_nan=False), encoding='utf8')
    def save(path, value):
        Path(path).write_bytes(pickle.dumps(value))
        saved[str(path)] = pickle.loads(Path(path).read_bytes())
        write_json(str(path) + '.sha256.json', {'sha256': digest(path), 'bytes': Path(path).stat().st_size})
    def initial(task, validated, identity, device, weights=None, resuming=False):
        model = Model(); instances['model'] = model
        return model, ({'mode': 'weights_only_transfer_new_experiment', 'source_weights_sha256': 'a' * 64}
                       if weights else None)
    def constructor(kind):
        def create(*args, **kwargs):
            instance = Stateful(); instances[kind] = instance
            return instance
        return create
    def step(model, optimizer, scaler, batch, config, context, callback):
        model.state['weight'].value += 1
        model.state['batchnorm'].value += 1
        optimizer.step()
        return Value(1), {'total': Value(1)}, {'overflow_retries': 0}
    pending_scores = iter(scores)
    eval_calls = []
    def evaluate(model, loader, *args):
        eval_calls.append({'split': loader if isinstance(loader, str) else 'train',
                           'weight': model.state['weight'].value})
        score = next(pending_scores)
        return {'selection_score': score, 'loss': {'total': score}, 'split': loader}
    identity = {'task': 'speed', 'point_names': ['pivot', 'tip'], 'input_hw': [512, 512], 'code': 'v3'}
    def load(path, target_identity, config):
        return validate_checkpoint(pickle.loads(Path(path).read_bytes()), target_identity, asdict(config))
    fake_torch = SimpleNamespace(Generator=Generator,
        optim=SimpleNamespace(AdamW=constructor('optimizer'),
                              lr_scheduler=SimpleNamespace(CosineAnnealingLR=constructor('scheduler'))),
        amp=SimpleNamespace(GradScaler=constructor('scaler')),
        cuda=SimpleNamespace(empty_cache=lambda: None, get_device_name=lambda device: 'FAKE-CUDA'),
        __version__='mock')
    space.update(torch=fake_torch, require_colab_cuda=lambda: 'fake-cuda', _identity=lambda *args: identity,
                 _seed=lambda value: None, _initial_model=initial, _configure_train=lambda *args: None,
                 _make_loaders=lambda *args: ({'train': [{'id': ['one']}], 'val': 'val', 'test': 'test'}, {}),
                 _to_device=lambda batch, device: batch, _checked_training_step=step, evaluate=evaluate,
                 batch_lighting_groups=lambda light, size: {'day': [0]},
                 _rng_state=lambda: dict.fromkeys(('python', 'numpy', 'torch_cpu', 'torch_cuda'), 0),
                 _restore_rng=lambda value: None, sha256=digest, save_torch=save, write_json=write_json,
                 _load_resume=load, emit=lambda event, callback: events.append(copy.deepcopy(event)))
    validated = {'tasks': {'speed': {'counts': {'train': 1, 'val': 1, 'test': 1}}}}
    config = space['TrainConfig'](epochs=2, min_epochs=2, patience=2, warmup_epochs=0, num_workers=0)
    def run(output, **kwargs):
        callback = kwargs.pop('checkpoint_callback', lambda *args: callbacks.append(copy.deepcopy(args)))
        return space['train_task']('speed', 'unused', output, config=config, validated=validated,
                                   checkpoint_callback=callback, **kwargs)
    return SimpleNamespace(run=run, space=space, saved=saved, events=events, callbacks=callbacks,
                           instances=instances, eval_calls=eval_calls, identity=identity, config=config)


class BaselineSelectionTests(unittest.TestCase):
    def test_every_worse_epoch_exports_intact_baseline_not_last_epoch(self):
        env = environment([.5, .7, .8, .6])
        with tempfile.TemporaryDirectory() as root:
            report = env.run(root, initialization_weights='verified-initializer')
            directory = Path(root) / 'speed'
            baseline = env.saved[str(directory / 'epoch_0000.pt')]
            latest = env.saved[str(directory / 'epoch_0002.pt')]
            selected = env.saved[str(directory / 'best_model.pt')]
            self.assertEqual(report['best_epoch'], 0)
            self.assertEqual(report['selection_origin'], 'initialization')
            self.assertEqual(report['best_validation']['selection_score'], .5)
            self.assertEqual(report['epochs_completed'], 2)
            self.assertEqual([row['is_best'] for row in latest['history']], [False, False])
            self.assertEqual(selected['model_state_dict']['weight'].value, 10)
            self.assertEqual(selected['model_state_dict']['batchnorm'].value, 7)
            self.assertEqual(latest['model_state_dict']['weight'].value, 12)
            self.assertEqual(latest['optimizer_state_dict']['steps'], 2)
            self.assertEqual(latest['best_state_dict']['weight'].value, 10)
            self.assertEqual(baseline['optimizer_state_dict']['steps'], 0)
            self.assertEqual(baseline['scheduler_state_dict']['steps'], 0)
            self.assertEqual(baseline['history'], [])
            self.assertEqual(json.loads((directory / 'best.json').read_text())['epoch'], 0)
            self.assertEqual([event['epoch'] for _, _, event in env.callbacks], [0, 1, 2, 2])
            self.assertEqual(env.eval_calls[-1], {'split': 'test', 'weight': 10})

    def test_improved_epoch_replaces_baseline_with_matching_report(self):
        env = environment([.5, .7, .3, .4])
        with tempfile.TemporaryDirectory() as root:
            report = env.run(root, initialization_weights='verified-initializer')
            self.assertEqual(report['best_epoch'], 2)
            self.assertEqual(report['best_validation']['selection_score'], .3)
            self.assertEqual(report['selection_origin'], 'trained_epoch')
            self.assertEqual(env.eval_calls[-1]['weight'], 12)

    def test_best_restart_epoch_zero_restores_zero_optimizer_steps(self):
        first = environment([.5, .7, .8, .6])
        resumed = environment([.4, .6, .7])
        with tempfile.TemporaryDirectory() as root:
            first.run(Path(root) / 'first', initialization_weights='verified-initializer')
            source = Path(root) / 'first/speed/epoch_0000.pt'
            report = resumed.run(Path(root) / 'branch', resume='best', resume_checkpoint=source)
            checkpoint = resumed.saved[str(Path(root) / 'branch/speed/epoch_0001.pt')]
            self.assertEqual(checkpoint['model_state_dict']['weight'].value, 11)
            self.assertEqual(checkpoint['optimizer_state_dict']['steps'], 1)
            self.assertEqual(checkpoint['scheduler_state_dict']['steps'], 1)
            self.assertEqual(checkpoint['initialization']['validation_before_finetuning']['selection_score'], .5)
            self.assertEqual(report['best_epoch'], 1)
            self.assertEqual(len(resumed.eval_calls), 3)  # No re-evaluation/replacement of saved baseline.

    def test_latest_resume_keeps_later_optimizer_and_earlier_best_weights(self):
        first = environment([.5, .7])
        resumed = environment([.8, .6])
        with tempfile.TemporaryDirectory() as root:
            def stop(task, path, metadata):
                if metadata['epoch'] == 1: raise RuntimeError('simulated upload interruption')
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                first.run(Path(root) / 'first', initialization_weights='verified-initializer', checkpoint_callback=stop)
            report = resumed.run(Path(root) / 'resume', resume='latest',
                                 resume_checkpoint=Path(root) / 'first/speed/epoch_0001.pt')
            checkpoint = resumed.saved[str(Path(root) / 'resume/speed/epoch_0002.pt')]
            self.assertEqual(checkpoint['optimizer_state_dict']['steps'], 2)
            self.assertEqual(checkpoint['model_state_dict']['weight'].value, 12)
            self.assertEqual(report['best_epoch'], 0)
            self.assertEqual(resumed.eval_calls[-1]['weight'], 10)

    def test_failed_baseline_save_callback_prevents_any_training_update(self):
        env = environment([.5])
        with tempfile.TemporaryDirectory() as root:
            def stop(*args): raise RuntimeError('upload failed')
            with self.assertRaisesRegex(RuntimeError, 'upload failed'):
                env.run(root, initialization_weights='verified-initializer', checkpoint_callback=stop)
            self.assertEqual(env.instances['optimizer'].steps, 0)
            self.assertEqual(env.instances['model'].state['weight'].value, 10)

    def test_new_heads_still_select_a_trained_epoch(self):
        env = environment([.7, .8, .6])
        with tempfile.TemporaryDirectory() as root:
            report = env.run(root)
            self.assertEqual(report['best_epoch'], 1)
            self.assertFalse((Path(root) / 'speed/epoch_0000.pt').exists())

    def test_invalid_or_unproven_epoch_zero_checkpoint_rejected(self):
        env = environment([.5, .7, .8, .6])
        with tempfile.TemporaryDirectory() as root:
            env.run(root, initialization_weights='verified-initializer')
            original = env.saved[str(Path(root) / 'speed/epoch_0000.pt')]
            for mutate in (lambda p: p['initialization'].pop('baseline_eligible_for_selection'),
                           lambda p: p.update(best_score=.4), lambda p: p.update(training_complete=True),
                           lambda p: p.update(epochs_without_improvement=1)):
                bad = copy.deepcopy(original); mutate(bad)
                with self.assertRaises(ValueError): validate_checkpoint(bad, env.identity, asdict(env.config))


if __name__ == '__main__': unittest.main()
