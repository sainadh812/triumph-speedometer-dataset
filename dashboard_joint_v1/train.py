"""Colab-only sequential all-task training with auditable epoch checkpoints.

checkpoint_callback(task, checkpoint_path, metadata) runs synchronously after an
atomic epoch save. The caller uploads and verifies that file before returning;
any callback failure stops this workflow and leaves local artifacts intact.
At finalization it is called again with phase='complete' and final report/model
paths. Remote recovery supplies resume_checkpoints={task: restored_local_path}.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import gc
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .dataset import TaskDataset, validate_dataset, sha256, PREPROCESSING_ID
    from .metrics import MetricAccumulator, heatmap_to_points
    from .model import InstrumentModel, require_colab_cuda, training_loss, ARCHITECTURE_ID
    from .state import CHECKPOINT_KIND, FORMAT_VERSION, identity_digest, validate_checkpoint, should_stop, RESUME_MODES
except ImportError:
    from dataset import TaskDataset, validate_dataset, sha256, PREPROCESSING_ID
    from metrics import MetricAccumulator, heatmap_to_points
    from model import InstrumentModel, require_colab_cuda, training_loss, ARCHITECTURE_ID
    from state import CHECKPOINT_KIND, FORMAT_VERSION, identity_digest, validate_checkpoint, should_stop, RESUME_MODES


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    min_epochs: int = 10
    patience: int = 8
    batch_size: int = 16
    learning_rate: float = 3e-4
    encoder_learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    amp: bool = True
    num_workers: int = 2
    seed: int = 812
    heatmap_sigma: float = 1.5
    gradient_clip: float = 5.
    min_improvement: float = 1e-4
    progress_every_batches: int = 20

    def checked(self):
        if not (1 <= self.min_epochs <= self.epochs and self.patience >= 1 and self.batch_size >= 1):
            raise ValueError('Invalid epoch/patience/batch configuration')
        if self.num_workers < 0 or self.warmup_epochs < 0 or self.progress_every_batches < 1:
            raise ValueError('Invalid loader/warmup/progress configuration')
        for key in ('learning_rate', 'encoder_learning_rate', 'heatmap_sigma', 'gradient_clip'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError('Invalid positive configuration: ' + key)
        if self.weight_decay < 0 or self.min_improvement < 0:
            raise ValueError('Weight decay/minimum improvement cannot be negative')
        return self


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    _atomic(path, lambda stream: stream.write((json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf8')))


def save_torch(path, value):
    _atomic(path, lambda stream: torch.save(value, stream))
    write_json(str(path) + '.sha256.json', {'sha256': sha256(path), 'bytes': Path(path).stat().st_size})


def emit(event, callback):
    print(json.dumps(event, allow_nan=False), flush=True)
    if callback is not None:
        callback(event)


def _rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all()}


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])


def _seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _identity(validated, task):
    directory = Path(__file__).resolve().parent
    return {'task': task, 'architecture_id': ARCHITECTURE_ID, 'preprocessing_id': PREPROCESSING_ID,
            'dataset_manifest_sha256': validated['manifest_sha256'],
            'samples_sha256': validated['tasks'][task]['samples_sha256'],
            'input_hw': list(validated['tasks'][task]['input_hw']),
            'point_names': validated['tasks'][task]['point_names'],
            'code_sha256': {name: sha256(directory / name) for name in
                            ('model.py', 'dataset.py', 'metrics.py', 'state.py', 'train.py')}}


def _configure_train(model, epoch, config):
    model.train()
    # Small source datasets: freeze all encoder weights for a short decoder warmup.
    train_encoder = epoch > config.warmup_epochs
    model.features.train(train_encoder)
    for parameter in model.features.parameters():
        parameter.requires_grad_(train_encoder)
    # Keep pretrained BatchNorm population estimates stable for small mixed crops.
    for module in model.features.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


@torch.no_grad()
def evaluate(model, loader, device, config, names):
    model.eval()
    metrics = MetricAccumulator(names)
    sums, count = {}, 0
    for raw in loader:
        batch = _to_device(raw, device)
        with torch.autocast('cuda', enabled=config.amp):
            output = model(batch['image'])
            loss, components = training_loss(output, batch, model.input_hw, config.heatmap_sigma)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite evaluation loss')
        size = len(raw['id'])
        count += size
        for key, value in components.items():
            sums[key] = sums.get(key, 0.) + float(value) * size
        points = heatmap_to_points(output['heatmap_logits'].float().cpu().numpy(), model.input_hw)
        kwargs = {}
        if model.task == 'localizer':
            kwargs = {'presence': output['presence_logits'].float().sigmoid().cpu().numpy(),
                      'present': raw['present'].numpy(),
                      'masks': output['lcd_logits'].float().sigmoid().cpu().numpy() >= .5,
                      'mask_targets': raw['mask'].numpy() >= .5,
                      'boxes': output['dashboard_box'].float().cpu().numpy(), 'box_targets': raw['box'].numpy()}
        metrics.add(points, raw['points'].numpy(), raw['visible'].numpy(),
                    output['visibility_logits'].float().sigmoid().cpu().numpy(), **kwargs)
    result = metrics.result()
    result['loss'] = {key: value / count for key, value in sums.items()}
    # Validation losses are comparable only within a task/identity, not across tasks.
    result['selection_score'] = result['loss']['total']
    result['selection_definition'] = 'Minimum combined validation loss; never use test values for selection.'
    return result


def _load_resume(path, identity, config):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Restored resume checkpoint is missing: ' + str(path))
    marker = path.with_name(path.name + '.sha256.json')
    if marker.exists():
        expected = json.loads(marker.read_text())
        if sha256(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError('Checkpoint checksum mismatch')
    # Only use this workflow's own locally produced or checksum-verified restored files.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return validate_checkpoint(payload, identity, asdict(config))


def _resume_path(directory, mode, provided):
    if mode not in RESUME_MODES:
        raise ValueError('resume must be fresh/latest/best')
    if mode == 'fresh':
        if provided:
            raise ValueError('fresh mode cannot load a checkpoint')
        if any(directory.glob('epoch_*.pt')) or (directory / 'report.json').exists():
            raise ValueError('Fresh training needs an empty new task output directory')
        return None
    if provided:
        return Path(provided)
    pointer = directory / (mode + '.json')
    if not pointer.is_file():
        raise FileNotFoundError('No local resume pointer; restore a verified Drive epoch and pass resume_checkpoint')
    value = json.loads(pointer.read_text())
    path = directory / value['filename']
    if not path.resolve().is_relative_to(directory.resolve()) or sha256(path) != value['sha256']:
        raise ValueError('Invalid local checkpoint pointer')
    return path


def train_task(task, dataset_root, output_root, config=None, resume='fresh', resume_checkpoint=None,
               checkpoint_callback=None, progress_callback=None, validated=None):
    device = require_colab_cuda()
    config = (config or TrainConfig(batch_size={'localizer': 16, 'speed': 32, 'rpm': 64}[task])).checked()
    validated = validated or validate_dataset(dataset_root)
    if task not in validated['tasks']:
        raise ValueError('Unknown training task')
    identity = _identity(validated, task)
    directory = Path(output_root) / task
    directory.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(config)
    completed = directory / 'report.json'
    if completed.exists() and resume != 'fresh':
        result = json.loads(completed.read_text())
        if result.get('status') == 'complete' and result['identity'] == identity and result['config'] == config_dict:
            if checkpoint_callback:
                checkpoint_callback(task, Path(result['last_checkpoint']),
                                    {'phase': 'complete', 'completed': True, 'identity': identity,
                                     'epoch': result['epochs_completed'], 'best_epoch': result['best_epoch'],
                                     'artifact_paths': [str(path) for path in directory.iterdir() if path.is_file()]})
            emit({'event': 'already_complete', 'task': task, 'report': str(completed)}, progress_callback)
            return result
        raise ValueError('Existing report identity/configuration differs')
    resume_path = _resume_path(directory, resume, resume_checkpoint)
    payload = _load_resume(resume_path, identity, config) if resume_path else None
    if payload and not payload['training_complete']:
        if any(int(path.stem.split('_')[-1]) > payload['epoch'] for path in directory.glob('epoch_*.pt')):
            raise ValueError('Resuming an older/best epoch branches history; use a new output directory to preserve later epochs')
    _seed(config.seed)
    generators = {split: torch.Generator().manual_seed(config.seed + offset) for offset, split in enumerate(('train', 'val', 'test'))}
    loaders = {split: DataLoader(TaskDataset(validated, task, split), batch_size=config.batch_size,
                                 shuffle=split == 'train', num_workers=config.num_workers, pin_memory=True,
                                 persistent_workers=False, generator=generators[split], drop_last=False)
               for split in generators}
    model = InstrumentModel(task, validated['tasks'][task]['input_hw'], pretrained=payload is None).to(device)
    encoder, heads = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith('features.') else heads).append(parameter)
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': config.encoder_learning_rate, 'name': 'encoder'},
                                   {'params': heads, 'lr': config.learning_rate, 'name': 'task_heads'}],
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=config.amp)
    history, best_state, best_epoch, best_score, waiting, start_epoch = [], None, 0, float('inf'), 0, 1
    initialization = {'mode': 'ImageNet encoder; newly initialized task heads', 'old_task_heads_loaded': False,
                      'weights_enum': 'MobileNet_V3_Small_Weights.IMAGENET1K_V1'}
    training_complete = False
    if payload:
        model.load_state_dict(payload['model_state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        scheduler.load_state_dict(payload['scheduler_state_dict'])
        scaler.load_state_dict(payload['scaler_state_dict'])
        history, best_state = payload['history'], payload['best_state_dict']
        best_epoch, best_score = payload['best_epoch'], payload['best_score']
        waiting, start_epoch = payload['epochs_without_improvement'], payload['epoch'] + 1
        initialization, training_complete = payload['initialization'], payload['training_complete']
        for split, state in payload['loader_generator_states'].items():
            generators[split].set_state(state)
        _restore_rng(payload['rng_state'])
    emit({'event': 'task_start', 'task': task, 'input_hw': identity['input_hw'],
          'counts': validated['tasks'][task]['counts'], 'resume': resume, 'next_epoch': start_epoch,
          'batch_size': config.batch_size, 'amp': config.amp}, progress_callback)
    last_checkpoint = resume_path
    for epoch in range(start_epoch, config.epochs + 1) if not training_complete else ():
        epoch_started = time.perf_counter()
        _configure_train(model, epoch, config)
        total, sample_count, components_sum = 0., 0, {}
        for step, raw in enumerate(loaders['train'], 1):
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=config.amp):
                output = model(batch['image'])
                loss, components = training_loss(output, batch, model.input_hw, config.heatmap_sigma)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss; previous epoch checkpoints remain resumable')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            if not torch.isfinite(norm):
                raise RuntimeError('Nonfinite gradients; previous epoch checkpoints remain resumable')
            scaler.step(optimizer)
            scaler.update()
            size = len(raw['id'])
            sample_count += size
            total += float(loss.detach()) * size
            for key, value in components.items():
                components_sum[key] = components_sum.get(key, 0.) + float(value.detach()) * size
            if step % config.progress_every_batches == 0 or step == len(loaders['train']):
                emit({'event': 'batch', 'task': task, 'epoch': epoch, 'batch': step,
                      'batches': len(loaders['train']), 'loss': total / sample_count}, progress_callback)
        validation = evaluate(model, loaders['val'], device, config, identity['point_names'])
        score = validation['selection_score']
        improved = score < best_score - config.min_improvement
        if improved:
            best_state, best_epoch, best_score, waiting = _cpu_state(model), epoch, score, 0
        else:
            waiting += 1
        scheduler.step()
        training_complete = should_stop(epoch, waiting, config_dict)
        record = {'epoch': epoch, 'train_loss': {key: value / sample_count for key, value in components_sum.items()},
                  'validation': validation, 'seconds': time.perf_counter() - epoch_started,
                  'best_epoch': best_epoch, 'is_best': improved,
                  'next_learning_rates': [group['lr'] for group in optimizer.param_groups]}
        history.append(record)
        last_checkpoint = directory / f'epoch_{epoch:04d}.pt'
        if last_checkpoint.exists():
            raise FileExistsError('Refusing to overwrite an epoch; use a new output directory when branching from best')
        state = {'checkpoint_kind': CHECKPOINT_KIND, 'format_version': FORMAT_VERSION,
                 'identity': identity, 'identity_sha256': identity_digest(identity), 'config': config_dict,
                 'epoch': epoch, 'model_state_dict': _cpu_state(model), 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict(),
                 'rng_state': _rng_state(), 'loader_generator_states': {key: gen.get_state() for key, gen in generators.items()},
                 'history': history, 'best_state_dict': best_state, 'best_epoch': best_epoch, 'best_score': best_score,
                 'epochs_without_improvement': waiting, 'training_complete': training_complete,
                 'initialization': initialization}
        save_torch(last_checkpoint, state)
        pointer = {'filename': last_checkpoint.name, 'sha256': sha256(last_checkpoint), 'epoch': epoch,
                   'identity_sha256': identity_digest(identity)}
        write_json(directory / 'latest.json', pointer)
        if improved:
            write_json(directory / 'best.json', pointer)
        write_json(directory / 'history.json', history)
        metadata = {'phase': 'epoch', 'epoch': epoch, 'best_epoch': best_epoch, 'best_score': best_score,
                    'is_best': improved, 'training_complete': training_complete, 'identity': identity,
                    'artifact_paths': [str(last_checkpoint), str(last_checkpoint) + '.sha256.json',
                                       str(directory / 'latest.json'), str(directory / 'history.json')]}
        if (directory / 'best.json').exists():
            metadata['artifact_paths'].append(str(directory / 'best.json'))
        if checkpoint_callback:
            checkpoint_callback(task, last_checkpoint, metadata)
        emit({'event': 'epoch_saved', 'task': task, 'epoch': epoch, 'validation_loss': score,
              'best_epoch': best_epoch, 'checkpoint': str(last_checkpoint), 'training_complete': training_complete}, progress_callback)
        del state
        if training_complete:
            break
    if best_state is None or last_checkpoint is None:
        raise RuntimeError('Training produced no valid epoch')
    model.load_state_dict(best_state, strict=True)
    # The test set is consulted once, only after validation-selected weights are final.
    test_marker = directory / 'test_once.json'
    if test_marker.exists():
        cached = json.loads(test_marker.read_text())
        if cached['identity_sha256'] != identity_digest(identity) or cached['best_epoch'] != best_epoch:
            raise ValueError('A different selected model already used this output test record')
        test = cached['metrics']
    else:
        test = evaluate(model, loaders['test'], device, config, identity['point_names'])
        write_json(test_marker, {'identity_sha256': identity_digest(identity), 'best_epoch': best_epoch, 'metrics': test})
    weights_path = directory / 'best_model.pt'
    save_torch(weights_path, {'checkpoint_kind': 'dashboard_joint_v1_inference', 'format_version': 1,
                              'identity': identity, 'model_state_dict': best_state, 'best_epoch': best_epoch,
                              'validation_selection_score': best_score, 'initialization': initialization})
    result = {'status': 'complete', 'task': task, 'identity': identity, 'config': config_dict,
              'best_epoch': best_epoch, 'epochs_completed': history[-1]['epoch'],
              'best_validation': history[best_epoch - 1]['validation'], 'test_once': test,
              'last_checkpoint': str(last_checkpoint), 'best_model': str(weights_path), 'report': str(completed),
              'metric_scope': 'Development time-block split; no unseen-recording, speed-kmh, or physical-RPM accuracy claim.',
              'initialization': initialization, 'training_location': 'Google Colab CUDA',
              'torch_version': torch.__version__, 'cuda_device': torch.cuda.get_device_name(device)}
    write_json(completed, result)
    if checkpoint_callback:
        checkpoint_callback(task, last_checkpoint, {'phase': 'complete', 'completed': True, 'identity': identity,
                            'epoch': history[-1]['epoch'], 'best_epoch': best_epoch,
                            'artifact_paths': [str(completed), str(weights_path), str(weights_path) + '.sha256.json',
                                               str(test_marker), str(directory / 'history.json')]})
    emit({'event': 'task_complete', 'task': task, 'best_epoch': best_epoch, 'report': str(completed)}, progress_callback)
    del model, optimizer, scheduler, scaler, loaders, payload, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def train_all(dataset_root, output_root, configs=None, resume_modes=None, resume_checkpoints=None,
              checkpoint_callback=None, progress_callback=None):
    require_colab_cuda()
    validated = validate_dataset(dataset_root)
    configs, resume_modes, resume_checkpoints = configs or {}, resume_modes or {}, resume_checkpoints or {}
    results = {}
    for task in ('localizer', 'speed', 'rpm'):
        config = configs.get(task, TrainConfig(batch_size={'localizer': 16, 'speed': 32, 'rpm': 64}[task]))
        if isinstance(config, dict):
            config = TrainConfig(**config)
        results[task] = train_task(task, dataset_root, output_root, config,
                                   resume_modes.get(task, 'fresh'), resume_checkpoints.get(task),
                                   checkpoint_callback, progress_callback, validated)
    write_json(Path(output_root) / 'training_complete.json', {'status': 'complete', 'tasks': results,
                                                            'dataset_manifest_sha256': validated['manifest_sha256']})
    return results


def smoke_test(dataset_root, output_path=None, configs=None, progress_callback=None):
    """One temporary train step and one validation batch per task, in Colab only.

    This exercises the full unfrozen encoder and AdamW allocation at requested
    batch sizes. It writes no epoch/task checkpoints, restores caller RNG states,
    and discards all temporary weights. The following fresh training initializes
    again from cached ImageNet weights. These one-step timings include warmup;
    they are not sustained inference-FPS or model-accuracy measurements.
    """
    device = require_colab_cuda()
    saved_rng = _rng_state()
    cudnn_state = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    validated = validate_dataset(dataset_root)
    configs = configs or {}
    report = {'status': 'running', 'dataset_manifest_sha256': validated['manifest_sha256'],
              'execution_location': 'Google Colab CUDA', 'device': torch.cuda.get_device_name(device),
              'task_results': {}, 'task_checkpoints_written': False,
              'scope': 'One train forward/backward/AdamW step plus one validation batch per task. Warmup included; not sustained FPS/accuracy.'}
    model = optimizer = scaler = output = batch = loss = components = None
    val_output = val_batch = val_loss = val_components = train_loader = val_loader = None
    task = None
    try:
        for task in ('localizer', 'speed', 'rpm'):
            config = configs.get(task, TrainConfig(batch_size={'localizer': 16, 'speed': 32, 'rpm': 64}[task]))
            if isinstance(config, dict):
                config = TrainConfig(**config)
            config.checked()
            _seed(config.seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            train_loader = DataLoader(TaskDataset(validated, task, 'train'), batch_size=config.batch_size,
                                      shuffle=False, num_workers=0, pin_memory=True)
            val_loader = DataLoader(TaskDataset(validated, task, 'val'), batch_size=config.batch_size,
                                    shuffle=False, num_workers=0, pin_memory=True)
            model = InstrumentModel(task, validated['tasks'][task]['input_hw'], pretrained=True).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            scaler = torch.amp.GradScaler('cuda', enabled=config.amp)
            _configure_train(model, config.warmup_epochs + 1, config)
            raw = next(iter(train_loader))
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=config.amp):
                output = model(batch['image'])
                loss, components = training_loss(output, batch, model.input_hw, config.heatmap_sigma)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite smoke training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError('Nonfinite smoke gradients')
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            train_seconds = time.perf_counter() - started
            train_count, train_loss = len(raw['id']), float(loss.detach())
            optimizer.zero_grad(set_to_none=True)
            output = batch = loss = components = None
            model.eval()
            val_raw = next(iter(val_loader))
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            val_batch = _to_device(val_raw, device)
            with torch.no_grad(), torch.autocast('cuda', enabled=config.amp):
                val_output = model(val_batch['image'])
                val_loss, val_components = training_loss(val_output, val_batch, model.input_hw, config.heatmap_sigma)
            if not torch.isfinite(val_loss):
                raise RuntimeError('Nonfinite smoke validation loss')
            torch.cuda.synchronize(device)
            validation_seconds = time.perf_counter() - started
            result = {'status': 'passed', 'input_hw': list(model.input_hw), 'configured_batch_size': config.batch_size,
                      'actual_training_batch_size': train_count, 'actual_validation_batch_size': len(val_raw['id']),
                      'amp': config.amp, 'training_step_seconds': train_seconds,
                      'single_training_step_samples_per_second': train_count / train_seconds,
                      'validation_batch_seconds': validation_seconds,
                      'single_validation_batch_samples_per_second': len(val_raw['id']) / validation_seconds,
                      'training_loss': train_loss, 'validation_loss': float(val_loss),
                      'gradient_norm_before_clipping': float(gradient_norm),
                      'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
                      'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device),
                      'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
                      'heatmap_shape': list(val_output['heatmap_logits'].shape),
                      'fresh_training_weights_modified': False}
            report['task_results'][task] = result
            emit({'event': 'smoke_task_passed', 'task': task, **result}, progress_callback)
            model = optimizer = scaler = val_output = val_batch = val_loss = val_components = None
            train_loader = val_loader = None
            del raw, val_raw, gradient_norm
            gc.collect()
            torch.cuda.empty_cache()
        report['status'] = 'passed'
        return report
    except BaseException as error:
        report.update(status='failed', failed_task=task, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        model = optimizer = scaler = output = batch = loss = components = None
        val_output = val_batch = val_loss = val_components = train_loader = val_loader = None
        gc.collect()
        torch.cuda.empty_cache()
        _restore_rng(saved_rng)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = cudnn_state
        report['caller_rng_restored'] = True
        if output_path is not None:
            write_json(output_path, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--resume', choices=RESUME_MODES, default='fresh')
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    configs = {task: replace(TrainConfig(batch_size={'localizer': 16, 'speed': 32, 'rpm': 64}[task]), epochs=args.epochs,
                             min_epochs=min(10, args.epochs)) for task in ('localizer', 'speed', 'rpm')}
    train_all(args.dataset_root, args.output_root, configs=configs,
              resume_modes={task: args.resume for task in configs})
