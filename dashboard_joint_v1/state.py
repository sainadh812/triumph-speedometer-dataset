"""Pure checkpoint identity helpers, independently testable without PyTorch."""
from __future__ import annotations

import hashlib
import json

CHECKPOINT_KIND = 'dashboard_joint_v1_epoch_full_state'
FORMAT_VERSION = 1
RESUME_MODES = ('fresh', 'latest', 'best')


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_checkpoint(payload, expected_identity, expected_config):
    required = {'checkpoint_kind', 'format_version', 'identity', 'identity_sha256', 'config', 'epoch',
                'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict',
                'rng_state', 'loader_generator_states', 'history', 'best_state_dict', 'best_epoch',
                'best_score', 'epochs_without_improvement', 'training_complete', 'initialization'}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError('Incomplete resumable epoch checkpoint')
    if payload['checkpoint_kind'] != CHECKPOINT_KIND or payload['format_version'] != FORMAT_VERSION:
        raise ValueError('Checkpoint family/version mismatch')
    if payload['identity'] != expected_identity or payload['identity_sha256'] != identity_digest(expected_identity):
        raise ValueError('Dataset, architecture or preprocessing identity changed; start a new fresh experiment')
    if payload['config'] != expected_config:
        raise ValueError('Training configuration changed; exact optimizer resume is not permitted')
    epoch = payload['epoch']
    if not isinstance(epoch, int) or epoch < 1 or [row['epoch'] for row in payload['history']] != list(range(1, epoch + 1)):
        raise ValueError('Checkpoint epoch/history is incomplete')
    if not 1 <= payload['best_epoch'] <= epoch or not payload['best_state_dict']:
        raise ValueError('Checkpoint best weights are missing')
    if set(payload['loader_generator_states']) != {'train', 'val', 'test'}:
        raise ValueError('Loader generator state is incomplete')
    if not {'python', 'numpy', 'torch_cpu', 'torch_cuda'}.issubset(payload['rng_state']):
        raise ValueError('RNG state is incomplete')
    return payload


def should_stop(epoch, epochs_without_improvement, config):
    return epoch >= config['epochs'] or (epoch >= config['min_epochs'] and
                                         epochs_without_improvement >= config['patience'])
