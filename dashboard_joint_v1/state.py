"""Pure checkpoint identity helpers, independently testable without PyTorch."""
from __future__ import annotations

import hashlib
import json
import math

CHECKPOINT_KIND = 'dashboard_joint_v1_epoch_full_state'
FORMAT_VERSION = 1
RESUME_MODES = ('fresh', 'latest', 'best')


def selected_validation(history, initialization, best_epoch):
    """Epoch zero is the unchanged transferred model, never a trained epoch."""
    if type(best_epoch) is not int or best_epoch < 0:
        raise ValueError('Invalid selected epoch')
    if best_epoch == 0:
        if initialization.get('baseline_eligible_for_selection') is not True:
            raise ValueError('Epoch zero requires an eligible initialization baseline')
        result = initialization.get('validation_before_finetuning')
    else:
        matches = [row for row in history if row.get('epoch') == best_epoch]
        if len(matches) != 1:
            raise ValueError('Selected epoch is absent or ambiguous in training history')
        result = matches[0].get('validation')
    if not isinstance(result, dict):
        raise ValueError('Selected validation metrics are missing')
    score = result.get('selection_score')
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError('Selected validation score must be finite')
    return result


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
    if type(epoch) is not int or epoch < 0 or [row['epoch'] for row in payload['history']] != list(range(1, epoch + 1)):
        raise ValueError('Checkpoint epoch/history is incomplete')
    if type(payload['best_epoch']) is not int or not 0 <= payload['best_epoch'] <= epoch or not payload['best_state_dict']:
        raise ValueError('Checkpoint best weights are missing')
    if payload['best_epoch'] == 0:
        baseline = selected_validation(payload['history'], payload['initialization'], 0)
        if baseline['selection_score'] != payload['best_score']:
            raise ValueError('Selected baseline score differs from saved validation')
    if epoch == 0 and (payload['training_complete'] or payload['epochs_without_improvement'] != 0):
        raise ValueError('Epoch zero must precede every optimizer update and stopping decision')
    if set(payload['loader_generator_states']) != {'train', 'val', 'test'}:
        raise ValueError('Loader generator state is incomplete')
    if not {'python', 'numpy', 'torch_cpu', 'torch_cuda'}.issubset(payload['rng_state']):
        raise ValueError('RNG state is incomplete')
    return payload


def should_stop(epoch, epochs_without_improvement, config):
    return epoch >= config['epochs'] or (epoch >= config['min_epochs'] and
                                         epochs_without_improvement >= config['patience'])
