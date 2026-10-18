"""Weights-only transfer into a NEW experiment; never a resume shortcut.

Import and payload validation are pure Python. Loading a real weights file is
lazy and uses PyTorch's restricted weights-only reader. This module constructs
no models and restores no optimizer, scheduler, RNG, scaler, or history.
"""
from collections.abc import Mapping
import copy
import hashlib
import json
import math
from pathlib import Path
import re

COMPATIBILITY_FIELDS = ('task', 'architecture_id', 'preprocessing_id', 'input_hw', 'point_names')
FORBIDDEN_STATE_FIELDS = {'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict',
                          'rng_state', 'loader_generator_states', 'history', 'best_state_dict'}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified_weight_file(path):
    """Require the exact checksum marker extracted from the verified archive."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Initialization weights are missing')
    marker_path = path.with_name(path.name + '.sha256.json')
    if not marker_path.is_file():
        raise ValueError('Initialization requires its verified checksum marker')
    marker = json.loads(marker_path.read_text(encoding='utf8'))
    digest, size = marker.get('sha256'), marker.get('bytes')
    if (not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) or
            type(size) is not int or size <= 0 or path.stat().st_size != size or _sha256(path) != digest):
        raise ValueError('Initialization weights checksum or byte count mismatch')
    return {'sha256': digest, 'size': size}


def validate_initialization_payload(payload, target_identity, source_sha256, source_size):
    """Return ONLY model parameters and transfer provenance.

    Dataset/sample/code identity changes are intentional for transfer. Structural
    task semantics and preprocessing must match exactly. Full-state epoch files
    are rejected even if someone changes their checkpoint_kind discriminator.
    """
    if (not isinstance(payload, dict) or payload.get('checkpoint_kind') != 'dashboard_joint_v1_inference' or
            type(payload.get('format_version')) is not int or payload['format_version'] != 1):
        raise ValueError('Initialization requires an exported best-model weights payload')
    if FORBIDDEN_STATE_FIELDS.intersection(payload):
        raise ValueError('Full training state cannot be used for weights-only initialization')
    source = payload.get('identity')
    if not isinstance(source, dict) or not isinstance(target_identity, dict):
        raise ValueError('Initialization and target identities are required')
    for field in COMPATIBILITY_FIELDS:
        if field not in source or field not in target_identity or source[field] != target_identity[field]:
            raise ValueError('Initialization compatibility mismatch: ' + field)
    for field in ('dataset_manifest_sha256', 'samples_sha256'):
        for identity in (source, target_identity):
            if not isinstance(identity.get(field), str) or not re.fullmatch('[0-9a-f]{64}', identity[field]):
                raise ValueError('Initialization dataset provenance is incomplete: ' + field)
    if (not isinstance(source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', source_sha256) or
            type(source_size) is not int or source_size <= 0):
        raise ValueError('Initialization file provenance is invalid')
    epoch, score = payload.get('best_epoch'), payload.get('validation_selection_score')
    if (type(epoch) is not int or epoch < 1 or isinstance(score, bool) or
            not isinstance(score, (int, float)) or not math.isfinite(score)):
        raise ValueError('Selected source epoch/validation provenance is invalid')
    state = payload.get('model_state_dict')
    if not isinstance(state, Mapping) or not state or any(not isinstance(key, str) for key in state):
        raise ValueError('Initialization lacks a model state dictionary')
    provenance = {
        'mode': 'weights_only_transfer_new_experiment', 'old_task_heads_loaded': True,
        'source_weights_sha256': source_sha256, 'source_weights_size': source_size,
        'source_best_epoch': epoch, 'source_validation_selection_score': score,
        'source_identity': copy.deepcopy(source),
        'target_dataset_manifest_sha256': target_identity['dataset_manifest_sha256'],
        'target_samples_sha256': target_identity['samples_sha256'],
        'optimizer_scheduler_scaler_rng_history_restored': False,
    }
    return state, provenance


def load_initialization(path, target_identity, loader=None):
    """Checksum-verify, restricted-load, then validate weights-only transfer.

    loader(path, map_location='cpu', weights_only=True) may be injected by pure
    tests. Train code must still call model.load_state_dict(state, strict=True).
    The caller creates entirely new optimizer/scheduler/scaler/history/RNG state.
    """
    reference = verified_weight_file(path)
    if loader is None:
        import torch
        loader = torch.load
    payload = loader(Path(path), map_location='cpu', weights_only=True)
    return validate_initialization_payload(payload, target_identity, reference['sha256'], reference['size'])
