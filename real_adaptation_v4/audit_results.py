"""Independent accounting/provenance audit for a compact V4 Colab results ZIP.

This reads inference weights and reports. It cannot certify the existence or
contents of Drive full-state checkpoints that are not included in the archive.
No training state, implementation, dataset, or checkpoint is modified.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys
import zipfile

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

EXPECTED_ARCHITECTURE = 'mobilenetv3small_shared_encoder_dual_decoder_needle2_ticks10_v4'
EXPECTED_PREPROCESSING = 'gray512_day0_stabilized_fixed_lcd128_mask_v4_imagenet_normalized'
EXPECTED_KIND = 'joint_needle_ticks_inference_v4'
KEYPOINT_NAMES = ['pivot', 'needle_tip'] + [f'tick_{speed}' for speed in range(0, 181, 20)]
EXPECTED_AVAILABILITY = 'amodal_in_frame_landmark_availability'
EXPECTED_IMPLEMENTATIONS = {'real_adaptation_v4/train_landmarks.py', 'real_adaptation_v4/geometry.py',
    'real_adaptation_v4/stabilize_mask.py', 'real_adaptation_v3/train_needle.py', 'training_pilot/train_core.py'}
SOURCE_SHA256 = '8dcfa269848bac5a46c8ff39d14df4d3470e67905085fb0a662ee45b9bc99d10'
REQUIRED = {'best_model.pt', 'best_model.complete.json', 'training_report.json', 'history.json',
            'before_validation_metrics.json', 'after_validation_metrics.json',
            'after_validation_predictions.json', 'dataset_integrity.json',
            'best_checkpoint.json', 'latest_checkpoint.json'}
ALLOWED = REQUIRED | {'training_history.png', 'drive_checkpoint_audit.json'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=_reject_json_constant)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, atol=1e-5):
    if expected is None:
        require(actual is None, label + ' must be null')
        return
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool), label + ' is not numeric')
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), label + ' is nonfinite')
    require(math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=atol), f'{label} mismatch: {actual} versus {expected}')


def compare_json(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label + ' keys differ')
        for key, value in expected.items():
            compare_json(actual[key], value, label + '.' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ' list length differs')
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_json(left, right, f'{label}[{index}]')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        close(actual, expected, label)
    else:
        require(actual == expected, label + ' differs')


def extract_compact_zip(archive_path, output_dir, expected_sha256=None):
    archive_path, output_dir = Path(archive_path), Path(output_dir)
    digest = sha256(archive_path)
    if expected_sha256 is not None:
        require(digest == expected_sha256, 'Results ZIP SHA256 differs from expected download')
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        require(len(names) == len(set(name.casefold() for name in names)), 'Duplicate ZIP member names')
        require(REQUIRED <= set(names) <= ALLOWED, 'Compact ZIP has missing or unexpected members')
        require(sum(item.file_size for item in members) < 1024 ** 3, 'Compact ZIP exceeds expected size')
        for item in members:
            require(item.filename in ALLOWED and not item.is_dir() and not item.flag_bits & 1
                    and not stat.S_ISLNK((item.external_attr >> 16) & 0xFFFF), 'Unsafe ZIP member')
        require(archive.testzip() is None, 'ZIP CRC check failed')
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in members:
            data = archive.read(item)
            target = output_dir / item.filename
            if target.exists():
                require(target.is_file() and hashlib.sha256(data).hexdigest() == sha256(target),
                        'Refusing to overwrite a different extracted file: ' + target.name)
            else:
                target.write_bytes(data)
    return digest


def error_summary(errors):
    values = np.asarray(errors, dtype=float)
    if not len(values):
        return {'count': 0, 'mae_kmh': None, 'rmse_kmh': None, 'p95_kmh': None,
                'within_2_kmh_percent': None, 'within_5_kmh_percent': None}
    return {'count': len(values), 'mae_kmh': float(values.mean()),
            'rmse_kmh': float(np.sqrt(np.mean(values ** 2))), 'p95_kmh': float(np.percentile(values, 95)),
            'within_2_kmh_percent': float(100 * np.mean(values <= 2)),
            'within_5_kmh_percent': float(100 * np.mean(values <= 5))}


def check_metric_consistency(metrics, expected_count, penalty):
    primary = metrics['primary_speed']
    require(metrics['count'] == expected_count and primary['eligible_count'] == expected_count,
            'Metric counts differ from this all-speed-labelled validation partition')
    accepted, rejected = primary['accepted'], primary['rejected']
    require(type(accepted) is int and type(rejected) is int and min(accepted, rejected) >= 0
            and accepted + rejected == expected_count, 'Accepted/rejected counts do not cover validation')
    close(primary['coverage_percent'], 100 * accepted / expected_count, 'Coverage')
    require(sum(primary['rejection_reasons'].values()) == rejected, 'Rejection reason totals differ')
    require(all(type(value) is int and value >= 0 for value in primary['rejection_reasons'].values()), 'Invalid rejection reason count')
    summary = primary['accepted_only']
    require(summary['count'] == accepted, 'Accepted-only speed count differs')
    if not accepted:
        compare_json(summary, error_summary([]), 'Empty accepted-only summary')
        close(primary['within_2_kmh_all_eligible_inputs_percent'], 0., 'All-input within2')
    else:
        for key in ('within_2_kmh_percent', 'within_5_kmh_percent'):
            require(0 <= summary[key] <= 100, 'Invalid accepted-only percentage')
        close(primary['within_2_kmh_all_eligible_inputs_percent'],
              summary['within_2_kmh_percent'] * accepted / expected_count, 'All-input within2 denominator')
    for key in ('needle_mean_error_px', 'tick_mean_error_px', 'combined_loss'):
        require(math.isfinite(metrics[key]) and metrics[key] >= 0, 'Invalid ' + key)
    score = metrics['needle_mean_error_px'] + metrics['tick_mean_error_px'] + penalty * (1 - primary['within_2_kmh_all_eligible_inputs_percent'] / 100)
    close(metrics['selection_score'], score, 'Selection score')
    require(metrics['uses_target_transform_for_speed'] is False and metrics['uses_target_ticks_for_speed'] is False,
            'Evaluation declares target-derived speed inputs')


def audit_prediction_accounting(predictions, records, metrics, calibration, penalty):
    from real_adaptation_v4.geometry import read_speed
    by_id = {row['sample_id']: row for row in records}
    require(len(by_id) == len(records), 'Duplicate validation labels')
    require(len(predictions) == len(records), 'Prediction row count differs')
    require({row['sample_id'] for row in predictions} == set(by_id), 'Prediction IDs do not cover validation')
    require(len({row['sample_id'] for row in predictions}) == len(predictions), 'Duplicate prediction IDs')
    errors, point_errors, reasons, clamped = [], [], Counter(), 0
    for row in predictions:
        target = by_id[row['sample_id']]
        require(target.get('speed_valid', True) and all(target['visibility']), 'Audit expects all available amodal targets')
        points, availability = np.asarray(row['predicted_points_xy']), np.asarray(row['predicted_visibility'])
        require(points.shape == (12, 2) and availability.shape == (12,) and np.isfinite(points).all()
                and np.isfinite(availability).all(), 'Malformed/nonfinite predicted landmarks')
        require(((points >= -1e-5) & (points <= 511.00001)).all()
                and ((availability >= 0) & (availability <= 1)).all(), 'Predictions outside normalized output range')
        close(row['target_speed_kmh'], target['speed_kmh'], 'Prediction target speed')
        per_point = np.linalg.norm(points - np.asarray(target['points_xy']), axis=1)
        require(np.allclose(per_point, row['point_errors_px'], rtol=1e-6, atol=1e-5), 'Stored landmark errors differ from labels')
        point_errors.append(per_point)
        # Replay consumes predicted points, predicted availability and static
        # calibration only. Rendering transforms/target tick points are absent.
        replay = read_speed(points, visibility=availability, calibration=calibration)
        stored = row['reading']
        require(type(stored['accepted']) is bool and stored['accepted'] == replay['accepted'], 'Stored acceptance differs from geometry replay')
        require(stored['reason'] == replay['reason'], 'Stored rejection reason differs from geometry replay')
        close(stored['speed_kmh'], replay['speed_kmh'], 'Replayed indicated speed')
        require(stored['endpoint_clamped'] == replay['endpoint_clamped'], 'Endpoint clamp differs from replay')
        clamped += int(stored['endpoint_clamped'])
        if stored['accepted']:
            require(0 <= stored['speed_kmh'] <= 180, 'Accepted speed outside calibrated range')
            error = abs(stored['speed_kmh'] - target['speed_kmh'])
            close(row['speed_error_kmh'], error, 'Stored speed error')
            errors.append(error)
        else:
            require(stored['speed_kmh'] is None and row['speed_error_kmh'] is None, 'Rejected reading carries a speed/error')
            reasons[stored['reason']] += 1
    array = np.asarray(point_errors)
    close(metrics['needle_mean_error_px'], array[:, :2].mean(), 'Needle pixel error')
    close(metrics['needle_pivot_mean_error_px'], array[:, 0].mean(), 'Pivot pixel error')
    close(metrics['needle_tip_mean_error_px'], array[:, 1].mean(), 'Tip pixel error')
    close(metrics['needle_tip_p95_error_px'], np.percentile(array[:, 1], 95), 'Tip p95 pixel error')
    close(metrics['tick_mean_error_px'], array[:, 2:].mean(), 'Tick pixel error')
    close(metrics['tick_p95_error_px'], np.percentile(array[:, 2:], 95), 'Tick p95 pixel error')
    for index, name in enumerate(KEYPOINT_NAMES):
        close(metrics['per_landmark_mean_error_px'][name], array[:, index].mean(), name + ' pixel error')
    expected_primary = {'accepted': len(errors), 'rejected': len(predictions) - len(errors),
        'coverage_percent': 100 * len(errors) / len(predictions), 'eligible_count': len(predictions),
        'within_2_kmh_all_eligible_inputs_percent': 100 * sum(error <= 2 for error in errors) / len(predictions),
        'accepted_only': error_summary(errors), 'rejection_reasons': dict(reasons)}
    compare_json(metrics['primary_speed'], expected_primary, 'Recomputed primary speed metrics')
    check_metric_consistency(metrics, len(predictions), penalty)
    return {'predictions_checked': len(predictions), 'geometry_replayed': len(predictions),
            'accepted': len(errors), 'rejected': len(predictions) - len(errors), 'endpoint_clamped': clamped,
            'rejection_reasons': dict(reasons),
            'all_input_within_2_kmh_percent': expected_primary['within_2_kmh_all_eligible_inputs_percent']}


def check_drive_audit(drive_audit, report, latest_pointer):
    """Check the included remote audit report without pretending to reread Drive."""
    require(drive_audit['status'] == 'passed' and drive_audit['run_directory'] == report['run_dir'], 'Remote Drive audit run/status differs')
    require(drive_audit['completed_epoch'] == report['completed_epoch']
            and drive_audit['best_epoch'] == report['best_epoch'], 'Remote Drive audit epochs differ')
    files = drive_audit['epoch_files']
    expected_epochs = list(range(report['completed_epoch'] + 1))
    require([row['epoch'] for row in files] == expected_epochs
            and drive_audit['epoch_checkpoint_count'] == len(expected_epochs), 'Remote Drive audit does not cover every epoch')
    require(drive_audit['all_epoch_files_and_sizes_verified'] is True, 'Remote audit did not verify every file/size')
    for row in files:
        require(isinstance(row['file'], str) and row['file'].startswith(f"epoch_{row['epoch']:05d}_")
                and Path(row['file']).name == row['file'] and row['file'].endswith('.pt')
                and type(row['bytes']) is int and row['bytes'] > 0
                and re.fullmatch('[0-9a-f]{64}', row['sha256_recorded']) is not None, 'Malformed remote checkpoint inventory')
    checks = drive_audit['best_and_latest_full_state_checks']
    require(len(checks) == 2 and {row['role'] for row in checks} == {'best', 'latest'}, 'Remote audit lacks best/latest full-state checks')
    for row in checks:
        epoch = report['best_epoch'] if row['role'] == 'best' else report['completed_epoch']
        pointer = report['best_checkpoint_path'] if row['role'] == 'best' else report['latest_checkpoint_path']
        inventory = files[epoch]
        require(row['epoch'] == epoch and row['file'] == inventory['file'] == Path(pointer).name
                and row['sha256'] == inventory['sha256_recorded'] and row['full_state_verified'] is True
                and type(row['optimizer_state_entries']) is int and row['optimizer_state_entries'] > 0,
                'Remote best/latest full-state verification differs')
    require(files[-1]['sha256_recorded'] == latest_pointer['sha256'], 'Remote latest hash differs from completed checkpoint marker')
    require(drive_audit['best_export_matches_selected_full_checkpoint_exactly'] is True, 'Remote audit did not confirm inference export matches best full state')
    return {'included': True, 'report_consistency_verified_locally': True,
            'epoch_checkpoint_count': len(files), 'epochs': expected_epochs,
            'all_epoch_files_markers_and_sizes_checked_remotely': True,
            'all_epoch_hashes_recomputed_remotely': drive_audit['all_epoch_hashes_recomputed_in_this_audit'],
            'best_and_latest_full_states_verified_remotely': checks,
            'export_matches_best_full_state_remotely': True,
            'local_auditor_read_full_state_file_bytes': False}


def check_geometry_runtime(identity):
    require(cv2.__version__ == identity.get('opencv_version'),
            f"OpenCV runtime differs: local {cv2.__version__}, training {identity.get('opencv_version')}. "
            'Use the pinned inference runtime before replay; geometry gates must not be weakened.')
    return {'opencv_local': cv2.__version__, 'opencv_training': identity['opencv_version'],
            'numpy_local': np.__version__, 'numpy_training': identity['numpy_version'],
            'opencv_matches': True, 'numpy_matches': np.__version__ == identity['numpy_version']}


def audit_results(archive_path, output_dir, data_dir, expected_sha256=None):
    import torch
    from real_adaptation_v4.train_landmarks import LandmarkModel
    from real_adaptation_v4.geometry import load_calibration
    torch.set_num_threads(2)
    archive_path, output_dir, data_dir = Path(archive_path), Path(output_dir), Path(data_dir).resolve()
    archive_sha = extract_compact_zip(archive_path, output_dir, expected_sha256)
    report = read_json(output_dir / 'training_report.json')
    before = read_json(output_dir / 'before_validation_metrics.json')
    after = read_json(output_dir / 'after_validation_metrics.json')
    history = read_json(output_dir / 'history.json')
    predictions = read_json(output_dir / 'after_validation_predictions.json')
    integrity = read_json(output_dir / 'dataset_integrity.json')
    marker = read_json(output_dir / 'best_model.complete.json')
    weights_path = output_dir / 'best_model.pt'
    require(marker['checkpoint'] == weights_path.name and marker['sha256'] == sha256(weights_path)
            and marker['bytes'] == weights_path.stat().st_size, 'Inference checkpoint checksum/size marker differs')
    require(marker['checkpoint_kind'] == EXPECTED_KIND and marker['format_version'] == 1, 'Wrong checkpoint marker family')
    payload = torch.load(weights_path, map_location='cpu', weights_only=True)
    require(payload.get('checkpoint_kind') == EXPECTED_KIND and payload.get('format_version') == 1, 'Wrong checkpoint payload family')
    require(payload.get('architecture') == EXPECTED_ARCHITECTURE and payload.get('preprocessing') == EXPECTED_PREPROCESSING,
            'Unexpected model architecture/preprocessing')
    require(payload.get('keypoint_names') == KEYPOINT_NAMES and payload.get('landmark_availability_semantics') == EXPECTED_AVAILABILITY,
            'Wrong landmark order/availability semantics')
    require(not {'optimizer_state_dict', 'scaler_state_dict', 'rng_state', 'training_state'} & payload.keys(),
            'Compact model is not the expected inference-only export')
    model = LandmarkModel()
    model.load_state_dict(payload['model_state_dict'], strict=True)
    require(all(torch.isfinite(tensor).all().item() for tensor in payload['model_state_dict'].values()), 'Nonfinite model weights')
    identity, config = payload['identity'], payload['config']
    runtime = check_geometry_runtime(identity)
    require(identity['architecture'] == EXPECTED_ARCHITECTURE and identity['preprocessing'] == EXPECTED_PREPROCESSING,
            'Identity architecture/preprocessing differs')
    compare_json(identity['dataset'], integrity, 'Payload dataset identity')
    compare_json(report['dataset_integrity'], integrity, 'Report dataset identity')
    for key, relative in (('manifest_sha256', 'manifest.json'), ('annotation_sha256', 'annotations.json'), ('calibration_sha256', 'calibration.json')):
        require(integrity[key] == sha256(data_dir / relative), 'Checkpoint dataset digest differs: ' + relative)
    manifest = read_json(data_dir / 'manifest.json')
    annotations = read_json(data_dir / 'annotations.json')
    preprocessing = read_json(data_dir / 'preprocessing_config.json')
    compare_json(integrity['preprocessing'], preprocessing, 'Preprocessing identity')
    compare_json(manifest['preprocessing'], preprocessing, 'Manifest preprocessing')
    require(integrity['landmark_supervision'] == EXPECTED_AVAILABILITY, 'Wrong dataset supervision')
    for path_key, hash_key in (('fixed_mask_path', 'fixed_mask_sha256'), ('reference_image_path', 'reference_sha256')):
        asset = (data_dir / preprocessing[path_key]).resolve()
        require(asset.is_relative_to(data_dir) and sha256(asset) == preprocessing[hash_key], 'Preprocessing asset differs')
    require(preprocessing['canonical_calibration_sha256'] == integrity['calibration_sha256'], 'Static calibration digest differs')
    require(set(identity['implementation_hashes']) == EXPECTED_IMPLEMENTATIONS, 'Implementation identity is incomplete/unexpected')
    for relative, digest in identity['implementation_hashes'].items():
        implementation = (PROJECT / relative).resolve()
        require(implementation.is_relative_to(PROJECT) and sha256(implementation) == digest, 'Training implementation differs: ' + relative)
    require(preprocessing['stabilize_mask_script_sha256'] == sha256(PROJECT / 'real_adaptation_v4/stabilize_mask.py'),
            'Stabilization implementation differs')
    settings = {key: value for key, value in config.items() if key not in ('data_dir', 'output_dir', 'device')}
    compare_json(identity['config'], settings, 'Configuration identity')
    records = [row for row in annotations['samples'] if row['split'] == 'validation']
    require(len(records) == integrity['sample_counts']['validation'], 'Validation partition count differs')
    require(annotations['keypoint_names'] == KEYPOINT_NAMES, 'Dataset landmark order differs')
    compare_json(report['before_metrics'], before, 'Before metrics file/report')
    compare_json(report['after_metrics'], after, 'After metrics file/report')
    compare_json(payload['metrics'], after, 'Best checkpoint/after metrics')
    for metrics in [before, after, *[row['validation_metrics'] for row in history]]:
        check_metric_consistency(metrics, len(records), config['geometry_penalty_px'])
    completed, target = report['completed_epoch'], report['target_epoch']
    require(type(completed) is int and type(target) is int and 0 <= completed <= target == config['epochs'], 'Invalid completed/target epoch')
    require([row['epoch'] for row in history] == list(range(1, completed + 1)), 'History has missing or duplicated epochs')
    scored = [(0, before)] + [(row['epoch'], row['validation_metrics']) for row in history]
    selected_epoch, selected_metrics = min(scored, key=lambda row: (row[1]['selection_score'], row[0]))
    require(report['best_epoch'] == payload['best_epoch'] == selected_epoch, 'Best epoch is not the earliest minimum selection score')
    compare_json(selected_metrics, after, 'Selected history/after metrics')
    require(report['training_complete'] == (completed == target or report['stop_reason'] == 'early_stopping'), 'Completion/early-stop state inconsistent')
    best_pointer, latest_pointer = read_json(output_dir / 'best_checkpoint.json'), read_json(output_dir / 'latest_checkpoint.json')
    require(best_pointer['epoch'] == selected_epoch and best_pointer['path'] == report['best_checkpoint_path'], 'Best Drive pointer differs')
    close(best_pointer['selection_loss'], selected_metrics['selection_score'], 'Best Drive pointer selection score')
    require(latest_pointer['epoch'] == completed and latest_pointer['path'] == report['latest_checkpoint_path'], 'Latest Drive pointer differs')
    compare_json(latest_pointer['identity'], identity, 'Latest Drive pointer identity')
    require(latest_pointer['checkpoint_kind'] == 'joint_needle_ticks_full_state_v4', 'Latest pointer does not name full state')
    close(latest_pointer['selection_loss'], (history[-1]['validation_metrics'] if history else before)['selection_score'], 'Latest Drive pointer metric')
    compare_json(report['initialization'], payload['initialization'], 'Initialization provenance')
    initialization = payload['initialization']
    if initialization['mode'] == 'transfer':
        require(initialization['source_sha256'] == SOURCE_SHA256 and initialization['source_epoch'] == 8
                and initialization['optimizer_restored'] is False, 'Wrong V3 transfer provenance')
    resume = report['resume_metadata']
    is_resume = resume['requested_mode'] in ('resume_latest', 'resume_best')
    require(resume['full_state_restored'] == is_resume and resume['optimizer_restored'] == is_resume, 'Resume claim inconsistent with requested mode')
    require(report['epochs_completed_this_call'] == completed - (resume['source_epoch'] if is_resume else 0), 'Epoch continuation count differs')
    accounting = audit_prediction_accounting(predictions, records, after, load_calibration(data_dir / 'calibration.json'), config['geometry_penalty_px'])
    drive_audit_path = output_dir / 'drive_checkpoint_audit.json'
    drive_audit = (check_drive_audit(read_json(drive_audit_path), report, latest_pointer)
                   if drive_audit_path.is_file() else {'included': False})
    audit = {'status': 'passed', 'archive_path': str(archive_path.resolve()), 'archive_sha256': archive_sha,
             'external_archive_checksum_verified': expected_sha256 is not None,
             'weights_sha256': marker['sha256'], 'dataset_manifest_sha256': integrity['manifest_sha256'],
             'completed_epoch': completed, 'best_epoch': selected_epoch, 'target_epoch': target,
             'training_complete': report['training_complete'], 'before_metrics': before, 'after_metrics': after,
             'prediction_accounting': accounting, 'model_tensor_count': len(payload['model_state_dict']),
             'dataset_preprocessing_and_implementation_identity_verified': True,
             'geometry_replay_runtime': runtime,
             'all_dataset_image_bytes_rehashed': False,
             'inference_export_contains_optimizer_state': False,
             'full_state_drive_checkpoint_contents_verified_locally': False,
             'full_state_note': 'Drive full-state files were not included in this compact ZIP. Their bytes were not reread by this local auditor; any separate remote best/latest verification is recorded in included_remote_drive_audit.',
             'included_remote_drive_audit': drive_audit,
             'real_accuracy_established': False,
             'limitations': ['Same-recording synthetic development validation.',
                 'Geometry replay verifies prediction accounting; a separate real-video inference review is still required.',
                 'This audit verifies metadata, tensor compatibility and prediction records, not a new forward pass of all images.']}
    (output_dir / 'audit_results.json').write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True, dest='archive')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--expected-zip-sha256')
    args = parser.parse_args()
    result = audit_results(args.archive, args.output_dir, args.data_dir, args.expected_zip_sha256)
    print(json.dumps(result, indent=2))
