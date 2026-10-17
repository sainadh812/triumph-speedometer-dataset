"""Run in the completed Colab notebook, where REPORT and RUN are available."""
import json
from pathlib import Path
import torch
from real_adaptation_v3.train_needle import _read_complete
from real_adaptation_v4.train_landmarks import validate_payload

run_directory = Path(REPORT['run_dir'])
export_payload, _ = _read_complete(run_directory/'best_model.pt')
entries = []
for marker_file in sorted(run_directory.parent.rglob('epoch_*.complete.json')):
    metadata = json.loads(marker_file.read_text())
    if metadata.get('identity') != export_payload['identity']:
        continue
    checkpoint_file = marker_file.with_name(marker_file.name.replace('.complete.json', '.pt'))
    valid_size = checkpoint_file.is_file() and checkpoint_file.stat().st_size == metadata['bytes']
    entries.append({'epoch': metadata['epoch'], 'file': str(checkpoint_file.relative_to(run_directory.parent)),
                    'file_and_size_valid': valid_size,
                    'bytes': metadata['bytes'], 'sha256_recorded': metadata['sha256']})
available_epochs = {item['epoch'] for item in entries if item['file_and_size_valid']}
missing_epochs = sorted(set(range(REPORT['completed_epoch'] + 1)) - available_epochs)
all_sizes_valid = all(item['file_and_size_valid'] for item in entries)
verified_states = []
for role in ('best', 'latest'):
    checkpoint_file = Path(REPORT[role + '_checkpoint_path'])
    payload, marker = _read_complete(checkpoint_file)
    validate_payload(payload)
    assert payload['identity'] == export_payload['identity'], role + ' checkpoint identity differs from exported model'
    expected_epoch = REPORT['best_epoch'] if role == 'best' else REPORT['completed_epoch']
    assert payload['training_state']['epoch'] == expected_epoch
    if role == 'best':
        assert set(payload['model_state_dict']) == set(export_payload['model_state_dict'])
        assert all(torch.equal(value, export_payload['model_state_dict'][name])
                   for name, value in payload['model_state_dict'].items())
    verified_states.append({'role': role, 'file': checkpoint_file.name,
        'epoch': payload['training_state']['epoch'], 'sha256': marker['sha256'],
        'optimizer_state_entries': len(payload['optimizer_state_dict']['state']),
        'full_state_verified': True})
checkpoint_audit = {'status': 'passed' if all_sizes_valid and not missing_epochs else 'issues_found', 'run_directory': str(run_directory),
    'completed_epoch': REPORT['completed_epoch'], 'best_epoch': REPORT['best_epoch'],
    'epoch_checkpoint_count': len(entries), 'all_epoch_files_and_sizes_verified': all_sizes_valid,
    'missing_epoch_numbers': missing_epochs, 'compatible_runs_included': True,
    'all_epoch_hashes_recomputed_in_this_audit': False, 'epoch_files': entries,
    'best_export_matches_selected_full_checkpoint_exactly': True,
    'best_and_latest_full_state_checks': verified_states,
    'scope': 'Compatible run lineage epoch files/markers/sizes; best and latest hashes and complete resumable payloads.'}
audit_path = run_directory/'drive_checkpoint_audit.json'
audit_path.write_text(json.dumps(checkpoint_audit, indent=2))
print(json.dumps(checkpoint_audit, indent=2))
from google.colab import files
files.download(str(audit_path))
