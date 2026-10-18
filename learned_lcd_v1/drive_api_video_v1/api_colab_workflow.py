"""Private Drive API storage around the unchanged Colab video diagnostic.

Importing this module performs no network requests, credential reads or ML work.
Only the explicitly called GPU function invokes the existing evaluator.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
import uuid
import zipfile

from drive_api_store import DriveStore, DriveStoreError

WORKSPACE_NAME = 'SpeedometerTraining_API_v1'
ASSET_BUNDLE_NAME = 'verified_inference_assets_v1.zip'
ASSET_BUNDLE_SHA256 = '7db028c93ba38c653ee1b4a12603e04f462e2494f834527a2951e6726ddf5174'
ASSET_BUNDLE_BYTES = 39839268
CONTENT_ROOT = Path('/content')
PENDING_PATH = CONTENT_ROOT / 'triumph_drive_api_pending.json'


def helpers(runtime_dir):
    path = Path(runtime_dir) / 'colab_video_helpers.py'
    spec = importlib.util.spec_from_file_location('frozen_video_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def connect(secret_name='TRIUMPH_DRIVE_OAUTH'):
    store = DriveStore.from_colab_secret(secret_name)
    root = store.ensure_folder(WORKSPACE_NAME)
    return store, root


def bundle_metadata(store, root):
    assets = store.ensure_folder('assets', root['id'])
    item = store.find_file(ASSET_BUNDLE_NAME, assets['id'])
    if item is None:
        raise RuntimeError('Verified model assets have not been migrated to this app workspace.')
    return item


def restore_assets(store, root, destination, h):
    item = bundle_metadata(store, root)
    archive = destination.parent / ASSET_BUNDLE_NAME
    store.download_verified(item['id'], archive, ASSET_BUNDLE_SHA256, ASSET_BUNDLE_BYTES)
    with zipfile.ZipFile(archive) as bundle:
        infos = h.checked_zip_files(bundle)
        manifest_name = 'asset_manifest.json'
        if manifest_name not in infos or infos[manifest_name].file_size > 1_000_000:
            raise ValueError('Asset manifest is missing or oversized.')
        data = bundle.read(manifest_name)
        manifest = json.loads(data)
        rows = manifest['files']
        declared = {row['path']: row for row in rows}
        if manifest.get('schema_version') != 1 or len(rows) != len(declared):
            raise ValueError('Invalid asset manifest.')
        if set(infos) != set(declared) | {manifest_name}:
            raise ValueError('Asset archive inventory mismatch.')
        hashes = {name: row['sha256'] for name, row in declared.items()}
        hashes[manifest_name] = hashlib.sha256(data).hexdigest()
        sizes = {name: row['bytes'] for name, row in declared.items()}
        h.extract_files(bundle, infos, destination, hashes, sizes)
    return manifest


def preflight(runtime_dir, secret_name='TRIUMPH_DRIVE_OAUTH'):
    """CPU-only authentication and asset metadata check, without model imports."""
    h = helpers(runtime_dir)
    h.require_cpu()
    store, root = connect(secret_name)
    item = bundle_metadata(store, root)
    # Uploading and checksum verification were completed by the migration tool.
    if int(item.get('size', -1)) != ASSET_BUNDLE_BYTES:
        raise ValueError('Stored model package size differs from this release.')
    if item.get('sha256Checksum') and item['sha256Checksum'] != ASSET_BUNDLE_SHA256:
        raise ValueError('Stored model package checksum differs from this release.')
    print('Drive API authorization works; verified model package is available. No GPU used.')
    return {'workspace_id': root['id'], 'asset_id': item['id'], 'model_inference_performed': False}


def upload_video(runtime_dir, start_seconds=0.0, max_seconds=0.0,
                 secret_name='TRIUMPH_DRIVE_OAUTH'):
    h = helpers(runtime_dir)
    h.require_cpu()
    if PENDING_PATH.exists():
        raise RuntimeError('A runtime result still needs saving. Run retry_pending() before another upload.')
    for value in (start_seconds, max_seconds):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Start and duration must be finite nonnegative numbers; duration 0 means the remainder.')
    store, root = connect(secret_name)
    bundle_metadata(store, root)
    from google.colab import files
    local = CONTENT_ROOT / h.unique_id('triumph_browser_upload')
    local.mkdir(exist_ok=False)
    previous = Path.cwd()
    try:
        os.chdir(local)
        uploaded = files.upload()
    finally:
        os.chdir(previous)
    if len(uploaded) != 1:
        raise ValueError('Select exactly one video.')
    original_name, payload = next(iter(uploaded.items()))
    if not payload:
        raise ValueError('The uploaded video is empty.')
    extension = Path(original_name).suffix.lower()
    if not extension or len(extension) > 12 or not extension[1:].isalnum():
        extension = '.video'
    source = local / ('source' + extension)
    source.write_bytes(payload)
    del payload, uploaded
    jobs = store.ensure_folder('jobs', root['id'])
    job_id = h.unique_id('video')
    folder = store.ensure_folder(job_id, jobs['id'])
    remote = store.upload_verified(source, folder['id'])
    job = {'schema_version': 1, 'job_id': job_id, 'created_at': h.utc_now(),
           'asset_bundle_sha256': ASSET_BUNDLE_SHA256,
           'start_seconds': float(start_seconds), 'max_seconds': float(max_seconds),
           'source': {'file_id': remote['id'], 'original_name': str(original_name),
                      'filename': source.name, 'bytes': source.stat().st_size, 'sha256': h.sha256(source)},
           'training_performed': False}
    store.upload_json(job, folder['id'], 'job.json')  # Completion marker is last.
    print('Input video is saved and checksum-verified in private Drive.')
    print('Job:', job_id, '| folder ID:', folder['id'])
    print('Switch to T4 GPU and run cell 2. No new Drive sign-in or video upload is needed.')
    return job


def latest_job(store, root, folder_id=''):
    jobs = store.ensure_folder('jobs', root['id'])
    folders = [item for item in store.list_files(jobs['id'])
               if item['mimeType'] == 'application/vnd.google-apps.folder']
    if folder_id:
        folders = [item for item in folders if item['id'] == folder_id]
    folders.sort(key=lambda row: (row.get('createdTime', ''), row['name']), reverse=True)
    for folder in folders:
        item = store.find_file('job.json', folder['id'])
        if item is None:
            continue  # An interrupted source upload is not a submitted job.
        job = store.read_json(item['id'])
        if (job.get('schema_version') != 1 or job.get('job_id') != folder['name'] or
                job.get('asset_bundle_sha256') != ASSET_BUNDLE_SHA256):
            raise ValueError('Uploaded job identity or model package mismatch.')
        source = job['source']
        if Path(source['filename']).name != source['filename'] or '/' in source['filename'] or '\\' in source['filename']:
            raise ValueError('Unsafe source filename.')
        metadata = store.metadata(source['file_id'])
        if (metadata.get('name') != source['filename'] or folder['id'] not in metadata.get('parents', [])
                or int(metadata.get('size', -1)) != source['bytes']
                or (metadata.get('sha256Checksum') and metadata['sha256Checksum'] != source['sha256'])):
            raise ValueError('Saved source file does not match this uploaded job.')
        for value in (job['start_seconds'], job['max_seconds']):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid saved video interval.')
        return job, folder
    raise RuntimeError('No completed input upload was found. Run cell 1 on CPU first.')


def show_result(receipt):
    from IPython.display import HTML, display
    print('Drive API backup verified:', receipt['status'])
    if receipt.get('output_file_id'):
        # IDs come from the authenticated Drive response; never include OAuth tokens in URLs.
        url = 'https://drive.google.com/file/d/' + receipt['output_file_id'] + '/view'
        display(HTML('<a target="_blank" rel="noopener" href="' + url + '">Open or download labeled video</a>'))
        print('MP4 bytes:', receipt['output_bytes'], '| SHA256:', receipt['output_sha256'])
    else:
        print('Partial diagnostics saved; no completed video is claimed.')
    for key, value in receipt.get('timings', {}).items():
        print(key + ':', value)


def optional_show_result(receipt):
    """Rendering a link is optional after checksum-verified preservation."""
    try:
        show_result(receipt)
    except Exception:
        print('Drive result is verified; the optional display failed.', flush=True)
        print('Run folder ID:', receipt['run_folder_id'], flush=True)
        if receipt.get('output_file_id'):
            print('Video file ID:', receipt['output_file_id'], flush=True)


def job_identity(job, folder_id):
    return {'job_id': job['job_id'], 'job_folder_id': folder_id,
            'asset_bundle_sha256': job['asset_bundle_sha256'],
            'source_sha256': job['source']['sha256'], 'source_bytes': job['source']['bytes'],
            'source_file_id': job['source']['file_id'],
            'start_seconds': job['start_seconds'], 'max_seconds': job['max_seconds']}


def validate_identity(document, job, folder_id):
    if any(document.get(key) != value for key, value in job_identity(job, folder_id).items()):
        raise ValueError('Result identity does not match the source, interval and model assets of this job.')


def validate_completed_receipt(store, receipt, job, folder):
    validate_identity(receipt, job, folder['id'])
    if receipt.get('status') != 'complete_and_verified' or not receipt.get('run_id'):
        raise ValueError('Saved completion receipt is invalid.')
    matches = [row for row in receipt.get('files', []) if row.get('path') == receipt.get('output_relative')]
    if (len(matches) != 1 or matches[0].get('file_id') != receipt.get('output_file_id')
            or matches[0].get('sha256') != receipt.get('output_sha256')
            or matches[0].get('bytes') != receipt.get('output_bytes')):
        raise ValueError('Saved video does not match the immutable result inventory.')
    metadata = store.metadata(receipt['output_file_id'])
    if int(metadata.get('size', -1)) != receipt['output_bytes']:
        raise ValueError('Saved video byte count has changed.')
    if metadata.get('sha256Checksum'):
        if metadata['sha256Checksum'] != receipt['output_sha256']:
            raise ValueError('Saved video SHA256 has changed.')
    else:
        # This path verifies bytes only; it never loads a model or repeats inference.
        with tempfile.TemporaryDirectory(prefix='verified_saved_video_') as temporary:
            store.download_verified(receipt['output_file_id'], Path(temporary) / 'output.mp4',
                                    receipt['output_sha256'], receipt['output_bytes'])
    return receipt


def result_timings(summary):
    keys = ('source_frames_processed', 'reader_processing_fps',
            'decode_inference_render_encode_fps', 'elapsed_seconds', 'timing_scope')
    result = {key: summary[key] for key in keys if key in summary}
    phases = summary.get('phase_seconds', {})
    for key in ('reader_wall_seconds', 'render_seconds', 'decode_wait_seconds', 'encoder_pipe_seconds'):
        if key in phases:
            result[key] = phases[key]
    network = summary.get('network_timing', {})
    forwards = [row.get('forward_wall_seconds', 0.) for row in network.values()]
    if forwards and all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in forwards) and sum(forwards) > 0:
        result['combined_network_forward_only_fps'] = summary['source_frames_processed'] / sum(forwards)
    return result


def active_path():
    return PENDING_PATH.with_name('triumph_drive_api_active.json')


def save_pending(store, pending, h):
    """Upload every result first, publish the final receipt last. No inference."""
    directory = Path(pending['directory'])
    if not directory.is_dir():
        raise RuntimeError('Pending runtime artifacts are missing.')
    if pending['asset_bundle_sha256'] != ASSET_BUNDLE_SHA256:
        raise ValueError('Pending results use different model assets.')
    if pending['status'] not in ('complete_and_verified', 'failed_partial_outputs_unverified'):
        raise ValueError('Unknown pending publication status.')
    if pending['status'] == 'complete_and_verified':
        output_path = (directory / pending['output_relative']).resolve()
        if not output_path.is_relative_to(directory.resolve()):
            raise ValueError('Completed video escaped its result directory.')
        h.verify(output_path, pending['output_sha256'], pending['output_bytes'])
    runs = store.ensure_folder('runs', pending['job_folder_id'])
    run = store.ensure_folder(pending['run_id'], runs['id'])
    folders = {'': run['id']}
    rows, uploaded = [], {}
    files = sorted(path for path in directory.rglob('*') if path.is_file())
    # Legacy local receipts do not become a remote completion signal until all uploads verify.
    files.sort(key=lambda p: (p.name in ('result_manifest.json', 'publication_receipt.json'), p.as_posix()))
    for path in files:
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Pending result contains an escaping symbolic link.')
        relative = path.relative_to(directory)
        parent = ''
        for component in relative.parts[:-1]:
            next_parent = parent + '/' + component if parent else component
            if next_parent not in folders:
                folders[next_parent] = store.ensure_folder(component, folders[parent])['id']
            parent = next_parent
        metadata = store.upload_verified(path, folders[parent])
        uploaded[relative.as_posix()] = metadata
        rows.append({'path': relative.as_posix(), 'file_id': metadata['id'],
                     'bytes': path.stat().st_size, 'sha256': h.sha256(path)})
    receipt = {'schema_version': 1, 'status': pending['status'], 'job_id': pending['job_id'],
               'run_id': pending['run_id'], 'run_folder_id': run['id'], 'files': rows,
               'verified_at': h.utc_now(), 'training_performed': False,
               'raw_readings_unvalidated': True, 'timings': pending.get('timings', {})}
    for key in ('job_folder_id', 'asset_bundle_sha256', 'source_sha256', 'source_bytes',
                'source_file_id', 'start_seconds', 'max_seconds'):
        receipt[key] = pending[key]
    if pending['status'] == 'complete_and_verified':
        output = pending['output_relative']
        if output not in uploaded:
            raise ValueError('Completed video is absent from uploaded files.')
        receipt.update(output_relative=output, output_file_id=uploaded[output]['id'], output_bytes=pending['output_bytes'],
                       output_sha256=pending['output_sha256'])
    name = 'result.json' if pending['status'] == 'complete_and_verified' else pending['run_id'] + '_failed.json'
    # The stable receipt supports recovering after runtime replacement, without overwriting a prior run.
    existing = store.find_file(name, pending['job_folder_id'])
    if existing is not None:
        prior = store.read_json(existing['id'])
        if any(prior.get(key) != value for key, value in receipt.items() if key != 'verified_at'):
            raise ValueError('Existing final receipt does not match this run.')
        receipt = prior
    else:
        store.upload_json(receipt, pending['job_folder_id'], name)
    if PENDING_PATH.exists():
        PENDING_PATH.unlink()  # Only the runtime-local recovery pointer, after verified preservation.
    active_path().unlink(missing_ok=True)
    optional_show_result(receipt)
    return receipt


def retry_pending(runtime_dir, secret_name='TRIUMPH_DRIVE_OAUTH', auto_disconnect=True):
    h = helpers(runtime_dir)
    store, root = connect(secret_name)
    pending = json.loads(PENDING_PATH.read_text())
    job, folder = latest_job(store, root, pending['job_folder_id'])
    validate_identity(pending, job, folder['id'])
    receipt = save_pending(store, pending, h)
    if auto_disconnect and h.gpu_present():
        h.disconnect_runtime()
    return receipt


def run_video(runtime_dir, secret_name='TRIUMPH_DRIVE_OAUTH', job_folder_id='', auto_disconnect=True):
    """One Colab run. Frozen weights, full labels, and existing batch sizes remain unchanged."""
    h = helpers(runtime_dir)
    work = None
    saved = False
    had_gpu = h.gpu_present()
    try:
        h.require_gpu()
        if PENDING_PATH.exists():
            print('Recovering an existing result upload; model inference will not run again.')
            receipt = retry_pending(runtime_dir, secret_name, auto_disconnect=False)
            saved = True
            return receipt
        if active_path().exists():
            record = json.loads(active_path().read_text())
            raise RuntimeError('A previous run has local artifacts requiring manual recovery at ' +
                               str(record.get('work', '(recorded active run)')) +
                               '. Inference will not be repeated. No upload-retry pointer exists yet.')
        store, root = connect(secret_name)
        uploaded_job, folder = latest_job(store, root, job_folder_id)
        done = store.find_file('result.json', folder['id'])
        if done:
            receipt = store.read_json(done['id'])
            validate_completed_receipt(store, receipt, uploaded_job, folder)
            saved = True
            optional_show_result(receipt)
            return receipt
        work = CONTENT_ROOT / h.unique_id('triumph_api_inference')
        work.mkdir(exist_ok=False)
        h.DRIVE_ROOT = work / 'asset_and_job_staging'  # Local only: this is never described as a Drive backup.
        restore_assets(store, root, h.DRIVE_ROOT, h)
        run_id = h.unique_id('run')
        source_info = uploaded_job['source']
        job_dir = h.DRIVE_ROOT / h.JOBS_REL / 'jobs' / uploaded_job['job_id']
        source = job_dir / 'source' / source_info['filename']
        store.download_verified(source_info['file_id'], source, source_info['sha256'], source_info['bytes'])
        job = dict(uploaded_job, status='uploaded', source={
            'relative_path': h.drive_relative(source), 'original_name': source_info['original_name'],
            'sha256': source_info['sha256'], 'bytes': source_info['bytes']})
        job_path = job_dir / 'job.json'
        h.save_job(job, job_path)
        h.write_json(active_path(), {'work': str(work), 'job_folder_id': folder['id'],
                                    'job_id': uploaded_job['job_id'], 'run_id': run_id})
        summary = None
        failure = None
        original_stdout, original_stderr = sys.stdout, sys.stderr
        with (work / 'inference.log').open('x', encoding='utf8') as log:
            sys.stdout, sys.stderr = h.Tee(original_stdout, log), h.Tee(original_stderr, log)
            try:
                repo, inputs, dataset, provenance = h.prepare_assets(work, Path(runtime_dir))
                video = h.copy_verified(source, inputs / source.name, source_info['sha256'], source_info['bytes'])
                evaluator = repo / 'evaluate_uploaded_video.py'
                command = [sys.executable, '-u', str(evaluator), '--repository-root', str(repo),
                    '--provenance-manifest', str(provenance), '--video', str(video),
                    '--expected-video-sha256', source_info['sha256'], '--output-root', str(work / 'evaluation'),
                    '--lcd-checkpoint', str(inputs / 'lcd/best_model.pt'),
                    '--speed-checkpoint', str(inputs / 'speed/best_model.pt'),
                    '--speed-calibration', str(repo / 'mixed_view_v5/calibration.json'),
                    '--rpm-dataset', str(dataset), '--rpm-checkpoint', str(inputs / 'rpm/best_model.pt'),
                    '--rpm-reference', str(repo / 'rpm_exploration/review_220558_frame_0169.png'),
                    '--start-seconds', str(job['start_seconds']), '--max-seconds', str(job['max_seconds']), '--device', 'cuda']
                h.write_json(work / 'command.json', command)
                h.run_logged(command, cwd=repo)
                summary = h.validate_results(work / 'evaluation', job, h.sha256(evaluator))
            except BaseException as error:
                traceback.print_exc()
                failure = {'type': type(error).__name__, 'message': str(error)}
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
        local_receipt = h.publish_run(work, job, job_path, run_id, summary=summary, error=failure)
        directory = h.safe_drive_path(local_receipt['drive_directory_relative'])
        pending = {'job_folder_id': folder['id'], 'job_id': job['job_id'], 'run_id': local_receipt['run_id'],
                   'directory': str(directory), 'status': local_receipt['status'],
                   **job_identity(uploaded_job, folder['id'])}
        if summary is not None:
            pending.update(output_relative='evaluation/' + Path(summary['output']).name,
                output_bytes=summary['output_bytes'], output_sha256=summary['output_sha256'],
                timings=result_timings(summary))
        h.write_json(PENDING_PATH, pending)
        receipt = save_pending(store, pending, h)
        saved = True
        if failure:
            raise RuntimeError('Inference failed; partial diagnostics were saved to Drive. No completed video is claimed.')
        return receipt
    finally:
        has_outputs = bool(work is not None and (work / 'evaluation').exists()
                           and any(p.is_file() for p in (work / 'evaluation').rglob('*')))
        if work is not None and (saved or not has_outputs) and not PENDING_PATH.exists():
            active_path().unlink(missing_ok=True)
        unsaved = PENDING_PATH.exists() or active_path().exists() or (has_outputs and not saved)
        if auto_disconnect and had_gpu and not unsaved:
            h.disconnect_runtime()
        elif auto_disconnect and had_gpu:
            print('Runtime retained: results have not yet been verified in Drive.')
            if PENDING_PATH.exists():
                print('Retry saving with workflow.retry_pending(RUNTIME_DIR); this does not rerun inference.')
            else:
                print('No upload-retry pointer exists. Do not rerun inference; preserve/recover the runtime files first.')
                if work is not None:
                    print('Runtime work directory:', work)
