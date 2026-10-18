"""CPU upload/download and Colab-only inference orchestration.

Importing this module has no side effects and imports no ML/numeric libraries.
All input videos, checkpoint weights and results remain in the user's Drive.
"""
from __future__ import annotations
import bisect
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import quote
import uuid
import zipfile
from datetime import datetime, timezone

CODE_COMMIT = 'af991b11529aeabd6355732b7562e85d857caa86'
DRIVE_ROOT = Path('/content/drive/MyDrive/SpeedometerTraining')
PREVIOUS_REL = 'video_tests/lcd_speed_rpm/150802_20261018_095125_615d6a3a'
PROVENANCE_SHA256 = '4e8d74042c0585e954a657e858643a749041fe1344d0edd04e8f7d55be17b875'
PROVENANCE_BYTES = 39411
SOURCE_FOLDERS = {'learned_lcd_v1', 'mixed_view_v5', 'real_adaptation_v4',
                  'real_adaptation_v3', 'training_pilot', 'rpm_training_v1', 'rpm_prototype'}
LCD_RUN_REL = 'experiments/learned_lcd_v2_mirrors/runs/20261018_082714_561655_919223d0'
SPEED_RUN_REL = 'experiments/speedometer_mixed_views_v5/runs/20261018_060437_397258_dc0d47c3'
RPM_RUN_REL = 'experiments/triumph_rpm_strip_v1/runs/20261017_171941_010868_1cabaa1b'
LCD_SHA256 = '23daf88c88f0a4ff9960be82591ab9089aecc71bb0339e30b70c1ff6c0d23f44'
SPEED_SHA256 = 'de08c9772c4eacc2b78f9037ff6c45c980d63ee6223739199b4ad28fc4777e1a'
RPM_SHA256 = 'b3ed93a01d9668cf823c32d4c80e7dcd9ede0e6b69e8f5a66942bb5faa1c6016'
CHECKPOINTS = {
    'lcd': {'run': LCD_RUN_REL, 'sha256': LCD_SHA256, 'bytes': 6320317,
            'marker_sha256': '6eadb8fb7b06c7cecb2db982d469f1a554cc21952a11a523e146e57bdc36aa1d',
            'marker_bytes': 247, 'kind': 'learned_lcd_inference_v1', 'epoch': 9},
    'speed': {'run': SPEED_RUN_REL, 'sha256': SPEED_SHA256, 'bytes': 9084571,
              'marker_sha256': 'd6f4e1e62361b7ca3f009bd76954ba05a476a458b9776422322c11339e6fce3a',
              'marker_bytes': 265, 'kind': 'joint_needle_ticks_mixed_view_inference_v5', 'epoch': 11},
    'rpm': {'run': RPM_RUN_REL, 'sha256': RPM_SHA256, 'bytes': 680941,
            'marker_sha256': 'c31f0ac8fd02186cbb68b9532dbc7e5c36ca510a3bf6af2bcb6068e5b1ae8f85',
            'marker_bytes': 233, 'kind': 'rpm_strip_inference_weights_v1', 'epoch': 9}}
RPM_ZIP_NAME = 'rpm_runtime_assets_20261018.zip'
RPM_ZIP_SHA256 = '5b009c5a59bd510899c2e6bccc18c74771bb23fef90d9b0b0327d95c42836282'
RPM_ZIP_BYTES = 25330260
RPM_DATA_ZIP_REL = 'rpm_training_v1/rpm_synthetic_2000_v1.zip'
RPM_DATA_ZIP_SHA256 = 'b54b49783c1339c0bad1fab46843e6f8aecb7bd7259b0ae565bd4f9bf0524090'
RPM_DATA_MANIFEST_SHA256 = '878479797c1505411d89f4c9228df40e0477d1a83b8d8fd660e1a62001451433'
PINNED_SOURCE_HASHES = {
    'rpm_training_v1/train_rpm.py': '9fc3542719eeb574a98ff0b01177df5967e025349b1092d01db2529fd2ba34d2',
    'rpm_training_v1/preprocess.py': '414f42f9e42808b9579c502854d37fa33590efe83bcb4b94e025c37de28acced',
    'mixed_view_v5/calibration.json': '2581a049a418d0f337a9997dcd6d196aef7d49d4d9194acc2bc176f14441d2f5'}

JOBS_REL = 'video_tests/browser_upload_v1'
BATCH_SIZES = {'lcd': 64, 'speed': 16, 'rpm': 128}
RUNTIME_FILES = ('evaluate_uploaded_video.py', 'evaluate_lcd_speed_rpm_batched.py', 'benchmark_lcd_positioning.py')
BASE_SHA256 = '0c5df9a13e5cd904d76339a74de8d84675965041ee166eacf94e933214947df3'
POSITIONING_SHA256 = '68c299ada88ff433fea7e57ac379823ee9d6848d61f0605556642ecc3c5957e0'

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(path, digest, size=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if size is not None and path.stat().st_size != size:
        raise ValueError('Byte count mismatch: ' + str(path))
    if sha256(path) != digest:
        raise ValueError('SHA256 mismatch: ' + str(path))
    return path


def copy_verified(source, target, digest=None, size=None):
    source = Path(source)
    digest = digest or sha256(source)
    size = source.stat().st_size if size is None else size
    verify(source, digest, size)
    target = Path(target)
    if target.exists():
        return verify(target, digest, size)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.staged-' + uuid.uuid4().hex)
    with source.open('rb') as src, temporary.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    verify(temporary, digest, size)
    os.replace(temporary, target)
    return verify(target, digest, size)


def checked_zip_files(bundle):
    infos = bundle.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(name.casefold() for name in names)):
        raise ValueError('ZIP has duplicate/case-alias names')
    for info in infos:
        name = info.filename
        path = PurePosixPath(name)
        mode = stat.S_IFMT(info.external_attr >> 16)
        if (not name or path.is_absolute() or path.as_posix() != name or
                any(part in ('', '.', '..') for part in name.split('/')) or
                '\\' in name or ':' in name or '\x00' in name or
                info.is_dir() or mode not in (0, stat.S_IFREG) or info.flag_bits & 1):
            raise ValueError('Unsafe or unsupported ZIP member: ' + repr(name))
    return {info.filename: info for info in infos}


def extract_files(bundle, infos, destination, expected_hashes, expected_sizes=None):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name, info in infos.items():
        target = (destination / name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError('ZIP extraction escapes destination')
        if expected_sizes and name in expected_sizes and info.file_size != expected_sizes[name]:
            raise ValueError('ZIP member byte count mismatch: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with bundle.open(info) as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        verify(target, expected_hashes[name], info.file_size)
    return destination


def extract_rpm_assets(archive, destination):
    # The fixed outer archive hash authenticates the embedded per-file allowlist.
    verify(archive, RPM_ZIP_SHA256, RPM_ZIP_BYTES)
    with zipfile.ZipFile(archive) as bundle:
        infos = checked_zip_files(bundle)
        manifest_name = 'rpm_assets_manifest.json'
        if manifest_name not in infos or infos[manifest_name].file_size > 100000:
            raise ValueError('Missing/oversized RPM asset manifest')
        manifest_bytes = bundle.read(manifest_name)
        manifest = json.loads(manifest_bytes)
        rows = manifest['files']
        declared = {row['path']: row for row in rows}
        if (len(rows) != 15 or len(declared) != 15 or
                set(infos) != set(declared) | {manifest_name} or
                manifest['compatible_code_commit'] != CODE_COMMIT or manifest['rpm_best_epoch'] != 9):
            raise ValueError('Unexpected RPM package allowlist/identity')
        hashes = {name: row['sha256'] for name, row in declared.items()}
        hashes[manifest_name] = hashlib.sha256(manifest_bytes).hexdigest()
        sizes = {name: row['bytes'] for name, row in declared.items()}
        extract_files(bundle, infos, destination, hashes, sizes)
    return manifest


def extract_rpm_dataset(archive, repo, work):
    verify(archive, RPM_DATA_ZIP_SHA256, 24407949)
    with zipfile.ZipFile(archive) as bundle:
        infos = checked_zip_files(bundle)
        name = 'dataset/manifest.json'
        if name not in infos or infos[name].file_size > 2000000:
            raise ValueError('Missing/oversized nested RPM dataset manifest')
        data = bundle.read(name)
        if hashlib.sha256(data).hexdigest() != RPM_DATA_MANIFEST_SHA256:
            raise ValueError('Nested RPM manifest hash mismatch')
        manifest = json.loads(data)
        hashes = {'dataset/' + rel: digest for rel, digest in manifest['sha256'].items()}
        hashes[name] = RPM_DATA_MANIFEST_SHA256
        if len(hashes) != 2030 or set(infos) != set(hashes):
            raise ValueError('Nested RPM dataset has missing or extra files')
        staging = extract_files(bundle, infos, work / 'rpm_dataset_staging', hashes)
    target = repo / 'rpm_training_v1/dataset'
    if target.exists():
        raise FileExistsError('Refusing to replace an existing RPM dataset')
    # Both are new runtime-local paths, and neither contains original user files.
    os.replace(staging / 'dataset', target)
    return target


def run_logged(command, cwd=None, timeout_s=None):
    print('Running:', ' '.join(map(str, command)), flush=True)
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    expired = threading.Event()
    def stop_after_budget():
        if process.poll() is None:
            expired.set()
            process.kill()
    timer = threading.Timer(timeout_s, stop_after_budget) if timeout_s else None
    if timer:
        timer.daemon = True
        timer.start()
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
        if expired.is_set():
            raise TimeoutError('Bounded evaluator exceeded ' + str(timeout_s) + ' seconds; preserving partial diagnostics')
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if timer:
            timer.cancel()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def verify_checkpoint(weight, marker, spec):
    verify(weight, spec['sha256'], spec['bytes'])
    verify(marker, spec['marker_sha256'], spec['marker_bytes'])
    data = json.loads(marker.read_text())
    if spec['kind'] == 'rpm_strip_inference_weights_v1':
        if (data.get('filename') != weight.name or data.get('kind') != spec['kind'] or
                data.get('sha256') != spec['sha256'] or data.get('bytes') != spec['bytes'] or
                data.get('best_epoch') != spec['epoch'] or data.get('contains_optimizer_state') is not False):
            raise ValueError('RPM checkpoint completion marker mismatch: ' + str(marker))
        return
    if (data.get('checkpoint') != weight.name or data.get('sha256') != spec['sha256'] or
            data.get('bytes') != spec['bytes'] or data.get('checkpoint_kind') != spec['kind'] or
            data.get('format_version') != 1):
        raise ValueError('Checkpoint completion marker mismatch: ' + str(marker))


def restore_verified_pipeline(work, repo):
    manifest_path = copy_verified(DRIVE_ROOT / PREVIOUS_REL / 'result_manifest.json',
                                  work / 'previous_result_manifest.json', PROVENANCE_SHA256, PROVENANCE_BYTES)
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'complete_and_verified' or manifest['code_commit'] != CODE_COMMIT:
        raise ValueError('Previous saved pipeline provenance mismatch')
    paths, copied = set(), []
    for row in manifest['files']:
        relative = row['path']
        path = PurePosixPath(relative)
        if (path.is_absolute() or '..' in path.parts or '\\' in relative or ':' in relative or
                path.as_posix() != relative or relative.casefold() in paths):
            raise ValueError('Unsafe/duplicate saved pipeline inventory')
        paths.add(relative.casefold())
        if (len(path.parts) == 3 and path.parts[0] == 'pipeline' and path.parts[1] in SOURCE_FOLDERS and
                (path.suffix == '.py' or relative == 'pipeline/mixed_view_v5/calibration.json')):
            target = repo / Path(*path.parts[1:])
            copy_verified(DRIVE_ROOT / PREVIOUS_REL / relative, target, row['sha256'], row['bytes'])
            copied.append(row)
    required = ['learned_lcd_v1/inference.py', 'learned_lcd_v1/model.py',
                'mixed_view_v5/train_landmarks.py', 'mixed_view_v5/inference.py',
                'mixed_view_v5/calibration.json', 'real_adaptation_v4/geometry.py',
                'real_adaptation_v4/infer_video.py', 'real_adaptation_v4/train_landmarks.py',
                'real_adaptation_v3/train_needle.py', 'training_pilot/train_core.py',
                'rpm_training_v1/train_rpm.py', 'rpm_training_v1/preprocess.py']
    for relative in required:
        if not (repo / relative).is_file():
            raise FileNotFoundError('Missing saved dependency: ' + relative)
    for relative, digest in PINNED_SOURCE_HASHES.items():
        verify(repo / relative, digest)
    write_json(work / 'restored_pipeline_inventory.json', copied)
    return manifest_path


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        self.original.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.original.flush()
        self.log.flush()


def write_json(path, value):
    """Publish metadata atomically and verify its final bytes on mounted Drive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf8')
    temporary = path.with_name(path.name + '.staged-' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    verify(path, hashlib.sha256(data).hexdigest(), len(data))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def unique_id(prefix):
    return prefix + '_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]


def safe_drive_path(relative):
    relative = str(relative)
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or path.as_posix() != relative or
            any(part in ('', '.', '..') for part in relative.split('/')) or '\\' in relative or ':' in relative):
        raise ValueError('Unsafe private Drive relative path')
    target = (DRIVE_ROOT / Path(*path.parts)).resolve()
    if not target.is_relative_to(DRIVE_ROOT.resolve()):
        raise ValueError('Private Drive path escapes SpeedometerTraining')
    return target


def drive_relative(path):
    return Path(path).resolve().relative_to(DRIVE_ROOT.resolve()).as_posix()


def require_drive():
    if not DRIVE_ROOT.is_dir():
        raise RuntimeError('Mount the same Google Drive used by the previous training runs. '
                           'Its MyDrive/SpeedometerTraining folder must already exist.')


def gpu_present():
    executable = shutil.which('nvidia-smi')
    if not executable:
        return False
    result = subprocess.run([executable, '-L'], capture_output=True, text=True, timeout=15)
    return result.returncode == 0 and 'GPU' in result.stdout


def require_cpu():
    if gpu_present():
        raise RuntimeError('Upload/download uses CPU only. Choose Runtime > Change runtime type > '
                           'Hardware accelerator: None, reconnect, then run this cell.')


def require_gpu():
    if not gpu_present():
        raise RuntimeError('The video is already saved in Drive. Choose Runtime > Change runtime type > '
                           'T4 GPU, reconnect, then run only cell 2. Do not upload it again.')


def save_job(job, job_path):
    write_json(job_path, job)
    write_json(DRIVE_ROOT / JOBS_REL / 'current_job.json', {
        'schema_version': 1, 'job_path': drive_relative(job_path),
        'job_sha256': sha256(job_path), 'updated_at': utc_now()})


def load_job():
    require_drive()
    pointer_path = DRIVE_ROOT / JOBS_REL / 'current_job.json'
    if not pointer_path.is_file():
        raise RuntimeError('No uploaded job exists. Run cell 1 on CPU first.')
    pointer = json.loads(pointer_path.read_text())
    job_path = safe_drive_path(pointer['job_path'])
    if not job_path.is_relative_to((DRIVE_ROOT / JOBS_REL / 'jobs').resolve()):
        raise ValueError('Unexpected job location')
    verify(job_path, pointer['job_sha256'])
    job = json.loads(job_path.read_text())
    if job.get('schema_version') != 1 or job.get('job_id') != job_path.parent.name:
        raise ValueError('Job identity mismatch')
    source = safe_drive_path(job['source']['relative_path'])
    if not source.is_relative_to(job_path.parent.resolve()):
        raise ValueError('Uploaded source is outside its job folder')
    return job, job_path


def upload_video(start_seconds=0., max_seconds=0., force_new_upload=False):
    """Browser upload on CPU; persist the source and trim settings across restarts."""
    require_cpu()
    require_drive()
    for name, value in [('START_SECONDS', start_seconds), ('MAX_SECONDS', max_seconds)]:
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(name + ' must be a finite nonnegative number; MAX_SECONDS=0 means the entire remainder.')
    pointer = DRIVE_ROOT / JOBS_REL / 'current_job.json'
    if pointer.is_file() and not force_new_upload:
        job, job_path = load_job()
        if job['status'] in ('uploaded', 'failed', 'running'):
            source = job['source']
            verify(safe_drive_path(source['relative_path']), source['sha256'], source['bytes'])
            job.update(status='uploaded', start_seconds=float(start_seconds), max_seconds=float(max_seconds))
            save_job(job, job_path)
            print('Reusing verified upload:', source['original_name'])
            print('To choose another video set FORCE_NEW_UPLOAD=True. Otherwise switch to T4 and run cell 2.')
            return job
    from google.colab import files
    upload_root = Path('/content') / unique_id('browser_video_upload')
    upload_root.mkdir(exist_ok=False)
    previous_cwd = Path.cwd()
    try:
        os.chdir(upload_root)
        uploaded = files.upload()
    finally:
        os.chdir(previous_cwd)
    if len(uploaded) != 1:
        raise ValueError('Select exactly one video. Nothing was submitted for inference.')
    original_name, payload = next(iter(uploaded.items()))
    original_name = str(original_name)
    extension = Path(original_name).suffix.lower()
    if not extension or len(extension) > 12 or not extension[1:].isalnum():
        extension = '.video'
    local_source = upload_root / ('source' + extension)
    digest = hashlib.sha256(payload).hexdigest()
    size = len(payload)
    if size == 0:
        raise ValueError('Uploaded video is empty')
    local_source.write_bytes(payload)
    del payload, uploaded
    verify(local_source, digest, size)
    job_id = unique_id('video')
    job_dir = DRIVE_ROOT / JOBS_REL / 'jobs' / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    source = copy_verified(local_source, job_dir / 'source' / local_source.name, digest, size)
    job = {'schema_version': 1, 'job_id': job_id, 'status': 'uploaded', 'created_at': utc_now(),
           'source': {'original_name': original_name, 'relative_path': drive_relative(source),
                      'bytes': size, 'sha256': digest},
           'start_seconds': float(start_seconds), 'max_seconds': float(max_seconds),
           'scope': 'Existing LCD9, speed11 and RPM9 weights; no training. Fast LCD-based RPM positioning; raw unvalidated readings.'}
    save_job(job, job_dir / 'job.json')
    print('Upload saved and SHA256 verified in private Drive:', original_name)
    print('START_SECONDS=', job['start_seconds'], '| MAX_SECONDS=', job['max_seconds'], '(0 = entire remaining video)')
    print('Now choose T4 GPU, reconnect, and run only cell 2. The uploaded video survives the runtime change.')
    return job


def prepare_assets(work, runtime_dir):
    """Restore only protected runtime code and unchanged private checkpoints."""
    repo, inputs, rpm_staging = work / 'repo', work / 'inputs', work / 'rpm_asset_staging'
    expected = [DRIVE_ROOT / PREVIOUS_REL / 'result_manifest.json', DRIVE_ROOT / RPM_ZIP_NAME]
    for name in ('lcd', 'speed'):
        expected.extend(DRIVE_ROOT / CHECKPOINTS[name]['run'] / filename
                        for filename in ('best_model.pt', 'best_model.complete.json'))
    missing = [str(path.relative_to(DRIVE_ROOT)) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError('Pretrained assets are missing from this Drive. Use the same account as the previous runs.\n' + '\n'.join(missing))
    for name in ('lcd', 'speed'):
        spec = CHECKPOINTS[name]
        for filename, digest, size in [('best_model.pt', spec['sha256'], spec['bytes']),
                                      ('best_model.complete.json', spec['marker_sha256'], spec['marker_bytes'])]:
            copy_verified(DRIVE_ROOT / spec['run'] / filename, inputs / name / filename, digest, size)
    provenance = restore_verified_pipeline(work, repo)
    archive = copy_verified(DRIVE_ROOT / RPM_ZIP_NAME, work / RPM_ZIP_NAME, RPM_ZIP_SHA256, RPM_ZIP_BYTES)
    manifest = extract_rpm_assets(archive, rpm_staging)
    for row in manifest['files']:
        relative = row['path']
        target = inputs / 'rpm' / Path(relative).name if relative.startswith('experiments/') else repo / relative
        copy_verified(rpm_staging / relative, target, row['sha256'], row['bytes'])
    dataset = extract_rpm_dataset(repo / RPM_DATA_ZIP_REL, repo, work)
    for name, spec in CHECKPOINTS.items():
        verify_checkpoint(inputs / name / 'best_model.pt', inputs / name / 'best_model.complete.json', spec)
    for relative, digest in PINNED_SOURCE_HASHES.items():
        verify(repo / relative, digest)
    verify(runtime_dir / 'evaluate_lcd_speed_rpm_batched.py', BASE_SHA256, 55284)
    verify(runtime_dir / 'benchmark_lcd_positioning.py', POSITIONING_SHA256, 21561)
    for name in RUNTIME_FILES:
        copy_verified(runtime_dir / name, repo / name)
    requirements = []
    for package, version in [('numpy', '2.1.3'), ('opencv-python-headless', '4.14.0.94')]:
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            requirements.append(package + '==' + version)
    if requirements:
        run_logged([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps'] + requirements)
    versions = {}
    for name in ('torch', 'torchvision', 'numpy', 'opencv-python-headless'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError('Colab runtime is missing ' + name + '; use the standard T4 Python runtime.')
    write_json(work / 'runtime_versions.json', versions)
    return repo, inputs, dataset, provenance


def validate_results(evaluation, job, evaluator_digest):
    """Reconcile completed artifacts without importing a model or rerunning video."""
    summary_path = evaluation / 'summary.json'
    def reject_nonfinite(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    def read_json(path):
        return json.loads(Path(path).read_text(), parse_constant=reject_nonfinite)
    summary = read_json(summary_path)
    completion = read_json(evaluation / 'completion.json')
    if summary['status'] != 'complete' or summary['input_sha256'] != job['source']['sha256']:
        raise ValueError('Output status or source-video identity mismatch')
    if (summary['script_sha256'] != evaluator_digest or summary['protected_base_sha256'] != BASE_SHA256 or
            summary['protected_fast_positioning_sha256'] != POSITIONING_SHA256 or summary['code_commit'] != CODE_COMMIT or
            summary['training_performed'] is not False or summary['original_frozen_pipeline_files_modified'] is not False or
            summary['sift_calls'] != 0 or summary['maximum_model_batch_sizes'] != BATCH_SIZES or
            summary['requested_start_seconds'] != job['start_seconds'] or summary['requested_max_seconds'] != job['max_seconds']):
        raise ValueError('Evaluator identity or requested scope mismatch')
    metadata = summary['speed_reader_metadata']
    if (metadata['lcd_checkpoint_sha256'] != LCD_SHA256 or metadata['lcd_best_epoch'] != 9 or
            metadata['speed_checkpoint_sha256'] != SPEED_SHA256 or metadata['speed_best_epoch'] != 11 or
            summary['rpm_reader_identity']['checkpoint_sha256'] != RPM_SHA256 or summary['rpm_reader_identity']['best_epoch'] != 9):
        raise ValueError('Frozen checkpoint identity mismatch')
    count = summary['source_frames_processed']
    if not isinstance(count, int) or count <= 0:
        raise ValueError('No source frames were processed')
    if (completion['status'] != 'complete_and_fully_decoded' or
            completion['summary_sha256'] != sha256(summary_path) or
            completion['source_frames_processed'] != count or
            completion['model_samples'] != {name: count for name in BATCH_SIZES} or
            completion['output_frames'] != summary['output_frames'] or
            completion['output_sha256'] != summary['output_sha256'] or
            completion['output_bytes'] != summary['output_bytes'] or completion['script_sha256'] != evaluator_digest):
        raise ValueError('Completed video accounting mismatch')
    output = Path(summary['output']).resolve()
    if not output.is_relative_to(evaluation.resolve()):
        raise ValueError('Output video escaped its evaluation folder')
    verify(output, summary['output_sha256'], summary['output_bytes'])
    if set(summary['network_timing']) != set(BATCH_SIZES):
        raise ValueError('Unexpected model timing inventory')
    for name, timing in summary['network_timing'].items():
        if (timing['inference_samples'] != count or timing['selected_batch_size'] != BATCH_SIZES[name] or
                timing['dtype'] != 'float32' or timing['autocast'] is not False):
            raise ValueError('Model did not process every source frame: ' + name)
    indices, times = [], []
    with (evaluation / 'source_predictions.jsonl').open(encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line, parse_constant=reject_nonfinite)
            if (row['source_frame'] != summary['source_first_frame'] + len(indices) or
                    any(row[name]['model_evaluated'] is not True for name in BATCH_SIZES) or
                    row['rpm']['input_mapping']['sift_called'] is not False):
                raise ValueError('Missing/repeated source model evaluation or unexpected SIFT')
            timestamp = row['source_time_s']
            if not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp) or (times and timestamp <= times[-1]):
                raise ValueError('Invalid source timestamp sequence')
            indices.append(row['source_frame'])
            times.append(timestamp)
    if len(indices) != count or indices[-1] != summary['source_last_frame']:
        raise ValueError('Source prediction count differs from completed summary')
    ledger = {name: {'indices': [], 'calls': 0, 'last_samples': 0} for name in BATCH_SIZES}
    with (evaluation / 'batch_timings.jsonl').open(encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line, parse_constant=reject_nonfinite)
            name = row['model']
            if (name not in ledger or not 1 <= row['samples'] <= BATCH_SIZES[name] or
                    row['samples'] != len(row['source_indices'])):
                raise ValueError('Invalid inference batch record')
            ledger[name]['indices'].extend(row['source_indices'])
            ledger[name]['calls'] += 1
            ledger[name]['last_samples'] = row['samples']
    for name, rows in ledger.items():
        timing = summary['network_timing'][name]
        if (rows['indices'] != indices or rows['calls'] != timing['successful_forward_calls'] or
                rows['last_samples'] != timing['last_batch_samples']):
            raise ValueError('Batch ledger does not cover the source exactly once: ' + name)
    fps = summary['output_fps']
    if not isinstance(fps, (float, int)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid output playback FPS')
    relative_times = [value - times[0] for value in times]
    mapped_count = 0
    with (evaluation / 'playback_mapping.jsonl').open(encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line, parse_constant=reject_nonfinite)
            output_time = mapped_count / fps
            right = min(bisect.bisect_left(relative_times, output_time), count - 1)
            left = max(0, right - 1)
            chosen = left if abs(relative_times[left] - output_time) <= abs(relative_times[right] - output_time) else right
            if (row['output_frame'] != mapped_count or not math.isclose(row['output_time_s'], output_time, abs_tol=1e-7) or
                    row['source_frame'] != indices[chosen] or not math.isclose(row['source_time_s'], times[chosen], abs_tol=1e-7)):
                raise ValueError('Playback does not use the declared nearest source timestamp')
            mapped_count += 1
    if mapped_count != summary['output_frames'] or mapped_count <= 0:
        raise ValueError('Output playback frame count differs')
    for filename in ('predictions.csv', 'timing_summary.json', 'chunk_timings.json'):
        if not (evaluation / filename).is_file():
            raise FileNotFoundError('Missing completed evidence: ' + filename)
    return summary


def publish_run(work, job, job_path, run_id, summary=None, error=None):
    """Verify every result file in Drive before publishing a downloadable receipt."""
    evaluation, repo = work / 'evaluation', work / 'repo'
    if summary is None:
        run_id += '_incomplete'
    destination = job_path.parent / 'runs' / run_id
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    sources = []
    if evaluation.exists():
        sources.extend((path, Path('evaluation') / path.relative_to(evaluation))
                       for path in sorted(evaluation.rglob('*')) if path.is_file() and path.suffix.lower() != '.avi')
    for name in ('inference.log', 'command.json', 'runtime_versions.json', 'previous_result_manifest.json', 'restored_pipeline_inventory.json'):
        if (work / name).is_file():
            sources.append((work / name, Path(name)))
    sources.append((Path(__file__), Path('pipeline') / Path(__file__).name))
    for name in RUNTIME_FILES:
        if (repo / name).is_file():
            sources.append((repo / name, Path('pipeline') / name))
    for source, relative in sources:
        target = copy_verified(source, destination / relative)
        entries.append({'path': relative.as_posix(), 'bytes': target.stat().st_size, 'sha256': sha256(target)})
    status = 'complete_and_verified' if summary is not None else 'failed_partial_outputs_unverified'
    manifest = {'schema_version': 1, 'status': status, 'job_id': job['job_id'], 'run_id': run_id,
                'source': job['source'], 'start_seconds': job['start_seconds'], 'max_seconds': job['max_seconds'],
                'created_at': utc_now(), 'checkpoints': CHECKPOINTS, 'training_performed': False,
                'raw_outputs_unvalidated': True, 'ground_truth_accuracy': None,
                'previous_pipeline_manifest_sha256': PROVENANCE_SHA256,
                'summary': summary, 'error': error, 'files': entries}
    manifest_path = destination / 'result_manifest.json'
    write_json(manifest_path, manifest)
    receipt = {'schema_version': 1, 'status': status, 'job_id': job['job_id'], 'run_id': run_id,
               'drive_directory_relative': drive_relative(destination),
               'result_manifest_relative': drive_relative(manifest_path), 'result_manifest_sha256': sha256(manifest_path),
               'files_copied_and_hash_verified': True, 'source_sha256': job['source']['sha256'],
               'created_at': utc_now(), 'ground_truth_accuracy': None}
    if summary is not None:
        output = destination / 'evaluation' / Path(summary['output']).name
        verify(output, summary['output_sha256'], summary['output_bytes'])
        receipt.update(output_relative=drive_relative(output), output_sha256=summary['output_sha256'],
                       output_bytes=summary['output_bytes'], output_frames=summary['output_frames'],
                       source_frames_processed=summary['source_frames_processed'])
    receipt_path = destination / 'publication_receipt.json'
    write_json(receipt_path, receipt)
    job.update(status='complete' if summary is not None else 'failed', latest_run_id=run_id,
               latest_receipt=drive_relative(receipt_path), latest_receipt_sha256=sha256(receipt_path))
    save_job(job, job_path)
    if summary is not None:
        write_json(DRIVE_ROOT / JOBS_REL / 'latest_result.json', {
            'receipt_relative': drive_relative(receipt_path), 'receipt_sha256': sha256(receipt_path)})
    return receipt


def show_drive_result(receipt):
    """A private Drive search link survives disconnect; cell 3 downloads by exact path."""
    from IPython.display import HTML, display
    url = 'https://drive.google.com/drive/u/1/search?q=' + quote(receipt['run_id'], safe='')
    print('Drive backup verified:', receipt['drive_directory_relative'])
    display(HTML('<a target="_blank" rel="noopener" href="' + url + '">Open this private result folder in Drive</a>'))
    if receipt['status'] == 'complete_and_verified':
        print('MP4 SHA256:', receipt['output_sha256'], '| bytes:', receipt['output_bytes'])
        print('After GPU disconnect: choose CPU/None, reconnect, then run cell 3 to download the MP4.')
    else:
        print('The run failed. Partial diagnostics were saved; no completed video is claimed.')


def disconnect_runtime():
    print('Releasing the GPU runtime now. All completed results remain in private Drive.', flush=True)
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception as error:
        print('Automatic disconnect failed; use Runtime > Disconnect and delete runtime.', repr(error), flush=True)


def run_uploaded_video(runtime_dir, auto_disconnect=True):
    """Run one uploaded job; never retry inference or train models automatically."""
    work = None
    job = job_path = receipt = None
    had_gpu = gpu_present()
    try:
        require_drive()
        require_gpu()
        job, job_path = load_job()
        if job['status'] == 'complete':
            receipt_path = safe_drive_path(job['latest_receipt'])
            verify(receipt_path, job['latest_receipt_sha256'])
            receipt = json.loads(receipt_path.read_text())
            verify(safe_drive_path(receipt['output_relative']), receipt['output_sha256'], receipt['output_bytes'])
            return receipt
        run_id = unique_id('run')
        work = Path('/content') / unique_id('speedometer_upload_inference')
        work.mkdir(exist_ok=False)
        runtime_dir = Path(runtime_dir).resolve()
        job.update(status='running', latest_attempt_id=run_id)
        save_job(job, job_path)
        original_stdout, original_stderr = sys.stdout, sys.stderr
        with (work / 'inference.log').open('x', encoding='utf8') as log:
            sys.stdout, sys.stderr = Tee(original_stdout, log), Tee(original_stderr, log)
            try:
                print('Existing frozen models only. Raw readings are not validated accuracy measurements.', flush=True)
                repo, inputs, dataset, provenance = prepare_assets(work, runtime_dir)
                source = job['source']
                video = copy_verified(safe_drive_path(source['relative_path']), inputs / Path(source['relative_path']).name,
                                      source['sha256'], source['bytes'])
                evaluator = repo / 'evaluate_uploaded_video.py'
                command = [sys.executable, '-u', str(evaluator), '--repository-root', str(repo),
                           '--provenance-manifest', str(provenance), '--video', str(video),
                           '--expected-video-sha256', source['sha256'], '--output-root', str(work / 'evaluation'),
                           '--lcd-checkpoint', str(inputs / 'lcd/best_model.pt'),
                           '--speed-checkpoint', str(inputs / 'speed/best_model.pt'),
                           '--speed-calibration', str(repo / 'mixed_view_v5/calibration.json'),
                           '--rpm-dataset', str(dataset), '--rpm-checkpoint', str(inputs / 'rpm/best_model.pt'),
                           '--rpm-reference', str(repo / 'rpm_exploration/review_220558_frame_0169.png'),
                           '--start-seconds', str(job['start_seconds']), '--max-seconds', str(job['max_seconds']),
                           '--device', 'cuda']
                write_json(work / 'command.json', command)
                run_logged(command, cwd=repo)
                summary = validate_results(work / 'evaluation', job, sha256(evaluator))
                receipt = publish_run(work, job, job_path, run_id, summary=summary)
            except BaseException as error:
                traceback.print_exc()
                sys.stdout.flush()
                receipt = publish_run(work, job, job_path, run_id, error={
                    'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
                raise
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
    finally:
        try:
            if receipt is not None:
                try:
                    show_drive_result(receipt)
                except Exception as display_error:
                    print('Verified Drive receipt:', json.dumps(receipt, indent=2), flush=True)
                    print('The optional Drive link could not be displayed:', repr(display_error), flush=True)
            elif work is not None:
                print('No verified completed result was published. Read the error above; private inputs and weights are unchanged.', flush=True)
        finally:
            evaluation = work / 'evaluation' if work is not None else None
            has_unsaved_outputs = (receipt is None and evaluation is not None and evaluation.exists() and
                                   any(path.is_file() for path in evaluation.rglob('*')))
            if auto_disconnect and had_gpu and not has_unsaved_outputs:
                disconnect_runtime()
            elif auto_disconnect and had_gpu and has_unsaved_outputs:
                print('GPU kept connected because Drive preservation failed and runtime outputs would be lost.', flush=True)
                print('Save the files in', evaluation, 'and', work / 'inference.log', 'before disconnecting.', flush=True)
                print('Emergency download: from google.colab import files; files.download(str(Path(' +
                      repr(str(evaluation)) + ") / 'annotated_fast_lcd_speed_rpm.mp4'))", flush=True)
    return receipt


def download_latest_result():
    """Copy/verify the persisted MP4 on CPU; no model code or GPU is required."""
    require_cpu()
    require_drive()
    job, _ = load_job()
    if job['status'] != 'complete':
        raise RuntimeError('The current uploaded video has no completed result. An older video will not be downloaded instead.')
    pointer_path = DRIVE_ROOT / JOBS_REL / 'latest_result.json'
    if not pointer_path.is_file():
        raise RuntimeError('No completed output is saved yet. Run cell 2 with T4 first.')
    pointer = json.loads(pointer_path.read_text())
    receipt_path = safe_drive_path(pointer['receipt_relative'])
    verify(receipt_path, pointer['receipt_sha256'])
    receipt = json.loads(receipt_path.read_text())
    if (receipt['status'] != 'complete_and_verified' or receipt['files_copied_and_hash_verified'] is not True or
            receipt['job_id'] != job['job_id'] or receipt['run_id'] != job['latest_run_id']):
        raise ValueError('Latest output is not a verified complete video')
    manifest = safe_drive_path(receipt['result_manifest_relative'])
    verify(manifest, receipt['result_manifest_sha256'])
    source = safe_drive_path(receipt['output_relative'])
    local = Path('/content') / (receipt['job_id'] + '_' + receipt['run_id'] + '.mp4')
    copy_verified(source, local, receipt['output_sha256'], receipt['output_bytes'])
    print('Downloading verified MP4:', local.name)
    print('SHA256:', receipt['output_sha256'])
    print('CPU only. Keep this runtime connected until the browser download finishes; the Drive copy remains available.')
    from google.colab import files
    files.download(str(local))
    return receipt
