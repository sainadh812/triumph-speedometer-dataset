"""Private Drive orchestration; immutable attempts and preservation before release."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import uuid
import zipfile

TASKS = ('localizer', 'speed', 'rpm')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        entries, names = bundle.infolist(), set()
        if sum(entry.file_size for entry in entries) > 20 * 1024 ** 3:
            raise ValueError('Dataset exceeds extraction limit')
        for entry in entries:
            target = (destination / entry.filename).resolve()
            if (not target.is_relative_to(destination) or '\\' in entry.filename or
                    ((entry.external_attr >> 16) & 0o170000) == 0o120000 or target in names):
                raise ValueError('Unsafe or duplicate dataset archive entry')
            names.add(target)
        bundle.extractall(destination)


def prepare(store, workspace_id, dataset_reference, workdir='/content/triumph_joint_v1'):
    """CPU data preparation. Extraction is bound to the exact verified archive."""
    ref = dataset_reference
    if not re.fullmatch('[0-9a-f]{64}', ref['sha256']):
        raise ValueError('Invalid dataset archive digest')
    workdir = Path(workdir)
    cache = workdir / ('dataset_' + ref['sha256'])
    cache.mkdir(parents=True, exist_ok=True)
    archive, dataset = cache / 'dataset.zip', cache / 'extracted'
    if not archive.exists() or archive.stat().st_size != ref['size'] or sha256(archive) != ref['sha256']:
        store.download_verified(ref['id'], archive, ref['sha256'], ref['size'])
    with zipfile.ZipFile(archive) as bundle:
        expected_manifest = hashlib.sha256(bundle.read('dataset_manifest.json')).hexdigest()
    if not (dataset / 'dataset_manifest.json').exists():
        safe_extract(archive, dataset)
    if sha256(dataset / 'dataset_manifest.json') != expected_manifest:
        raise ValueError('Extracted dataset manifest does not belong to the verified archive')
    from .dataset import validate_dataset
    validate_dataset(dataset)
    root = store.ensure_folder('dashboard_joint_v1', workspace_id)
    receipt = store.upload_json({'dataset_sha256': ref['sha256'], 'dataset_size': ref['size'],
                                 'dataset_manifest_sha256': expected_manifest, 'prepared': True}, root['id'],
                                'dataset_' + ref['sha256'][:16] + '_prepared.json')
    print('Private dataset verified. Tasks:', ', '.join(TASKS), flush=True)
    return {'dataset_root': str(dataset), 'workdir': str(workdir), 'drive_root_id': root['id'],
            'dataset_reference': ref, 'dataset_manifest_sha256': expected_manifest,
            'prepared_receipt_id': receipt['id']}


def _file_digest(ref):
    digest = ref.get('verifiedSha256', ref.get('sha256Checksum'))
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('A saved artifact has no verified SHA256')
    return digest


def _download(store, ref, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    store.download_verified(ref['id'], path, _file_digest(ref), int(ref['size']))
    return path


def _restore(store, task_folder, target, mode, task=None):
    """Select immutable receipt deterministically; latest restores final test cache."""
    if mode not in ('fresh', 'latest', 'best'):
        raise ValueError('Invalid resume mode')
    target = Path(target)
    attempts = sorted((item for item in store.list_files(task_folder)
                       if item['name'].startswith('attempt_') and
                       item.get('mimeType', 'application/vnd.google-apps.folder') == 'application/vnd.google-apps.folder'),
                      key=lambda item: item['name'])
    documents = []
    for attempt in attempts:
        for item in store.list_files(attempt['id']):
            if not re.fullmatch(r'epoch_\d{4}_(epoch|complete)_receipt.json', item['name']):
                continue
            doc = store.read_json(item['id'])
            if doc.get('attempt') != attempt['name'][len('attempt_'):]:
                raise ValueError('Checkpoint receipt attempt identity mismatch')
            if task is not None and doc['metadata']['identity']['task'] != task:
                raise ValueError('Checkpoint receipt task identity mismatch')
            doc['_attempt_id'] = attempt['id']
            documents.append(doc)
    if mode == 'fresh':
        if documents:
            raise ValueError('Fresh mode requires a new RUN_ID; existing epochs are preserved.')
        return None
    if not documents:
        raise ValueError('No saved epoch available for ' + mode + ' restart.')
    if mode == 'best':
        candidates = [doc for doc in documents if doc['phase'] == 'epoch' and doc['metadata'].get('is_best')]
        if not candidates:
            raise ValueError('No saved best epoch available.')
        chosen = min(candidates, key=lambda doc: (doc['metadata']['best_score'], doc['attempt'], doc['metadata']['epoch']))
    else:
        chosen = max(documents, key=lambda doc: (doc['attempt'], doc['metadata']['epoch'], doc['phase'] == 'complete'))
    checkpoint_name = chosen['checkpoint']['name']
    if not re.fullmatch(r'epoch_\d{4}\.pt', checkpoint_name):
        raise ValueError('Invalid saved checkpoint filename')
    checkpoint = _download(store, chosen['checkpoint'], target / checkpoint_name)
    epoch = int(chosen['metadata']['epoch'])
    if mode == 'best':
        return str(checkpoint)
    prefix = f'epoch_{epoch:04d}_{chosen["phase"]}_'
    allowed = {'report.json', 'test_once.json', 'best_model.pt', 'best_model.pt.sha256.json', 'history.json',
               'latest.json', 'best.json', checkpoint_name + '.sha256.json'}
    downloaded = set()
    for ref in chosen.get('artifacts', []):
        name = ref.get('local_name') or (ref['name'][len(prefix):] if ref['name'].startswith(prefix) else '')
        if name not in allowed:
            continue
        _download(store, ref, target / name)
        downloaded.add(name)
    # Finalization uploads test_once first. Recover it even if its receipt upload
    # failed; train.py subsequently checks the cached model/dataset identity.
    if 'test_once.json' not in downloaded:
        name = f'epoch_{epoch:04d}_complete_test_once.json'
        matches = [item for item in store.list_files(chosen['_attempt_id']) if item['name'] == name]
        if len(matches) > 1:
            raise ValueError('Ambiguous saved test cache')
        if matches:
            _download(store, store.metadata(matches[0]['id']), target / 'test_once.json')
            downloaded.add('test_once.json')
    if chosen['phase'] == 'complete':
        required = {'report.json', 'test_once.json', 'best_model.pt', 'best_model.pt.sha256.json', 'history.json'}
        if not required.issubset(downloaded):
            raise ValueError('Completed task receipt lacks required final artifacts')
        report_path = target / 'report.json'
        report = json.loads(report_path.read_text())
        if report['identity'] != chosen['metadata']['identity'] or report.get('status') != 'complete':
            raise ValueError('Completed report does not match checkpoint identity')
        marker = json.loads((target / 'best_model.pt.sha256.json').read_text())
        if sha256(target / 'best_model.pt') != marker['sha256'] or (target / 'best_model.pt').stat().st_size != marker['bytes']:
            raise ValueError('Restored selected weights differ from their checksum marker')
        report.update(last_checkpoint=str(checkpoint), best_model=str(target / 'best_model.pt'), report=str(report_path))
        write_json(report_path, report)
    return str(checkpoint)


def _disconnect():
    """A runtime release error must never turn saved training into failure."""
    try:
        from google.colab import runtime
        runtime.unassign()
        return True
    except Exception:
        print('Artifacts are safe; automatic runtime release failed. Disconnect from the Runtime menu.', flush=True)
        return False


def run_all(store, prepared, run_id, resume_modes=None, configs=None, auto_disconnect=True):
    output = folder = None
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    verified_local, remote_attempts = {}, {}
    execution_started = False
    started = time.perf_counter()
    summary = None

    def remember(path):
        verified_local[str(Path(path).resolve())] = sha256(path)

    def upload(path, parent, name=None):
        ref = store.upload_verified(path, parent, name)
        remember(path)
        return ref

    try:
        from .train import train_all, smoke_test
        from .model import require_colab_cuda
        require_colab_cuda()
        if not re.fullmatch(r'[A-Za-z0-9_-]{4,100}', run_id):
            raise ValueError('Invalid run ID')
        modes = resume_modes or {task: 'fresh' for task in TASKS}
        if set(modes) != set(TASKS) or any(mode not in ('fresh', 'latest', 'best') for mode in modes.values()):
            raise ValueError('Specify fresh/latest/best for each task')
        folder = store.ensure_folder(run_id, prepared['drive_root_id'])
        folders = {task: store.ensure_folder(task, folder['id'])['id'] for task in TASKS}
        output = Path(prepared['workdir']) / 'runs' / run_id / attempt
        output.mkdir(parents=True, exist_ok=False)
        resume_paths = {}
        for task in TASKS:
            resume_paths[task] = _restore(store, folders[task], output / task, modes[task], task)
            # Record restored bytes immediately, including before later restore errors.
            for path in (output / task).rglob('*'):
                if path.is_file():
                    remember(path)
        remote_attempts = {task: store.ensure_folder('attempt_' + attempt, folders[task])['id'] for task in TASKS}

        def callback(task, path, metadata):
            epoch, phase = int(metadata['epoch']), metadata.get('phase', 'epoch')
            dest = remote_attempts[task]
            checkpoint = upload(path, dest)
            artifacts = []
            paths = [Path(raw) for raw in metadata.get('artifact_paths', [])]
            paths.sort(key=lambda item: (item.name != 'test_once.json', item.name == 'report.json', item.name))
            for artifact in paths:
                if artifact.resolve() == Path(path).resolve():
                    continue
                if not artifact.is_file():
                    raise FileNotFoundError('Required checkpoint artifact missing: ' + artifact.name)
                ref = upload(artifact, dest, f'epoch_{epoch:04d}_{phase}_{artifact.name}')
                artifacts.append(dict(ref, local_name=artifact.name))
            receipt = {'checkpoint': checkpoint, 'metadata': metadata, 'artifacts': artifacts,
                       'attempt': attempt, 'phase': phase, 'task': task}
            name = f'epoch_{epoch:04d}_{phase}_receipt.json'
            local_receipt = output / task / ('drive_' + name)
            write_json(local_receipt, receipt)
            store.upload_json(receipt, dest, name)
            remember(local_receipt)
            print(f'{task}: epoch {epoch} verified in Drive ({phase})', flush=True)

        def progress(event):
            print(json.dumps(event, allow_nan=False), flush=True)

        complete_tasks = [task for task in TASKS if (output / task / 'report.json').is_file()]
        execution_started = True
        if len(complete_tasks) != len(TASKS):
            smoke = smoke_test(prepared['dataset_root'], output_path=output / 'gpu_smoke.json',
                               configs=configs, progress_callback=progress)
            write_json(output / 'gpu_smoke.json', smoke)
            upload(output / 'gpu_smoke.json', folder['id'], attempt + '_gpu_smoke.json')
        result = train_all(prepared['dataset_root'], output, configs=configs, resume_modes=modes,
                           resume_checkpoints=resume_paths, checkpoint_callback=callback, progress_callback=progress)
        archive = output / 'training_results.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            for path in sorted(output.rglob('*')):
                if not path.is_file() or path == archive:
                    continue
                if path.suffix in ('.pt', '.pth') and 'best' not in path.name:
                    continue
                bundle.write(path, path.relative_to(output).as_posix())
        saved = upload(archive, folder['id'], attempt + '_training_results.zip')
        summary = {'status': 'complete', 'run_id': run_id, 'attempt': attempt,
                   'elapsed_seconds': time.perf_counter() - started, 'results': result,
                   'archive': saved, 'drive_run_folder_id': folder['id'], 'task_attempt_folder_ids': remote_attempts,
                   'all_epoch_checkpoints_preserved': True, 'no_old_model_comparison': True}
        write_json(output / 'COMPLETE.json', summary)
        store.upload_json(summary, folder['id'], attempt + '_COMPLETE.json')
        remember(output / 'COMPLETE.json')
    except BaseException as error:
        failure = {'status': 'failed', 'run_id': run_id, 'attempt': attempt, 'exception_type': type(error).__name__}
        unsaved = []
        if output is not None and execution_started:
            unsaved = [path for path in output.rglob('*') if path.is_file() and
                       verified_local.get(str(path.resolve())) != sha256(path)]
        preserved = not unsaved
        if unsaved and folder is not None:
            try:
                rescue = output / 'failure_recovery.zip'
                with zipfile.ZipFile(rescue, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
                    for path in unsaved:
                        if path != rescue:
                            bundle.write(path, path.relative_to(output).as_posix())
                failure['recovery_archive'] = upload(rescue, folder['id'], attempt + '_failure_recovery.zip')
                store.upload_json(failure, folder['id'], attempt + '_FAILED.json')
                preserved = True
            except Exception:
                print('Recovery upload is unverified. Runtime retained to protect artifacts; attention needed.', flush=True)
        if auto_disconnect and preserved:
            print('No unsaved runtime artifacts remain. Releasing accelerator.', flush=True)
            _disconnect()
        elif not preserved:
            print('Runtime retained: required artifacts still need a verified durable copy.', flush=True)
        raise
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    if auto_disconnect:
        print('All checkpoints/results verified in Drive. Releasing the accelerator.', flush=True)
        _disconnect()
    return summary
