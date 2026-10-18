"""Pure mocked Drive/control-flow tests. No Google, credentials, torch or models."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from test_pure_contract import fixture, refresh_manifest

HERE = Path(__file__).resolve().parent
PACKAGE = 'joint_workflow_pure_test'


class FakeStore:
    def __init__(self, events=None):
        self.items, self.payloads, self.events = {}, {}, events if events is not None else []
        self.fail_upload = None
        self.fail_folder = False

    def ensure_folder(self, name, parent):
        if self.fail_folder:
            raise RuntimeError('Simulated unavailable account')
        for item in self.items.values():
            if item['name'] == name and item['parents'] == [parent]:
                return dict(item)
        key = str(len(self.items) + 1)
        self.items[key] = {'id': key, 'name': name, 'parents': [parent], 'mimeType': 'application/vnd.google-apps.folder'}
        return dict(self.items[key])

    def _put(self, content, parent, name):
        digest = hashlib.sha256(content).hexdigest()
        for item in self.items.values():
            if item['name'] == name and item['parents'] == [parent]:
                if self.payloads[item['id']] != content:
                    raise ValueError('Immutable name collision')
                return dict(item)
        key = str(len(self.items) + 1)
        item = {'id': key, 'name': name, 'size': str(len(content)), 'parents': [parent],
                'mimeType': 'application/octet-stream', 'verifiedSha256': digest, 'sha256Checksum': digest}
        self.items[key], self.payloads[key] = item, content
        self.events.append('upload:' + name)
        return dict(item)

    def upload_verified(self, path, parent, name=None):
        name = name or Path(path).name
        if self.fail_upload and self.fail_upload(name):
            raise RuntimeError('Simulated upload failure')
        return self._put(Path(path).read_bytes(), parent, name)

    def upload_json(self, value, parent, name):
        return self._put(json.dumps(value, sort_keys=True).encode(), parent, name)

    def list_files(self, parent):
        return [dict(item) for item in self.items.values() if item['parents'] == [parent]]

    def read_json(self, key):
        return json.loads(self.payloads[key])

    def metadata(self, key):
        return dict(self.items[key])

    def download_verified(self, key, target, digest, size):
        content = self.payloads[key]
        if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('Corrupt download')
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(content)
        return {'local_path': str(target)}


@contextlib.contextmanager
def load_workflow(events, fake_train=None, fake_smoke=None, release_error=False):
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(HERE)]
    train = types.ModuleType(PACKAGE + '.train')
    train.train_all = fake_train or simulated_training
    train.smoke_test = fake_smoke or (lambda *a, **kw: {'status': 'passed', 'simulated': True})
    model = types.ModuleType(PACKAGE + '.model')
    model.require_colab_cuda = lambda: None
    google, colab, runtime = types.ModuleType('google'), types.ModuleType('google.colab'), types.ModuleType('google.colab.runtime')
    def unassign():
        events.append('disconnect')
        if release_error:
            raise RuntimeError('Simulated UI release failure')
    runtime.unassign = unassign
    colab.runtime, google.colab = runtime, colab
    injected = {PACKAGE: package, PACKAGE + '.train': train, PACKAGE + '.model': model,
                'google': google, 'google.colab': colab, 'google.colab.runtime': runtime}
    with patch.dict(sys.modules, injected), contextlib.redirect_stdout(io.StringIO()):
        spec = importlib.util.spec_from_file_location(PACKAGE + '.colab_workflow', HERE / 'colab_workflow.py')
        workflow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow)
        yield workflow


def simulated_training(dataset_root, output, checkpoint_callback, **kwargs):
    """Fake bytes and callbacks only; deliberately no ML import/execution."""
    results = {}
    for task in ('localizer', 'speed', 'rpm'):
        directory = Path(output) / task
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / 'report.json').exists():
            results[task] = json.loads((directory / 'report.json').read_text())
            continue
        checkpoint = directory / 'epoch_0001.pt'
        checkpoint.write_bytes(b'not-a-real-model-' + task.encode())
        identity = {'task': task, 'dataset': 'mock'}
        history = directory / 'history.json'
        history.write_text('[{"epoch":1}]')
        metadata = {'phase': 'epoch', 'epoch': 1, 'best_epoch': 1, 'best_score': .1,
                    'is_best': True, 'identity': identity, 'artifact_paths': [str(history)]}
        checkpoint_callback(task, checkpoint, metadata)
        best = directory / 'best_model.pt'
        best.write_bytes(b'fake-best-' + task.encode())
        marker = directory / 'best_model.pt.sha256.json'
        marker.write_text(json.dumps({'sha256': hashlib.sha256(best.read_bytes()).hexdigest(), 'bytes': best.stat().st_size}))
        test = directory / 'test_once.json'
        test.write_text('{"best_epoch":1,"fake_metric":true}')
        result = {'status': 'complete', 'identity': identity, 'config': {}, 'epochs_completed': 1, 'best_epoch': 1,
                  'last_checkpoint': str(checkpoint), 'best_model': str(best), 'report': str(directory / 'report.json')}
        (directory / 'report.json').write_text(json.dumps(result))
        metadata = {'phase': 'complete', 'epoch': 1, 'best_epoch': 1, 'identity': identity,
                    'artifact_paths': [str(directory / 'report.json'), str(test), str(best), str(marker), str(history)]}
        checkpoint_callback(task, checkpoint, metadata)
        results[task] = result
    return results


class WorkflowTests(unittest.TestCase):
    def prepared(self, root):
        return {'workdir': str(root), 'dataset_root': str(root / 'dummy_dataset'), 'drive_root_id': 'root'}

    def test_setup_auth_failure_releases_gpu_without_training(self):
        events, store = [], FakeStore()
        store.fail_folder = True
        with tempfile.TemporaryDirectory() as temporary, load_workflow(events) as workflow:
            with self.assertRaises(RuntimeError):
                workflow.run_all(store, self.prepared(Path(temporary)), 'test_run')
        self.assertEqual(events, ['disconnect'])

    def test_missing_resume_epoch_releases_gpu(self):
        events = []
        with tempfile.TemporaryDirectory() as temporary, load_workflow(events) as workflow:
            with self.assertRaises(ValueError):
                workflow.run_all(FakeStore(), self.prepared(Path(temporary)), 'test_run',
                                 resume_modes={task: 'latest' for task in workflow.TASKS})
        self.assertEqual(events, ['disconnect'])

    def test_all_epochs_complete_before_disconnect(self):
        events, store = [], FakeStore()
        store.events = events
        with tempfile.TemporaryDirectory() as temporary, load_workflow(events) as workflow:
            result = workflow.run_all(store, self.prepared(Path(temporary)), 'test_run')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(events[-1], 'disconnect')
        self.assertEqual(sum(name.endswith('epoch_receipt.json') for name in events), 3)
        self.assertEqual(sum(name.endswith('complete_receipt.json') for name in events), 3)
        self.assertTrue(events[-2].endswith('_COMPLETE.json'))

    def test_release_failure_does_not_reclassify_completed_training(self):
        events, store = [], FakeStore()
        store.events = events
        with tempfile.TemporaryDirectory() as temporary, load_workflow(events, release_error=True) as workflow:
            result = workflow.run_all(store, self.prepared(Path(temporary)), 'test_run')
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(any(event.endswith('_FAILED.json') for event in events))

    def test_failed_epoch_upload_saved_to_recovery_before_release(self):
        events, store = [], FakeStore()
        store.events = events
        store.fail_upload = lambda name: name == 'epoch_0001.pt'
        with tempfile.TemporaryDirectory() as temporary, load_workflow(events) as workflow:
            with self.assertRaises(RuntimeError):
                workflow.run_all(store, self.prepared(Path(temporary)), 'test_run')
        self.assertEqual(events[-1], 'disconnect')
        self.assertTrue(any('failure_recovery.zip' in event for event in events))
        self.assertTrue(events[-2].endswith('_FAILED.json'))

    def test_unverified_recovery_retains_runtime(self):
        events, store = [], FakeStore()
        store.events = events
        store.fail_upload = lambda name: name == 'epoch_0001.pt' or 'failure_recovery.zip' in name
        with tempfile.TemporaryDirectory() as temporary, load_workflow(events) as workflow:
            with self.assertRaises(RuntimeError):
                workflow.run_all(store, self.prepared(Path(temporary)), 'test_run')
        self.assertNotIn('disconnect', events)

    def test_latest_completed_restore_preserves_test_cache_and_rewrites_paths(self):
        events, store = [], FakeStore()
        with tempfile.TemporaryDirectory() as temporary, load_workflow(events) as workflow:
            root = Path(temporary)
            result = workflow.run_all(store, self.prepared(root), 'test_run', auto_disconnect=False)
            task_folder = store.ensure_folder('localizer', result['drive_run_folder_id'])
            target = root / 'restored'
            path = workflow._restore(store, task_folder['id'], target, 'latest', 'localizer')
            report = json.loads((target / 'report.json').read_text())
            self.assertEqual(report['last_checkpoint'], path)
            self.assertEqual(report['best_model'], str(target / 'best_model.pt'))
            self.assertTrue((target / 'test_once.json').exists())
            self.assertTrue((target / 'history.json').exists())
            branch = root / 'best_branch'
            workflow._restore(store, task_folder['id'], branch, 'best', 'localizer')
            self.assertFalse((branch / 'test_once.json').exists())
            self.assertFalse((branch / 'report.json').exists())

    def test_test_cache_recovered_when_complete_receipt_missing(self):
        events, store = [], FakeStore()
        with tempfile.TemporaryDirectory() as temporary, load_workflow(events) as workflow:
            root = Path(temporary)
            result = workflow.run_all(store, self.prepared(root), 'test_run', auto_disconnect=False)
            for key in list(store.items):
                if store.items[key]['name'].endswith('_complete_receipt.json'):
                    del store.items[key]
            task_folder = store.ensure_folder('localizer', result['drive_run_folder_id'])
            workflow._restore(store, task_folder['id'], root / 'partial_restored', 'latest', 'localizer')
            self.assertTrue((root / 'partial_restored/test_once.json').is_file())
            self.assertFalse((root / 'partial_restored/report.json').exists())

    def test_prepare_different_archive_never_reuses_old_dataset(self):
        events, store = [], FakeStore()
        with tempfile.TemporaryDirectory() as temporary, load_workflow(events) as workflow:
            root = Path(temporary)
            original = root / 'fixture'
            original.mkdir()
            manifest = fixture(original)
            references = []
            for number in (1, 2):
                manifest['dataset_id'] = 'version' + str(number)
                refresh_manifest(original, manifest)
                archive = root / f'data{number}.zip'
                with zipfile.ZipFile(archive, 'w') as bundle:
                    for path in original.rglob('*'):
                        if path.is_file():
                            bundle.write(path, path.relative_to(original).as_posix())
                ref = store.upload_verified(archive, 'root')
                references.append({'id': ref['id'], 'size': int(ref['size']), 'sha256': ref['verifiedSha256']})
            first = workflow.prepare(store, 'root', references[0], root / 'work')
            second = workflow.prepare(store, 'root', references[1], root / 'work')
            self.assertNotEqual(first['dataset_root'], second['dataset_root'])
            self.assertNotEqual(first['dataset_manifest_sha256'], second['dataset_manifest_sha256'])
            self.assertEqual(json.loads((Path(second['dataset_root']) / 'dataset_manifest.json').read_text())['dataset_id'], 'version2')


if __name__ == '__main__':
    unittest.main()
