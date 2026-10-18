"""Offline workflow tests: fake Drive, fake subprocess, no Google/ML imports."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import zipfile

import api_colab_workflow as workflow


FOLDER = 'application/vnd.google-apps.folder'
RESTORE_ASSETS = workflow.restore_assets


def digest(data):
    return hashlib.sha256(data).hexdigest()


class FakeStore:
    def __init__(self):
        self.items, self.payloads, self.events = {}, {}, []
        self.fail_upload_name = None
        self.lose_result_ack = False
        self.downloads = 0

    def _create(self, name, parent=None, data=None, mime='application/octet-stream'):
        ident = 'id_' + str(len(self.items) + 1)
        row = {'id': ident, 'name': name, 'mimeType': mime,
               'parents': [parent] if parent else [], 'createdTime': f'2026-10-18T00:00:{len(self.items):02d}Z'}
        if data is not None:
            row.update(size=str(len(data)), sha256Checksum=digest(data))
            self.payloads[ident] = data
        self.items[ident] = row
        return dict(row)

    def ensure_folder(self, name, parent=None):
        matches = [row for row in self.items.values() if row['name'] == name and row['parents'] == ([parent] if parent else [])]
        if len(matches) > 1:
            raise ValueError('Ambiguous fake folder')
        if matches:
            if matches[0]['mimeType'] != FOLDER:
                raise ValueError('Folder/file collision')
            return dict(matches[0])
        return self._create(name, parent, mime=FOLDER)

    def list_files(self, parent):
        return [dict(row) for row in self.items.values() if parent in row['parents']]

    def find_file(self, name, parent):
        rows = [row for row in self.list_files(parent) if row['name'] == name]
        if len(rows) > 1:
            raise ValueError('Duplicate fake file')
        return rows[0] if rows else None

    def metadata(self, ident):
        return dict(self.items[ident])

    def read_json(self, ident):
        return json.loads(self.payloads[ident])

    def _upload(self, data, parent, name):
        if name == self.fail_upload_name:
            raise workflow.DriveStoreError('Simulated transfer failure')
        existing = self.find_file(name, parent)
        if existing:
            if self.payloads[existing['id']] != data:
                raise workflow.DriveStoreError('Immutable fake file differs')
            return existing
        row = self._create(name, parent, data)
        self.events.append(('upload', parent, name))
        if name == 'result.json' and self.lose_result_ack:
            self.lose_result_ack = False
            raise workflow.DriveStoreError('Simulated lost final acknowledgement')
        return row

    def upload_verified(self, path, parent, name=None):
        path = Path(path)
        return self._upload(path.read_bytes(), parent, name or path.name)

    def upload_json(self, value, parent, name):
        return self._upload((json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode(), parent, name)

    def download_verified(self, ident, target, sha256, size):
        data = self.payloads[ident]
        if len(data) != size or digest(data) != sha256:
            raise ValueError('Fake download failed checksum')
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.downloads += 1
        return self.metadata(ident)


class FakeHelpers:
    JOBS_REL = 'video_tests/browser_upload_v1'

    def __init__(self):
        self.sequence = 0
        self.inference_calls = 0
        self.disconnections = 0
        self.publish_failure = False
        self.summary = None

    @staticmethod
    def sha256(path):
        return digest(Path(path).read_bytes())

    def verify(self, path, sha256, size=None):
        data = Path(path).read_bytes()
        if digest(data) != sha256 or (size is not None and len(data) != size):
            raise ValueError('Local checksum mismatch')

    @staticmethod
    def utc_now():
        return '2026-10-18T00:00:00Z'

    def unique_id(self, prefix):
        self.sequence += 1
        return prefix + '_' + str(self.sequence)

    @staticmethod
    def gpu_present():
        return True

    @staticmethod
    def require_gpu():
        return None

    def disconnect_runtime(self):
        self.disconnections += 1

    @staticmethod
    def write_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding='utf8')

    def save_job(self, job, path):
        self.write_json(path, job)

    def drive_relative(self, path):
        return Path(path).relative_to(self.DRIVE_ROOT).as_posix()

    def safe_drive_path(self, relative):
        return self.DRIVE_ROOT / relative

    class Tee:
        def __init__(self, original, log):
            self.original, self.log = original, log
        def write(self, value):
            self.original.write(value)
            self.log.write(value)
        def flush(self):
            self.original.flush()
            self.log.flush()

    def prepare_assets(self, work, runtime):
        repo, inputs, dataset = work / 'repo', work / 'inputs', work / 'dataset'
        for directory in (repo, inputs, dataset):
            directory.mkdir(parents=True)
        (repo / 'evaluate_uploaded_video.py').write_text('# Dummy, never executed\n')
        return repo, inputs, dataset, work / 'provenance.json'

    def copy_verified(self, source, target, sha256, size):
        self.verify(source, sha256, size)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return target

    def run_logged(self, command, cwd=None):
        # Tests exercise orchestration, not the actual subprocess/model program.
        self.inference_calls += 1
        output = Path(command[command.index('--output-root') + 1])
        output.mkdir()
        (output / 'annotated_fast_lcd_speed_rpm.mp4').write_bytes(b'FAKE_OUTPUT_NO_VIDEO_OR_MODEL')
        self.summary = {'output': str(output / 'annotated_fast_lcd_speed_rpm.mp4'),
                        'output_bytes': 29, 'output_sha256': self.sha256(output / 'annotated_fast_lcd_speed_rpm.mp4'),
                        'source_frames_processed': 10, 'reader_processing_fps': 5.,
                        'decode_inference_render_encode_fps': 2., 'elapsed_seconds': 6.,
                        'network_timing': {name: {'forward_wall_seconds': .1} for name in ('lcd', 'speed', 'rpm')},
                        'phase_seconds': {'render_seconds': 2., 'reader_wall_seconds': 2., 'decode_wait_seconds': .5,
                                          'encoder_pipe_seconds': .5},
                        'timing_scope': 'Fake reader and full pipeline are distinct.'}
        self.summary['output_bytes'] = Path(self.summary['output']).stat().st_size
        self.write_json(output / 'summary.json', self.summary)

    def validate_results(self, directory, job, evaluator_sha):
        return self.summary

    def publish_run(self, work, job, job_path, run_id, summary=None, error=None):
        if self.publish_failure:
            raise OSError('Simulated local publication failure')
        dest = job_path.parent / 'runs' / run_id
        shutil.copytree(work / 'evaluation', dest / 'evaluation')
        return {'run_id': run_id, 'status': 'complete_and_verified',
                'drive_directory_relative': self.drive_relative(dest)}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.store = FakeStore()
        self.root = self.store.ensure_folder(workflow.WORKSPACE_NAME)
        self.h = FakeHelpers()
        self.output = io.StringIO()
        for patch in (mock.patch.object(workflow, 'CONTENT_ROOT', self.directory),
                      mock.patch.object(workflow, 'PENDING_PATH', self.directory / 'pending.json'),
                      mock.patch.object(workflow, 'helpers', return_value=self.h),
                      mock.patch.object(workflow, 'connect', return_value=(self.store, self.root)),
                      mock.patch.object(workflow, 'restore_assets', return_value={}),
                      mock.patch.object(workflow, 'show_result')):
            patch.start()
            self.addCleanup(patch.stop)
        self.capture = contextlib.redirect_stdout(self.output)
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)
        self.job, self.folder = self.make_job('video_current')

    def make_job(self, name):
        jobs = self.store.ensure_folder('jobs', self.root['id'])
        folder = self.store.ensure_folder(name, jobs['id'])
        content = ('FAKE_SOURCE_' + name).encode()
        source = self.store._create('source.mp4', folder['id'], content)
        job = {'schema_version': 1, 'job_id': name, 'asset_bundle_sha256': workflow.ASSET_BUNDLE_SHA256,
               'start_seconds': 0., 'max_seconds': 0., 'source': {'file_id': source['id'],
               'filename': source['name'], 'original_name': 'user.mp4', 'sha256': digest(content), 'bytes': len(content)}}
        self.store.upload_json(job, folder['id'], 'job.json')
        return job, folder

    def pending(self):
        directory = self.directory / 'published_local'
        directory.mkdir()
        (directory / 'evaluation').mkdir()
        video = directory / 'evaluation' / 'annotated_fast_lcd_speed_rpm.mp4'
        video.write_bytes(b'FAKE_SAVED_RESULT')
        (directory / 'result_manifest.json').write_text('{"dummy":true}')
        pending = {**workflow.job_identity(self.job, self.folder['id']), 'directory': str(directory),
                   'run_id': 'run_one', 'status': 'complete_and_verified',
                   'output_relative': video.relative_to(directory).as_posix(),
                   'output_sha256': digest(video.read_bytes()), 'output_bytes': video.stat().st_size}
        self.h.write_json(workflow.PENDING_PATH, pending)
        return pending

    def test_latest_submitted_job_not_incomplete_upload_or_older_result(self):
        newer, newer_folder = self.make_job('video_newer')
        jobs = self.store.ensure_folder('jobs', self.root['id'])
        self.store.ensure_folder('interrupted_newest_upload', jobs['id'])
        found, folder = workflow.latest_job(self.store, self.root)
        self.assertEqual(found['job_id'], newer['job_id'])
        self.assertEqual(folder['id'], newer_folder['id'])
        self.store.items[newer['source']['file_id']]['parents'] = [self.folder['id']]
        with self.assertRaisesRegex(ValueError, 'source file'):
            workflow.latest_job(self.store, self.root)

    def test_upload_failure_keeps_pointer_and_never_publishes_completion(self):
        pending = self.pending()
        self.store.fail_upload_name = 'annotated_fast_lcd_speed_rpm.mp4'
        with self.assertRaises(workflow.DriveStoreError):
            workflow.save_pending(self.store, pending, self.h)
        self.assertTrue(workflow.PENDING_PATH.exists())
        self.assertIsNone(self.store.find_file('result.json', self.folder['id']))
        self.assertEqual(self.h.inference_calls, 0)

    def test_result_is_last_and_retry_after_lost_receipt_ack_is_immutable(self):
        pending = self.pending()
        self.store.lose_result_ack = True
        with self.assertRaises(workflow.DriveStoreError):
            workflow.save_pending(self.store, pending, self.h)
        self.assertTrue(workflow.PENDING_PATH.exists())
        self.assertEqual(self.store.events[-1][2], 'result.json')
        before = dict(self.store.payloads)
        receipt = workflow.retry_pending(self.directory)
        self.assertEqual(self.store.payloads, before)
        self.assertFalse(workflow.PENDING_PATH.exists())
        self.assertEqual(receipt['source_sha256'], self.job['source']['sha256'])
        self.assertEqual(self.h.disconnections, 1)
        self.assertEqual(self.h.inference_calls, 0)

    def test_optional_display_failure_does_not_block_safe_release(self):
        self.pending()
        with mock.patch.object(workflow, 'show_result', side_effect=RuntimeError('Display unavailable')):
            receipt = workflow.retry_pending(self.directory)
        self.assertEqual(receipt['status'], 'complete_and_verified')
        self.assertFalse(workflow.PENDING_PATH.exists())
        self.assertEqual(self.h.disconnections, 1)

    def test_retry_refuses_changed_job_identity_before_upload(self):
        pending = self.pending()
        pending['source_sha256'] = '0' * 64
        self.h.write_json(workflow.PENDING_PATH, pending)
        before = len(self.store.events)
        with self.assertRaisesRegex(ValueError, 'identity'):
            workflow.retry_pending(self.directory)
        self.assertEqual(len(self.store.events), before)
        self.assertTrue(workflow.PENDING_PATH.exists())
        self.assertEqual(self.h.disconnections, 0)

    def test_pending_video_change_is_rejected_before_upload(self):
        pending = self.pending()
        (Path(pending['directory']) / pending['output_relative']).write_bytes(b'CHANGED')
        before = len(self.store.events)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            workflow.save_pending(self.store, pending, self.h)
        self.assertEqual(len(self.store.events), before)

    def test_remote_save_failure_retry_does_not_repeat_inference(self):
        self.store.fail_upload_name = 'annotated_fast_lcd_speed_rpm.mp4'
        with self.assertRaises(workflow.DriveStoreError):
            workflow.run_video(self.directory)
        self.assertEqual(self.h.inference_calls, 1)
        self.assertEqual(self.h.disconnections, 0)
        self.assertTrue(workflow.PENDING_PATH.exists())
        self.store.fail_upload_name = None
        receipt = workflow.run_video(self.directory)
        self.assertEqual(self.h.inference_calls, 1)
        self.assertEqual(self.h.disconnections, 1)
        self.assertFalse(workflow.active_path().exists())
        self.assertEqual(receipt['timings']['reader_processing_fps'], 5.)
        self.assertEqual(receipt['timings']['decode_inference_render_encode_fps'], 2.)
        self.assertAlmostEqual(receipt['timings']['combined_network_forward_only_fps'], 10 / .3)

    def test_failed_local_publication_retains_runtime_and_blocks_repeat(self):
        self.h.publish_failure = True
        with self.assertRaises(OSError):
            workflow.run_video(self.directory)
        self.assertFalse(workflow.PENDING_PATH.exists())
        self.assertTrue(workflow.active_path().exists())
        self.assertEqual(self.h.disconnections, 0)
        self.assertNotIn('Retry saving with workflow.retry_pending', self.output.getvalue())
        with self.assertRaisesRegex(RuntimeError, 'manual recovery'):
            workflow.run_video(self.directory)
        self.assertEqual(self.h.inference_calls, 1)
        self.assertEqual(self.h.disconnections, 0)

    def test_completed_job_reuses_verified_output_without_inference(self):
        pending = self.pending()
        workflow.save_pending(self.store, pending, self.h)
        receipt = workflow.run_video(self.directory)
        self.assertEqual(receipt['run_id'], 'run_one')
        self.assertEqual(self.h.inference_calls, 0)
        self.assertEqual(self.h.disconnections, 1)

    def test_completed_receipt_cannot_relabel_source_or_changed_output(self):
        pending = self.pending()
        receipt = workflow.save_pending(self.store, pending, self.h)
        changed = dict(receipt, source_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'identity'):
            workflow.validate_completed_receipt(self.store, changed, self.job, self.folder)
        self.store.items[receipt['output_file_id']]['sha256Checksum'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            workflow.validate_completed_receipt(self.store, receipt, self.job, self.folder)

    def test_completed_checksum_fallback_download_has_no_inference(self):
        receipt = workflow.save_pending(self.store, self.pending(), self.h)
        self.store.items[receipt['output_file_id']]['sha256Checksum'] = None
        workflow.validate_completed_receipt(self.store, receipt, self.job, self.folder)
        self.assertEqual(self.store.downloads, 1)
        self.assertEqual(self.h.inference_calls, 0)

    def test_prior_final_receipt_with_wrong_identity_cannot_be_overwritten(self):
        pending = self.pending()
        receipt = workflow.save_pending(self.store, pending, self.h)
        wrong = dict(receipt, source_bytes=999)
        item = self.store.find_file('result.json', self.folder['id'])
        self.store.payloads[item['id']] = json.dumps(wrong).encode()
        self.h.write_json(workflow.PENDING_PATH, pending)
        with self.assertRaisesRegex(ValueError, 'Existing final receipt'):
            workflow.save_pending(self.store, pending, self.h)
        self.assertEqual(self.store.read_json(item['id']), wrong)
        self.assertTrue(workflow.PENDING_PATH.exists())

    def test_wrong_model_bundle_fails_before_inference(self):
        item = self.store.find_file('job.json', self.folder['id'])
        changed = dict(self.job, asset_bundle_sha256='f' * 64)
        self.store.payloads[item['id']] = json.dumps(changed).encode()
        with self.assertRaisesRegex(ValueError, 'model package'):
            workflow.run_video(self.directory)
        self.assertEqual(self.h.inference_calls, 0)
        self.assertEqual(self.h.disconnections, 1)

    def test_preserved_partial_failure_never_publishes_completed_receipt(self):
        pending = self.pending()
        pending['status'] = 'failed_partial_outputs_unverified'
        self.h.write_json(workflow.PENDING_PATH, pending)
        receipt = workflow.retry_pending(self.directory)
        self.assertEqual(receipt['status'], 'failed_partial_outputs_unverified')
        self.assertNotIn('output_file_id', receipt)
        self.assertIsNone(self.store.find_file('result.json', self.folder['id']))
        self.assertIsNotNone(self.store.find_file('run_one_failed.json', self.folder['id']))
        self.assertEqual(self.h.disconnections, 1)

    def test_restore_uses_exact_manifest_and_hashes_without_model_imports(self):
        helper_path = Path(__file__).resolve().parents[1] / 'upload_video_test_v1' / 'colab_video_helpers.py'
        spec = importlib.util.spec_from_file_location('offline_frozen_zip_helpers', helper_path)
        h = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(h)  # This helper has stdlib-only imports and no I/O at import.
        data = b'DUMMY_MODEL_BYTES_NEVER_LOADED'
        name = 'experiments/dummy/best_model.pt'
        manifest = {'schema_version': 1, 'files': [{'path': name, 'sha256': digest(data), 'bytes': len(data)}]}
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as bundle:
            bundle.writestr('asset_manifest.json', json.dumps(manifest))
            bundle.writestr(name, data)
        assets = self.store.ensure_folder('assets', self.root['id'])
        remote = self.store._create(workflow.ASSET_BUNDLE_NAME, assets['id'], archive.getvalue())
        with mock.patch.object(workflow, 'ASSET_BUNDLE_SHA256', digest(archive.getvalue())), \
             mock.patch.object(workflow, 'ASSET_BUNDLE_BYTES', len(archive.getvalue())):
            restored = RESTORE_ASSETS(self.store, self.root, self.directory / 'restore', h)
        self.assertEqual(restored, manifest)
        self.assertEqual((self.directory / 'restore' / name).read_bytes(), data)
        self.assertEqual(self.h.inference_calls, 0)
        self.assertEqual(self.store.downloads, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
