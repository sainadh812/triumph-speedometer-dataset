"""Pure/mock tests: no neural model loading, Google access, or GPU execution."""
import contextlib
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

import numpy as np

from test_workflow import FakeStore

HERE = Path(__file__).resolve().parent
PACKAGE = 'joint_video_pure_test'


@contextlib.contextmanager
def module():
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(HERE)]
    model = types.ModuleType(PACKAGE + '.model')
    model.require_colab_cuda = lambda: None
    inference = types.ModuleType(PACKAGE + '.inference')
    inference.DashboardPredictor = lambda *args, **kwargs: object()
    with patch.dict(sys.modules, {PACKAGE: package, PACKAGE + '.model': model,
                                 PACKAGE + '.inference': inference}), contextlib.redirect_stdout(io.StringIO()):
        spec = importlib.util.spec_from_file_location(PACKAGE + '.video_test', HERE / 'video_test.py')
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        yield value


def fixture(root, store):
    folder = store.ensure_folder('test_run', 'root')
    results = {}
    archive = root / 'training.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        for task in ('localizer', 'speed', 'rpm'):
            identity = {'task': task, 'architecture_id': 'mock_arch', 'preprocessing_id': 'mock_preprocess',
                        'dataset_manifest_sha256': 'c' * 64, 'code_sha256': {'model.py': 'a' * 64}}
            report = {'status': 'complete', 'task': task, 'identity': identity, 'best_epoch': 2}
            weights = ('FAKE_BYTES_NOT_MODEL_' + task).encode()
            bundle.writestr(task + '/best_model.pt', weights)
            bundle.writestr(task + '/best_model.pt.sha256.json', json.dumps({
                'sha256': hashlib.sha256(weights).hexdigest(), 'bytes': len(weights)}))
            bundle.writestr(task + '/report.json', json.dumps(report))
            bundle.writestr(task + '/epoch_0001.pt', b'do-not-extract')
            results[task] = report
    saved = store.upload_verified(archive, folder['id'], '20260101_training_results.zip')
    complete = {'status': 'complete', 'run_id': 'test_run', 'attempt': '20260101',
                'drive_run_folder_id': folder['id'], 'archive': saved, 'results': results}
    receipt = store.upload_json(complete, folder['id'], '20260101_COMPLETE.json')
    source = root / 'source.mp4'
    source.write_bytes(b'not-a-real-video')
    video = store.upload_verified(source, 'root')
    reference = {'id': video['id'], 'sha256': video['verifiedSha256'], 'size': int(video['size']),
                 'name': 'source.mp4', 'label': 'day'}
    return ({'drive_root_id': 'root', 'workdir': str(root / 'runtime'), 'dataset_manifest_sha256': 'c' * 64},
            {'run_id': 'test_run', 'complete_id': receipt['id']}, {'videos': [reference]}, complete, archive)


def fake_process(source, directory, predictor, batch_size, seconds, label):
    directory.mkdir()
    (directory / 'labeled.mp4').write_bytes(b'fake-encoded-frames')
    (directory / 'predictions.csv').write_text('frame_index,rpm\n0,2000\n')
    (directory / 'predictions.jsonl').write_text('{"frame_index":0,"rpm":2000}\n')
    report = {'status': 'complete', 'source_label': label, 'frames_processed': 1,
              'model_pipeline_fps': 20., 'end_to_end_processing_fps': 10.,
              'timing': {'total_processing_seconds': .1}}
    (directory / 'report.json').write_text(json.dumps(report))
    return report


class VideoTests(unittest.TestCase):
    def test_exact_complete_and_selected_extraction(self):
        with tempfile.TemporaryDirectory() as temp, module() as runner:
            root, store = Path(temp), FakeStore()
            prepared, run_ref, _, complete, archive = fixture(root, store)
            selected, _ = runner.resolve_complete(store, prepared, run_ref)
            self.assertEqual(selected, complete)
            paths, provenance = runner.extract_models(archive, root / 'models', complete,
                                                       {'dataset_manifest_sha256': 'c' * 64})
            self.assertEqual(set(paths), {'localizer', 'speed', 'rpm'})
            self.assertEqual(len(list((root / 'models').rglob('*.*'))), 9)
            self.assertFalse((root / 'models/localizer/epoch_0001.pt').exists())
            self.assertEqual(provenance['rpm']['best_epoch'], 2)

    def test_archive_unsafe_extra_entry_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, module() as runner:
            root = Path(temp)
            *_, complete, archive = fixture(root, FakeStore())
            with zipfile.ZipFile(archive, 'a') as bundle:
                bundle.writestr('../escape', 'bad')
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                runner.extract_models(archive, root / 'models', complete)
            self.assertFalse((root / 'escape').exists())

    def test_corrupt_checkpoint_and_wrong_dataset_fail_before_prediction(self):
        with tempfile.TemporaryDirectory() as temp, module() as runner:
            root = Path(temp)
            *_, complete, archive = fixture(root, FakeStore())
            with self.assertRaisesRegex(ValueError, 'training identity'):
                runner.extract_models(archive, root / 'wrong', complete,
                                      {'dataset_manifest_sha256': 'd' * 64})
            bad = root / 'bad.zip'
            with zipfile.ZipFile(archive) as original, zipfile.ZipFile(bad, 'w') as sink:
                for item in original.infolist():
                    data = b'corrupted' if item.filename == 'rpm/best_model.pt' else original.read(item)
                    sink.writestr(item, data)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                runner.extract_models(bad, root / 'bad_models', complete)

    def test_every_selected_frame_is_batched_ordered_and_encoded(self):
        class Capture:
            def __init__(self, path): self.index = 0
            def isOpened(self): return True
            def grab(self): self.index += 1; return True
            def read(self):
                frame = np.full((64, 80, 3), self.index, np.uint8)
                self.index += 1
                return True, frame
            def release(self): pass
        outputs, batch_sizes = [], []
        class Writer:
            def __init__(self, path, *args): Path(path).write_bytes(b'intermediate')
            def isOpened(self): return True
            def write(self, frame): outputs.append(frame.copy())
            def release(self): pass
        class Predictor:
            def predict_frames(self, batch):
                batch_sizes.append(len(batch))
                return [{'speed': {'value_kmh': float(frame[0, 0, 0]), 'reason': None,
                                   'point_visibility': {'pivot': .9, 'tip': .9}},
                         'rpm': {'value': None, 'reason': 'bar_not_visible' if frame[0, 0, 0] % 2 == 0 else 'rpm_roi_corner_not_visible',
                                 'point_visibility': {'bar_end': .1} if frame[0, 0, 0] % 2 == 0 else {}},
                         'lcd_status': 'unavailable'} for frame in batch]
        with tempfile.TemporaryDirectory() as temp, module() as runner:
            root = Path(temp)
            source = root / 'source.mp4'
            metadata = {'timestamps': [i / 2 for i in range(12)], 'last_frame_period_seconds': .5,
                        'fps': 2., 'variable_frame_rate': False}
            def probe(path):
                return metadata if Path(path) == source else {'decoded_timestamp_count': 6,
                                                              'stream_height': 64, 'stream_width': 80}
            with patch('cv2.VideoCapture', Capture), patch('cv2.VideoWriter', Writer), \
                    patch.object(runner, 'probe_video', side_effect=probe), \
                    patch.object(runner, '_encode_h264', side_effect=lambda a, b: Path(b).write_bytes(b'h264')):
                report = runner.process_video(source, root / 'out', Predictor(), batch_size=4, last_seconds=3.)
            self.assertEqual(batch_sizes, [4, 2])
            self.assertEqual(len(outputs), 6)
            self.assertTrue(all(frame.shape == (64, 80, 3) for frame in outputs))
            rows = [json.loads(row) for row in (root / 'out/predictions.jsonl').read_text().splitlines()]
            self.assertEqual([row['frame_index'] for row in rows], list(range(6, 12)))
            self.assertEqual([row['speed_kmh'] for row in rows], list(range(6, 12)))
            self.assertEqual(report['availability']['rpm_available'], 0)
            self.assertEqual(report['first_source_timestamp_seconds'], 3.)
            self.assertEqual(report['output_playback_fps'], 2.)
            self.assertEqual(report['actual_task_inference_frames'], {'localizer': 6, 'speed': 6, 'rpm': 3, 'all_three': 3})
            self.assertEqual(report['actual_task_inference_coverage']['rpm'], .5)
            self.assertGreaterEqual(report['timing']['initial_probe_seconds'], 0)
            self.assertAlmostEqual(report['timing']['decode_seconds'], sum(report['decode_timing_breakdown'].values()))
            self.assertEqual(report['prediction_pipeline_fps'], report['model_pipeline_fps'])
            self.assertFalse((root / 'out/labeled_intermediate.mp4').exists())

    def test_unavailable_reading_does_not_imply_reader_was_skipped(self):
        with module() as runner:
            self.assertEqual(runner._task_inference_flags({
                'speed': {'value_kmh': None, 'reason': 'predicted_pivot_inconsistent_with_tick_plane',
                          'point_visibility': {'pivot': .9}},
                'rpm': {'value': None, 'reason': 'rpm_roi_corner_not_visible', 'point_visibility': {}}
            }), {'localizer': True, 'speed': True, 'rpm': False})

    def test_all_results_verified_before_release_and_provenance_in_archive(self):
        events = []
        with tempfile.TemporaryDirectory() as temp, module() as runner:
            store = FakeStore(events)
            prepared, run_ref, config, _, _ = fixture(Path(temp), store)
            with patch.object(runner, 'process_video', side_effect=fake_process), \
                    patch.object(runner, '_disconnect', side_effect=lambda: events.append('disconnect')):
                result = runner.run_video_tests(store, prepared, run_ref, config)
            self.assertEqual(events[-1], 'disconnect')
            self.assertEqual(events[-2], 'upload:COMPLETE.json')
            self.assertTrue(result['all_result_files_verified_in_drive'])
            archive_bytes = store.payloads[result['results_archive']['id']]
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as bundle:
                self.assertIn('provenance.json', bundle.namelist())
                self.assertIn('video_01/predictions.jsonl', bundle.namelist())

    def test_failed_results_upload_preserves_recovery_before_release(self):
        events = []
        with tempfile.TemporaryDirectory() as temp, module() as runner:
            store = FakeStore(events)
            prepared, run_ref, config, _, _ = fixture(Path(temp), store)
            store.fail_upload = lambda name: name == 'video_01_labeled.mp4'
            with patch.object(runner, 'process_video', side_effect=fake_process), \
                    patch.object(runner, '_disconnect', side_effect=lambda: events.append('disconnect')):
                with self.assertRaises(RuntimeError):
                    runner.run_video_tests(store, prepared, run_ref, config)
            self.assertTrue(any('failure_recovery.zip' in event for event in events))
            self.assertEqual(events[-1], 'disconnect')

    def test_failed_recovery_retains_runtime(self):
        events = []
        with tempfile.TemporaryDirectory() as temp, module() as runner:
            store = FakeStore(events)
            prepared, run_ref, config, _, _ = fixture(Path(temp), store)
            store.fail_upload = lambda name: True
            with patch.object(runner, 'process_video', side_effect=fake_process), \
                    patch.object(runner, '_disconnect', side_effect=lambda: events.append('disconnect')):
                with self.assertRaises(RuntimeError):
                    runner.run_video_tests(store, prepared, run_ref, config)
            self.assertNotIn('disconnect', events)

    def test_missing_training_complete_releases_without_prediction(self):
        events = []
        with tempfile.TemporaryDirectory() as temp, module() as runner:
            store = FakeStore(events)
            prepared, run_ref, config, _, _ = fixture(Path(temp), store)
            run_ref['complete_id'] = 'nonexistent'
            with patch.object(runner, 'process_video') as process, \
                    patch.object(runner, '_disconnect', side_effect=lambda: events.append('disconnect')):
                with self.assertRaisesRegex(ValueError, 'COMPLETE'):
                    runner.run_video_tests(store, prepared, run_ref, config)
                process.assert_not_called()
            self.assertEqual(events[-1], 'disconnect')


if __name__ == '__main__':
    unittest.main()
