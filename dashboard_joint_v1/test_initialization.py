"""Pure payload, mocked loader, and mocked Drive tests; never execute a model."""
import copy
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import zipfile

from initialization import load_initialization, validate_initialization_payload, verified_weight_file
from test_workflow import FakeStore, load_workflow
from test_video_test import fixture as video_fixture


def identities(task='speed'):
    source = {'task': task, 'architecture_id': 'arch1', 'preprocessing_id': 'gray1',
              'input_hw': [512, 512], 'point_names': ['pivot', 'tip'],
              'dataset_manifest_sha256': 'a' * 64, 'samples_sha256': 'b' * 64,
              'code_sha256': {'train.py': 'c' * 64}}
    target = copy.deepcopy(source)
    target.update(dataset_manifest_sha256='d' * 64, samples_sha256='e' * 64,
                  code_sha256={'train.py': 'f' * 64})
    return source, target


def payload():
    source, target = identities()
    return {'checkpoint_kind': 'dashboard_joint_v1_inference', 'format_version': 1,
            'identity': source, 'model_state_dict': {'mock.parameter': 'FAKE_NOT_A_TENSOR'},
            'best_epoch': 11, 'validation_selection_score': 3.85}, target


class InitializationTests(unittest.TestCase):
    def test_training_model_factory_strictly_loads_selected_heads_without_pretrained_download(self):
        tree = ast.parse(Path(__file__).with_name('train.py').read_text())
        function = next(node for node in tree.body if getattr(node, 'name', None) == '_initial_model')
        model = Mock()
        constructor = Mock(return_value=Mock(to=Mock(return_value=model)))
        weights, provenance = {'heads': 'FAKE'}, {'mode': 'weights_only_transfer_new_experiment'}
        loader = Mock(return_value=(weights, provenance))
        space = {'InstrumentModel': constructor, 'load_initialization': loader}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'isolated-model-initialization', 'exec'), space)
        factory = space['_initial_model']
        validated = {'tasks': {'speed': {'input_hw': [512, 512]}}}
        identity = {'task': 'speed'}
        actual, record = factory('speed', validated, identity, 'mock_cuda', 'selected.pt')
        self.assertIs(actual, model)
        self.assertIs(record, provenance)
        constructor.assert_called_once_with('speed', [512, 512], pretrained=False)
        model.load_state_dict.assert_called_once_with(weights, strict=True)
        loader.assert_called_once_with('selected.pt', identity)
        # Incompatible tensor/head shapes must fail, not silently drop unmatched keys.
        model.load_state_dict.side_effect = RuntimeError('shape mismatch')
        with self.assertRaisesRegex(RuntimeError, 'shape mismatch'):
            factory('speed', validated, identity, 'mock_cuda', 'selected.pt')
        constructor.reset_mock(); loader.reset_mock(); model.reset_mock()
        with self.assertRaisesRegex(ValueError, 'combine'):
            factory('speed', validated, identity, 'mock_cuda', 'selected.pt', resuming=True)
        constructor.assert_not_called(); loader.assert_not_called()
        factory('speed', validated, identity, 'mock_cuda', resuming=True)
        constructor.assert_called_once_with('speed', [512, 512], pretrained=False)
        loader.assert_not_called(); model.load_state_dict.assert_not_called()
        constructor.reset_mock()
        factory('speed', validated, identity, 'mock_cuda')
        constructor.assert_called_once_with('speed', [512, 512], pretrained=True)

    def test_new_dataset_and_code_allowed_only_as_weights_transfer(self):
        value, target = payload()
        state, provenance = validate_initialization_payload(value, target, '0' * 64, 101)
        self.assertIs(state, value['model_state_dict'])
        self.assertFalse(provenance['optimizer_scheduler_scaler_rng_history_restored'])
        self.assertEqual(provenance['source_identity']['dataset_manifest_sha256'], 'a' * 64)
        self.assertEqual(provenance['target_dataset_manifest_sha256'], 'd' * 64)
        value['identity']['task'] = 'mutated'
        self.assertEqual(provenance['source_identity']['task'], 'speed')

    def test_all_structural_mismatches_rejected(self):
        for field, bad in [('task', 'rpm'), ('architecture_id', 'other'), ('preprocessing_id', 'rgb'),
                           ('input_hw', [384, 256]), ('point_names', ['tip', 'pivot'])]:
            with self.subTest(field=field):
                value, target = payload(); target[field] = bad
                with self.assertRaisesRegex(ValueError, field):
                    validate_initialization_payload(value, target, '0' * 64, 101)

    def test_full_epoch_and_spoofed_full_state_rejected(self):
        value, target = payload();value['checkpoint_kind'] = 'dashboard_joint_v1_epoch_full_state'
        with self.assertRaisesRegex(ValueError, 'exported best-model'):
            validate_initialization_payload(value, target, '0' * 64, 101)
        for key in ('optimizer_state_dict', 'history', 'rng_state', 'scaler_state_dict', 'best_state_dict'):
            value, target = payload();value[key] = {}
            with self.assertRaisesRegex(ValueError, 'Full training state'):
                validate_initialization_payload(value, target, '0' * 64, 101)

    def test_incomplete_provenance_and_empty_weights_rejected(self):
        for field, bad in [('best_epoch', 0), ('validation_selection_score', float('nan')),
                           ('model_state_dict', {}), ('format_version', True)]:
            value, target = payload();value[field] = bad
            with self.assertRaises(ValueError):
                validate_initialization_payload(value, target, '0' * 64, 101)
        value, target = payload();target.pop('samples_sha256')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            validate_initialization_payload(value, target, '0' * 64, 101)

    def test_selected_epoch_zero_transfers_only_with_matching_baseline_provenance(self):
        value, target = payload()
        value['best_epoch'] = 0
        value['initialization'] = {'baseline_eligible_for_selection': True,
                                  'validation_before_finetuning': {'selection_score': value['validation_selection_score']}}
        state, provenance = validate_initialization_payload(value, target, '0' * 64, 101)
        self.assertEqual(state, value['model_state_dict'])
        self.assertEqual(provenance['source_best_epoch'], 0)
        self.assertEqual(provenance['source_selection_origin'], 'initialization')
        value['initialization']['validation_before_finetuning']['selection_score'] += 1
        with self.assertRaisesRegex(ValueError, 'epoch zero'):
            validate_initialization_payload(value, target, '0' * 64, 101)

    def test_verified_load_uses_restricted_reader_and_never_restores_state(self):
        value, target = payload()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'best_model.pt';path.write_bytes(b'FAKE_WEIGHTS')
            marker={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
            path.with_name(path.name+'.sha256.json').write_text(json.dumps(marker))
            loader=Mock(return_value=value)
            state, provenance=load_initialization(path,target,loader=loader)
            loader.assert_called_once_with(path,map_location='cpu',weights_only=True)
            self.assertEqual(set(state),{'mock.parameter'})
            self.assertEqual(provenance['source_weights_sha256'],marker['sha256'])
            path.write_bytes(b'CHANGED')
            loader.reset_mock()
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_initialization(path,target,loader=loader)
            loader.assert_not_called()

    def test_missing_marker_rejected_before_loader(self):
        value,target=payload()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'best_model.pt';path.write_bytes(b'x');loader=Mock()
            with self.assertRaisesRegex(ValueError,'checksum marker'):
                load_initialization(path,target,loader=loader)
            loader.assert_not_called()

    def test_cpu_prepares_exact_completed_models_without_epoch_files(self):
        with tempfile.TemporaryDirectory() as temporary, load_workflow([]) as workflow:
            root,store=Path(temporary),FakeStore()
            prepared,reference,_,complete,_=video_fixture(root,store)
            reference['expected_identity']={'dataset_manifest_sha256':'c'*64}
            count_before=len(store.items)
            result=workflow.prepare_initialization(store,prepared,reference)
            self.assertEqual(len(store.items),count_before)  # read-only remote access
            self.assertEqual(set(result['weights']),set(workflow.TASKS))
            selected=Path(next(iter(result['weights'].values()))).parents[1]
            self.assertEqual(len(list(selected.rglob('*.*'))),6)
            self.assertFalse(list(selected.rglob('epoch_*.pt')))
            for task,path in result['weights'].items():
                self.assertEqual(verified_weight_file(path)['sha256'],result['models'][task]['sha256'])
            prepared['initialization']=result
            modes={task:'fresh' for task in workflow.TASKS};modes['speed']='latest'
            paths,_=workflow._prepared_initialization_weights(prepared,reference,modes,'new_run')
            self.assertEqual(set(paths),{'localizer','rpm'})
            with self.assertRaisesRegex(ValueError,'NEW run ID'):
                workflow._prepared_initialization_weights(prepared,reference,modes,'test_run')

    def test_explicit_reference_requires_cpu_preflight_and_exact_source(self):
        with tempfile.TemporaryDirectory() as temporary, load_workflow([]) as workflow:
            root,store=Path(temporary),FakeStore()
            prepared,reference,*_=video_fixture(root,store)
            with self.assertRaisesRegex(ValueError,'current runtime'):
                workflow._prepared_initialization_weights(prepared,reference,{t:'fresh' for t in workflow.TASKS},'new_run')
            with self.assertRaisesRegex(ValueError,'exact source'):
                workflow.prepare_initialization(store,prepared,{'run_id':'test_run'})
            wrong=dict(reference,complete_id='absent')
            with self.assertRaisesRegex(ValueError,'not in its source run'):
                workflow.prepare_initialization(store,prepared,wrong)
            wrong=dict(reference,expected_identity={'dataset_manifest_sha256':'f'*64})
            with self.assertRaisesRegex(ValueError,'source identity differs'):
                workflow.prepare_initialization(store,prepared,wrong)

    def test_unsafe_zip_and_checksum_mismatch_fail_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary, load_workflow([]) as workflow:
            root,store=Path(temporary),FakeStore()
            *_,complete,archive=video_fixture(root,store)
            bad=root/'bad.zip'
            with zipfile.ZipFile(archive) as source,zipfile.ZipFile(bad,'w') as sink:
                for item in source.infolist():
                    sink.writestr(item,b'corrupt' if item.filename=='rpm/best_model.pt' else source.read(item))
            with self.assertRaisesRegex(ValueError,'checksum'):
                workflow._extract_initialization_models(bad,root/'bad_models',complete)
            self.assertFalse((root/'bad_models').exists())
            with zipfile.ZipFile(archive,'a') as bundle:bundle.writestr('../escape','bad')
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                workflow._extract_initialization_models(archive,root/'unsafe_models',complete)
            self.assertFalse((root/'unsafe_models').exists())

    def test_cached_weights_tamper_blocks_gpu_path(self):
        with tempfile.TemporaryDirectory() as temporary, load_workflow([]) as workflow:
            root,store=Path(temporary),FakeStore();prepared,reference,*_=video_fixture(root,store)
            result=workflow.prepare_initialization(store,prepared,reference);prepared['initialization']=result
            Path(result['weights']['speed']).write_bytes(b'tamper')
            with self.assertRaisesRegex(ValueError,'checksum'):
                workflow._prepared_initialization_weights(prepared,reference,{t:'fresh' for t in workflow.TASKS},'new_run')

    def test_all_resume_needs_no_initialization_reference_or_cache(self):
        with tempfile.TemporaryDirectory() as temporary, load_workflow([]) as workflow:
            prepared = {'workdir': temporary}
            modes = {'localizer': 'latest', 'speed': 'best', 'rpm': 'latest'}
            # A stale reference and an erased /content cache cannot block exact resume.
            reference = {'run_id': 'old_source', 'complete_id': 'unused'}
            self.assertEqual(workflow._prepared_initialization_weights(prepared, reference, modes, 'ongoing_run'),
                             ({}, None))
            prepared['initialization'] = {'reference': reference, 'weights': {'speed': 'missing'}}
            self.assertEqual(workflow._prepared_initialization_weights(prepared, reference, modes, 'ongoing_run'),
                             ({}, None))

    def test_mixed_resume_verifies_only_fresh_task_weights(self):
        with tempfile.TemporaryDirectory() as temporary, load_workflow([]) as workflow:
            root, store = Path(temporary), FakeStore()
            prepared, reference, *_ = video_fixture(root, store)
            prepared['initialization'] = workflow.prepare_initialization(store, prepared, reference)
            Path(prepared['initialization']['weights']['speed']).unlink()
            modes = {'localizer': 'fresh', 'speed': 'latest', 'rpm': 'fresh'}
            weights, provenance = workflow._prepared_initialization_weights(prepared, reference, modes, 'new_run')
            self.assertEqual(set(weights), {'localizer', 'rpm'})
            self.assertEqual(provenance['reference'], reference)

    def test_workflow_passes_weights_to_smoke_and_training(self):
        events=[];calls=[]
        def smoke(*args,**kwargs):
            calls.append(('smoke',kwargs['initialization_weights']));return {'status':'passed'}
        from test_workflow import simulated_training
        def train(*args,**kwargs):
            calls.append(('train',kwargs['initialization_weights']));return simulated_training(*args,**kwargs)
        with tempfile.TemporaryDirectory() as temporary,load_workflow(events,fake_train=train,fake_smoke=smoke) as workflow:
            root,store=Path(temporary),FakeStore(events);prepared,reference,*_=video_fixture(root,store)
            prepared['dataset_root']=str(root/'dummy_dataset')
            prepared['initialization']=workflow.prepare_initialization(store,prepared,reference)
            result=workflow.run_all(store,prepared,'new_run',initialization_reference=reference)
            self.assertEqual([x[0] for x in calls],['smoke','train'])
            self.assertEqual(set(calls[0][1]),set(workflow.TASKS))
            self.assertEqual(result['initialization']['reference'],reference)
            self.assertEqual(events[-1],'disconnect')


if __name__=='__main__':unittest.main()
