"""Build a self-contained Colab notebook from tested sources and frozen dataset."""
from pathlib import Path
import base64
import gzip
import hashlib
import json
import textwrap

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOURCES = ('ring_training_v2/model.py', 'ring_training_v2/data.py', 'ring_training_v2/train.py',
           'ring_training_v2/evaluation.py', 'ring_training/ring_model.py', 'ring_training/ring_data.py',
           'ring_training/ring_train.py', 'ring_training/ring_eval.py', 'training_pilot/train_core.py',
           'ring_geometry_pilot/geometry_reader.py')


def build():
    sources = {p: (PROJECT/p).read_bytes() for p in SOURCES}
    hashes = {p: hashlib.sha256(v).hexdigest() for p,v in sources.items()}
    packed = base64.b64encode(gzip.compress(json.dumps({p:base64.b64encode(v).decode() for p,v in sources.items()}).encode(),mtime=0)).decode()
    archive = HERE/'arc_needle_augmented700.zip'
    zip_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    labels_hash = hashlib.sha256((HERE/'dataset_v2/labels.json').read_bytes()).hexdigest()
    cells = []
    def add(kind, text, hidden=False):
        text = textwrap.dedent(text).strip()+'\n'
        c = dict(cell_type=kind, source=text.splitlines(keepends=True),
                 metadata={'cellView':'form'} if hidden else {}, id=hashlib.sha256(text.encode()).hexdigest()[:12])
        if kind == 'code':
            compile(text, f'cell_{len(cells)}', 'exec'); c.update(execution_count=None, outputs=[])
        cells.append(c)
    def md(s): add('markdown',s)
    def code(s,hidden=False): add('code',s,hidden)

    md('''
    # Triumph speedometer — augmented arc and tick training

    **500 training / 100 validation / 100 test images, grayscale 512×512.** Pose families are assigned to splits before appearance variants are made. All images still come from one reconstructed dashboard; synthetic test results do not establish real-camera accuracy.

    Changes: a dedicated multiscale tick decoder, visibility training weighted using training labels, and a separate frozen epoch-19 needle reader. Arc/direct-speed features initialize from the previous arc model's best epoch 33. The new architecture starts a new optimizer; subsequent restarts restore complete V2 state.

    The dataset includes translation, mild shear, perspective, full 360° roll, scale, motion/defocus blur, lighting/noise and controlled tick occlusion. Mirrors are excluded because they reverse the dial. Image transforms are also applied to needle/tick labels and masks; intended speed stays unchanged.

    Select **T4 GPU**, then **Run all** and connect Drive. Every completed epoch, including zero, is saved. The test set is disabled by default: inspect validation and finish model decisions before enabling the final test. No ImageGen calls are needed.
    ''')
    md('''
    ## 1. Runtime and restart choices
    `resume_latest` continues the most recent complete epoch; on the first-ever run it initializes from the two approved models. `resume_best` restores the best full state. `transfer` requires a new experiment. `fresh` uses random weights in a uniquely named experiment, including a trainable needle branch.

    Saved experiment settings control resumed runs. The fixed target prevents accidental extra epochs. A changed architecture, dataset, code or runtime requires an explicitly new experiment; invalid checkpoints never silently start fresh.
    ''')
    code('''
    #@title Experiment settings
    TRAIN_MODE = "resume_latest" #@param ["resume_latest", "resume_best", "transfer", "fresh"]
    EXPERIMENT_NAME = "speedometer_arc_v2_augmented" #@param {type:"string"}
    BATCH_SIZE = 8 #@param {type:"integer"}
    HEAD_EPOCHS = 3 #@param {type:"integer"}
    FINETUNE_EPOCHS = 37 #@param {type:"integer"}
    RUN_FINAL_TEST = False #@param {type:"boolean"}

    import os, sys, json, hashlib, base64, gzip, shutil, uuid, re, stat, zipfile, urllib.request
    from pathlib import Path, PurePosixPath, PureWindowsPath
    from datetime import datetime, timezone
    from dataclasses import asdict
    import numpy as np
    import torch, torchvision, PIL, cv2
    from torch.utils.data import DataLoader
    from IPython.display import display, Image as DisplayImage
    LOCAL_SMOKE = os.environ.get("RING_V2_LOCAL_SMOKE", "0") == "1"
    if LOCAL_SMOKE:
        torch.set_num_threads(4)
    TRAIN_MODE = os.environ.get("RING_V2_TRAIN_MODE", TRAIN_MODE)
    EXPERIMENT_NAME = os.environ.get("RING_V2_EXPERIMENT", EXPERIMENT_NAME)
    assert TRAIN_MODE in {"transfer", "fresh", "resume_latest", "resume_best"}
    assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", EXPERIMENT_NAME)
    WORK_DIR = Path(os.environ.get("RING_V2_WORK_DIR", "/content/speedometer_arc_v2_work"))
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU")
    print("PyTorch", torch.__version__, "Torchvision", torchvision.__version__)
    print("Each full epoch: 500 training images; validation uses 100 separate images.")
    ''')
    md('''
    ## 2. Connect Drive
    Private source weights and all new checkpoints stay in MyDrive/SpeedometerTraining. The notebook is configured to read the verified arc epoch33 and needle epoch19 checkpoints. GPU computation uses local dataset files; Drive stores durable results.
    ''')
    code('''
    if LOCAL_SMOKE:
        DRIVE_ROOT = Path(os.environ["RING_V2_DRIVE_ROOT"])
    else:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").is_dir():
            drive.mount("/content/drive")
        assert Path("/content/drive/MyDrive").is_dir(), "Drive must be connected before training"
        DRIVE_ROOT = Path("/content/drive/MyDrive/SpeedometerTraining")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    RING_CHECKPOINT = Path(os.environ.get("RING_V2_RING_CHECKPOINT", str(DRIVE_ROOT / "experiments/speedometer_arc_v1/runs/20261017_111703_294103_b07af1/best_model.pt")))
    NEEDLE_CHECKPOINT = Path(os.environ.get("RING_V2_NEEDLE_CHECKPOINT", str(DRIVE_ROOT / "experiments/speedometer_rotation_v2_from47/runs/20261017_100006_266720_f43a79f6/best_model.pt")))
    RING_SHA = "9dbe35f135b91823a6a65c201c6d3ec557aa11c87a23ef4a335657de6c4720c9"
    NEEDLE_SHA = "ba47ae2a3c07fd9a490342eebb8a01ac160dfe5e9dbca24dd13014ed68cac270"
    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""): digest.update(block)
        return digest.hexdigest()
    print("Checkpoints will be saved under", DRIVE_ROOT)
    ''')
    md('''
    ## 3. Restore the exact implementation
    This cell restores the sources embedded in the notebook and checks their hashes. Dataset archives contain data only. Sources are copied into Drive with each experiment so the implementation remains reviewable.
    ''')
    code('''
    #@title Restore verified sources
    SOURCE_HASHES = __HASHES__
    packed = "__PACKED__"
    payload = json.loads(gzip.decompress(base64.b64decode(packed)))
    assert set(payload) == set(SOURCE_HASHES)
    CODE_ROOT = WORK_DIR / ("code_" + hashlib.sha256(json.dumps(SOURCE_HASHES, sort_keys=True).encode()).hexdigest()[:12])
    for relative, encoded in payload.items():
        raw = base64.b64decode(encoded, validate=True)
        assert hashlib.sha256(raw).hexdigest() == SOURCE_HASHES[relative]
        target = CODE_ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): assert target.read_bytes() == raw
        else: target.write_bytes(raw)
    sys.path.insert(0, str(CODE_ROOT))
    from ring_training_v2 import model, data, train as trainer, evaluation
    from ring_training import ring_train as previous_trainer
    from ring_training_v2.train import TrainConfig, train, load_model
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in {"ring_training_v2", "ring_training", "training_pilot", "ring_geometry_pilot", "geometry_reader", "ring_model", "ring_data", "ring_train", "train_core"} and getattr(module,"__file__",None):
            assert Path(module.__file__).resolve().is_relative_to(CODE_ROOT.resolve()), "Restart runtime to change implementations"
    print("Verified", len(SOURCE_HASHES), "implementation files")
    '''.replace('__HASHES__',repr(hashes)).replace('__PACKED__',packed),hidden=True)
    md('''
    ## 4. Download and check the dataset
    Images are high-quality grayscale JPEG; supervision masks are lossless PNG. Transformed keypoints, arc masks, occlusion and validity masks are checked. Related appearance variants stay in the same split. Test images are excluded from all optimizer updates and checkpoint selection.
    ''')
    code('''
    DATASET_URL = "https://raw.githubusercontent.com/sainadh812/triumph-speedometer-dataset/main/ring_training_v2/arc_needle_augmented700.zip"
    ZIP_SHA = "__ZIP_SHA__"
    LABELS_SHA = "__LABELS_SHA__"
    ZIP_PATH = Path(os.environ.get("RING_V2_DATASET_ZIP", str(WORK_DIR/"arc_needle_augmented700.zip")))
    if not ZIP_PATH.exists():
        partial = ZIP_PATH.with_suffix('.partial')
        with urllib.request.urlopen(DATASET_URL,timeout=120) as response, partial.open('wb') as output:
            shutil.copyfileobj(response, output)
        assert sha256(partial) == ZIP_SHA, "Dataset download checksum mismatch"
        os.replace(partial,ZIP_PATH)
    assert sha256(ZIP_PATH) == ZIP_SHA
    extraction = WORK_DIR / ("dataset_"+uuid.uuid4().hex[:12])
    extraction.mkdir()
    with zipfile.ZipFile(ZIP_PATH) as archive:
        assert len(archive.infolist()) < 10000
        assert sum(i.file_size for i in archive.infolist()) < 1024**3
        for item in archive.infolist():
            parts=PurePosixPath(item.filename).parts
            assert parts and not PureWindowsPath(item.filename).drive and not item.filename.startswith('/')
            assert '\\\\' not in item.filename and '..' not in parts and not stat.S_ISLNK(item.external_attr>>16)
            assert (extraction/item.filename).resolve().is_relative_to(extraction.resolve())
        assert archive.testzip() is None
        archive.extractall(extraction)
    DATA_DIR = extraction / "dataset_v2"
    assert sha256(DATA_DIR/'labels.json') == LABELS_SHA
    RECORDS, INTEGRITY = data.verify_dataset(DATA_DIR)
    print(json.dumps(INTEGRITY,indent=2))
    '''.replace('__ZIP_SHA__',zip_hash).replace('__LABELS_SHA__',labels_hash))
    md('''
    ## 5. Select the persistent experiment
    New tick/visibility parameters learn first, then the final encoder blocks are fine-tuned at a lower rate. The transferred needle branch stays frozen throughout. Visibility class weighting uses training labels only.
    ''')
    code('''
    def atomic_json(path,value):
        path=Path(path); temp=path.with_name('.tmp_'+uuid.uuid4().hex+'.partial')
        with temp.open('x',encoding='utf-8') as f:
            json.dump(value,f,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
    RUNTIME={'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),'numpy':str(np.__version__),
             'pillow':str(PIL.__version__),'opencv':str(cv2.__version__),
             'device':DEVICE,'device_name':torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'cpu'}
    if TRAIN_MODE=='fresh':
        EXPERIMENT_NAME=EXPERIMENT_NAME[:70]+'_fresh_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    EXPERIMENT_DIR=DRIVE_ROOT/'experiments'/EXPERIMENT_NAME
    MANIFEST=EXPERIMENT_DIR/'experiment_config.json'
    RUNS_DIR=EXPERIMENT_DIR/'runs'
    if EXPERIMENT_DIR.exists():
        assert TRAIN_MODE in {'resume_latest','resume_best'}, 'Use a new experiment name for initialization'
        assert MANIFEST.is_file(), 'Existing experiment has no valid manifest; no fresh fallback'
        manifest=json.loads(MANIFEST.read_text())
        for key,value in {'source_hashes':SOURCE_HASHES,'dataset_zip_sha256':ZIP_SHA,'dataset_labels_sha256':LABELS_SHA,'runtime':RUNTIME}.items():
            assert manifest[key]==value, 'Saved experiment differs in '+key
        settings=dict(manifest['config']); settings.update(data_dir=str(DATA_DIR),output_dir=str(RUNS_DIR),device=DEVICE)
        CONFIG=TrainConfig(**settings); EFFECTIVE_MODE=TRAIN_MODE
    else:
        assert TRAIN_MODE!='resume_best', 'No previous best exists'
        assert type(BATCH_SIZE) is int and BATCH_SIZE>0
        assert type(HEAD_EPOCHS) is int and HEAD_EPOCHS>=0
        assert type(FINETUNE_EPOCHS) is int and FINETUNE_EPOCHS>=0
        assert HEAD_EPOCHS+FINETUNE_EPOCHS>0
        EFFECTIVE_MODE='transfer' if TRAIN_MODE=='resume_latest' else TRAIN_MODE
        if EFFECTIVE_MODE=='transfer':
            assert sha256(RING_CHECKPOINT)==RING_SHA and sha256(NEEDLE_CHECKPOINT)==NEEDLE_SHA
        CONFIG=TrainConfig(data_dir=str(DATA_DIR),output_dir=str(RUNS_DIR),batch_size=BATCH_SIZE,
                           head_epochs=HEAD_EPOCHS,finetune_epochs=FINETUNE_EPOCHS,device=DEVICE)
        EXPERIMENT_DIR.mkdir(parents=True); RUNS_DIR.mkdir()
        manifest={'source_hashes':SOURCE_HASHES,'dataset_zip_sha256':ZIP_SHA,'dataset_labels_sha256':LABELS_SHA,
                  'runtime':RUNTIME,'config':asdict(CONFIG),'initial_mode':EFFECTIVE_MODE,
                  'source_ring_sha256':RING_SHA if EFFECTIVE_MODE=='transfer' else None,
                  'source_needle_sha256':NEEDLE_SHA if EFFECTIVE_MODE=='transfer' else None}
        atomic_json(MANIFEST,manifest)
    for relative,expected in SOURCE_HASHES.items():
        saved=EXPERIMENT_DIR/'source'/relative; saved.parent.mkdir(parents=True,exist_ok=True)
        if saved.exists(): assert sha256(saved)==expected
        else: shutil.copy2(CODE_ROOT/relative,saved)
    print('Experiment:',EXPERIMENT_NAME,'Mode:',EFFECTIVE_MODE)
    print('Fixed target:',CONFIG.head_epochs+CONFIG.finetune_epochs,'epochs; batch:',CONFIG.batch_size)
    ''')
    md('''
    ## 6. Train and select using validation
    **Tick error** measures distance to the named landmark in input pixels; lower is better. **Arc IoU** measures mask overlap; higher is better. **Visibility balanced accuracy** checks visible and hidden ticks equally. **Geometry coverage** is the percentage of inputs yielding an accepted reading; rejected readings count against the all-input ±2 km/h metric.

    Selection combines validation tick error and geometric reading success. Each epoch saves weights, optimizer, mixed-precision scaler, random states and history. An interrupted epoch restarts from the previous completed epoch. Improving total loss alone does not prove improved speed reading.
    ''')
    code('''
    stop_after=int(os.environ['RING_V2_STOP_AFTER_EPOCHS']) if LOCAL_SMOKE and 'RING_V2_STOP_AFTER_EPOCHS' in os.environ else None
    REPORT=train(CONFIG,mode=EFFECTIVE_MODE,
                 ring_checkpoint=str(RING_CHECKPOINT) if EFFECTIVE_MODE=='transfer' else None,
                 needle_checkpoint=str(NEEDLE_CHECKPOINT) if EFFECTIVE_MODE=='transfer' else None,
                 search_root=str(RUNS_DIR) if EFFECTIVE_MODE.startswith('resume_') else None,
                 stop_after_epochs=stop_after)
    SESSION_DIR=Path(REPORT['run_dir'])
    atomic_json(SESSION_DIR/'experiment_config_snapshot.json',manifest)
    print(json.dumps({k:REPORT[k] for k in ('best_epoch','completed_epoch','target_epoch','training_complete','best_model_path','latest_checkpoint_path')},indent=2))
    ''')
    md('''
    ## 7. Inspect validation predictions
    The previous arc model and selected improved model are evaluated on the same validation images. Yellow points should match the green tick crosses. Preserve needle accuracy as well as improving ticks. All geometric readings use only predicted landmarks and predicted visibility.
    ''')
    code('''
    validation=data.AugmentedRingDataset(DATA_DIR,split='val')
    val_loader=DataLoader(validation,batch_size=CONFIG.batch_size,shuffle=False,num_workers=0)
    if RING_CHECKPOINT.is_file() and sha256(RING_CHECKPOINT)==RING_SHA:
        previous,_=previous_trainer.load_model(RING_CHECKPOINT,device=DEVICE)
    else:
        previous,_=load_model(REPORT['source_baseline_checkpoint_path'],device=DEVICE)
    BEFORE,before_rows=evaluation.evaluate(previous,val_loader,DEVICE)
    evaluation.save_evaluation(SESSION_DIR,'previous_validation',BEFORE,before_rows)
    del previous
    if DEVICE=='cuda': torch.cuda.empty_cache()
    best_model,best_payload=load_model(REPORT['best_model_path'],device=DEVICE)
    AFTER,after_rows=evaluation.evaluate(best_model,val_loader,DEVICE)
    evaluation.save_evaluation(SESSION_DIR,'improved_validation',AFTER,after_rows)
    evaluation.comparison_plot(BEFORE,AFTER,SESSION_DIR/'validation_comparison.png')
    evaluation.prediction_sheet(best_model,validation,DEVICE,SESSION_DIR/'validation_predictions.png')
    evaluation.history_plot(REPORT['history_path'],SESSION_DIR/'tick_learning.png')
    print('Previous validation:',json.dumps(BEFORE,indent=2))
    print('Improved validation:',json.dumps(AFTER,indent=2))
    display(DisplayImage(filename=str(SESSION_DIR/'validation_comparison.png')))
    display(DisplayImage(filename=str(SESSION_DIR/'validation_predictions.png')))
    REPORT.update(previous_validation=BEFORE,improved_validation=AFTER)
    atomic_json(SESSION_DIR/'experiment_report.json',REPORT)
    ''')
    md('''
    ## 8. Final synthetic test — enable after model decisions are finished
    Leave RUN_FINAL_TEST false while changing the model. Once validation inspection is complete, set it true in the first cell, run that settings cell, then run this test cell. The fixed test report is cached by selected model hash and does not affect checkpoint selection. A synthetic test from this one source still cannot prove performance on a real live feed.
    ''')
    code('''
    if RUN_FINAL_TEST and REPORT['training_complete']:
        digest=hashlib.sha256((model.ARCHITECTURE_ID+LABELS_SHA+'test').encode())
        for name,tensor in sorted(best_model.state_dict().items()):
            digest.update(json.dumps([name,str(tensor.dtype),list(tensor.shape)]).encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        selected_sha=digest.hexdigest()
        final_path=EXPERIMENT_DIR/'final_test'/selected_sha
        required=('improved_test_metrics.json','improved_test_predictions.json','improved_test_predictions.csv','test_predictions.png','provenance.json')
        if all((final_path/name).is_file() for name in required):
            provenance=json.loads((final_path/'provenance.json').read_text())
            assert provenance['selected_tensor_and_dataset_sha256']==selected_sha and provenance['labels_sha256']==LABELS_SHA
            TEST_METRICS=json.loads((final_path/'improved_test_metrics.json').read_text())
            print('Using existing test report for this exact model')
        else:
            test=data.AugmentedRingDataset(DATA_DIR,split='test')
            loader=DataLoader(test,batch_size=CONFIG.batch_size,shuffle=False,num_workers=0)
            TEST_METRICS,test_rows=evaluation.evaluate(best_model,loader,DEVICE)
            evaluation.save_evaluation(final_path,'improved_test',TEST_METRICS,test_rows)
            evaluation.prediction_sheet(best_model,test,DEVICE,final_path/'test_predictions.png')
            atomic_json(final_path/'provenance.json',{'selected_tensor_and_dataset_sha256':selected_sha,'labels_sha256':LABELS_SHA,'selection_used_test':False})
        print(json.dumps(TEST_METRICS,indent=2))
        display(DisplayImage(filename=str(final_path/'test_predictions.png')))
        REPORT['final_test']=TEST_METRICS
        atomic_json(SESSION_DIR/'experiment_report.json',REPORT)
    else:
        print('Test set remains unevaluated. Inspect validation before enabling the final test.')
    ''')
    md('''
    ## 9. Save and download results
    Every full epoch checkpoint remains in Drive. The compact ZIP contains selected weights, metrics, predictions, plots and exact source files. It includes private paths/weights and is not uploaded publicly. If automatic download is blocked, open the printed ZIP in Google Drive and select Download.
    ''')
    code('''
    for relative in SOURCE_HASHES:
        target=SESSION_DIR/'source'/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(CODE_ROOT/relative,target)
    atomic_json(SESSION_DIR/'source_hashes.json',SOURCE_HASHES)
    atomic_json(EXPERIMENT_DIR/'latest_result.json',REPORT)
    for name in ('source_annotations_snapshot.json','calibration_snapshot.json'):
        shutil.copy2(DATA_DIR/name,SESSION_DIR/name)
    if RUN_FINAL_TEST and 'final_path' in globals():
        shutil.copytree(final_path,SESSION_DIR/'final_test',dirs_exist_ok=True)
    exports=EXPERIMENT_DIR/'exports'; exports.mkdir(exist_ok=True)
    archive_path=exports/(SESSION_DIR.name+'_results.zip')
    partial=archive_path.with_name('.tmp_'+uuid.uuid4().hex+'.partial')
    with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for p in sorted(SESSION_DIR.rglob('*')):
            if p.is_file() and not p.name.startswith(('epoch_','.tmp_')) and not p.name.endswith('.partial'):
                archive.write(p,p.relative_to(SESSION_DIR).as_posix())
    os.replace(partial,archive_path)
    atomic_json(archive_path.with_suffix('.sha256.json'),{'sha256':sha256(archive_path),'bytes':archive_path.stat().st_size})
    print('Results saved:',archive_path)
    print('Full epoch checkpoints:',RUNS_DIR)
    print('Resume experiment name:',EXPERIMENT_NAME)
    if not LOCAL_SMOKE:
        from google.colab import files
        local_download=WORK_DIR/archive_path.name; shutil.copy2(archive_path,local_download)
        files.download(str(local_download))
    ''')
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU','colab':{'name':'Speedometer_Arc_Ticks_Augmented_Colab.ipynb','provenance':[]}})
    path=HERE/'Speedometer_Arc_Ticks_Augmented_Colab.ipynb'
    path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'notebook':str(path),'dataset_sha256':zip_hash,'labels_sha256':labels_hash,'source_hashes':hashes},indent=2))
    return notebook


if __name__=='__main__': build()
