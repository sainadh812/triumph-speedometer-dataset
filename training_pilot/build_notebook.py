"""Build the self-contained Drive-persistent Colab notebook."""
from pathlib import Path
import hashlib
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
core = (ROOT / "train_core.py").read_text(encoding="utf-8")
workflow = (ROOT / "notebook_workflow.py").read_text(encoding="utf-8")
core_hash = hashlib.sha256(core.encode()).hexdigest()
workflow_hash = hashlib.sha256(workflow.encode()).hexdigest()
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source, hidden=False):
    cell = nbf.v4.new_code_cell(source.strip())
    if hidden:
        cell.metadata["cellView"] = "form"
    cells.append(cell)


md("""
# Triumph speedometer — rotation training from epoch 47

**Start from the best epoch-47 model, train with full-circle dashboard rotations, and measure the change.**

1. Choose **Runtime → Change runtime type → T4 GPU**, then **Run all**.
2. Authorize the Google Drive mount when Colab asks. Dataset files download automatically from the public release.
3. Leave **`resume_latest`** selected to continue the newest completed epoch. Full training state is saved to Drive after every completed epoch.

Default experiment: `speedometer_rotation_v2_from47`. Its first run imports the verified best epoch-47 model into a **new dataset experiment**, resets optimizer/random-generator state once, and evaluates the source model before any updates. The new experiment has a fixed budget of **20 epochs, numbered 0 → 20**, subject to early stopping. Source epoch 47 remains in its provenance. Rerunning retains that target; it does not add another 20 epochs.

Later restarts restore model, optimizer, mixed-precision scaler, random generators, data-loader state, training history and early-stopping counters. Old runs remain intact. A disconnect during an epoch resumes from the last verified completed epoch.

The main dataset has **6,000 images: 4,800 training + 1,200 validation**, at 12 angles separated by 30°. Original 0° images are retained. All variants of a pose stay in one split. A separate **1,200-image robustness set** uses the 12 unseen offset angles 15°,45°,… and never affects model selection or early stopping. Before/after reports show 0°, rotated images and each angle separately. This tests a broad range of rotations; it does not establish perfect performance at every possible angle.

The **98% within ±2 km/h** target still needs independently labeled real footage. Validation and robustness images share the synthetic master and are not a real-camera test. The original 500-image notebook is preserved as `Speedometer_Baseline_500_Colab.ipynb`.

[Latest notebook in Colab](https://colab.research.google.com/github/sainadh812/triumph-speedometer-dataset/blob/main/Speedometer_Training_Colab.ipynb) · [Dataset release](https://github.com/sainadh812/triumph-speedometer-dataset/releases/tag/v2.0.0)
""")
md("""
## 1. Runtime and training mode

- **`resume_latest`** restores the most recently published complete epoch, including its optimizer state.
- **`resume_best`** restores the complete epoch with the lowest validation MAE. This intentionally rewinds training state and retains the original target epoch.
- **`fresh`** initializes a new model from ImageNet weights, or random weights if `FRESH_PRETRAINED=False`. Use a **new experiment name**, for example `speedometer_fresh_v2`; that configuration is reused on later resumes.

Fresh experiments require no private epoch-47 checkpoint. They report after-training validation and robustness metrics, with no source-model before/after claim.

Existing experiments reuse their saved configuration; changing a fresh-only setting does not alter a resumed run. Exact continuation requires matching dependency versions, dataset, device type and training settings. Broken uploads are skipped only when a verified compatible recovery checkpoint exists. No valid checkpoint means an explicit error, never a silent fresh model.
""")
code("""
#@title Training mode and experiment
TRAIN_MODE = "resume_latest" #@param ["resume_latest", "resume_best", "fresh"]
EXPERIMENT_NAME = "speedometer_rotation_v2_from47" #@param {type:"string"}
FRESH_PRETRAINED = True #@param {type:"boolean"}
# Fresh requires a NEW experiment name; later resume that same name.
import os, sys, json, time, hashlib, zipfile, importlib.util, shutil
from pathlib import Path
import torch, torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Image as DisplayImage

IN_COLAB = importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
RUN_MODE = os.environ.get('SPEEDOMETER_RUN_MODE', 'pilot')
assert RUN_MODE in ('pilot', 'smoke')
TRAIN_MODE = os.environ.get('SPEEDOMETER_TRAIN_MODE', TRAIN_MODE)
EXPERIMENT_NAME = os.environ.get('SPEEDOMETER_EXPERIMENT', 'smoke_rotation_persistence_v2' if RUN_MODE == 'smoke' else EXPERIMENT_NAME)
IMPORT_PILOT_IF_MISSING = True
# Leave at 0 for normal resume. A positive value explicitly adds a NEW budget.
EXTEND_BY_EPOCHS = 0

WORK_DIR = Path(os.environ.get('SPEEDOMETER_WORK_DIR', '/content/speedometer_work' if IN_COLAB else str(Path.cwd() / 'speedometer_work')))
WORK_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = Path(os.environ.get('SPEEDOMETER_DATASET_ZIP', '/content/speedometer_bw_rotation_v2.zip' if IN_COLAB else str(Path.cwd() / 'speedometer_bw_rotation_v2.zip')))
DATASET_URL = 'https://github.com/sainadh812/triumph-speedometer-dataset/releases/download/v2.0.0/speedometer_bw_rotation_v2.zip'
EXPECTED_ZIP_SHA256 = 'ce4d6d102adac02603c0bdc964b0832d0ba661ec70e8401c6d4f8d195add2aa4'
EXPECTED_LABELS_SHA256 = 'a6e15644d5fa7098b5ea6ba1625b39b21c0c04c4a3d122356cc923dbf9aad4b6'
EXPECTED_ROBUSTNESS_LABELS_SHA256 = '4f6ebb8055d116d4f5522d48b6514df0e8c3e1968afd673530bc21122c621f1d'
EXPECTED_SEED_SHA256 = 'a284ecb0ceb4892074feea3802d9e79b9e4bc9892c25113b8934b5428a16e97c'
print('Mode:', RUN_MODE, TRAIN_MODE, '| torch:', torch.__version__, '| torchvision:', torchvision.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not connected')
if not hasattr(torch.amp, 'GradScaler'):
    raise RuntimeError('Use a current Colab PyTorch runtime with torch.amp.GradScaler.')
if RUN_MODE == 'pilot' and not torch.cuda.is_available():
    raise RuntimeError('Select Runtime > Change runtime type > T4 GPU and Run all again.')
if RUN_MODE == 'smoke':
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    print('Local smoke check only: 128 pixels, 24/24 images, 2 epochs, random weights. Not an accuracy result.')
""")
md("""
## 2. Mount Drive and load the implementation

Authorize Drive access so checkpoints can survive a runtime reset. Files are saved under **MyDrive/SpeedometerTraining/experiments/**. Each invocation receives a unique run folder. Local smoke mode uses a mock Drive directory and does not request Google authorization.

The complete implementation is embedded here. Inputs remain 512 × 512; grayscale is repeated into three channels and ImageNet-normalized. A spatial decoder predicts pivot/tip heatmaps, and a speed head predicts continuous km/h. Batch-normalization running statistics stay frozen. Existing image augmentations and labels are preserved.
""")
code("""
if RUN_MODE == 'smoke':
    DRIVE_ROOT = Path(os.environ.get('SPEEDOMETER_DRIVE_ROOT', str(WORK_DIR / 'mock_drive' / 'SpeedometerTraining')))
elif IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    if not os.path.ismount('/content/drive') or not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Google Drive is not mounted. Check authorization and rerun; no temporary checkpoint folder was created.')
    DRIVE_ROOT = Path('/content/drive/MyDrive/SpeedometerTraining')
else:
    raise RuntimeError('The full pilot uses Colab with mounted Drive. Local mode is for smoke checks.')
SEED_CHECKPOINT = DRIVE_ROOT / 'experiments/speedometer_400_continuation_v1/runs/20261017_090632_970209_082ca611/best_model.pt'
print('Persistent experiment root:', DRIVE_ROOT)
""")
code("#@title Load embedded training and persistence modules\nTRAINING_CORE_SOURCE = " + repr(core)
     + "\nWORKFLOW_SOURCE = " + repr(workflow)
     + "\nEXPECTED_CORE_SHA256 = " + repr(core_hash)
     + "\nEXPECTED_WORKFLOW_SHA256 = " + repr(workflow_hash) + "\n" + """
def load_embedded(name, filename, source, expected_hash):
    assert hashlib.sha256(source.encode('utf-8')).hexdigest() == expected_hash
    path = WORK_DIR / filename
    path.write_text(source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, path

training, module_path = load_embedded('speedometer_core', 'train_core.py', TRAINING_CORE_SOURCE, EXPECTED_CORE_SHA256)
workflow, workflow_path = load_embedded('speedometer_workflow', 'notebook_workflow.py', WORKFLOW_SOURCE, EXPECTED_WORKFLOW_SHA256)
workflow.ensure_writable(DRIVE_ROOT)
print('Drive write/read check passed; embedded modules loaded.')
""", hidden=True)
md("""
## 3. Download and verify the fixed dataset

The public rotation archive downloads only when a verified local copy is unavailable. Archive and label SHA256, individual input hashes, transformed coordinates and group separation are checked. No token or dataset upload is required. The separate unseen-angle set is kept out of training and model selection.
""")
code("""
import tempfile
import requests

def download_dataset(url, destination, expected_sha256, attempts=3):
    destination = Path(destination)
    if destination.is_file() and workflow.file_sha256(destination) == expected_sha256:
        print('Using the verified cached dataset:', destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        with tempfile.NamedTemporaryFile(prefix=destination.name + '.', suffix='.part', dir=destination.parent, delete=False) as handle:
            partial = Path(handle.name)
        try:
            print(f'Downloading dataset: attempt {attempt}/{attempts}')
            with requests.get(url, stream=True, timeout=(30,60)) as response:
                response.raise_for_status()
                with partial.open('wb') as target:
                    for block in response.iter_content(1024*1024):
                        if block:
                            target.write(block)
            if workflow.file_sha256(partial) != expected_sha256:
                raise ValueError('Dataset checksum mismatch')
            partial.replace(destination)
            return
        except (requests.RequestException, OSError, ValueError) as error:
            if attempt == attempts:
                raise RuntimeError('Dataset download failed; check the connection and rerun this cell.') from error
            time.sleep(attempt*2)
        finally:
            partial.unlink(missing_ok=True)

download_dataset(DATASET_URL, ZIP_PATH, EXPECTED_ZIP_SHA256)
with zipfile.ZipFile(ZIP_PATH) as archive:
    if archive.testzip() is not None:
        raise ValueError('Dataset archive is damaged')
    for entry in archive.infolist():
        destination = (WORK_DIR / entry.filename).resolve()
        if not destination.is_relative_to(WORK_DIR.resolve()) or not entry.filename.startswith('bw_dataset_rotation_v2/'):
            raise ValueError('Unexpected dataset archive path')
    archive.extractall(WORK_DIR)
DATA_DIR = WORK_DIR / 'bw_dataset_rotation_v2'
ROBUSTNESS_DIR = DATA_DIR / 'robustness'
if workflow.file_sha256(DATA_DIR / 'labels.json') != EXPECTED_LABELS_SHA256:
    raise ValueError('Dataset label provenance mismatch')
records, integrity = training.verify_dataset(DATA_DIR)
assert integrity['split_counts'] == {'train':4800, 'validation':1200}
validation_records = [r for r in records if r['dataset_split'] == 'validation']
assert len({r['group_id'] for r in validation_records}) == 50
if workflow.file_sha256(ROBUSTNESS_DIR / 'labels.json') != EXPECTED_ROBUSTNESS_LABELS_SHA256:
    raise ValueError('Robustness label provenance mismatch')
print('Verified main dataset: 4800 training, 1200 validation, 50 validation poses. Separate robustness:1200 images.')
if (DATA_DIR / 'overview.png').is_file():
    display(DisplayImage(filename=str(DATA_DIR / 'overview.png'), width=700))
""")
md("""
## 4. Resolve the experiment and continue training

The saved experiment manifest is authoritative on resume. Default transfer uses 512 pixels, batch size 8, all 4,800 training images, 1,200 validation images and 20 fine-tuning epochs. Fresh experiments use 5 head-training plus 25 fine-tuning epochs. Early stopping uses **only main validation**, and remains enabled. The source model is evaluated on unseen angles before training, and the selected best model is evaluated on exactly those images afterward.

Full checkpoints are staged, copied and rehashed, then published with a `.complete.json` sidecar. Drive needs space for all epoch saves. Incomplete/corrupt files are preserved and skipped if an older verified compatible checkpoint exists. With no valid recovery state, the notebook stops instead of importing legacy weights or starting fresh. The Drive mount cannot guarantee remote upload completion at the instant of a disconnect, so earlier recovery files are retained.
""")
code("""
existing_manifest_path = DRIVE_ROOT / 'experiments' / EXPERIMENT_NAME / 'experiment_config.json'
existing_kind = json.loads(existing_manifest_path.read_text(encoding='utf-8')).get('initialization_kind') if existing_manifest_path.exists() else None
COMPARE_SOURCE_MODEL = TRAIN_MODE != 'fresh' and existing_kind != 'fresh'
evaluation_seed = None
seed_hash = None
if COMPARE_SOURCE_MODEL and RUN_MODE == 'smoke':
    evaluation_seed = WORK_DIR / 'smoke_random_evaluation_seed.pt'
    if not evaluation_seed.exists():
        torch.manual_seed(42)
        random_seed_model = training.SpeedometerModel(pretrained=False)
        torch.save({
            'format_version':1, 'model_state_dict':random_seed_model.state_dict(),
            'config':{'input_size':128,'pretrained':False}, 'best_epoch':0,
            'architecture':training._ARCHITECTURE_ID, 'preprocessing':training._PREPROCESSING_ID,
            'keypoint_order':['pivot','tip'], 'speed_range_kmh':[0,180],
            'labels_sha256':'0'*64, 'validation_metrics':{'mae_kmh':90.0},
            'scope':'Random software-test fixture; no source model performance claim',
        },evaluation_seed)
        del random_seed_model
    seed_hash = workflow.file_sha256(evaluation_seed)
elif COMPARE_SOURCE_MODEL:
    evaluation_seed = SEED_CHECKPOINT
    seed_hash = EXPECTED_SEED_SHA256
    if not evaluation_seed.is_file() or workflow.file_sha256(evaluation_seed) != seed_hash:
        raise ValueError('The source epoch-47 checkpoint is missing or changed; before/after comparison cannot proceed.')
prepared = workflow.prepare_persistent_run(
    training, data_dir=DATA_DIR, drive_root=DRIVE_ROOT, experiment_name=EXPERIMENT_NAME,
    requested_mode=TRAIN_MODE, run_mode=RUN_MODE, fresh_pretrained=FRESH_PRETRAINED,
    import_pilot_if_missing=IMPORT_PILOT_IF_MISSING, warm_start_path=evaluation_seed or SEED_CHECKPOINT,
    extend_by_epochs=EXTEND_BY_EPOCHS, core_sha256=EXPECTED_CORE_SHA256,
    first_import_mode='warm_start_transfer', approved_source_sha256=seed_hash or EXPECTED_SEED_SHA256,
    source_epoch=0 if RUN_MODE=='smoke' else 47, additional_epochs=2 if RUN_MODE=='smoke' else 20, train_sample_limit=None,
    smoke_train_samples=24, smoke_val_samples=24, dataset_name='rotation_v2',
    allow_smoke_transfer=(RUN_MODE=='smoke'),
)
config = prepared['config']
EVALUATIONS_DIR = Path(prepared['experiment_dir']) / 'evaluations'
baseline_robustness = None
if COMPARE_SOURCE_MODEL:
    baseline_robustness = workflow.evaluate_rotation_checkpoint(
        training, checkpoint_path=evaluation_seed, data_dir=ROBUSTNESS_DIR,
        output_dir=EVALUATIONS_DIR / 'source47_unseen_angles',
        expected_labels_sha256=EXPECTED_ROBUSTNESS_LABELS_SHA256,
        batch_size=config.batch_size, max_samples=24 if RUN_MODE=='smoke' else None,
    )
    print('Source model before training — unseen-angle robustness:')
    display(pd.DataFrame(baseline_robustness['breakdown']).T)
else:
    print('Fresh initialization: no private epoch-47 seed is needed. After-training metrics will be reported; no source47 comparison is claimed.')
print('Effective configuration:')
display(pd.Series({key:getattr(config,key) for key in ('input_size','batch_size','head_epochs','finetune_epochs','max_train_samples','max_val_samples','seed')}))
result = training.train_persistent(config, **prepared['kwargs'])
result['run_mode'] = RUN_MODE
SESSION_DIR = Path(prepared['run_dir'])
(SESSION_DIR / 'experiment_results.json').write_text(json.dumps([result], indent=2), encoding='utf-8')
print('Latest complete training state:', result['latest_checkpoint_path'])
print('Best inference weights:', result['checkpoint_path'])
print('Epoch:', result['epochs_run'], '| saved target:', result['target_epoch'], '| epochs this invocation:', result['epochs_this_call'])
""")
md("""
## 5. Inspect the saved model

MAE measures average error; **within ±2 km/h** measures the fraction meeting that tolerance. These are different metrics. Inspect worst predictions and tip errors as well as averages. Reports separate lighting/blur profiles, speed ranges and the inferred 60-tick region. No learned confidence or unreadable-frame detector is claimed.
""")
code("""
m = result['metrics']
summary = pd.DataFrame([{
    'mode':prepared['actual_mode'], 'training_images':result['training_samples'],
    'validation_images':result['validation_samples'], 'validation_poses':m['pose_groups'],
    'MAE_kmh':m['mae_kmh'], 'RMSE_kmh':m['rmse_kmh'], 'within_2_kmh_percent':m['within_2_kmh_percent'],
    'P95_error_kmh':m['p95_absolute_error_kmh'], 'tip_error_px':m['tip_mae_px'],
    'best_epoch':result['best_epoch'], 'latest_epoch':result['epochs_run'], 'target_epoch':result['target_epoch'],
    'epochs_this_call':result['epochs_this_call'], 'scope':'synthetic validation' if RUN_MODE=='pilot' else 'software smoke only',
}])
summary.to_csv(SESSION_DIR / 'run_summary.csv', index=False)
display(summary)
display(DisplayImage(filename=result['learning_curves_path']))
display(DisplayImage(filename=result['worst_predictions_path'], width=1100))
display(pd.DataFrame(result['metrics_by_profile']).T)
print('Real-camera accuracy remains unmeasured.' if RUN_MODE=='pilot' else 'SMOKE ONLY — not a trained-model accuracy claim.')
""")
md("""
## 6. Compare source epoch 47 with the selected model

Both models see the **same images** in each comparison. Main validation includes angle 0° and the 30° grid used for model selection. The separate unseen-angle robustness set uses the offset 15° grid and is evaluated only before/after training. Its performance never changes the optimizer, early stopping or selected checkpoint. These are synthetic robustness checks, not independent real recordings.

Fresh experiments have no private source-model baseline and report after-training robustness metrics instead.

A negative MAE change means improvement; a positive change in the percentage within ±2 km/h means improvement. Inspect 0° separately to identify any loss of the original dashboard-reading performance.
""")
code("""
main_before_path = result.get('baseline_predictions_path')
if COMPARE_SOURCE_MODEL and main_before_path is None:
    raise RuntimeError('The source-model main-validation baseline is missing; no before/after claim can be made.')
main_comparison = []
if COMPARE_SOURCE_MODEL:
    main_comparison = workflow.save_rotation_comparison(
        training, before_predictions=main_before_path, after_predictions=result['predictions_path'],
        output_path=SESSION_DIR / 'comparison_main_validation.csv', evaluation_name='main_validation',
    )
selected_robustness = workflow.evaluate_rotation_checkpoint(
    training, checkpoint_path=result['checkpoint_path'], data_dir=ROBUSTNESS_DIR,
    output_dir=SESSION_DIR / 'robustness_selected',
    expected_labels_sha256=EXPECTED_ROBUSTNESS_LABELS_SHA256,
    batch_size=config.batch_size, max_samples=24 if RUN_MODE=='smoke' else None,
)
robustness_comparison = []
if COMPARE_SOURCE_MODEL:
    robustness_comparison = workflow.save_rotation_comparison(
        training, before_predictions=baseline_robustness['predictions_path'],
        after_predictions=selected_robustness['predictions_path'],
        output_path=SESSION_DIR / 'comparison_unseen_angles.csv', evaluation_name='unseen_angle_robustness',
    )
comparison = pd.DataFrame(main_comparison + robustness_comparison)
if COMPARE_SOURCE_MODEL:
    comparison.to_csv(SESSION_DIR / 'rotation_comparison.csv',index=False)
    display(comparison)
    shutil.copytree(EVALUATIONS_DIR / 'source47_unseen_angles', SESSION_DIR / 'robustness_source',dirs_exist_ok=True)
    figure,axes=plt.subplots(1,2,figsize=(14,4))
    for axis,rows,title in zip(axes,[main_comparison,robustness_comparison],['Main validation','Unseen-angle robustness']):
        angles=sorted((float(row['group'].removeprefix('angle_')),row) for row in rows if row['group'].startswith('angle_'))
        axis.plot([angle for angle,row in angles],[row['before_mae_kmh'] for angle,row in angles],'o-',label='Source before training')
        axis.plot([angle for angle,row in angles],[row['after_mae_kmh'] for angle,row in angles],'o-',label='Selected after training')
        axis.set(xlabel='Clockwise image rotation (degrees)',ylabel='MAE (km/h)',title=title)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(SESSION_DIR / 'rotation_before_after.png',dpi=140)
    plt.close(figure)
    display(DisplayImage(filename=str(SESSION_DIR / 'rotation_before_after.png')))
else:
    after_only = pd.DataFrame(selected_robustness['breakdown']).T
    after_only.to_csv(SESSION_DIR / 'robustness_after_training.csv')
    display(after_only)
    (SESSION_DIR / 'comparison_availability.json').write_text(json.dumps({
        'source_comparison_available':False,'reason':'Fresh initialization has no private epoch-47 baseline.'
    },indent=2),encoding='utf-8')
    print('Fresh experiment: after-training evaluation only; no before/after improvement claim.')
""")
code("""
saved_model, checkpoint = training.load_model(result['checkpoint_path'])
example = validation_records[0]
prediction = training.predict_image(saved_model, DATA_DIR / example['image'], input_size=checkpoint['config']['input_size'])
with Image.open(DATA_DIR / example['image']) as image:
    pixels = np.asarray(image)
figure, axis = plt.subplots(figsize=(6,6))
axis.imshow(pixels, cmap='gray', vmin=0, vmax=255)
for source,color,style in [(example['needle_position_normalized'],'lime','-'),(prediction['keypoints_normalized'],'magenta','--')]:
    points = np.array([source['pivot'],source['tip']])*511
    axis.plot(points[:,0],points[:,1],style,color=color,marker='o',linewidth=1.5)
axis.set_title(f"Label {example['speed_kmh']:.1f} km/h | prediction {prediction['speed_kmh']:.1f} km/h")
axis.axis('off')
figure.tight_layout()
figure.savefig(SESSION_DIR / 'reloaded_model_example.png',dpi=130)
plt.close(figure)
display(DisplayImage(filename=str(SESSION_DIR / 'reloaded_model_example.png')))
""")
md("""
## 7. Export a compact results ZIP

The ZIP contains the best inference model, metrics, predictions, plots, source and a manifest of Drive locations. **Full per-epoch training states remain on Drive** and are excluded from the download to avoid duplicating many large files. Retain the Drive experiment for exact continuation; the compact ZIP alone is an inference/report backup.

After a reset, reopen this notebook, mount the same Drive, retain the experiment name, choose `resume_latest` and Run all. If its fixed budget is complete, no extra epochs run. Use `resume_best` only to rewind to the best complete state. A new training budget requires an explicit positive `EXTEND_BY_EPOCHS`; leave it at 0 on ordinary restarts.
""")
code("""
shutil.copyfile(module_path, SESSION_DIR / 'train_core.py')
shutil.copyfile(workflow_path, SESSION_DIR / 'notebook_workflow.py')
shutil.copyfile(prepared['manifest_path'], SESSION_DIR / 'experiment_config_snapshot.json')
drive_manifest = {
    'drive_root':str(DRIVE_ROOT), 'experiment_dir':prepared['experiment_dir'], 'run_dir':str(SESSION_DIR),
    'experiment_config':prepared['manifest_path'], 'latest_full_checkpoint':result['latest_checkpoint_path'],
    'best_inference_checkpoint':result['checkpoint_path'], 'dataset_labels_sha256':EXPECTED_LABELS_SHA256,
    'robustness_labels_sha256':EXPECTED_ROBUSTNESS_LABELS_SHA256, 'source_checkpoint_sha256':seed_hash,
    'source_epoch':(0 if RUN_MODE=='smoke' else 47) if COMPARE_SOURCE_MODEL else None,
    'training_core_sha256':EXPECTED_CORE_SHA256, 'workflow_sha256':EXPECTED_WORKFLOW_SHA256,
    'target_epoch':result['target_epoch'], 'run_mode':RUN_MODE,
    'scope':'Synthetic development validation; independent real footage still required',
    'archive_excludes':'All epoch_*.pt, epoch_*.complete.json and partial files; retain Drive for full resume',
}
(SESSION_DIR / 'drive_paths.json').write_text(json.dumps(drive_manifest,indent=2),encoding='utf-8')
RESULTS_ZIP = SESSION_DIR.with_name(SESSION_DIR.name+'_results.zip')
with zipfile.ZipFile(RESULTS_ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for path in sorted(SESSION_DIR.rglob('*')):
        if path.is_file() and not path.name.startswith('epoch_') and '.partial' not in path.name:
            archive.write(path,path.relative_to(SESSION_DIR))
print('Compact results archive:',RESULTS_ZIP)
print('Full training states remain under:',prepared['experiment_dir'])
if IN_COLAB and RUN_MODE!='smoke':
    from google.colab import files
    files.download(str(RESULTS_ZIP))
""")

notebook = nbf.v4.new_notebook(cells=cells)
if any("PENDING_ROTATION_" in cell.source or "PENDING_ROBUSTNESS_" in cell.source for cell in cells):
    raise RuntimeError("Rotation release checksums are not confirmed yet; notebook publication is blocked.")
notebook.metadata.update({
    "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11"},
    "colab":{"name":"Speedometer_Training_Colab.ipynb","provenance":[]}, "accelerator":"GPU",
    "speedometer":{"core_sha256":core_hash,"workflow_sha256":workflow_hash,
                   "dataset_images":6000,"robustness_images":1200,"input_size":512,"drive_persistent":True,
                   "default_experiment":"speedometer_rotation_v2_from47"},
})
nbf.validate(notebook)
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"notebook_cell_{index}", "exec")
target = PROJECT / "Speedometer_Training_Colab.ipynb"
with target.open("w",encoding="utf-8",newline="\n") as handle:
    nbf.write(notebook,handle)
print(target)
print(f"Cells: {len(cells)}; core SHA256: {core_hash}; workflow SHA256: {workflow_hash}")
