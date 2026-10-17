"""Build the auditable V4 Colab notebook; fill publication pins before release."""
import json
from pathlib import Path

cells = []
def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip().splitlines(True)})
def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': text.strip().splitlines(True), 'execution_count': None, 'outputs': []})

md('''# Triumph V4: learn the needle and the speed scale together

This experiment starts from the verified V3 epoch-8 needle weights. A new branch detects ten semantic scale ticks (0–180 km/h in steps of 20). Speed is calculated from predicted ticks and the predicted needle. Ground-truth rendering transforms are never used for reading speed.

Tick locations are supervised even when the needle covers the printed stroke: the model learns where that landmark lies from the surrounding scale. The legacy-named visibility head predicts in-frame landmark availability, not literal visibility or calibrated confidence. Speed still requires geometrically consistent endpoint ticks.

**512×512 grayscale, camera stabilized to the current daytime reference, then a reviewed fixed LCD mask.** All images, masks and points undergo spatial augmentation together. Two warmup epochs train the new tick branch; subsequent epochs gently fine-tune the needle and late encoder layers. Maximum 20 epochs, with early stopping after 6 epochs without meaningful improvement. This is same-video development data, not evidence of independent real-video accuracy.

Choose **T4 GPU** and **Run all**. Full-state checkpoints are saved to Drive after epoch 0 and every completed epoch. The modes below let you start from V3, resume a V4 run, or start with random weights.''')
code('''#@title Experiment settings
TRAIN_MODE = "transfer" #@param ["transfer", "resume_latest", "resume_best", "fresh"]
EXPERIMENT_NAME = "speedometer_day_arc_needle_v4" #@param {type:"string"}
EPOCHS = 20 #@param {type:"integer"}
BATCH_SIZE = 8 #@param {type:"integer"}
SOURCE_CHECKPOINT = "/content/drive/MyDrive/SpeedometerTraining/experiments/speedometer_real_needle_v3/runs/20261017_144300_919847_94361a85/best_model.pt" #@param {type:"string"}
DOWNLOAD_RESULTS = True #@param {type:"boolean"}
CODE_COMMIT = "5d8093de966eacb27ace59c48878104413ae1f16"
DATA_MANIFEST_SHA256 = "8546b2a6f73bf3ec5898a90800b6d102f79f9aa82b3eb79d1f2c62a4f3dd83a4"
DATA_MANIFEST_FILENAME = "data_download_manifest.json"
SOURCE_SHA256 = "8dcfa269848bac5a46c8ff39d14df4d3470e67905085fb0a662ee45b9bc99d10"
''')
md('''## 1. GPU and Google Drive
`transfer` creates the named experiment once. Select `resume_latest` after an interruption, or `resume_best` to continue from the best completed epoch with its matching optimizer. Resume restores the original settings and epoch target. `fresh` creates a separately named experiment with no previous weights. No mode silently falls back to fresh training.''')
code('''import os,sys,json,re,hashlib,subprocess,uuid,zipfile,urllib.request
from pathlib import Path
from dataclasses import asdict
import torch,torchvision,cv2,numpy as np
if not torch.cuda.is_available():
    raise RuntimeError("Choose Runtime > Change runtime type > T4 GPU, reconnect and Run all. Training has not started.")
from google.colab import drive
if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')
DRIVE_ROOT = Path('/content/drive/MyDrive/SpeedometerTraining')
if not Path('/content/drive/MyDrive').is_dir():
    raise RuntimeError('Connect Drive before training so checkpoints persist.')
DRIVE_ROOT.mkdir(parents=True,exist_ok=True)
probe=DRIVE_ROOT/('.write_probe_'+uuid.uuid4().hex)
probe.write_text('ok'); assert probe.read_text()=='ok'; probe.unlink()
WORK=Path('/content/speedometer_joint_landmarks_v4_work')
WORK.mkdir(parents=True,exist_ok=True)
print('GPU:',torch.cuda.get_device_name(0))
''')
md('''## 2. Load pinned code and verified data
The manifest verifies every image and label. The mask and camera-stabilization metadata must match this experiment. Data are stored in the temporary runtime; full checkpoints are stored in Drive.''')
code('''for name,value,length in [('CODE_COMMIT',CODE_COMMIT,40),('DATA_MANIFEST_SHA256',DATA_MANIFEST_SHA256,64)]:
    if re.fullmatch(rf'[0-9a-f]{{{length}}}',value) is None:
        raise ValueError(name+' is not finalized; use the published notebook.')
if globals().get('_V4_IMPORTED_COMMIT',CODE_COMMIT)!=CODE_COMMIT:
    raise RuntimeError('Restart runtime before changing the implementation commit.')
REPO=WORK/'repo'
if not REPO.exists():
    REPO.mkdir()
    subprocess.run(['git','-C',str(REPO),'init'],check=True)
    subprocess.run(['git','-C',str(REPO),'remote','add','origin','https://github.com/sainadh812/triumph-speedometer-dataset.git'],check=True)
subprocess.run(['git','-C',str(REPO),'sparse-checkout','init','--no-cone'],check=True)
subprocess.run(['git','-C',str(REPO),'sparse-checkout','set','--no-cone','/real_adaptation_v4/*.py','/real_adaptation_v3/*.py','/training_pilot/*.py'],check=True)
subprocess.run(['git','-C',str(REPO),'config','remote.origin.promisor','true'],check=True)
subprocess.run(['git','-C',str(REPO),'config','remote.origin.partialclonefilter','blob:none'],check=True)
subprocess.run(['git','-C',str(REPO),'fetch','--depth=1','--filter=blob:none','origin',CODE_COMMIT],check=True)
subprocess.run(['git','-C',str(REPO),'checkout','--detach',CODE_COMMIT],check=True)
actual=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
assert actual==CODE_COMMIT
sys.path.insert(0,str(REPO))
from real_adaptation_v4.train_landmarks import TrainConfig,train,verify_dataset,discover_checkpoints,_sha
from real_adaptation_v3.train_needle import _read_complete
from real_adaptation_v3.notebook_helpers import download_verified
from real_adaptation_v4.notebook_helpers import prepare_dataset
_V4_IMPORTED_COMMIT=CODE_COMMIT
DATA_BASE_URL=f'https://raw.githubusercontent.com/sainadh812/triumph-speedometer-dataset/{CODE_COMMIT}/real_adaptation_v4'
manifest_path=download_verified(DATA_BASE_URL+'/'+DATA_MANIFEST_FILENAME,WORK/DATA_MANIFEST_FILENAME,DATA_MANIFEST_SHA256)
DATA_DIR=prepare_dataset(manifest_path,DATA_BASE_URL,WORK/'data')
partitions,integrity=verify_dataset(DATA_DIR)
print(json.dumps(integrity,indent=2))
''')
md('''## 3. Dataset preview and training configuration
Cyan dots and a line mark the pivot and needle tip; amber dots mark the ten labelled scale ticks. This preview is a label-quality check. Same-video validation is used for checkpoint selection; independent videos are still required to establish accuracy.''')
code('''import matplotlib.pyplot as plt
from PIL import Image
fig,axes=plt.subplots(2,3,figsize=(15,10))
for ax,row in zip(axes.flat,partitions['train'][::max(1,len(partitions['train'])//6)][:6]):
    ax.imshow(Image.open(row['image']),cmap='gray',vmin=0,vmax=255)
    p=np.asarray(row['points_xy'])
    ax.plot(p[:2,0],p[:2,1],'-o',color='cyan',markersize=3)
    ax.scatter(p[2:,0],p[2:,1],s=12,c='orange')
    ax.set_title(f"{row['speed_kmh']:.1f} km/h | {row['sample_id']}");ax.axis('off')
plt.tight_layout();plt.show()
EXPERIMENT=DRIVE_ROOT/'experiments'/EXPERIMENT_NAME
if TRAIN_MODE=='fresh':
    EXPERIMENT=EXPERIMENT.with_name(EXPERIMENT.name+'_fresh_'+uuid.uuid4().hex[:8])
if TRAIN_MODE=='transfer' and EXPERIMENT.exists():
    raise FileExistsError('Experiment exists. Choose resume_latest/resume_best or a new experiment name.')
if TRAIN_MODE.startswith('resume_'):
    candidates,rejected=discover_checkpoints(EXPERIMENT/'runs')
    selected=max(candidates,key=lambda x:x['created_at_ns']) if TRAIN_MODE=='resume_latest' else min(candidates,key=lambda x:(x['selection_loss'],-x['created_at_ns']))
    payload,_=_read_complete(selected['path'])
    settings=dict(payload['config'])
    settings.update(data_dir=str(DATA_DIR),output_dir=str(EXPERIMENT/'runs'),device='auto')
    CONFIG=TrainConfig(**settings)
    print('Requested resume:',selected['path'],'epoch',selected['epoch'])
else:
    CONFIG=TrainConfig(str(DATA_DIR),str(EXPERIMENT/'runs'),epochs=EPOCHS,batch_size=BATCH_SIZE)
    if TRAIN_MODE=='transfer':
        if not Path(SOURCE_CHECKPOINT).is_file() or _sha(SOURCE_CHECKPOINT)!=SOURCE_SHA256:
            raise ValueError('Verified V3 epoch-8 weights were not found at SOURCE_CHECKPOINT.')
        _read_complete(SOURCE_CHECKPOINT)
EXPERIMENT.mkdir(parents=True,exist_ok=True)
print(json.dumps(asdict(CONFIG),indent=2))
print('Every completed epoch will be saved under',EXPERIMENT/'runs')
''')
md('''## 4. Train
The primary metric counts rejected readings as failures. Coverage alone is not accuracy. Tick error and needle error are measured in pixels; lower is better. All-input ±2 km/h is the percentage of eligible validation images whose accepted speed is within 2 km/h of its synthetic label; higher is better.''')
code('''if globals().get('_V4_TRAIN_CALL_STARTED',False):
    raise RuntimeError('Training was already invoked in this runtime. Restart, select resume_latest/resume_best, then Run all to continue deliberately.')
_V4_TRAIN_CALL_STARTED=True
REPORT=train(CONFIG,mode=TRAIN_MODE,source_checkpoint=SOURCE_CHECKPOINT if TRAIN_MODE=='transfer' else None,
             search_root=EXPERIMENT/'runs' if TRAIN_MODE.startswith('resume_') else None)
print(json.dumps(REPORT,indent=2))
''')
md('''## 5. Review and download
Every epoch's resumable state remains in Drive. The compact ZIP contains the selected inference weights, checksum marker, metrics, history and predictions. It does not replace the full-state checkpoints needed to resume training.''')
code('''RUN=Path(REPORT['run_dir'])
history=json.loads((RUN/'history.json').read_text())
if history:
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['validation_metrics']['needle_mean_error_px'] for r in history],label='Needle')
    axes[0].plot(epochs,[r['validation_metrics']['tick_mean_error_px'] for r in history],label='Scale ticks')
    axes[0].set(xlabel='Epoch',ylabel='Mean point error (pixels)');axes[0].legend()
    axes[1].plot(epochs,[r['validation_metrics']['primary_speed']['within_2_kmh_all_eligible_inputs_percent'] for r in history])
    axes[1].set(xlabel='Epoch',ylabel='All-input ±2 km/h (%)',ylim=(0,100))
    fig.suptitle('Same-video synthetic validation — not independent real-video accuracy')
    plt.tight_layout();plt.savefig(RUN/'training_history.png',dpi=140);plt.show()
RESULT_ZIP=WORK/(RUN.name+'_v4_results.zip')
include=['best_model.pt','best_model.complete.json','training_report.json','history.json',
         'before_validation_metrics.json','after_validation_metrics.json','after_validation_predictions.json',
         'dataset_integrity.json','training_history.png','best_checkpoint.json','latest_checkpoint.json']
with zipfile.ZipFile(RESULT_ZIP,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name in include:
        path=RUN/name
        if path.is_file(): archive.write(path,name)
print('Best epoch:',REPORT['best_epoch'],'| Results:',RESULT_ZIP,'| SHA256:',_sha(RESULT_ZIP))
print('Drive run:',RUN)
if DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(str(RESULT_ZIP))
''')

notebook = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
    'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}},
    'nbformat': 4, 'nbformat_minor': 5}
target = Path(__file__).with_name('Speedometer_Joint_Landmarks_V4_Colab.ipynb')
target.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
print(target)
