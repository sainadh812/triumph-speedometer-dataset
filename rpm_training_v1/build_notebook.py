"""Build a Colab notebook pinned to reviewed RPM code and data."""
from pathlib import Path
import argparse
import ast
import json
import re

HERE=Path(__file__).resolve().parent


def build(commit,manifest_sha):
    if not re.fullmatch('[0-9a-f]{40}',commit) or not re.fullmatch('[0-9a-f]{64}',manifest_sha):
        raise ValueError('Final publication pins are required')
    cells=[]
    def md(text):cells.append({'cell_type':'markdown','id':f'rpm-{len(cells):02d}','metadata':{},'source':text.strip().splitlines(True)})
    def code(text):
        ast.parse(text.strip())
        cells.append({'cell_type':'code','id':f'rpm-{len(cells):02d}','metadata':{},'source':text.strip().splitlines(True),'execution_count':None,'outputs':[]})
    md('''# Triumph RPM — first training pilot

Train a small, separate CNN on **2,000 grayscale images (1,600 train / 400 validation), 256 high × 192 wide**. OpenCV rendered known filled-bar positions on your real LCD backgrounds. No ImageGen was used. Gear, trip, clock and fuel pixels are excluded from the model input.

The model predicts the bar's position on a draft printed scale and a readability score. **RPM labels are approximate scale values, not measured engine RPM.** Validation measures agreement with synthetic targets; it cannot establish real-video accuracy. Startup detection and LCD alignment remain external full-LCD steps.

Choose a **T4 GPU**, then **Run all**. A new RPM network starts with random weights; speedometer weights are incompatible. Every completed epoch is saved to Google Drive.''')
    code(f'''#@title Experiment settings
TRAIN_MODE = "fresh" #@param ["fresh", "latest", "best", "explicit"]
EXPERIMENT_NAME = "triumph_rpm_strip_v1" #@param {{type:"string"}}
EPOCHS = 20 #@param {{type:"integer"}}
BATCH_SIZE = 32 #@param {{type:"integer"}}
CHECKPOINT_PATH = "" #@param {{type:"string"}}
DOWNLOAD_RESULTS = True #@param {{type:"boolean"}}
CODE_COMMIT = "{commit}"
DATA_MANIFEST_SHA256 = "{manifest_sha}"
''')
    md('''## 1. Runtime and Google Drive

`fresh` creates a new run with random weights. `latest` restores the latest full state; `best` restores the best full state; `explicit` loads the checkpoint path entered above. Resume includes optimizer, scheduler, epoch, random states and AMP scaler. Incompatible or incomplete checkpoints fail without silently starting fresh. EPOCHS is the total target epoch, not the number of additional epochs.

Checkpoints stay in `MyDrive/SpeedometerTraining/experiments/triumph_rpm_strip_v1/runs/`. The previous speedometer experiments remain separate.''')
    code('''import os,sys,json,re,hashlib,subprocess,uuid,zipfile,urllib.request,importlib,shutil
from pathlib import Path
expected={'numpy':'2.1.3','cv2':'4.14.0'}
loaded_mismatch=[k for k,v in expected.items() if k in sys.modules and getattr(sys.modules[k],'__version__',None)!=v]
probe_code="import json,numpy,cv2; print(json.dumps({'numpy':numpy.__version__,'cv2':cv2.__version__}))"
probe=subprocess.run([sys.executable,'-c',probe_code],text=True,capture_output=True)
try: disk=json.loads(probe.stdout.strip().splitlines()[-1]) if probe.returncode==0 else {}
except (ValueError,IndexError): disk={}
if disk!=expected:
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','numpy==2.1.3','opencv-python-headless==4.14.0.94'],check=True)
if loaded_mismatch:
    raise RuntimeError('Matching packages are installed. Restart the runtime and Run all before training: '+str(loaded_mismatch))
import numpy as np,cv2,torch
assert {'numpy':np.__version__,'cv2':cv2.__version__}==expected
if not torch.cuda.is_available():
    raise RuntimeError('Select Runtime > Change runtime type > T4 GPU. Training has not started.')
from google.colab import drive
if not Path('/content/drive/MyDrive').is_dir():drive.mount('/content/drive')
assert Path('/content/drive/MyDrive').is_dir(),'Google Drive must be connected before training'
if not re.fullmatch(r'[A-Za-z0-9_-]+',EXPERIMENT_NAME):raise ValueError('Use letters, digits, underscores or hyphens in the experiment name')
EXPERIMENT=Path('/content/drive/MyDrive/SpeedometerTraining/experiments')/EXPERIMENT_NAME
EXPERIMENT.mkdir(parents=True,exist_ok=True)
test=EXPERIMENT/('.write_probe_'+uuid.uuid4().hex);test.write_text('ok');assert test.read_text()=='ok';test.unlink()
WORK=Path('/content/rpm_training_v1_work');WORK.mkdir(exist_ok=True)
print('GPU:',torch.cuda.get_device_name(0))
print('Checkpoint experiment:',EXPERIMENT)
''')
    md('''## 2. Verified code and dataset from GitHub

An immutable commit fixes the implementation. Each archive part and the reconstructed archive are checked before extraction, then every input image and label is verified. Train/validation procedural families are disjoint, but source images still come from the same dashboard recording.''')
    code('''if globals().get('_RPM_CODE_COMMIT',CODE_COMMIT)!=CODE_COMMIT:raise RuntimeError('Restart runtime before changing code version')
REPO=WORK/'repo'
if not REPO.exists():
    REPO.mkdir();subprocess.run(['git','-C',str(REPO),'init'],check=True)
    subprocess.run(['git','-C',str(REPO),'remote','add','origin','https://github.com/sainadh812/triumph-speedometer-dataset.git'],check=True)
subprocess.run(['git','-C',str(REPO),'sparse-checkout','init','--no-cone'],check=True)
subprocess.run(['git','-C',str(REPO),'sparse-checkout','set','--no-cone','/rpm_training_v1/*.py','/rpm_training_v1/calibration/'],check=True)
subprocess.run(['git','-C',str(REPO),'config','remote.origin.promisor','true'],check=True)
subprocess.run(['git','-C',str(REPO),'config','remote.origin.partialclonefilter','blob:none'],check=True)
subprocess.run(['git','-C',str(REPO),'fetch','--depth=1','--filter=blob:none','origin',CODE_COMMIT],check=True)
subprocess.run(['git','-C',str(REPO),'checkout','--detach',CODE_COMMIT],check=True)
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==CODE_COMMIT
sys.path.insert(0,str(REPO))
from rpm_training_v1.notebook_helpers import download_verified,prepare_dataset,sha256
from rpm_training_v1.train_rpm import TrainConfig,train,verify_dataset,load_checkpoint
_RPM_CODE_COMMIT=CODE_COMMIT
BASE=f'https://raw.githubusercontent.com/sainadh812/triumph-speedometer-dataset/{CODE_COMMIT}/rpm_training_v1'
manifest=download_verified(BASE+'/data_download_manifest.json',WORK/'data_download_manifest.json',DATA_MANIFEST_SHA256)
DATA=prepare_dataset(manifest,BASE,WORK/'data')
ROWS,DATA_IDENTITY=verify_dataset(DATA)
print(json.dumps({'counts':DATA_IDENTITY['counts'],'verified_files':DATA_IDENTITY['verified_file_count'],'input_hw':DATA_IDENTITY['input_hw'],'scope':DATA_IDENTITY['provenance_limit']},indent=2))
''')
    md('''## 3. Check training inputs

Dots show known rendered edge targets; they are absent from the model inputs. Negative examples have no numeric RPM label. Printed-scale RPM conversion is provisional. Startup examples appear only in the separate real-frame diagnostics because the strip alone cannot distinguish startup animation from a running engine.''')
    code('''import matplotlib.pyplot as plt
from PIL import Image
chosen=[r for r in ROWS if r['split']=='train' and r['id'].endswith('_0')][:10]
fig,axes=plt.subplots(2,5,figsize=(12,6))
for ax,row in zip(axes.flat,chosen):
    ax.imshow(Image.open(DATA/row['image']),cmap='gray',vmin=0,vmax=255)
    if row['edge_xy']:ax.scatter(*row['edge_xy'],s=14,c='cyan')
    ax.set_title(f"~{row['indicated_rpm_approx']:.0f} RPM" if row['readable'] else 'No numeric target')
    ax.axis('off')
fig.suptitle('Synthetic targets on draft calibration');plt.tight_layout();plt.show()
CONFIG=TrainConfig(dataset=str(DATA),output=str(EXPERIMENT),epochs=EPOCHS,batch_size=BATCH_SIZE,
                   resume=TRAIN_MODE,checkpoint=CHECKPOINT_PATH or None,device='cuda')
print('Mode:',TRAIN_MODE,'| Total epoch target:',EPOCHS,'| Batch:',BATCH_SIZE)
''')
    md('''## 4. Train and save every epoch

**Fill MAE** is average error on the normalized scale (lower is better). **Approximate RPM MAE** multiplies that error by 12,000; it is synthetic agreement, not measured engine-RPM accuracy. **Readability balanced accuracy** balances accepted and rejected examples. **Coverage** is the fraction of inputs receiving a number, not accuracy.

Best-model selection combines fill error on every readable target and readability loss on all targets. Rejecting difficult positives cannot improve the fill metric by hiding their errors.''')
    code('''if globals().get('_RPM_TRAINING_STARTED',False):raise RuntimeError('Training already invoked. Restart and choose latest/best to resume, or fresh for a new run.')
_RPM_TRAINING_STARTED=True
REPORT=train(CONFIG)
RUN=Path(REPORT['run_dir'])
print(json.dumps({k:REPORT[k] for k in ['run_dir','best_epoch','completed_epoch','stop_reason','best_validation']},indent=2))
''')
    md('''## 5. Audit Drive checkpoints and inspect real frames

Every epoch has a full resumable checkpoint. The separate best-model export is for inference, not resume. The real frames below are unlabeled development examples from the source recordings: inspect them visually; no real accuracy percentage is calculated.''')
    code('''audit=[]
for marker in sorted((RUN/'checkpoints').glob('epoch_*.complete.json')):
    item=json.loads(marker.read_text());path=marker.with_suffix('').with_suffix('.pt')
    assert path.is_file() and sha256(path)==item['sha256']
    audit.append({'epoch':item['epoch'],'path':str(path),'sha256':item['sha256']})
for role in ('best_checkpoint','latest_checkpoint'):
    load_checkpoint(REPORT[role]['path'],REPORT['identity'])
(RUN/'drive_checkpoint_audit.json').write_text(json.dumps({'verified_current_run_epochs':audit,'best_latest_full_state_verified':True},indent=2))
print('Checkpoint hashes verified:',len(audit),'| Best epoch:',REPORT['best_epoch'])
from rpm_training_v1.evaluate_real_diagnostics import evaluate_real
REAL=evaluate_real(DATA,REPORT['best_model'],RUN)
print(json.dumps(REAL,indent=2))
from IPython.display import display,Image as IPImage
display(IPImage(filename=REAL['contact_sheet']))
''')
    md('''## 6. Results and download

Inspect the synthetic validation curve and real-frame predictions before deciding whether to add data or change the model. The result ZIP contains selected weights and reports; every full-state checkpoint remains in Drive.''')
    code('''history=json.loads((RUN/'history.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,4))
epochs=[r['epoch'] for r in history]
axes[0].plot(epochs,[r['validation']['approx_indicated_rpm_mae'] for r in history])
axes[0].set(xlabel='Epoch',ylabel='Synthetic approximate RPM MAE')
axes[1].plot(epochs,[100*r['validation']['readability_balanced_accuracy'] for r in history])
axes[1].set(xlabel='Epoch',ylabel='Readability balanced accuracy (%)',ylim=(0,100))
fig.suptitle('Synthetic validation — not independent real RPM accuracy')
plt.tight_layout();plt.savefig(RUN/'training_history.png',dpi=140);plt.show()
include=['best_model.pt','best_model.complete.json','training_report.json','results.json','history.json',
         'config.json','identity.json','initialization.json','validation_predictions.json','drive_checkpoint_audit.json',
         'training_history.png','real_development_predictions.json','real_development_predictions.jpg','best.json','latest.json']
RESULT_ZIP=WORK/(RUN.name+'_rpm_v1_results.zip')
with zipfile.ZipFile(RESULT_ZIP,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in include:
        if (RUN/name).is_file():archive.write(RUN/name,name)
DRIVE_ZIP=RUN/RESULT_ZIP.name;shutil.copy2(RESULT_ZIP,DRIVE_ZIP)
assert sha256(DRIVE_ZIP)==sha256(RESULT_ZIP)
print('Drive results:',DRIVE_ZIP)
print('Best checkpoint:',REPORT['best_checkpoint']['path'])
print('Latest checkpoint:',REPORT['latest_checkpoint']['path'])
if DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(str(RESULT_ZIP))
''')
    nb={'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{'accelerator':'GPU','colab':{'provenance':[],'gpuType':'T4'},
       'kernelspec':{'display_name':'Python 3','name':'python3'},'language_info':{'name':'python'}}}
    path=HERE/'Triumph_RPM_Training_V1_Colab.ipynb'
    path.write_text(json.dumps(nb,indent=1)+'\n',encoding='utf-8')
    print(path)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True);parser.add_argument('--manifest-sha',required=True)
    args=parser.parse_args();build(args.commit,args.manifest_sha)
