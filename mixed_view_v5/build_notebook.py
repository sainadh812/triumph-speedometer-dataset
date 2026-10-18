"""Build the published V5 notebook from exact code/data publication pins."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).parent

def build(commit,manifest_sha):
    cells=[]
    def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s.strip().splitlines(True)))
    def code(s): cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.strip().splitlines(True)))
    md('''# Triumph V5 — straight, left and right camera views

7,500 images derived from the two reviewed speed datasets. The image, LCD mask and all 12 landmarks rotate together; numbers are never mirrored. Parent images and every derivative retain the original train/validation split. Batches balance both source views and all three tilt groups.

Inputs remain full-frame, grayscale and letterboxed to 512×512. Registration projects the LCD mask only; it does **not** rotate the input into one fixed camera view. The model predicts the needle and ten scale landmarks; geometry converts their relative positions into indicated speed.

This is a new experiment initialized from all V4 epoch-17 weights. Every completed epoch saves full training state to Drive. Validation and both supplied videos come from the source recordings, so these are development results, not independent road accuracy. The ignition sweep video shows indicated needle speed, not physical vehicle speed. All model training and video testing execute in Colab.''')
    code('''#@title Settings
TRAIN_MODE = "transfer_v4" #@param ["transfer_v4", "resume_latest", "resume_best", "fresh"]
EXPERIMENT_NAME = "speedometer_mixed_views_v5" #@param {type:"string"}
EPOCHS = 12 #@param {type:"integer"}
BATCH_SIZE = 12 #@param [6,12] {type:"raw"}
SOURCE_CHECKPOINT = "/content/drive/MyDrive/SpeedometerTraining/experiments/speedometer_day_arc_needle_v4/runs/20261017_155953_221267_c1b59ef0/best_model.pt"
SOURCE_SHA256 = "ed7f27c44e89f4caebefddc79fb1481786efc2e6fc3ddd103f7bcb3b2625857d"
CODE_COMMIT = '''+repr(commit)+'''
DATA_MANIFEST_SHA256 = '''+repr(manifest_sha)+'''
''')
    md('''## 1. GPU and checkpoint storage
Choose **T4 GPU**. The setup checks the GPU and mounts your existing Drive storage. Original videos and checkpoints stay private in Drive; only the synthetic dataset and code are public.''')
    code('''import sys, subprocess, json, hashlib, shutil, uuid, zipfile, os
from pathlib import Path
from dataclasses import asdict
subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-deps','numpy==2.1.3','opencv-python-headless==4.14.0.94'],check=True)
import torch, cv2, numpy as np
assert cv2.__version__=='4.14.0' and np.__version__=='2.1.3', 'Restart the runtime to load matching numeric libraries'
assert torch.cuda.is_available(), 'Choose T4 GPU before training'
from google.colab import drive
if not Path('/content/drive/MyDrive').is_dir(): drive.mount('/content/drive')
DRIVE_ROOT=Path('/content/drive/MyDrive/SpeedometerTraining')
DRIVE_ROOT.mkdir(parents=True,exist_ok=True)
WORK=Path('/content/speedometer_mixed_views_v5_work');WORK.mkdir(exist_ok=True)
print('GPU:',torch.cuda.get_device_name(0),'| OpenCV:',cv2.__version__)
print('Checkpoint storage:',DRIVE_ROOT)
''')
    md('''## 2. Verified dataset and code
The pinned commit and SHA256 hashes make this run reproducible. Every image and annotation is checked before training.''')
    code('''REPO=WORK/'repo'
if not REPO.exists():
    REPO.mkdir()
    subprocess.run(['git','-C',str(REPO),'init'],check=True)
    subprocess.run(['git','-C',str(REPO),'remote','add','origin','https://github.com/sainadh812/triumph-speedometer-dataset.git'],check=True)
subprocess.run(['git','-C',str(REPO),'sparse-checkout','init','--no-cone'],check=True)
subprocess.run(['git','-C',str(REPO),'sparse-checkout','set','--no-cone','/mixed_view_v5/*.py','/real_adaptation_v4/*.py','/real_adaptation_v3/*.py','/training_pilot/*.py'],check=True)
subprocess.run(['git','-C',str(REPO),'config','remote.origin.promisor','true'],check=True)
subprocess.run(['git','-C',str(REPO),'config','remote.origin.partialclonefilter','blob:none'],check=True)
subprocess.run(['git','-C',str(REPO),'fetch','--depth=1','--filter=blob:none','origin',CODE_COMMIT],check=True)
subprocess.run(['git','-C',str(REPO),'checkout','--detach',CODE_COMMIT],check=True)
sys.path.insert(0,str(REPO))
from real_adaptation_v3.notebook_helpers import download_verified,sha256
from mixed_view_v5.notebook_helpers import prepare_dataset
BASE=f'https://raw.githubusercontent.com/sainadh812/triumph-speedometer-dataset/{CODE_COMMIT}/mixed_view_v5'
manifest=download_verified(BASE+'/data_download_manifest.json',WORK/'data_download_manifest.json',DATA_MANIFEST_SHA256)
DATA_DIR=prepare_dataset(manifest,BASE,WORK/'data')
print('Verified dataset:',DATA_DIR)
from mixed_view_v5.train_landmarks import TrainConfig,train,verify_dataset
partitions,integrity=verify_dataset(DATA_DIR)
print(json.dumps(integrity,indent=2))
''')
    md('''## 5. Train and save every epoch
`transfer_v4` starts this mixed-view experiment from the existing weights with a new optimizer. `resume_latest` restores the last completed V5 epoch; `resume_best` restores the best V5 checkpoint. `fresh` uses no previous weights and gets its own experiment folder.

Checkpoint selection compares the two source views equally. Per-view and per-tilt results reveal regressions that a pooled score could hide. Rejected readings count as failures for the all-input ±2 km/h score.''')
    code('''EXPERIMENT=DRIVE_ROOT/'experiments'/EXPERIMENT_NAME
if TRAIN_MODE=='fresh': EXPERIMENT=EXPERIMENT.with_name(EXPERIMENT.name+'_fresh_'+uuid.uuid4().hex[:8])
if TRAIN_MODE=='transfer_v4' and EXPERIMENT.exists():
    raise FileExistsError('Experiment exists: select resume_latest/resume_best or a new experiment name')
if TRAIN_MODE=='transfer_v4': assert sha256(SOURCE_CHECKPOINT)==SOURCE_SHA256, 'Unexpected source weights'
CONFIG=TrainConfig(data_dir=str(DATA_DIR),output_dir=str(EXPERIMENT/'runs'),epochs=EPOCHS,batch_size=BATCH_SIZE)
print(json.dumps(asdict(CONFIG),indent=2))
REPORT=train(CONFIG,mode=TRAIN_MODE,source_checkpoint=SOURCE_CHECKPOINT if TRAIN_MODE=='transfer_v4' else None,
             search_root=EXPERIMENT/'runs' if TRAIN_MODE.startswith('resume_') else None)
RUN=Path(REPORT['run_dir'])
print(json.dumps(REPORT,indent=2))
''')
    md('''## 3. Upload the two original videos privately
Reuses matching copies in your Drive. On first use select the 17:43:59 daytime video and the 18:13:23 straight-view video. Their hashes are checked. No input crops or artificial tilts are applied during testing.''')
    code('''VIDEO_DIR=DRIVE_ROOT/'private_test_videos';VIDEO_DIR.mkdir(exist_ok=True)
VIDEO_SPECS={
 'WhatsApp Video 2026-10-17 at 5.43.59 PM.mp4':'f6aa7bcd1b4dc2f5e93ad6a91cf2e865f2f05391485710cb716d568116dc6828',
 'WhatsApp Video 2026-10-17 at 6.13.23 PM.mp4':'6125503654f3d2a06ac67765bfc4f5af670f027335a501d50e2a88412b8ffaf9'}
missing=[n for n,h in VIDEO_SPECS.items() if not (VIDEO_DIR/n).is_file()]
if missing:
    from google.colab import files
    print('Select:',missing)
    uploaded=files.upload()
    for name,blob in uploaded.items():
        if name not in VIDEO_SPECS or hashlib.sha256(blob).hexdigest()!=VIDEO_SPECS[name]:
            raise ValueError('Unexpected video: '+name)
        (VIDEO_DIR/name).write_bytes(blob)
for name,h in VIDEO_SPECS.items(): assert sha256(VIDEO_DIR/name)==h, 'Video missing or changed: '+name
print('Both private source videos verified.')
''')
    md('''## 6. Colab video evaluation
Run before/after models using the same V5 input pipeline. Outputs retain the full original frame, with needle/tick overlays, indicated speed or an explicit rejection reason. Coverage is reported separately from accuracy. Both videos are source-recording checks; no 98% accuracy claim can be inferred from them.''')
    code('''EVAL_ROOT=WORK/('video_evaluation_'+RUN.name)
EVAL_ROOT.mkdir(exist_ok=False)
shutil.copytree(BASELINE_ROOT,EVAL_ROOT/'before')
for role,checkpoint,baseline in [('after',REPORT['best_model_path'],False)]:
    command=[sys.executable,'-u','-m','mixed_view_v5.evaluate_videos',
             '--checkpoint',str(checkpoint),'--dataset',str(DATA_DIR),
             '--output-root',str(EVAL_ROOT/role),'--device','cuda']
    for name in VIDEO_SPECS: command.extend(['--video',str(VIDEO_DIR/name)])
    if baseline: command.append('--baseline-v4')
    with subprocess.Popen(command,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,bufsize=1) as process:
        for line in process.stdout: print(line,end='',flush=True)
        returncode=process.wait()
        if returncode: raise subprocess.CalledProcessError(returncode,command)
print('Before/after video evaluation complete:',EVAL_ROOT)
''')
    md('''## 7. Save and download the inspection results
The ZIP includes both before/after videos, frame predictions, metrics and selected weights. Full resumable checkpoints remain in the Drive run folder. Each MP4 is decoded to verify its frame count and 30 FPS timeline before it is marked complete.''')
    code('''RESULT_ZIP=WORK/(RUN.name+'_v5_results.zip')
with zipfile.ZipFile(RESULT_ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
    for p in RUN.iterdir():
        if p.is_file() and (p.suffix=='.json' or p.name in ['best_model.pt','best_model.complete.json','best_worst_view_model.pt','best_worst_view_model.complete.json']):
            archive.write(p,'training/'+p.name)
    for p in EVAL_ROOT.rglob('*'):
        if p.is_file() and p.suffix!='.avi': archive.write(p,'video_evaluation/'+p.relative_to(EVAL_ROOT).as_posix())
DRIVE_ZIP=RUN/RESULT_ZIP.name
shutil.copy2(RESULT_ZIP,DRIVE_ZIP)
assert sha256(DRIVE_ZIP)==sha256(RESULT_ZIP)
print('Verified results archive:',DRIVE_ZIP,'| SHA256:',sha256(RESULT_ZIP))
from google.colab import files
files.download(str(RESULT_ZIP))
''')
    # Inspect the full video input pipeline before spending GPU time training.
    end=len(cells)
    md('''## 4. Baseline video check before training
Test the existing V4 weights with the same V5 preprocessing used after training. This isolates the effect of new weights from the change in input handling. The original V4 checkpoints remain unchanged.''')
    code('''assert sha256(SOURCE_CHECKPOINT)==SOURCE_SHA256, 'Unexpected baseline weights'
BASELINE_ROOT=WORK/('baseline_v4_in_v5_'+uuid.uuid4().hex[:8])
command=[sys.executable,'-u','-m','mixed_view_v5.evaluate_videos',
         '--checkpoint',str(SOURCE_CHECKPOINT),'--dataset',str(DATA_DIR),
         '--output-root',str(BASELINE_ROOT),'--device','cuda','--baseline-v4']
for name in VIDEO_SPECS: command.extend(['--video',str(VIDEO_DIR/name)])
with subprocess.Popen(command,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                      text=True,bufsize=1) as process:
    for line in process.stdout: print(line,end='',flush=True)
    returncode=process.wait()
    if returncode: raise subprocess.CalledProcessError(returncode,command)
print('Baseline complete:',BASELINE_ROOT)
''')
    cells=cells[:6]+cells[8:10]+cells[end:]+cells[6:8]+cells[10:end]
    notebook=dict(cells=cells,metadata=dict(accelerator='GPU',colab=dict(provenance=[],gpuType='T4'),
                  kernelspec=dict(display_name='Python 3',name='python3'),language_info=dict(name='python')),
                  nbformat=4,nbformat_minor=5)
    for i,c in enumerate(cells):c['id']=f'v5-{i:02d}'
    dest=ROOT/'Triumph_Mixed_Views_V5_Colab.ipynb'
    dest.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
    print(dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--manifest-sha',required=True)
    a=p.parse_args();build(a.commit,a.manifest_sha)
