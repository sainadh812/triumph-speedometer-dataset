"""Build the standalone Colab notebook; publication pins are supplied by root."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent


def build(commit, manifest_sha):
    cells = []
    def md(source): cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.strip().splitlines(True)})
    def code(source): cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source.strip().splitlines(True)})
    md('''# Triumph — learned LCD detection + frozen V5 speed

Train a small LCD segmentation model on reviewed **unmasked, full-frame** images. At runtime its predicted mask hides only the LCD before the unchanged V5 epoch-11 speed model predicts the needle and scale points. There is no SIFT alignment, input crop, image warp, reference-pose lookup, or previous-frame reading fallback.

All model training and video tests run in Colab. Every completed detector epoch is saved to Drive with optimizer, scaler, RNG, and sampler state. The speed weights remain frozen. Validation chooses LCD thresholds before either video is tested; if its integration criteria are not met, videos still show diagnostic LCD predictions but speed readings remain disabled.

LCD training data covers the full 360° rotation range. The frozen V5 speed model was trained on approximately ±25° camera tilts, so this experiment does not establish end-to-end speed reading at arbitrary angles.

These source recordings and synthetic variants provide development evidence, not a claim of independent road accuracy. Startup needle sweeps indicate dial position, not vehicle motion.''')
    code('''#@title Settings
TRAIN_MODE = "imagenet" #@param ["imagenet", "fresh", "resume_latest", "resume_best", "resume_explicit"]
EXPERIMENT_NAME = "learned_lcd_v1" #@param {type:"string"}
EPOCHS = 20 #@param {type:"integer"}
BATCH_SIZE = 12 #@param [4,8,12] {type:"raw"}
RESUME_CHECKPOINT = "" #@param {type:"string"}
CODE_COMMIT = '''+repr(commit)+'''
DOWNLOAD_MANIFEST_SHA256 = '''+repr(manifest_sha)+'''
SPEED_CHECKPOINT = "/content/drive/MyDrive/SpeedometerTraining/experiments/speedometer_mixed_views_v5/runs/20261018_060437_397258_dc0d47c3/best_model.pt"
SPEED_CHECKPOINT_SHA256 = "de08c9772c4eacc2b78f9037ff6c45c980d63ee6223739199b4ad28fc4777e1a"
''')
    md('''## 1. T4 GPU and Google Drive
Choose a T4 runtime. `imagenet` initializes only the new LCD model from ImageNet; `fresh` initializes it randomly. Resume modes restore full detector training state without changing the frozen speed model.''')
    code('''import sys, subprocess, json, hashlib, shutil, uuid, zipfile, re
from pathlib import Path
from dataclasses import asdict
assert re.fullmatch(r"[0-9a-f]{40}",CODE_COMMIT), 'Notebook publication commit has not been pinned'
assert re.fullmatch(r"[0-9a-f]{64}",DOWNLOAD_MANIFEST_SHA256), 'Dataset manifest has not been pinned'
subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-deps','numpy==2.1.3','opencv-python-headless==4.14.0.94'],check=True)
import torch, cv2, numpy as np
assert torch.cuda.is_available(), 'Select a T4 GPU runtime'
assert cv2.__version__=='4.14.0' and np.__version__=='2.1.3', 'Restart runtime to load the pinned numeric libraries'
from google.colab import drive
if not Path('/content/drive/MyDrive').is_dir(): drive.mount('/content/drive',timeout_ms=300000)
DRIVE_ROOT=Path('/content/drive/MyDrive/SpeedometerTraining');DRIVE_ROOT.mkdir(parents=True,exist_ok=True)
WORK=Path('/content/learned_lcd_v1_work');WORK.mkdir(exist_ok=True)
print('GPU:',torch.cuda.get_device_name(0),'| OpenCV:',cv2.__version__)
''')
    md('''## 2. Verified code, dataset, and frozen speed checkpoint
The pinned code commit and dataset hashes identify this experiment. The LCD model sees original LCD pixels during training. Public dataset assets contain the reviewed source frames and their labels; original video files and Drive checkpoints remain private.''')
    code('''REPO=WORK/'repo'
if not REPO.exists():
    REPO.mkdir()
    subprocess.run(['git','-C',str(REPO),'init'],check=True)
    subprocess.run(['git','-C',str(REPO),'remote','add','origin','https://github.com/sainadh812/triumph-speedometer-dataset.git'],check=True)
subprocess.run(['git','-C',str(REPO),'sparse-checkout','init','--no-cone'],check=True)
subprocess.run(['git','-C',str(REPO),'sparse-checkout','set','--no-cone','/learned_lcd_v1/*.py','/mixed_view_v5/*.py','/mixed_view_v5/calibration.json','/real_adaptation_v4/*.py','/real_adaptation_v3/*.py','/training_pilot/*.py'],check=True)
subprocess.run(['git','-C',str(REPO),'config','remote.origin.promisor','true'],check=True)
subprocess.run(['git','-C',str(REPO),'config','remote.origin.partialclonefilter','blob:none'],check=True)
subprocess.run(['git','-C',str(REPO),'fetch','--depth=1','--filter=blob:none','origin',CODE_COMMIT],check=True)
subprocess.run(['git','-C',str(REPO),'checkout','--detach',CODE_COMMIT],check=True)
sys.path.insert(0,str(REPO))
from learned_lcd_v1.download_dataset import download_dataset,sha256
BASE=f'https://raw.githubusercontent.com/sainadh812/triumph-speedometer-dataset/{CODE_COMMIT}/learned_lcd_v1/'
DATA_DIR=download_dataset(BASE+'download_manifest.json',WORK/'data',DOWNLOAD_MANIFEST_SHA256)
SPEED_CALIBRATION=REPO/'mixed_view_v5/calibration.json'
assert sha256(SPEED_CHECKPOINT)==SPEED_CHECKPOINT_SHA256, 'Unexpected frozen speed weights'
assert Path(SPEED_CHECKPOINT).with_suffix('.complete.json').is_file(), 'Missing checkpoint completion marker'
assert sha256(SPEED_CALIBRATION)=='2581a049a418d0f337a9997dcd6d196aef7d49d4d9194acc2bc176f14441d2f5'
print('Dataset:',DATA_DIR)
print('Frozen V5 epoch11 speed checkpoint verified:',SPEED_CHECKPOINT)
''')
    md('''## 3. Train the LCD detector and save every epoch
Training validates images, masks, source groups and hashes. Selection and threshold calibration use validation only. LCD foreground precision, recall, IoU, and boundary error describe mask quality; accepted coverage describes how often the gate permits the speed model to run. A segmentation score is not calibrated certainty.

The validation gate requires at least **98% accepted-mask foreground precision, 90% accepted-mask foreground recall, and 80% positive-frame acceptance**. Recall prevents a tiny precise fragment from qualifying while leaving most of the LCD visible. Thresholds stay fixed during video testing. Failure to meet these criteria produces a diagnostic-only export instead of silently relaxing the rules.''')
    code('''from learned_lcd_v1.train import TrainConfig,train
EXPERIMENT=DRIVE_ROOT/'experiments'/EXPERIMENT_NAME
if TRAIN_MODE=='fresh': EXPERIMENT=EXPERIMENT.with_name(EXPERIMENT.name+'_fresh_'+uuid.uuid4().hex[:8])
if TRAIN_MODE=='imagenet' and (EXPERIMENT/'runs').exists():
    raise FileExistsError('Experiment already exists: choose a resume mode or a new experiment name')
CONFIG=TrainConfig(data_dir=str(DATA_DIR),output_dir=str(EXPERIMENT/'runs'),epochs=EPOCHS,batch_size=BATCH_SIZE)
print(json.dumps(asdict(CONFIG),indent=2))
REPORT=train(CONFIG,mode=TRAIN_MODE,checkpoint=RESUME_CHECKPOINT if TRAIN_MODE=='resume_explicit' else None)
RUN=Path(REPORT['best_model_path']).parent
print(json.dumps({
    'best_epoch':REPORT['best_epoch'],'completed_epoch':REPORT['completed_epoch'],
    'ready_for_integration':REPORT['ready_for_integration'],
    'validation_iou':REPORT['validation_metrics']['raw']['macro_iou'],
    'validation_gate_coverage':REPORT['validation_metrics']['coverage_fraction'],
    'gate_selection':REPORT['gate_selection']['selected_validation_result'],
    'best_model_path':REPORT['best_model_path'],'training_report':str(RUN/'training_report.json')},indent=2))
print('Every completed epoch remains in:',RUN)
''')
    md('''## 4. Verify the two private test videos
Reuse the original 17:43:59 daytime video and 18:13:23 straight-view video already in Drive. If one is missing, use Colab's Files panel to upload it into `/content`, then rerun this cell. These are development videos; they do not supply labels to inference.''')
    code('''VIDEO_DIR=DRIVE_ROOT/'private_test_videos';VIDEO_DIR.mkdir(exist_ok=True)
VIDEO_SPECS={
 'WhatsApp Video 2026-10-17 at 5.43.59 PM.mp4':'f6aa7bcd1b4dc2f5e93ad6a91cf2e865f2f05391485710cb716d568116dc6828',
 'WhatsApp Video 2026-10-17 at 6.13.23 PM.mp4':'6125503654f3d2a06ac67765bfc4f5af670f027335a501d50e2a88412b8ffaf9'}
for name,digest in VIDEO_SPECS.items():
    local_upload=Path('/content')/name
    if not (VIDEO_DIR/name).is_file() and local_upload.is_file():
        assert sha256(local_upload)==digest, 'Unexpected uploaded video: '+name
        shutil.copy2(local_upload,VIDEO_DIR/name)
missing=[name for name in VIDEO_SPECS if not (VIDEO_DIR/name).is_file()]
if missing: raise FileNotFoundError('Upload these videos into /content using the Colab Files panel, then rerun: '+str(missing))
for name,digest in VIDEO_SPECS.items(): assert sha256(VIDEO_DIR/name)==digest, 'Video missing or changed: '+name
print('Both original private videos verified.')
''')
    md('''## 5. Run both videos at 30 FPS and inspect the actual input
The output preserves the full original frame with a cyan LCD outline and needle/tick predictions. The inset shows the actual masked speed input, or the LCD diagnostic map when speed is withheld. Gray128 masking happens only after the current LCD gate passes. A model that failed the validation integration criteria produces diagnostic videos with speed disabled.

Coverage is compared with the previous frozen-V5 test: 602/602 accepted daytime frames and 340/540 straight-view frames. This comparison measures availability, not correctness. No speed accuracy or LCD IoU can be measured on an unannotated video.''')
    code('''EVAL_ROOT=WORK/('video_evaluation_'+RUN.name+'_'+uuid.uuid4().hex[:8])
command=[sys.executable,'-u','-m','learned_lcd_v1.evaluate_videos',
         '--lcd-checkpoint',str(REPORT['best_model_path']),
         '--speed-checkpoint',SPEED_CHECKPOINT,'--speed-calibration',str(SPEED_CALIBRATION),
         '--output-root',str(EVAL_ROOT),'--device','cuda']
for name in VIDEO_SPECS: command.extend(['--video',str(VIDEO_DIR/name)])
with subprocess.Popen(command,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
    for line in process.stdout: print(line,end='',flush=True)
    returncode=process.wait()
    if returncode: raise subprocess.CalledProcessError(returncode,command)
print('Verified complete 30FPS inspection videos:',EVAL_ROOT)
''')
    md('''## 6. Save results to Drive and download
The archive includes videos, predictions, gate reasons, training metrics, the selected LCD weights, and pipeline source. Full resumable epoch checkpoints remain in the Drive run directory. H264 outputs preserve audio when present and are fully decoded to check frame count, duration, and 30 FPS before completion.''')
    code('''RESULT_ZIP=WORK/(RUN.name+'_learned_lcd_results.zip')
with zipfile.ZipFile(RESULT_ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
    for path in RUN.rglob('*'):
        if path.is_file() and (path.suffix in ['.json','.jsonl','.csv','.png'] or path.name in ['best_model.pt','best_model.complete.json']):
            archive.write(path,'training/'+path.relative_to(RUN).as_posix())
    for path in EVAL_ROOT.rglob('*'):
        if path.is_file() and path.suffix!='.avi':
            archive.write(path,'video_evaluation/'+path.relative_to(EVAL_ROOT).as_posix())
    for path in sorted((REPO/'learned_lcd_v1').glob('*.py')):
        archive.write(path,'pipeline/learned_lcd_v1/'+path.name)
    archive.write(SPEED_CALIBRATION,'pipeline/speed_calibration.json')
    archive.writestr('pipeline/publication.json',json.dumps({'code_commit':CODE_COMMIT,
        'download_manifest_sha256':DOWNLOAD_MANIFEST_SHA256,'speed_checkpoint_sha256':SPEED_CHECKPOINT_SHA256},indent=2))
DRIVE_ZIP=RUN/RESULT_ZIP.name
shutil.copy2(RESULT_ZIP,DRIVE_ZIP)
assert sha256(DRIVE_ZIP)==sha256(RESULT_ZIP)
print('Verified Drive archive:',DRIVE_ZIP,'| SHA256:',sha256(RESULT_ZIP))
from google.colab import files
files.download(str(RESULT_ZIP))
''')
    for index, cell in enumerate(cells): cell['id'] = f'learned-lcd-{index:02d}'
    notebook = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}},
                'nbformat': 4, 'nbformat_minor': 5}
    destination = ROOT/'Triumph_Learned_LCD_V1_Colab.ipynb'
    destination.write_text(json.dumps(notebook, indent=1)+'\n', encoding='utf-8')
    (ROOT/'colab_cells.json').write_text(json.dumps([''.join(cell['source']) for cell in cells
                                                  if cell['cell_type']=='code'], indent=2)+'\n', encoding='utf-8')
    print(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', default='SET_AFTER_PUBLICATION')
    parser.add_argument('--manifest-sha', default='SET_AFTER_DATASET_PUBLICATION')
    args = parser.parse_args()
    build(args.commit, args.manifest_sha)
