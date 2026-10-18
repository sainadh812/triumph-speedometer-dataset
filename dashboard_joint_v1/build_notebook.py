"""Create the public template or a private configured Colab notebook.

Private dataset identifiers are supplied in an external config. OAuth values
must never appear in config files or notebook cells; Colab Secrets provides them.
"""
import argparse
import hashlib
import json
from pathlib import Path

RUNTIME_FILES=['preprocess.py','dataset.py','model.py','metrics.py','state.py','sampling.py','initialization.py','train.py','colab_workflow.py']

def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def bootstrap(commit,hashes):
    return f'''import sys, subprocess, hashlib, urllib.request, json
from pathlib import Path
CODE_COMMIT = {commit!r}
CODE_SHA256 = {hashes!r}
runtime_signature = hashlib.sha256(json.dumps({{'commit': CODE_COMMIT, 'files': CODE_SHA256}}, sort_keys=True).encode()).hexdigest()
loaded_runtime_modules = [name for name in sys.modules if name == 'drive_api_store' or name == 'dashboard_joint_v1' or name.startswith('dashboard_joint_v1.')]
if loaded_runtime_modules and globals().get('_TRIUMPH_JOINT_CODE_SIGNATURE') != runtime_signature:
    raise RuntimeError('Pinned training code changed in this kernel. Restart the runtime before loading this revision.')
code_root = Path('/content/triumph_joint_code')
code_root.mkdir(exist_ok=True)
for relative, expected in CODE_SHA256.items():
    target = code_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        url = 'https://raw.githubusercontent.com/sainadh812/triumph-speedometer-dataset/' + CODE_COMMIT + '/' + relative
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError('Pinned code verification failed')
        target.write_bytes(data)
(code_root / 'dashboard_joint_v1' / '__init__.py').touch()
if str(code_root) not in sys.path: sys.path.insert(0, str(code_root))
transport = code_root / 'learned_lcd_v1' / 'drive_api_video_v1'
if str(transport) not in sys.path: sys.path.insert(0, str(transport))
from drive_api_store import DriveStore
from dashboard_joint_v1.colab_workflow import prepare, run_all
_TRIUMPH_JOINT_CODE_SIGNATURE = runtime_signature
store = DriveStore.from_colab_secret('TRIUMPH_DRIVE_OAUTH')
'''

def build(repository,commit,config=None):
    repo=Path(repository); cfg=config or {}
    paths=['dashboard_joint_v1/'+p for p in RUNTIME_FILES]+['learned_lcd_v1/drive_api_video_v1/drive_api_store.py']
    hashes={p:file_hash(repo/p) for p in paths}
    setup=bootstrap(commit,hashes)
    reference=cfg.get('dataset_reference',{'id':'SET_PRIVATE_DATASET_FILE_ID','sha256':'SET_DATASET_SHA256','size':0})
    workspace=cfg.get('workspace_id','SET_PRIVATE_WORKSPACE_FOLDER_ID')
    run_id=cfg.get('run_id','joint_v1_first_run')
    settings=f'WORKSPACE_ID = {workspace!r}\nDATASET_REFERENCE = {reference!r}\n'
    description='''# Train dashboard/LCD, masked speed, and RPM together

This workflow trains **all three components sequentially in one Colab run**. It starts from ImageNet features with new task heads; it does not load or compare the old readers.

- **Localizer:** full video frame → dashboard box/presence, precise LCD mask, four semantic LCD corners and four RPM-region corners. The RPM region can remain usable when the far LCD corner is off screen.
- **Speed:** dashboard view, exact LCD interior filled gray128 → pivot, needle tip, 0–180 scale ticks.
- **RPM:** wider aligned LCD scale view → anchors0,2,4,6,8,10,12 and fill-bar endpoint. Full scale and bar remain visible.

Reviewed day/night frames are augmented with label-consistent rotation, translation, mild shear/perspective, exposure, reflections, noise and blur. Horizontal/vertical mirror cases keep their semantic labels. These transforms preserve the actual observed needle/bar state. Additional procedural RPM-bar examples, when included, are explicitly marked synthetic; their scalar is a rendering target, not measured engine RPM. Only training uses those renders. Source time blocks never cross train/validation/test. This is development evaluation within two recordings, not an independent accuracy certificate.

Run cell1 on **CPU** first. Switch to **T4 GPU** only after it passes, then run cell2. The runtime can be reset between cells: code and data are reloaded from verified persistent sources. Existing Colab secret `TRIUMPH_DRIVE_OAUTH` provides authorized private Drive API access; no Drive mount or credentials in cells.

Every epoch saves weights, optimizer, scheduler, AMP and RNG state to Drive, verified by SHA256/size. Fresh requires a new run ID. Latest resumes interruption; best resumes a selected checkpoint in a preserved branch. All three tasks use validation to select weights, then evaluate development test data. Results and best weights are exported; the GPU releases automatically only after verified saving.
'''
    cpu='''#@title 1. CPU: verify private dataset, labels and Drive access
import torch
if torch.cuda.is_available():
    raise RuntimeError('Use Hardware accelerator=None for preparation; GPU is not needed here.')
'''+setup+settings+'''prepared = prepare(store, WORKSPACE_ID, DATASET_REFERENCE)
print('CPU preflight passed. The dataset and all task splits are verified. Now select T4 and run cell2.')
'''
    gpu=f'''#@title 2. T4: train all components and save every epoch
RUN_ID = {run_id!r} #@param {{type:"string"}}
LOCALIZER_RESUME = "fresh" #@param ["fresh", "latest", "best"]
SPEED_RESUME = "fresh" #@param ["fresh", "latest", "best"]
RPM_RESUME = "fresh" #@param ["fresh", "latest", "best"]
AUTO_DISCONNECT = True #@param {{type:"boolean"}}
'''+'''# Bootstrap/preparation failures have no new training results to preserve.
try:
'''+''.join('    '+line+'\n' for line in (setup+settings).splitlines())+'''    prepared = prepare(store, WORKSPACE_ID, DATASET_REFERENCE)
except BaseException:
    from google.colab import runtime
    print('Preparation failed before training; releasing the runtime. No new training artifacts exist.')
    runtime.unassign()
    raise
configs = {
    'localizer': {'batch_size':16, 'epochs':40, 'min_epochs':10, 'patience':8},
    'speed': {'batch_size':32, 'epochs':40, 'min_epochs':10, 'patience':8},
    'rpm': {'batch_size':64, 'epochs':40, 'min_epochs':10, 'patience':8},
}
summary = run_all(store, prepared, RUN_ID,
    resume_modes={'localizer':LOCALIZER_RESUME,'speed':SPEED_RESUME,'rpm':RPM_RESUME},
    configs=configs, auto_disconnect=AUTO_DISCONNECT)
'''
    end='''## Results and restart

Private Drive path: `SpeedometerTraining_API_v1 / dashboard_joint_v1 / RUN_ID`.

Each task folder retains every resumable epoch. Each attempt has its own history and receipt files. A final `<attempt>_COMPLETE.json` identifies the checksum-verified results ZIP and all three task reports. The ZIP contains the best weights, task metrics and training history; the individual full epoch checkpoints remain in Drive.

If interrupted, keep the same RUN_ID. Choose **latest** for tasks with a saved checkpoint (including finished tasks), and **fresh** for tasks that have never reached an epoch checkpoint. Finished tasks restore their saved report instead of rerunning their completed training/test. To branch from validation-best choose **best**. For a completely fresh experiment choose a **new RUN_ID** and **fresh** for all tasks. Resume requires the same data, architecture, preprocessing and training configuration.

Landmark pixel errors, LCD mask IoU, box IoU and visibility/presence metrics describe this run. They are not yet a claim of physical RPM accuracy or98% real-world speed accuracy. Inspect new video results before making that conclusion. No old-model comparison is performed.
'''
    cells=[]
    for ident,kind,source in [('joint_intro','markdown',description),('joint_cpu','code',cpu),('joint_train','code',gpu),('joint_results','markdown',end)]:
        cell={'cell_type':kind,'metadata':{'id':ident},'source':source.splitlines(keepends=True),'id':ident}
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Triumph_All_Models_Training.ipynb'},
             'kernelspec':{'name':'python3','display_name':'Python3'},'language_info':{'name':'python'}},'cells':cells}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--commit',required=True)
    p.add_argument('--config');p.add_argument('--output',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text()) if a.config else None
    Path(a.output).write_text(json.dumps(build(a.repo,a.commit,config),indent=2),encoding='utf8',newline='\n')
