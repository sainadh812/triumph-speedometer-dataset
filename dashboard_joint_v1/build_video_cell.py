"""Build an appended Colab cell for the two private post-training video tests."""
import hashlib
import json
from pathlib import Path


FILES = ['preprocess.py', 'dataset.py', 'model.py', 'metrics.py', 'state.py',
         'sampling.py', 'initialization.py', 'train.py', 'colab_workflow.py', 'inference.py', 'video_test.py']


def build(repository, commit, workspace_id, run_reference, video_config):
    root = Path(repository)
    paths = ['dashboard_joint_v1/' + name for name in FILES]
    paths += ['learned_lcd_v1/drive_api_video_v1/drive_api_store.py']
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}
    source = '''#@title 3. Test the last two minutes of both videos: dashboard + LCD + speed + RPM
# Uses saved NEW weights. This cell performs inference only, without retraining.
import sys, hashlib, urllib.request, json
from pathlib import Path
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Select T4 GPU, then run this cell. Training cells do not need to run again.')
'''
    source += f'VIDEO_COMMIT = {commit!r}\nVIDEO_CODE_SHA256 = {hashes!r}\n'
    source += '''code_root = Path('/content/triumph_joint_code')
code_root.mkdir(parents=True, exist_ok=True)
try:
    # Permit adding inference to a training kernel only when every already-loaded
    # module still has the exact pinned bytes originally loaded by that kernel.
    previous_hashes = globals().get('CODE_SHA256', {}) | globals().get('_VIDEO_LOADED_HASHES', {})
    for relative, expected in VIDEO_CODE_SHA256.items():
        module_name = 'drive_api_store' if relative.endswith('/drive_api_store.py') else relative[:-3].replace('/', '.')
        module = sys.modules.get(module_name)
        if module is not None and previous_hashes.get(relative) != expected:
            raise RuntimeError('An already loaded model module differs. Restart the runtime and run only cell3.')
        target = code_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            url = 'https://raw.githubusercontent.com/sainadh812/triumph-speedometer-dataset/' + VIDEO_COMMIT + '/' + relative
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError('Video-test code checksum mismatch')
            target.write_bytes(data)
    (code_root / 'dashboard_joint_v1' / '__init__.py').touch()
    if str(code_root) not in sys.path: sys.path.insert(0, str(code_root))
    transport = code_root / 'learned_lcd_v1' / 'drive_api_video_v1'
    if str(transport) not in sys.path: sys.path.insert(0, str(transport))
    from drive_api_store import DriveStore
    from dashboard_joint_v1.video_test import run_video_tests
    _VIDEO_LOADED_HASHES = VIDEO_CODE_SHA256.copy()
    store = DriveStore.from_colab_secret('TRIUMPH_DRIVE_OAUTH')
'''
    source += f'    WORKSPACE_ID = {workspace_id!r}\n'
    source += '''    video_root = store.ensure_folder('dashboard_joint_v1', WORKSPACE_ID)
except BaseException:
    from google.colab import runtime
    print('Video setup failed before inference; no new video results exist. Releasing the runtime.')
    runtime.unassign()
    raise
'''
    source += f'RUN_REFERENCE = {run_reference!r}\nVIDEO_CONFIG = {video_config!r}\n'
    source += '''VIDEO_CONFIG['code_commit'] = VIDEO_COMMIT
video_results = run_video_tests(store,
    {'drive_root_id':video_root['id'], 'workdir':'/content/triumph_joint_video_test'},
    RUN_REFERENCE, VIDEO_CONFIG, auto_disconnect=True, download=False)
# Download the saved labeled MP4s from Drive after the runtime releases.
'''
    return {'cell_type': 'code', 'id': 'joint_video_test', 'metadata': {'id': 'joint_video_test'},
            'source': source.splitlines(keepends=True), 'outputs': [], 'execution_count': None}
