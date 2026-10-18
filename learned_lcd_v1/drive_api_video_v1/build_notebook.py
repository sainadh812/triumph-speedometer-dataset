"""Build a credential-free Colab notebook pinned to reviewed public code."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import textwrap

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NEW_DIR = 'learned_lcd_v1/drive_api_video_v1'
OLD_DIR = 'learned_lcd_v1/upload_video_test_v1'
RUNTIME_PATHS = {
    'drive_api_store.py': NEW_DIR + '/drive_api_store.py',
    'api_colab_workflow.py': NEW_DIR + '/api_colab_workflow.py',
    **{name: OLD_DIR + '/' + name for name in (
        'colab_video_helpers.py', 'evaluate_uploaded_video.py',
        'evaluate_lcd_speed_rpm_batched.py', 'benchmark_lcd_positioning.py')},
}


def bootstrap(commit, inventory):
    return textwrap.dedent('''\
        # This public notebook contains no credentials, weights, or private file IDs.
        from pathlib import Path
        import hashlib, importlib.util, os, sys, urllib.request, uuid
        CODE_COMMIT = __COMMIT__
        EXPECTED_FILES = __FILES__
        RUNTIME_DIR = Path('/content/triumph_api_runtime') / CODE_COMMIT
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        for name, metadata in EXPECTED_FILES.items():
            target = RUNTIME_DIR / name
            if target.is_file():
                data = target.read_bytes()
            else:
                url = 'https://raw.githubusercontent.com/sainadh812/triumph-speedometer-dataset/' + CODE_COMMIT + '/' + metadata['path']
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        data = response.read()
                except Exception:
                    raise RuntimeError('Could not download reviewed public code: ' + name) from None
            if len(data) != metadata['bytes'] or hashlib.sha256(data).hexdigest() != metadata['sha256']:
                raise RuntimeError('Public runtime checksum mismatch: ' + name)
            if not target.exists():
                temporary = target.with_name(name + '.' + uuid.uuid4().hex + '.tmp')
                temporary.write_bytes(data)
                os.replace(temporary, target)
        for name in ('drive_api_store', 'api_colab_workflow'):
            specification = importlib.util.spec_from_file_location(name, RUNTIME_DIR / (name + '.py'))
            module = importlib.util.module_from_spec(specification)
            sys.modules[name] = module
            specification.loader.exec_module(module)
        workflow = sys.modules['api_colab_workflow']
        ''').replace('__COMMIT__', repr(commit)).replace('__FILES__', repr(inventory))


def build(commit, output):
    if not re.fullmatch(r'[a-f0-9]{40}', commit):
        raise ValueError('A full immutable runtime code commit is required.')
    inventory = {}
    for name, relative in RUNTIME_PATHS.items():
        data = (REPO / relative).read_bytes()
        ast.parse(data, filename=name)
        inventory[name] = {'path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    setup = bootstrap(commit, inventory)
    intro = '''# Test a video: LCD + speed + RPM, using the Drive API

Run the cells individually. Start with **Hardware accelerator: None**.

This notebook uses the existing LCD epoch 9, speed epoch 11 and RPM epoch 9 checkpoints. It preserves the previous diagnostic settings, full input/output labels and larger RPM text. It does not train or improve those models. Raw readings remain unvalidated; known tick-position and RPM-crop errors remain.

**One-time setup:** in Colab's key/Secrets panel, add the private OAuth JSON under **TRIUMPH_DRIVE_OAUTH**, then allow this notebook to access that secret. Never paste credentials into a code cell or notebook output. Use the configured Google account. The secret is read only in memory; the public notebook contains none of its values. An External/Testing OAuth grant expires after about seven days and needs renewal; this is not permanent authorization.

The first CPU check verifies account access and the migrated model package. To browse for a video, turn **SETUP_CHECK_ONLY** off and rerun cell 1. The upload is saved through the Drive API before you switch to T4. Cell 2 downloads verified assets, runs the unchanged models, saves all results privately, then releases the GPU. There is no Drive mount.
'''
    cpu = ('#@title 1. CPU setup check / browse and save a video\n'
           'SETUP_CHECK_ONLY = True #@param {type:"boolean"}\n'
           'START_SECONDS = 0.0 #@param {type:"number"}\n'
           'MAX_SECONDS = 0.0 #@param {type:"number"}\n'
           '# MAX_SECONDS=0 processes the entire remainder. Set 4 for a short experiment.\n'
           + setup + '\n'
           'if SETUP_CHECK_ONLY:\n'
           '    setup_result = workflow.preflight(RUNTIME_DIR)\n'
           '    print("To browse for a video, turn SETUP_CHECK_ONLY off and rerun this cell.")\n'
           'else:\n'
           '    job = workflow.upload_video(RUNTIME_DIR, START_SECONDS, MAX_SECONDS)\n')
    gpu_action = setup + '\nworkflow_started = True\nreceipt = workflow.run_video(RUNTIME_DIR, job_folder_id=JOB_FOLDER_ID, auto_disconnect=AUTO_DISCONNECT)\n'
    gpu = ('#@title 2. Switch to T4, then process the saved video\n'
           '# Blank JOB_FOLDER_ID selects the newest completed input upload.\n'
           '# An already completed job is displayed without running inference again.\n'
           'JOB_FOLDER_ID = "" #@param {type:"string"}\n'
           'AUTO_DISCONNECT = True #@param {type:"boolean"}\n'
           'workflow_started = False\ntry:\n' + textwrap.indent(gpu_action, '    ') +
           'except BaseException:\n'
           '    if not workflow_started and AUTO_DISCONNECT:\n'
           '        print("Public code setup failed before any inference. Releasing this runtime.")\n'
           '        try:\n'
           '            from google.colab import runtime\n'
           '            runtime.unassign()\n'
           '        except Exception:\n'
           '            print("Use Runtime > Disconnect and delete runtime if still connected.")\n'
           '    raise\n')
    outro = '''### 3. Download the labeled video

Wait for **Drive API backup verified** in cell 2. Open **Open or download labeled video**, then download the MP4 directly from Drive. The GPU can remain disconnected during that download. Predictions, timing reports and model provenance are stored alongside it.

Reader FPS measures model-pipeline processing; end-to-end FPS includes video decoding, drawing labels and encoding. Video playback FPS is a separate quantity. Drive upload time is additional.

If saving fails, runtime results are retained. Rerun cell 2 to retry an existing pending upload without repeating inference. Do not change or delete that runtime until the results are saved. If a failure happens before a recovery pointer can be written, the error prints the exact runtime folder to preserve manually.

Private files live under **My Drive / SpeedometerTraining_API_v1**. Existing folders and checkpoints from earlier accounts are preserved.
'''
    cells = []
    for index, (kind, source) in enumerate((('markdown', intro), ('code', cpu), ('code', gpu), ('markdown', outro))):
        cell = {'cell_type': kind, 'metadata': {'id': 'drive_api_step_' + str(index)}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            ast.parse(source)
            cell.update(execution_count=None, outputs=[])
            cell['metadata']['cellView'] = 'form'
        cells.append(cell)
    document = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
                'metadata': {'colab': {'name': output.name, 'provenance': []},
                             'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'},
                             'runtime_release': {'code_commit': commit, 'files': inventory}}}
    output.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    return {'path': str(output), 'code_cells': 2, 'default_accelerator': None,
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'code_commit': commit}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'Triumph_Drive_API_Video_Test.ipynb')
    args = parser.parse_args()
    print(json.dumps(build(args.code_commit, args.output), indent=2))
