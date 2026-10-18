"""Build CPU-upload / T4-inference cells and runtime-free Drive download instructions.

Only writes JSON. It never mounts Drive, uploads a video or invokes a model.
Publish runtime code first, then supply that immutable commit to this builder.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parent
RUNTIME_FILES = ('colab_video_helpers.py', 'evaluate_uploaded_video.py',
                 'evaluate_lcd_speed_rpm_batched.py', 'benchmark_lcd_positioning.py')
PUBLIC_DIRECTORY = 'learned_lcd_v1/upload_video_test_v1'

DRIVE_DOWNLOAD_INSTRUCTIONS = '''### 3. Download the finished video directly from Google Drive

**No cell to run, no CPU switch, and no runtime connection needed.**

Wait until cell 2 prints **Drive backup verified**. Cell 2 already saves the labeled MP4, predictions and timings to Drive and verifies their hashes before automatic GPU disconnect. Changing or deleting the Colab runtime does not delete these saved Drive files.

1. Use **Open this private result folder in Drive** in cell 2's output. If that output is unavailable, [find the browser_upload_v1 folder in Drive](https://drive.google.com/drive/search?q=browser_upload_v1).
2. Use the same Google account as the training runs. Open `jobs` → your video's `video_...` folder → `runs` → the completed `run_...` folder → `evaluation`.
3. Right-click **annotated_fast_lcd_speed_rpm.mp4** and choose **Download**.

Full location: `My Drive/SpeedometerTraining/video_tests/browser_upload_v1/jobs/<job_id>/runs/<run_id>/evaluation/annotated_fast_lcd_speed_rpm.mp4`

Download the completed run matching your uploaded video. A run ending in `_incomplete` is a failed partial run. If cell 2 is still processing, wait; a download step cannot finish the video. If you turned **AUTO_DISCONNECT** off, disconnect and delete the GPU runtime after **Drive backup verified** appears. The Drive download continues independently of Colab.
'''


def bootstrap(commit, repository, files, cpu_only):
    return textwrap.dedent('''\
        # Public code is pinned; videos and weights remain in your private Drive.
        from pathlib import Path
        import hashlib, importlib.util, os, urllib.request, uuid
        CODE_COMMIT = __COMMIT__
        REPOSITORY = __REPOSITORY__
        EXPECTED_FILES = __FILES__
        RUNTIME_DIR = Path('/content/speedometer_upload_runtime') / CODE_COMMIT
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        for name, metadata in EXPECTED_FILES.items():
            destination = RUNTIME_DIR / name
            if destination.is_file():
                data = destination.read_bytes()
            else:
                url = 'https://raw.githubusercontent.com/' + REPOSITORY + '/' + CODE_COMMIT + '/__PUBLIC_DIRECTORY__/' + name
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        data = response.read()
                except Exception as error:
                    raise RuntimeError('Could not fetch pinned public runtime code: ' + name) from error
            if len(data) != metadata['bytes'] or hashlib.sha256(data).hexdigest() != metadata['sha256']:
                raise ValueError('Runtime file differs from the reviewed release: ' + name)
            if not destination.exists():
                temporary = destination.with_name(name + '.staged-' + uuid.uuid4().hex)
                temporary.write_bytes(data)
                os.replace(temporary, destination)
        specification = importlib.util.spec_from_file_location('speedometer_upload_helpers', RUNTIME_DIR / 'colab_video_helpers.py')
        helpers = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(helpers)  # Import-only; no model work.
        __RUNTIME_CHECK__
        from google.colab import drive
        if not Path('/content/drive/MyDrive').is_dir():
            drive.mount('/content/drive', timeout_ms=300000)
        ''').replace('__COMMIT__', repr(commit)).replace('__REPOSITORY__', repr(repository)).replace(
            '__FILES__', repr(files)).replace('__PUBLIC_DIRECTORY__', PUBLIC_DIRECTORY).replace(
                '__RUNTIME_CHECK__', 'helpers.require_cpu()' if cpu_only else 'helpers.require_gpu()')


def build(commit, repository, output):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('--code-commit must be the full immutable runtime-code commit')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
        raise ValueError('--repository must be owner/repository')
    files = {}
    for name in RUNTIME_FILES:
        path = ROOT / name
        data = path.read_bytes()
        ast.parse(data, filename=name)
        files[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    cpu_setup = bootstrap(commit, repository, files, cpu_only=True)
    gpu_setup = bootstrap(commit, repository, files, cpu_only=False)
    gpu_action = (gpu_setup + '\n' +
                  'inference_started = True\n' +
                  'receipt = helpers.run_uploaded_video(RUNTIME_DIR, auto_disconnect=AUTO_DISCONNECT)\n')
    guarded_gpu_action = ('inference_started = False\ntry:\n' + textwrap.indent(gpu_action, '    ') +
        'except BaseException as setup_error:\n'
        '    if not inference_started and AUTO_DISCONNECT:\n'
        '        print("Setup failed before inference; no runtime result was produced:", repr(setup_error), flush=True)\n'
        '        try:\n'
        '            import shutil, subprocess\n'
        '            executable = shutil.which("nvidia-smi")\n'
        '            gpu = subprocess.run([executable, "-L"], capture_output=True, text=True, timeout=15) if executable else None\n'
        '            if gpu is not None and gpu.returncode == 0 and "GPU" in gpu.stdout:\n'
        '                from google.colab import runtime\n'
        '                runtime.unassign()\n'
        '        except Exception as release_error:\n'
        '            print("Use Runtime > Disconnect and delete runtime if still connected:", repr(release_error))\n'
        '    raise\n')
    sources = [
        '#@title 1. Upload a video on CPU and save it to Drive\n'
        '# Run the cells individually. Use the same Drive as your previous model runs.\n'
        '# Keep Hardware accelerator = None until the upload is fully saved.\n'
        'START_SECONDS = 0.0 #@param {type:"number"}\n'
        'MAX_SECONDS = 0.0 #@param {type:"number"}\n'
        'FORCE_NEW_UPLOAD = False #@param {type:"boolean"}\n'
        '# MAX_SECONDS=0 processes the entire remaining video. No old128-frame limit.\n'
        + cpu_setup + '\n'
        'job = helpers.upload_video(START_SECONDS, MAX_SECONDS, FORCE_NEW_UPLOAD)\n',
        '#@title 2. Switch to T4 GPU, then run the saved video\n'
        '# Runtime > Change runtime type > T4 GPU. Reconnect and run only this cell.\n'
        '# Existing LCD9 + speed11 + RPM9 weights. Fast LCD positioning; no training.\n'
        '# Large RPM label plus raw speed. Old-model readings remain unvalidated.\n'
        'AUTO_DISCONNECT = True #@param {type:"boolean"}\n'
        + guarded_gpu_action,
    ]
    introduction = (
        '# Upload a video and test the existing models\n\n'
        '**Run the two code cells individually:** CPU upload → T4 inference → download directly from Drive. '
        'Use the same Google Drive as the previous training runs; the pretrained assets must already exist there.\n\n'
        'The default processes the entire uploaded video. Cell 2 saves and verifies the output in private Drive, '
        'then releases the GPU. Step 3 opens Drive; no runtime switch or reconnection is needed. If Drive preservation fails, the GPU stays connected '
        'so you can save the remaining runtime files.\n\n'
        'This uses the existing LCD9, speed11 and RPM9 models with fast LCD-based RPM positioning. '
        '**No retraining; raw readings remain unvalidated.** The upright LCD corner assumption and unmasked speed '
        'input retain the known diagnostic limitations. Videos and weights are never published.\n')
    cells = [{'cell_type': 'markdown', 'metadata': {'id': 'workflow_instructions'},
              'source': introduction.splitlines(keepends=True)}]
    for index, source in enumerate(sources, 1):
        ast.parse(source, filename=f'cell_{index}')
        cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {'id': f'upload_video_step_{index}', 'cellView': 'form'},
                      'outputs': [], 'source': source.splitlines(keepends=True)})
    cells.append({'cell_type': 'markdown', 'metadata': {'id': 'upload_video_step_3'},
                  'source': DRIVE_DOWNLOAD_INSTRUCTIONS.splitlines(keepends=True)})
    document = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'colab': {'name': output.name, 'provenance': []},
                             'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'},
                             'runtime_release': {'repository': repository, 'code_commit': commit, 'files': files},
                             'workflow': 'Run two code cells individually: CPU upload, T4 inference with verified Drive save and automatic unassign. Download directly from Drive without a runtime.'},
                'cells': cells}
    # Deliberately omit accelerator/GPU metadata: uploads start on CPU.
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    return {'path': str(output.resolve()), 'code_cells': 2, 'markdown_cells': 2, 'default_accelerator': None,
            'code_commit': commit, 'notebook_bytes': output.stat().st_size,
            'notebook_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'runtime_files': files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--repository', default='sainadh812/triumph-speedometer-dataset')
    parser.add_argument('--output', type=Path, default=ROOT / 'Triumph_Upload_Video_Test.ipynb')
    args = parser.parse_args()
    result = build(args.code_commit, args.repository, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
