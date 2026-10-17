"""Build a self-contained, uploadable Colab notebook from the tested core."""
from pathlib import Path
import hashlib
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
core = (ROOT / "train_core.py").read_text(encoding="utf-8")
core_hash = hashlib.sha256(core.encode("utf-8")).hexdigest()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text, hidden=False):
    cell = nbf.v4.new_code_cell(text.strip())
    if hidden:
        cell.metadata.update({"cellView": "form"})
    cells.append(cell)


md("""
# Triumph speedometer — first training experiment

**500 grayscale images: 400 training + 100 synthetic validation.** This notebook trains a pretrained MobileNetV3-Small with a speed output and two needle keypoint heatmaps. It compares nested training sets of **100, 250, and 400 images**, while keeping the same validation set.

1. In Colab, choose **Runtime → Change runtime type → T4 GPU** (or another available GPU).
2. Choose **Runtime → Run all**.
3. The notebook automatically downloads **`speedometer_bw500.zip`** from the public GitHub release and verifies its checksum. No dataset upload or GitHub token is needed.

[Open the latest notebook from GitHub in Colab](https://colab.research.google.com/github/sainadh812/triumph-speedometer-dataset/blob/main/Speedometer_Training_Colab.ipynb). A notebook previously saved to Drive is a separate copy and will not update automatically.

**Output:** best checkpoints, speed/keypoint metrics, prediction tables, learning curves, worst-case overlays, and a results ZIP. The ZIP is offered for download after training. Keep it before disconnecting the runtime.

The proposed target is **98% of real readable frames within ±2 km/h of the verified dial reading**. This notebook reports the same tolerance on **synthetic development validation only**. Its 100 validation images represent 50 paired poses from the same reconstructed source; they do not establish live-camera accuracy. Real footage has not been supplied.

Sources: [Colab runtime FAQ](https://research.google.com/colaboratory/faq.html), [MobileNetV3-Small pretrained weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).
""")
md("""
## 1. Check the runtime

The default run requires a GPU. Colab's existing PyTorch installation is used without replacing its CUDA packages. A separate smoke mode exists for local software checks; smoke results are **not** model performance.
""")
code("""
import os, sys, json, time, hashlib, zipfile, csv, importlib.util
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
WORK_DIR = Path(os.environ.get('SPEEDOMETER_WORK_DIR', '/content/speedometer_work' if IN_COLAB else str(Path.cwd() / 'speedometer_work')))
WORK_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = Path(os.environ.get('SPEEDOMETER_DATASET_ZIP', '/content/speedometer_bw500.zip' if IN_COLAB else str(Path.cwd() / 'speedometer_bw500.zip')))
DATASET_URL = 'https://github.com/sainadh812/triumph-speedometer-dataset/releases/download/v1.0.0/speedometer_bw500.zip'
EXPECTED_ZIP_SHA256 = '92e773ec1ccee15a455a78453e8e5cfe81d04e2d93a57ff83fcea807b6ab3e27'
EXPECTED_LABELS_SHA256 = 'e77ea6f7354597008bec89b9c98acf8eca03ca76310921c53b2c822f1d2ef195'

print('Mode:', RUN_MODE, '| Python:', sys.version.split()[0], '| PyTorch:', torch.__version__, '| torchvision:', torchvision.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not connected')
if not hasattr(torch.amp, 'GradScaler'):
    raise RuntimeError('This notebook needs PyTorch 2.3 or newer; reconnect to a current Colab runtime.')
if RUN_MODE == 'pilot' and not torch.cuda.is_available():
    raise RuntimeError('Select Runtime > Change runtime type > T4 GPU, reconnect, then Run all. No CPU pilot was started.')
if RUN_MODE == 'smoke':
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    print('SMOKE CHECK ONLY: short random-weight run, not an accuracy experiment.')
""")
md("""
## 2. Download and verify the dataset

The public [GitHub release](https://github.com/sainadh812/triumph-speedometer-dataset/releases/tag/v1.0.0) supplies the fixed 500-image archive (about 101 MB). A verified local copy is reused on reruns. Otherwise, the archive is downloaded to a temporary file, checked, and then moved into place. Interrupted downloads can be retried by rerunning this cell.

All 500 model inputs and the original split labels will be checked. Captions and preview images are excluded from training. No login or token is required.
""")
code("""
import tempfile
import requests

def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def download_dataset(url, destination, expected_sha256, attempts=3):
    destination = Path(destination)
    if destination.is_file() and file_sha256(destination) == expected_sha256:
        print('Using the verified cached dataset:', destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        with tempfile.NamedTemporaryFile(prefix=destination.name + '.', suffix='.part',
                                         dir=destination.parent, delete=False) as temporary:
            partial_path = Path(temporary.name)
        try:
            print(f'Downloading dataset from GitHub (attempt {attempt}/{attempts})...')
            downloaded = 0
            next_progress = 25 * 1024 * 1024
            with requests.get(url, stream=True, timeout=(30, 60)) as response:
                response.raise_for_status()
                with partial_path.open('wb') as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
                            downloaded += len(block)
                            if downloaded >= next_progress:
                                print(f'  {downloaded / 1_000_000:.0f} MB downloaded')
                                next_progress += 25 * 1024 * 1024
            if file_sha256(partial_path) != expected_sha256:
                raise ValueError('Downloaded dataset checksum does not match the approved archive.')
            partial_path.replace(destination)
            print(f'Downloaded and verified {downloaded / 1_000_000:.1f} MB.')
            return
        except (requests.RequestException, OSError, ValueError) as error:
            if attempt == attempts:
                raise RuntimeError('Dataset download failed. Check the connection and rerun this cell.') from error
            print('Download did not complete or verify; retrying...')
            time.sleep(2 * attempt)
        finally:
            partial_path.unlink(missing_ok=True)

download_dataset(DATASET_URL, ZIP_PATH, EXPECTED_ZIP_SHA256)

if file_sha256(ZIP_PATH) != EXPECTED_ZIP_SHA256:
    raise ValueError('Dataset checksum mismatch; rerun this cell to download the approved archive.')

with zipfile.ZipFile(ZIP_PATH) as archive:
    if archive.testzip() is not None:
        raise ValueError('The dataset archive failed its integrity check.')
    for entry in archive.infolist():
        target = (WORK_DIR / entry.filename).resolve()
        if not target.is_relative_to(WORK_DIR.resolve()) or not entry.filename.startswith('bw_dataset_500/'):
            raise ValueError('Unexpected path in the dataset archive.')
    archive.extractall(WORK_DIR)

DATA_DIR = WORK_DIR / 'bw_dataset_500'
if file_sha256(DATA_DIR / 'labels.json') != EXPECTED_LABELS_SHA256:
    raise ValueError('Label provenance mismatch.')
print('Verified dataset archive:', ZIP_PATH.name)
display(DisplayImage(filename=str(DATA_DIR / 'overview.png'), width=700))
""")
md("""
## 3. Load the training implementation

The complete source is embedded below and saved into the runtime. No extra code upload is needed. The model preserves the full 512 × 512 crop. Grayscale values are repeated into three channels to reuse pretrained image features; no color information is introduced.

Training uses speed regression plus supervised pivot/tip heatmaps. It does not convert the raw perspective-distorted needle angle into speed. Pretrained features are frozen initially, then the final blocks are fine-tuned. Batch-normalization statistics stay frozen for the small batches.
""")
code("#@title Load the embedded training module\nTRAINING_CORE_SOURCE = " + repr(core) + "\n" +
     "EXPECTED_CORE_SHA256 = " + repr(core_hash) + "\n" + """
assert hashlib.sha256(TRAINING_CORE_SOURCE.encode('utf-8')).hexdigest() == EXPECTED_CORE_SHA256
module_path = WORK_DIR / 'train_core.py'
module_path.write_text(TRAINING_CORE_SOURCE, encoding='utf-8')
spec = importlib.util.spec_from_file_location('speedometer_core', module_path)
training = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = training
spec.loader.exec_module(training)
print('Training module loaded; source SHA256:', EXPECTED_CORE_SHA256)
""", hidden=True)
code("""
records, integrity = training.verify_dataset(DATA_DIR)
assert integrity['split_counts'] == {'train': 400, 'validation': 100}
train_records = [r for r in records if r['dataset_split'] == 'train']
validation_records = [r for r in records if r['dataset_split'] == 'validation']
assert len({r['group_id'] for r in validation_records}) == 50
assert not ({r['speed_kmh'] for r in train_records} & {r['speed_kmh'] for r in validation_records})

SEED = 42
previous_ids = set()
subset_table = []
for size in (100, 250, 400):
    subset = training.select_group_subset(train_records, size, SEED)
    ids = {r['sample_id'] for r in subset}
    assert len(subset) == size and previous_ids.issubset(ids)
    previous_ids = ids
    subset_table.append({'training_images': size, 'training_poses': len({r['group_id'] for r in subset}),
                         'validation_images': 100, 'validation_poses': 50})
display(pd.DataFrame(subset_table))
print('All input hashes, keypoint coordinates, and split groups verified.')
print('Validation speed range:', min(r['speed_kmh'] for r in validation_records), 'to', max(r['speed_kmh'] for r in validation_records), 'km/h')
print('Exact 0 and 180 km/h examples are training-only; the validation set does not test those exact endpoints.')
""")
md("""
## 4. Train the pilot

Default: **100, 250, and 400 training images**, 512-pixel input, batch size 8, up to 5 frozen-feature epochs followed by 25 fine-tuning epochs. Each experiment starts from the same pretrained weights and seed. The same 100 validation images are used throughout, and the best checkpoint is selected by validation mean absolute speed error.

To run just the full-data experiment, set `RUN_LEARNING_CURVE = False`. If the GPU reports an out-of-memory error, lower `BATCH_SIZE` to 4 and rerun this section. Each execution creates a new output directory; previous checkpoints are preserved. Availability and runtime depend on the assigned Colab GPU.
""")
code("""
RUN_LEARNING_CURVE = True
BATCH_SIZE = 8
HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 25
PATIENCE = 7

from datetime import datetime, timezone
run_stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
SESSION_DIR = WORK_DIR / ('smoke_' if RUN_MODE == 'smoke' else 'pilot_') / run_stamp
SESSION_DIR.mkdir(parents=True, exist_ok=False)
training_sizes = [100, 250, 400] if RUN_LEARNING_CURVE else [400]
if RUN_MODE == 'smoke':
    training_sizes = [8]

results = []
for size in training_sizes:
    config = training.TrainConfig(
        data_dir=str(DATA_DIR), output_dir=str(SESSION_DIR / f'train_{size}'),
        input_size=128 if RUN_MODE == 'smoke' else 512,
        batch_size=2 if RUN_MODE == 'smoke' else BATCH_SIZE,
        head_epochs=1 if RUN_MODE == 'smoke' else HEAD_EPOCHS,
        finetune_epochs=1 if RUN_MODE == 'smoke' else FINETUNE_EPOCHS,
        patience=PATIENCE, seed=SEED,
        pretrained=(RUN_MODE != 'smoke'), max_train_samples=size,
        max_val_samples=4 if RUN_MODE == 'smoke' else None,
    )
    print()
    print(f'Starting {size}-image experiment')
    result = training.train_experiment(config)
    result['run_mode'] = RUN_MODE
    results.append(result)
    print('Saved:', result['checkpoint_path'])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

(SESSION_DIR / 'experiment_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print('All requested experiments finished:', SESSION_DIR)
""")
md("""
## 5. Compare results and inspect mistakes

**Within ±2 km/h** is the proportion of validation predictions whose absolute speed error is at most 2 km/h. MAE is the average absolute speed error. Also inspect the 95th-percentile error and the needle tip error.

These are raw single-frame predictions, before temporal smoothing. No learned confidence or unreadable-frame detector is claimed. Paired images share pose groups, so the 100 validation frames are not 100 independent real-world trials. Reports also separate appearance profiles, speed bins, and the inferred 60-tick region.
""")
code("""
comparison_rows = []
for result in results:
    m = result['metrics']
    comparison_rows.append({
        'training_images': result['training_samples'], 'validation_images': result['validation_samples'],
        'validation_poses': m['pose_groups'], 'MAE_kmh': m['mae_kmh'],
        'RMSE_kmh': m['rmse_kmh'], 'within_2_kmh_percent': m['within_2_kmh_percent'],
        'P95_error_kmh': m['p95_absolute_error_kmh'], 'max_error_kmh': m['max_absolute_error_kmh'],
        'tip_error_px': m['tip_mae_px'], 'best_epoch': result['best_epoch'],
        'epochs_run': result['epochs_run'], 'run_mode': RUN_MODE,
    })
comparison = pd.DataFrame(comparison_rows).sort_values('training_images')
comparison.to_csv(SESSION_DIR / 'learning_curve_comparison.csv', index=False)
display(comparison)
if RUN_MODE == 'smoke':
    print('SMOKE RESULTS ONLY — verifies execution, not speed-reading accuracy.')
else:
    print('SYNTHETIC VALIDATION ONLY — real-camera accuracy remains unmeasured.')

selected_result = next((r for r in results if r['training_samples'] == 400), results[-1])
display(DisplayImage(filename=selected_result['learning_curves_path']))
display(DisplayImage(filename=selected_result['worst_predictions_path'], width=1100))
display(pd.DataFrame(selected_result['metrics_by_profile']).T)
""")
md("""
## 6. Check a saved model and export the results

This cell reloads the selected checkpoint and predicts one validation image using the same preprocessing. Green shows the supplied needle label; magenta shows the model's prediction. Later, actual camera crops must match the training framing, and real speed labels are needed for evaluation.
""")
code("""
saved_model, checkpoint = training.load_model(selected_result['checkpoint_path'])
example = validation_records[0]
prediction = training.predict_image(saved_model, DATA_DIR / example['image'], input_size=checkpoint['config']['input_size'])
with Image.open(DATA_DIR / example['image']) as im:
    example_pixels = np.asarray(im)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(example_pixels, cmap='gray', vmin=0, vmax=255)
for source, color, style in [(example['needle_position_normalized'], 'lime', '-'),
                             (prediction['keypoints_normalized'], 'magenta', '--')]:
    points = np.array([source['pivot'], source['tip']]) * 511
    ax.plot(points[:, 0], points[:, 1], style, color=color, marker='o', linewidth=1.5)
ax.set_title(f"Label {example['speed_kmh']:.1f} km/h | prediction {prediction['speed_kmh']:.1f} km/h")
ax.axis('off')
fig.tight_layout()
fig.savefig(SESSION_DIR / 'reloaded_model_example.png', dpi=130)
plt.close(fig)
display(DisplayImage(filename=str(SESSION_DIR / 'reloaded_model_example.png')))
""")
code("""
import shutil
shutil.copyfile(module_path, SESSION_DIR / 'train_core.py')
export_note = {
    'run_mode': RUN_MODE, 'dataset_labels_sha256': EXPECTED_LABELS_SHA256,
    'training_core_sha256': EXPECTED_CORE_SHA256,
    'scope': 'Synthetic development validation, not real-camera testing',
    'real_accuracy_measured': False,
    'next_decision': 'Inspect errors and learning curves; collect independently labeled real footage before deployment.',
}
(SESSION_DIR / 'run_scope.json').write_text(json.dumps(export_note, indent=2), encoding='utf-8')
RESULTS_ZIP = Path(shutil.make_archive(str(SESSION_DIR) + '_results', 'zip', root_dir=SESSION_DIR))
print('Results archive:', RESULTS_ZIP)
print('Checkpoint, inference code, metrics, prediction CSVs, and visual reports are included.')
if IN_COLAB and RUN_MODE != 'smoke':
    from google.colab import files
    files.download(str(RESULTS_ZIP))
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "colab": {"name": "Speedometer_Training_Colab.ipynb", "provenance": []},
    "accelerator": "GPU",
    "speedometer": {"core_sha256": core_hash, "dataset_images": 500, "input_size": 512},
})
nbf.validate(notebook)
target = PROJECT / "Speedometer_Training_Colab.ipynb"
with target.open('w', encoding='utf-8', newline='\n') as handle:
    nbf.write(notebook, handle)
print(target)
print(f"Cells: {len(cells)}; embedded core SHA256: {core_hash}")
