# Full-circle rotation training from the best epoch-47 model

`Speedometer_Training_Colab.ipynb` starts from the existing best epoch-47 model, trains on the rotation dataset, and compares source and selected models on identical images. The previous 500-image workflow is preserved in `Speedometer_Baseline_500_Colab.ipynb`.

## Run in Colab

1. Open the [latest notebook in Colab](https://colab.research.google.com/github/sainadh812/triumph-speedometer-dataset/blob/main/Speedometer_Training_Colab.ipynb).
2. Select **Runtime > Change runtime type > T4 GPU**, then **Run all**.
3. Authorize the Google Drive mount when prompted. The notebook verifies the mount and tests writing and reading its checkpoint folder.
4. Keep `TRAIN_MODE = "resume_latest"` and `EXPERIMENT_NAME = "speedometer_rotation_v2_from47"` for the default experiment.
5. Inspect the before/after reports. A compact results ZIP is saved to Drive and offered for download.

The dataset downloads from the [v2.0.0 release](https://github.com/sainadh812/triumph-speedometer-dataset/releases/tag/v2.0.0), with no dataset upload or GitHub token required. Archive and label checksums are embedded in the notebook. A notebook previously copied into Drive does not update automatically; open the current linked notebook when adopting this workflow.

The source model is read from:

```text
/content/drive/MyDrive/SpeedometerTraining/experiments/speedometer_400_continuation_v1/runs/20261017_090632_970209_082ca611/best_model.pt
```

Its required SHA256 is `a284ecb0ceb4892074feea3802d9e79b9e4bc9892c25113b8934b5428a16e97c`. Earlier experiments and their checkpoints are preserved.

## Dataset and evaluation design

| Set | Images | Purpose |
|---|---:|---|
| Main training | 4,800 | Optimizer updates |
| Main validation | 1,200 | Model selection and early stopping |
| Unseen-angle robustness | 1,200 | Before/after evaluation only |

The main set rotates each of the original 500 images through 12 angles: **0°,30°,60°,90°,120°,150°,180°,-150°,-120°,-90°,-60°,-30°**, with positive angles clockwise. Original 0° images remain included. Needle pivot/tip labels transform with each image, while the speed label remains unchanged. All variants of a clean pose retain the original train/validation split.

The separate robustness set rotates the 100 original validation parents through **15°,45°,75°,105°,135°,165°,-165°,-135°,-105°,-75°,-45°,-15°**. Those angles are absent from the training rotation grid. This set never changes optimizer updates, early stopping or checkpoint selection.

The many rotations still represent the same original pose groups; main validation has **50 groups**, not 1,200 independent real trials. The robustness set also shares these parents and the reconstructed master. It checks interpolation to unseen rotations rather than independent scene generalization. A finite angle grid cannot establish perfect accuracy at every possible camera angle.

## New experiment, fixed budget

Changing the dataset requires an explicit **transfer warm start**, not exact resumption of the old optimizer trajectory. The notebook imports epoch-47 weights, resets optimizer/RNG state once, and records the source checkpoint, source dataset and epoch in provenance.

Before the first update, the source model is evaluated on the **new main validation set**. Its new baseline, rather than its historical metric on the original dataset, becomes the comparison threshold for the best checkpoint. The initial state is saved as **new experiment epoch 0**. The default fixed target is **epoch 20**, subject to early stopping. The source epoch remains 47 in provenance; it is not confused with new-experiment epoch numbering.

The source model is also evaluated on unseen-angle robustness before training. After training, the selected best checkpoint is evaluated on the same robustness images. Cached evaluations are bound to the exact model hash, label hash, selected sample IDs and dependency versions.

## Three notebook modes

| Mode | Behavior |
|---|---|
| `resume_latest` | Restore the most recently published compatible full state. On this new experiment's first run only, import the verified epoch-47 seed. |
| `resume_best` | Restore the complete state with the lowest main-validation MAE, intentionally rewinding optimizer and counters together. |
| `fresh` | Initialize a new model from ImageNet, or random weights when `FRESH_PRETRAINED=False`. Requires a new experiment name. |

For a fresh experiment, choose a name such as `speedometer_rotation_fresh_v1`. Later resume that same name. Fresh runs use 5 head-training and up to 25 fine-tuning epochs; transfer runs use up to 20 fine-tuning epochs. A saved `experiment_config.json` governs subsequent resumes, so changing fresh-only controls does not alter an existing model.

Fresh experiments and their later resumes require no private epoch-47 seed. They report main validation and after-training robustness results. Source-model comparisons are explicitly omitted rather than presenting an unsupported improvement claim.

Ordinary restart does **not** add another 20 epochs. It retains the saved target and early-stopping decision. A completed run is evaluated without additional training. `EXTEND_BY_EPOCHS` defaults to **0**; a positive value explicitly adds a budget. Reset it to 0 for ordinary restarts.

Exact continuation checks the dataset, selected sample IDs, configuration, device type and recorded versions of PyTorch, torchvision, NumPy and Pillow. Dependency mismatch produces an explicit error. Restore the compatible environment rather than treating a reset model as a continuation. Source hashes are recorded; use the same implementation when reproducing a trajectory.

## Persistence and recovery

Each invocation receives a unique folder beneath:

```text
MyDrive/SpeedometerTraining/experiments/<experiment_name>/runs/<unique_run_id>/
```

Every completed epoch saves model weights, optimizer, AMP scaler, Python/NumPy/PyTorch/CUDA RNG states, data-loader generators, history, stage progress, early-stopping counters, best state, transfer baseline and fixed target. Files are staged, copied, rehashed and published with `.complete.json` sidecars.

An interrupted epoch resumes from the last verified completed epoch. Incomplete or corrupt saves are preserved and skipped when an older compatible checkpoint exists. If none is usable, the notebook stops; it never silently imports seed weights or starts fresh. Drive's mounted filesystem cannot guarantee remote upload completion at the instant of disconnection, so earlier recovery files remain intact. Keep enough Drive space for per-epoch states.

## Model and reports

The architecture remains MobileNetV3-Small with continuous speed regression and auxiliary pivot/tip heatmaps. Inputs are **512 × 512 grayscale**, repeated into three channels and ImageNet-normalized. The batch size is 8. Final encoder blocks and output heads are fine-tuned with AdamW; encoder batch-normalization running statistics remain frozen. No additional training-time geometry changes are applied to these labeled images.

`rotation_comparison.csv` and `rotation_before_after.png` compare source and selected models. `comparison_main_validation.csv` separates original 0°, all rotated images and each angle. `comparison_unseen_angles.csv` covers the held-out offset grid. Negative MAE change means improvement; positive change in the percentage within ±2 km/h means improvement. Inspect original 0° performance for regressions.

The run also exports metrics by appearance/speed/calibration region, per-frame predictions, training history, worst-prediction overlays, source provenance and inference examples. The green overlay is the supplied needle label; magenta is the predicted needle.

The compact ZIP includes the best inference checkpoint, evaluation CSVs, reports, plots, source code and Drive paths. Full per-epoch states remain on Drive and are excluded from the ZIP. **Retain the Drive experiment folder for exact training resume.**

## Accuracy interpretation and checks

The proposed real-world target is **98% of readable frames within ±2 km/h of a verified dial reading**, with unreadable frames reported separately. MAE is average absolute error and is different from the fraction meeting that tolerance. Neither synthetic validation nor synthetic rotation robustness establishes real-camera accuracy. Independent labeled dashboard footage remains necessary; camera crops must match training framing. No confidence score or unreadability detector is claimed.

`build_notebook.py` embeds the current training and workflow modules. The local notebook smoke check uses a random test fixture, 128-pixel inputs, complete subsets of 24 training/24 validation images, 24 robustness images and two epochs. It exercises transfer, complete-state saves, subsequent resume and before/after artifacts. Its metrics are software checks, not results from the trained epoch-47 model. Remote Drive authorization/upload behavior and GPU performance require actual Colab execution.
