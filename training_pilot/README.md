# Speedometer training with Google Drive recovery

The self-contained `Speedometer_Training_Colab.ipynb` continues the existing 400-image pilot and saves complete training state to Google Drive after each completed epoch. The same 100 synthetic validation images, representing 50 pose groups, are retained.

## Start or resume in Colab

1. Open the [latest notebook in Colab](https://colab.research.google.com/github/sainadh812/triumph-speedometer-dataset/blob/main/Speedometer_Training_Colab.ipynb).
2. Select **Runtime > Change runtime type > T4 GPU**, then **Run all**.
3. Authorize the Google Drive mount when prompted. The notebook verifies that its Drive folder can be written and read.
4. Leave `TRAIN_MODE = "resume_latest"` and `EXPERIMENT_NAME = "speedometer_400_continuation_v1"` for the default continuation.
5. Inspect the final metrics and prediction overlays. A compact results ZIP is saved to Drive and offered for download.

The fixed 500-image dataset downloads automatically from the [public release](https://github.com/sainadh812/triumph-speedometer-dataset/releases/tag/v1.0.0). Its archive, source labels and model inputs are checked against recorded checksums. No dataset upload or GitHub token is needed. A previously copied notebook in Drive does not update automatically; use the latest linked notebook when adopting this workflow.

The approved legacy seed must be at:

```text
/content/drive/MyDrive/SpeedometerTraining/pilot_best_epoch29.pt
```

Its required SHA256 is `a2ed22a3feb4d4d5f8552bc31af6c5051b66503840ab8a22d3f361a5e6a1dd36`. The notebook verifies those bytes before importing them.

## Three training modes

| Mode | Behavior |
|---|---|
| `resume_latest` | Restore the most recently published verified, compatible full-state checkpoint. This is the normal restart setting. |
| `resume_best` | Restore the complete checkpoint with the lowest validation MAE. This rewinds model, optimizer and training counters together, while retaining the saved target epoch. |
| `fresh` | Create a new experiment from ImageNet weights, or random weights when `FRESH_PRETRAINED = False`. A new experiment name is required. |

For a fresh experiment, choose a new name such as `speedometer_fresh_v2`. Later switch to `resume_latest` while keeping that name. Existing experiment folders and checkpoints are preserved. Fresh training never overwrites them.

The first setup writes `experiment_config.json`. Resume uses that saved configuration, including batch size, losses, epoch-stage budgets, sample selection and seed. Changing `FRESH_PRETRAINED` does not change an already initialized model. Input and output paths are rebound to the current session and its new run folder.

Exact resume rejects a changed dataset, incompatible training configuration, changed device type, or different recorded versions of PyTorch, torchvision, NumPy or Pillow. A new Colab runtime may update dependencies; if versions differ, the notebook shows the saved and current versions and stops. Restore the compatible environment before continuing. It does not automatically replace CUDA packages or pretend that a new initialization is a continuation. Source hashes are recorded for provenance; use the same implementation for a reproducible trajectory.

## The initial legacy import and fixed budget

The existing best checkpoint was saved at **epoch 29**. It contains model weights and metrics, but lacks optimizer, mixed-precision scaler and random-generator state. Therefore the first import is an explicit **warm start with a new optimizer**, rather than an exact continuation of the original optimizer trajectory.

The default continuation uses **20 additional fine-tuning epochs**, giving a fixed absolute target of **epoch 49**, with early stopping enabled. A full-state recovery checkpoint is saved at the imported boundary before new epochs run. Subsequent sessions restore full state from these new checkpoints.

Rerunning `resume_latest` does **not** grant another 20 epochs. It finishes the remaining saved budget, or performs evaluation without additional training if that budget is already complete. A completed early-stopping decision is also retained. `EXTEND_BY_EPOCHS` defaults to **0**; a positive value deliberately adds a new budget. Reset that setting to 0 before ordinary restarts so the extension is not requested again.

Legacy import occurs only when an experiment has no full-state checkpoint artifacts and its manifest permits that import. If files exist but none is a verified compatible full-state checkpoint, the notebook fails clearly. It never treats corruption as permission to restart from legacy or fresh weights.

## What survives a disconnect

Each invocation receives a unique folder under:

```text
MyDrive/SpeedometerTraining/experiments/<experiment_name>/runs/<unique_run_id>/
```

Every completed epoch saves model weights, optimizer state, mixed-precision scaler state, Python/NumPy/PyTorch/CUDA random-generator states, data-loader generator states, training history, stage progress, early-stopping counters, the best model so far and the fixed target epoch.

Checkpoints are staged, copied to Drive, verified by SHA256 and then published with an `epoch_*.complete.json` sidecar. Discovery verifies the bytes again before restoration. Incomplete or corrupt uploads are preserved and skipped when an older compatible verified checkpoint exists; their paths are printed. If no valid checkpoint remains, training stops for recovery rather than silently resetting.

A mid-epoch interruption loses that unfinished epoch's work; recovery starts from the last completed verified epoch. Drive's mounted filesystem cannot guarantee that a remote upload finishes at the exact instant of disconnection. Retaining previous epoch files provides independent recovery points. Keep sufficient Drive space for the growing set of full-state files.

## Model and settings

| Setting | Default continuation |
|---|---|
| Input | 512 × 512 grayscale dashboard crop |
| Encoder | MobileNetV3-Small, initialized from the trained legacy model |
| Outputs | Continuous speed, needle pivot and needle tip |
| Spatial supervision | Two 128 × 128 heatmaps and coordinate loss |
| Samples | 400 training / 100 synthetic validation |
| Batch size | 8 |
| Training | Fine-tune the final four encoder blocks and output heads |
| Optimizer | AdamW, with a smaller learning rate for encoder parameters |
| Stopping | Saved target or early stopping after 7 epochs without material MAE improvement |

Fresh experiments start with 5 head-training epochs and up to 25 fine-tuning epochs. Grayscale is repeated into three identical channels and ImageNet-normalized to reuse pretrained features; no color information is added. Encoder batch-normalization running statistics stay frozen. The dataset already contains lighting, blur, noise and camera variations, so the training module applies no additional geometry that could misalign labels.

The speed head combines image features and predicted needle coordinates. It does not directly convert an uncorrected perspective angle to speed. The auxiliary heatmaps support inspection; they are not confidence scores or an unreadability detector.

## Inspect and export results

Each run produces `metrics.json`, per-frame `validation_predictions.csv`, `history.csv`, learning curves, worst-prediction overlays, configuration and provenance reports, and the selected `best_model.pt` for inference. Metrics also separate appearance profiles, speed bins and the region affected by the inferred 60 km/h anchor. The green line in visual reports is the label; magenta is the predicted needle.

The compact results ZIP includes the best inference checkpoint, reports, plots, source code and `drive_paths.json`. It excludes duplicate per-epoch full-state checkpoints, which remain in Drive. **The ZIP alone does not provide exact training resume; retain the Drive experiment folder.** Local smoke runs do not trigger browser downloads.

The proposed real-world target remains **at least 98% of readable frames within ±2 km/h of a verified dashboard reading**, with unreadable frames reported separately. MAE is average absolute error and is different from the percentage meeting that tolerance. Reports include both, plus RMSE, 95th-percentile error, maximum error and keypoint distances.

Synthetic validation is development evidence, not a real-world test. Independent real dashboard footage has not yet been supplied. Camera crops must match training framing, and real labeled recording sessions are still needed before judging live-feed accuracy or adding temporal smoothing.

## Developer checks

`build_notebook.py` embeds the current `train_core.py` and `notebook_workflow.py` and validates notebook code cells. `check_notebook.py` executes the generated notebook against a local mock Drive using 128-pixel inputs, 8 training images, 4 validation images, random initialization and two epochs. It then reruns with `resume_latest`, verifies the fixed target does not expand, checks full-state fields and hashes, and inspects compact ZIP contents. This is a software smoke check, not a model-performance experiment or a remote Drive upload test.

The persistent core API is:

```python
result = train_persistent(
    config,
    mode="resume_latest",  # fresh, resume_latest, resume_best, or explicit warm_start
    search_root=experiment_directory,
    extend_by_epochs=0,
)
```

For inference, `load_model(best_model_path)` and `predict_image(model, image, input_size=512)` apply the saved model to a matched dashboard crop. Keypoint pixel coordinates refer to the resized input. The model does not locate the dashboard in an arbitrary full camera frame.
