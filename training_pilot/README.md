# Speedometer training pilot

This pilot trains a compact neural network to read the synthetic Triumph dashboard dataset. It predicts a continuous speed in km/h and the needle pivot and tip positions. The first full training run belongs in Google Colab; no full-training result or real-camera accuracy is claimed here.

## Run in Google Colab

1. Open the [training notebook directly in Google Colab](https://colab.research.google.com/github/sainadh812/triumph-speedometer-dataset/blob/main/Speedometer_Training_Colab.ipynb).
2. Choose **Runtime > Change runtime type > T4 GPU**, then save the setting. An available CUDA GPU can also be used.
3. Choose **Runtime > Run all**. The notebook automatically downloads `speedometer_bw500.zip` from the public GitHub release and verifies its SHA-256 checksum before extracting it. It reuses a cached ZIP only when its checksum matches.
4. Keep the notebook running through its three experiments and result export. Each experiment prints training progress and synthetic validation errors.
5. The notebook creates a results ZIP and requests its download at completion. Keep that ZIP before disconnecting the Colab runtime. Automatic download is skipped in local mode.

An existing notebook saved in Google Drive is a separate copy and does not automatically receive these changes. Open the GitHub notebook link above to use the updated download code; save a new copy to Drive if desired.

The dataset comes from the [v1.0.0 release asset](https://github.com/sainadh812/triumph-speedometer-dataset/releases/download/v1.0.0/speedometer_bw500.zip); no GitHub sign-in or manual dataset upload is required. The notebook contains its training code, so no repository checkout or separate upload of `train_core.py` is needed. Its first pretrained-model initialization also needs internet access to obtain MobileNet weights. If that download fails, training stops with a clear error instead of silently starting with random weights.

## What the pilot measures

The dataset has **400 training images and 100 validation images**. Variants of a shared clean needle pose stay in one split. The 100 validation images represent **50 pose groups**, and all images were derived from one synthetic source dashboard.

The notebook trains fresh models on **100, 250, and 400 training images**, always evaluating against the same 100 validation images. These training subsets are nested, preserve complete pose groups, and use the same random seed. The smaller subset is contained in the larger one. This comparison shows whether adding images from the present generator is helping; it does not establish generalization to a real camera.

Do not move validation samples into training or choose different validation images between experiments. These validation results help select the model and improve the dataset. They are not a final independent test.

The proposed accuracy target is **at least 98% of readable real test frames within ±2 km/h of a verified dashboard reading**, with unreadable frames reported separately. The notebook reports the percentage within ±1, ±2, and ±5 km/h. It also reports MAE, RMSE, 95th-percentile absolute error, maximum error, bias, and pivot/tip position error. **A low MAE is not the same as 98% within ±2 km/h.**

No real dashboard video has been provided yet. A separate real recording session is still needed for development validation, followed by an independent session for final testing. Synthetic validation alone cannot certify the 98% target or measure unreadable-frame handling.

## Model and training settings

| Setting | Default |
|---|---|
| Input | 512 × 512 grayscale dashboard image |
| Encoder | ImageNet-pretrained MobileNetV3-Small |
| Speed output | Continuous, sigmoid-bounded to 0–180 km/h |
| Needle outputs | Pivot and tip heatmaps at 128 × 128, decoded to subpixel coordinates |
| Batch size | 8 |
| First stage | Up to 5 epochs training the new heads with the encoder frozen |
| Second stage | Up to 25 epochs fine-tuning the final four encoder blocks and the heads |
| Optimizer | AdamW; lower learning rate for the unfrozen encoder |
| Stopping rule | Stop a stage after 7 epochs without material validation MAE improvement |
| Saved model | Checkpoint with the lowest validation MAE across both stages |

Grayscale pixels are repeated into three identical channels, then normalized with the ImageNet channel means and standard deviations. This retains compatibility with the pretrained encoder; it does not add color information. The source 512 × 512 model inputs are not center-cropped.

The spatial decoder learns where the needle pivot and tip appear. The speed head uses encoder features and the predicted keypoint coordinates. Training combines speed Huber loss, Gaussian heatmap supervision, and coordinate loss. The auxiliary needle outputs support visual inspection; they are not a calibrated confidence score or an unreadability detector.

Batch-normalization running statistics in the encoder remain frozen, including during fine-tuning, because the pilot uses small batches. The decoder uses group normalization. Mixed precision is enabled on CUDA GPUs. Seeds are recorded, although hardware and library differences can still cause small run-to-run differences.

The generator already applied lighting, blur, noise, and camera variations. The training module adds no further geometric augmentation, so the supplied keypoint labels stay aligned with their images. Hash, image-mode, coordinate, sample-count, and pose-group checks run before training.

## Results to inspect

Each experiment exports:

- `best_model.pt`: the selected weights plus configuration, preprocessing notes, and source-label checksum.
- `metrics.json`: overall metrics and breakdowns by appearance profile, speed range, and the region influenced by the inferred 60 km/h anchor.
- `validation_predictions.csv`: target/predicted speeds and needle coordinates for every validation sample.
- `worst_predictions.png`: the largest speed errors, with green label overlays and magenta prediction overlays.
- `learning_curves.png` and `history.csv`: training loss and validation progress.
- `config.json` and `data_integrity.json`: settings, checksums, and exact sample membership.

The notebook also exports a comparison CSV and packages its results into a ZIP. Compare the **±2 km/h percentage and error distribution** across dataset sizes, then inspect the worst examples. Persistent errors around a particular speed or appearance may need better labels or new source examples, rather than more copies of similar images.

## Local checks and inference

`train_core.py` exposes `TrainConfig`, `train_experiment`, `load_model`, and `predict_image`. For example:

```python
from train_core import TrainConfig, train_experiment

result = train_experiment(TrainConfig(
    data_dir="../bw_dataset_500",
    output_dir="./runs/pilot_400",
))
```

Use a new output directory for every experiment; an existing best checkpoint is protected from accidental overwrite. The portable data loader defaults to zero worker processes, which also works on Windows.

A short local smoke configuration uses **128 × 128 inputs, 8 training images, 4 validation images, two epochs, and random initial weights**. Its purpose is to check that loading, gradients, checkpoint saving, evaluation, and plotting work. Smoke-test metrics are not evidence of model quality, and smoke outputs must remain separate from Colab experiment results.

For inference, load a completed full-run checkpoint and pass a PIL image or image path:

```python
from train_core import load_model, predict_image

model, checkpoint = load_model("./runs/pilot_400/best_model.pt")
prediction = predict_image(
    model,
    "dashboard_crop.png",
    input_size=checkpoint["config"]["input_size"],
)
```

Real frames must be cropped and framed like the training dashboard. The current model does not detect the dashboard in an arbitrary full camera frame. Returned pixel keypoints refer to the resized model input; normalized keypoints use the same zero-based endpoint convention as the training labels. Evaluate raw frame predictions before adding temporal smoothing to a live feed.
