# Mixed camera views, V5

This experiment combines the reviewed straight-view V3 and daytime tilted-view V4 datasets. Each of the 2,500 parents produces a left, approximately straight, and right roll variant, giving 7,500 images (6,000 train / 1,500 validation). The generator uses deterministic OpenCV transforms, not new ImageGen calls. Numbers are never mirrored. Speed stays unchanged while all 12 landmarks and the LCD mask follow the exact image transform.

Each full parent image fits inside the 512×512 output. Perspective already present in the source remains; these are camera-roll variants, not new photographs from the opposite physical viewpoint. Both source views and all three roll groups receive equal representation in training batches. Original source groups and parent splits are retained.

## Input and model

The existing MobileNetV3-small 12-landmark network is retained: pivot, needle tip and ten semantic scale ticks (0–180 km/h). All V4 epoch-17 weights transfer into a new experiment; optimizer state starts fresh because the dataset and preprocessing changed. The first epoch trains the prediction heads with the encoder and its BatchNorm statistics frozen, then gently fine-tunes late encoder blocks.

At inference the full sensor frame is padded and resized, preserving its orientation. Matching against two static references locates the LCD mask only. It does not warp the image into a fixed camera pose. Ambiguous or unsafe masks produce a documented rejection. Speed uses predicted tick geometry and the predicted needle, never source annotations or the image's known rendering transform.

A reliably registered LCD may extend beyond the camera frame. Its mask is intersected with the visible sensor area; off-screen LCD vertices do not by themselves reject a reading. The visible-area minimum, analog needle/tick protection, and reference-agreement checks still apply. Evaluation metadata records the clipping policy and inference-code hash. Before/after comparisons must use the same inference version.

## Colab workflow

Open `Triumph_Mixed_Views_V5_Colab.ipynb`, select T4 GPU, and connect your Drive. The notebook downloads pinned code and hash-verified dataset parts. Every completed epoch saves the full optimizer, model, AMP scaler, sampler and random state to Drive. The modes are `transfer_v4`, `resume_latest`, `resume_best`, and `fresh`; resume never silently starts over. Both an equal-group best checkpoint and a worst-source-view best checkpoint are retained.

The video evaluator is restricted to Colab. It tests both original videos at a 30 FPS sampling/playback timeline, keeping the full original frame and showing predicted landmarks, speed or rejection reason, and actual model input. Original videos and weights remain in private Drive storage.

## What the metrics mean

Landmark error is measured in input-image pixels. The all-input ±2 km/h score counts rejected eligible images as failures. Acceptance coverage is not accuracy. Scores are reported separately for both source views and each roll group, plus a balanced aggregate.

These images and the two requested videos originate from the same two recordings. Their results measure development performance, not independent real-world accuracy. Shared plate repairs and calibration also limit independence between synthetic train and validation groups. The ignition-sweep recording shows indicated needle positions, not actual road speed. Independent recordings with reviewed labels remain necessary to establish deployment accuracy.

The reused V3 and V4 draft calibrations differ by less than 1 km/h on the known V3 labels; see the dataset report for exact checks and annotation limitations.

## Files

- `dataset_builder.py`: deterministic generation, source provenance and label/mask QA.
- `train_landmarks.py`: balanced training, per-view metrics, durable checkpoints.
- `inference.py`: native-orientation input and image-driven LCD-mask projection.
- `evaluate_videos.py`: Colab-only full-frame before/after video testing.
- `data_download_manifest.json`: multipart archive hashes.
- `DATASET_REPORT.json`: generation and static validation results.

The new experiment preserves the original V3/V4 datasets, code and checkpoints.
