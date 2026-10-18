# Learned LCD detection with frozen V5 speed

This experiment learns the visible LCD region from unmasked dashboard images. The current-frame prediction is gated, clipped to visible sensor pixels, and filled with gray128 before the unchanged V5 epoch11 model estimates the needle and ten speed-scale landmarks. The existing geometry decoder converts those predicted points into indicated speed.

Input framing and orientation remain unchanged: grayscale, square letterbox, 512×512, repeated RGB with ImageNet normalization. Inference uses no SIFT, reference pose, image alignment, source labels, previous-frame mask, carried-forward speed, or assumed zero. The original frame stays visible in the video output.

The LCD dataset covers 360° rotations. The frozen V5 speed model's training scope is approximately ±25° camera tilts. This experiment therefore does not establish end-to-end speed reading at arbitrary angles; detecting the LCD across rotations does not demonstrate that the speed landmarks generalize equally.

[Open the notebook in Colab](https://colab.research.google.com/github/sainadh812/triumph-speedometer-dataset/blob/main/learned_lcd_v1/Triumph_Learned_LCD_V1_Colab.ipynb)

## Colab workflow

1. Open `Triumph_Learned_LCD_V1_Colab.ipynb` with a T4 runtime and mount Drive.
2. Verify the pinned code and dataset manifest. Publication placeholders intentionally prevent an unpinned run.
3. Train the LCD model using `imagenet`, or choose `fresh` for random initialization. `resume_latest`, `resume_best`, and `resume_explicit` restore complete detector training state.
4. Keep every completed epoch checkpoint on Drive. The selected export includes the validation-selected mask threshold and fixed gates.
5. Process both private source videos in Colab, saving full-frame H264 overlays, source audio, CSV/JSONL predictions, selected inputs, and summary metrics. Download the verified result ZIP.

`build_notebook.py --commit <git-commit> --manifest-sha <download-manifest-sha256>` regenerates the notebook and `colab_cells.json`. Model training and video inference are not run locally.

## Detector gate and limitations

The gate checks finite probabilities, visible area, component dominance, mean foreground probability and boundary uncertainty. It can accept an LCD clipped by the image edge. It never expands the predicted region with a convex hull or dilation; tiny disconnected fragments are removed only when the dominant component passes its configured share threshold. Boundary uncertainty excludes the artificial sensor-cut edge.

Threshold candidates are evaluated on grouped development validation before video testing. Integration readiness requires **accepted-mask macro foreground precision ≥98%, accepted-mask macro foreground recall ≥90%, and accepted-positive frame coverage ≥80%**. Recall prevents a tiny high-precision fragment from qualifying while leaving most of the LCD visible. If no threshold qualifies, the export remains diagnostic-only: videos still show LCD candidates and rejection reasons while speed stays disabled.

Probability and boundary scores are **not calibrated confidence**. A single segmentation head cannot guarantee that a wrong mask preserves the needle. Subsequent predicted-needle/tick overlap and geometry checks provide additional rejection tests, but they can miss a feature already hidden by a wrong mask. Masks are not adjusted afterward to force speed agreement.

Foreground IoU, precision/recall and boundary errors require reviewed labels. Reports distinguish annotated evaluation from unannotated video coverage. The videos used here are development recordings; acceptance is not accuracy, and ignition needle sweeps are not road speed.

## Frozen speed identity

- V5 selected epoch: **11**
- Weight SHA256: `de08c9772c4eacc2b78f9037ff6c45c980d63ee6223739199b4ad28fc4777e1a`
- Canonical calibration SHA256: `2581a049a418d0f337a9997dcd6d196aef7d49d4d9194acc2bc176f14441d2f5`
- New adapter ID: `learned_lcd_mask128_to_frozen_v5_speed_v1`

The adapter verifies completion markers, original speed implementation hashes, calibration, landmark order and OpenCV version. It records both model hashes and preserves the original speed-training preprocessing identity as provenance. The learned mask is a new input adapter; it is not represented as the original SIFT preprocessing.

Video summaries compare accepted coverage with the completed V5 epoch11/SIFT test using the same source-video hashes: daytime602/602 and straight340/540. This comparison occurs after prediction and does not supply information to either model.

## Code interfaces

- `inference.gate_lcd_probability(probability512, valid_sensor_mask512, gate_config)` is the same pure NumPy/OpenCV gate used during validation and inference. A binary nonrectangular valid-pixel map is supported for synthetic perspective augmentation.
- `inference.LearnedLCDSpeedReader(lcd_checkpoint, speed_checkpoint, speed_calibration, device='cuda')` predicts from the current raw frame only.
- `python -m learned_lcd_v1.evaluate_videos --lcd-checkpoint ... --speed-checkpoint ... --speed-calibration ... --video ... --video ... --output-root ...` runs only in Colab.

Dataset review, training and publication pins are finalized separately from this adapter. Do not change gate implementation or threshold configuration between validation and video testing without a new recorded evaluation.

## Reviewed dataset

The dataset contains 4,426 grayscale512 images: 3,600 training, 614 validation, and 212 held-out test samples. These come from58 manually reviewed source frames (32/14/12), selected from68 audited frames. Ten uncertain or temporal-gap frames are excluded from training and evaluation. Each source retains an LCD polygon/mask, valid-sensor mask, split identity, and augmentation transform; protected analog-region labels are available for the daytime and straight-view speed recordings.

Augmentation uses OpenCV, without ImageGen, LCD content repainting, cropping, or mirroring. Shared transforms keep each image and mask aligned through 360-degree rotation, perspective, translation, and scale. Lighting, blur, noise, and compression vary appearance; reflections and glare from the real source frames are retained. Every source remains in a single split; validation uses separated temporal blocks from the development recordings, and test uses one separate close-up recording. Synthetic variants are not independent real-world recordings.

The existing speed and RPM datasets and checkpoints remain separate. This experiment learns LCD location; it does not retrain the RPM reader or recognize gear digits.
