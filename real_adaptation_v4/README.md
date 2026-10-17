# V4: joint needle and speed-scale landmarks

This experiment adapts the Triumph speedometer reader to the daytime video recorded at 17:43:59. It adds learned scale landmarks to the working V3 needle model so that speed can be calculated from the predicted needle and tick positions.

## Input and LCD masking

Inputs are 512 × 512 grayscale images. Real frames are first stabilized to the reviewed daytime reference using image-only SIFT matching on the static speed scale. A fixed polygon then masks the LCD to gray value 128. This replaces the misplaced mask derived from the earlier camera view. A failed stabilization is reported as a rejection rather than applying an uncertain mask over the needle.

During synthetic generation, the image, LCD mask and all labels receive the same geometric transformation. The LCD pixels are restored to exactly 128 after appearance augmentation. Before training, the loader checks every image and mask: masks must be binary, visible landmarks and the needle shaft must remain outside them, and hidden pixels must equal 128. Reference images, calibration, preprocessing configuration and the stabilization implementation are checksum protected.

## Dataset and labels

The dataset contains **2,000 images: 1,600 training and 400 validation**, generated with OpenCV from six stabilized real source plates. Variations include needle speed, rotation, translation, scale, shear, mild perspective, lighting, shadows, reflections, blur, motion blur and compression. This version does not mirror the printed dial.

Each sample has twelve pixel-coordinate landmarks in this order:

`pivot, needle_tip, tick_0, tick_20, tick_40, tick_60, tick_80, tick_100, tick_120, tick_140, tick_160, tick_180`

`annotations.json` also records indicated speed, source frame/group, split, image and LCD-mask paths, and generation provenance. Tick coordinates are **amodal**: a tick remains a supervised target when the needle covers its printed stroke. The legacy-named `visibility` field means an in-frame landmark label is available; `observed_visibility` separately records synthetic needle occlusion. The model's corresponding head is not calibrated confidence.

Training and validation use different source plates and donor groups. **All plates still come from one short recording**, so neighbouring frames and dashboard appearance remain correlated. Validation is a development check, not independent real-camera accuracy. Rendered speed labels are exact relative to a manually reviewed draft calibration; that calibration has approximate pixel uncertainty and is not a road-speed ground truth.

## Model and speed calculation

The model uses a MobileNetV3-Small encoder with two spatial decoders. The original decoder predicts the needle pivot and tip; a new decoder predicts ten semantic scale ticks. Both produce 128 × 128 heatmaps from a 512 × 512 input. The original needle predictions are preserved exactly when the verified V3 epoch-8 weights are loaded.

Speed decoding fits a projective mapping from **predicted tick points** to the static canonical calibration, rectifies the predicted needle, and interpolates its angle between calibrated speed ticks. It never consumes a sample's rendering transform or target coordinates. The retained direct speed-regression head is frozen and is not the primary speed reader.

Geometric checks reject inconsistent tick layouts, insufficient support, unavailable or outlying scale endpoints, excessive reprojection error, implausible pivots/needle lengths and readings outside the calibrated arc. Both 0 and 180 km/h endpoints must be supported. The old 100-pixel fixed-camera displacement gate is absent. Small endpoint excursions within 2 km/h can be clamped and are reported explicitly. These checks measure geometric consistency; they cannot prove a prediction is correct.

## Colab and checkpoints

Open `Speedometer_Joint_Landmarks_V4_Colab.ipynb`, select a T4 GPU, connect Google Drive and run all cells. The published notebook pins an exact code commit and dataset-download manifest. It verifies each archive part, joins the parts, verifies the full archive and safely extracts it before checking the dataset.

The recorded Colab runtime used **OpenCV 4.14.0, NumPy 2.1.3, PyTorch 2.11.0+cu130, TorchVision 0.26.0+cu130 and Pillow 11.3.0**. The notebook verifies the exact OpenCV/NumPy versions before importing the frozen training implementation. When needed, it installs `opencv-python-headless==4.14.0.94` and `numpy==2.1.3` with `--no-deps`, preserving other installed packages. If mismatched libraries are already loaded, restart the runtime when prompted and run all cells again. Resume additionally requires the checkpoint's other recorded runtime and configuration values; it never substitutes a fresh run.

This version check matters: OpenCV 4.6.0 and 4.14.0 produced different RANSAC acceptance decisions from the same stored predicted landmarks. Matching model weights and source code alone does not reproduce speed-decoder behavior.

The default experiment is `speedometer_day_arc_needle_v4`, with batch size 8 and a maximum of 20 epochs. The first two epochs train the new tick decoder and landmark-availability head. Later epochs gently fine-tune the needle decoder and final four encoder blocks. Early stopping uses six epochs without meaningful validation improvement. The new architecture starts a new optimizer while transferring the verified V3 needle weights.

| Mode | Behavior |
| --- | --- |
| `transfer` | Start a new V4 experiment from the verified private V3 epoch-8 checkpoint. |
| `resume_latest` | Restore the newest compatible completed V4 epoch. |
| `resume_best` | Restore the best compatible epoch with that epoch's optimizer and random state. |
| `fresh` | Create a separate experiment with random weights and no previous checkpoint. |

Epoch 0 and every completed epoch save full state to `MyDrive/SpeedometerTraining/experiments/speedometer_day_arc_needle_v4/runs/`. States include model, optimizer, AMP scaler, random generators, history and best weights. Completion markers verify file hashes; incompatible or incomplete checkpoints do not silently trigger fresh training. Resume retains the original epoch target rather than adding another 20 epochs. Private source weights and training checkpoints remain in Drive.

## Reading the results

- **Needle/tick error:** pixel distance from the synthetic target; lower is better.
- **Coverage:** percentage of inputs that pass geometry checks; this is not accuracy.
- **All-input ±2 km/h:** percentage of eligible inputs with an accepted reading within 2 km/h of the target. Rejections count as failures.
- **Accepted-only speed error:** useful diagnostically, but excludes failed readings.

Model selection combines needle error, tick error and the all-input speed metric. Results must then be inspected on the complete real video, followed by separately recorded clips before making any accuracy claim. A 98% real-video target has not been established by this dataset.

The local verification suite covers preserved V3 predictions, actual model gradients, mask placement, exact interrupted/resumed training, corrupt-checkpoint recovery, perspective-aware geometry and multipart archive verification:

```bash
python -m unittest real_adaptation_v4.test_training real_adaptation_v4.test_geometry real_adaptation_v4.test_notebook_helpers -v
```

The private-checkpoint transfer test is skipped when the verified V3 checkpoint is unavailable. The public dataset and fresh mode can be used without those private weights.

## Real-video inference

Use an isolated Python environment with working PyTorch/TorchVision and the matching numerical libraries. Install these into that activated environment, then restart any already-running notebook kernel:

```bash
python -m pip install --no-deps numpy==2.1.3 opencv-python-headless==4.14.0.94
```

Keep the private `best_model.pt` together with its `best_model.complete.json` marker. Use the verified V4 dataset and the unchanged frozen implementation files. FFmpeg and FFprobe must be available on PATH. From the repository root:

```bash
python -m real_adaptation_v4.infer_video --checkpoint /path/to/best_model.pt --dataset /path/to/dataset2000 --video /path/to/input.mp4 --output /path/to/predictions.mp4 --run-dir /path/to/new_inference_run
```

Inference checks OpenCV against the version recorded in the checkpoint before processing video. It also verifies model, dataset, calibration, preprocessing and implementation hashes. The output is a 30 FPS sampling/playback timeline, not a promise of live 30 FPS processing. Predicted landmarks, rejected readings and per-frame results remain visible for inspection.

Optional visual-reference comparison requires explicit private annotations:

```bash
python -m real_adaptation_v4.compare_real_reference --predictions /path/to/new_inference_run/predictions.jsonl --reference /path/to/visual_reference.json
```

Approximate same-recording visual labels are a development diagnostic, not independent accuracy ground truth. Original videos, manual source-frame annotations and trained weights are not included in this repository update.
