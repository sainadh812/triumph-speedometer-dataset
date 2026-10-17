# Real dashboard adaptation V3

This experiment addresses the poor needle detections on the fixed-camera night video. It uses the approved real-dial background repairs, masks the LCD, and fine-tunes the previously frozen needle model. It does not retrain the earlier ring/tick branch.

## Data

- 500 single-channel 512 x 512 synthetic PNGs: 400 training, 100 validation.
- Speeds span 0–180 km/h, including fractional values. Labels contain speed, pivot/tip, tick landmarks, needle/LCD masks, and rendering provenance.
- Appearance varies across clear, dim, bright, reflection, shadow, blur, motion blur, and compression conditions. Camera changes include ±3 degrees, small scale/perspective changes, and a mixture of small and larger translations covering movement observed in the video. This is a fixed-camera experiment, not a new full-rotation robustness claim.
- Training uses source plates 0 and 180. Validation uses plate 204. They share the same recording, calibration, and some repair donors; validation is a limited synthetic diagnostic.
- LCD pixels are constant gray 128. Warning lamps inside the needle sweep remain visible, because a fixed mask there would also erase the needle at some speeds.
- The private Colab run additionally uses ten approved, manually annotated real frames for training. Nine later frames from the same clip are diagnostics, with one glare-obscured frame excluded from precise speed scoring. These private examples and the original video are not in this public repository.

`real_dashboard_lcd_mask_500_v3.zip` is rooted at `dataset500/`. The manifest records its SHA-256 and QA results. The archive includes input and label contact sheets. There is no independent test split.

## Training and speed measurement

The notebook extracts the legacy `needle_branch` weights from the private best V2 combined checkpoint, verified by checksum. This starts a new experiment: its optimizer and epoch counter reset. The decoder, prediction heads, and last four encoder layers are trainable; the remaining encoder layers stay frozen. The new model retains the existing MobileNetV3-Small architecture.

Training uses needle heatmap and coordinate losses, plus an auxiliary direct-speed loss. Main speed estimates use predicted needle endpoints and the manually calibrated dial. Image features estimate the camera transform to the reference dial; the known rendering transform and per-image target ticks are never used as an inference shortcut. Failed registration and implausible geometry reject a reading.

A fixed 2 km/h endpoint allowance handles the small offset of a resting needle from the visually estimated zero tick. Raw extrapolated readings from -2 to 0 or 180 to 182 km/h are clamped to the physical 0–180 range and flagged in reports; larger excursions are rejected. Raw values and clamp counts remain visible. This is a calibration policy developed on the same clip, not evidence that the neural detector improved.

The default pilot is 25 epochs, batch size 8. Reports compare the starting model with the selected best model and show needle-point pixel error, geometric speed coverage, success within ±2 km/h with rejections counted as failures, and accepted-only speed error. The direct-speed head is reported separately.

## Colab and checkpoints

Open `Speedometer_Real_Adaptation_V3_Colab.ipynb` in Colab, choose a GPU, and run all cells. Public data and code are pinned and verified. The private masked-frame ZIP is requested only when it is absent from the user's Drive.

Every completed epoch is saved to a new Drive experiment with full model, optimizer, AMP scaler, loader/random-number states, and completion metadata. Learning rates stay fixed; there is no scheduler. Modes are:

- `transfer`: create this experiment once from the approved old needle weights.
- `resume_latest`: continue the latest completed state after interruption.
- `resume_best`: branch from the best completed state.
- `fresh`: create a separate experiment without previous trained weights.

A missing or incompatible checkpoint does not silently start over. Old experiments remain intact. Training weights and full checkpoints stay private.

## Completed pilot: 17 October 2026

The Colab T4 run completed all 25 epochs and selected epoch 8 using synthetic validation only. Every completed epoch and the best inference model were saved privately to Drive. The downloaded compact export passed archive/checkpoint checksum, provenance, metric-denominator and best-selection checks; one real-frame CPU inference replay also passed.

| Development measure | Starting weights | Selected best |
| --- | ---: | ---: |
| Synthetic validation: within ±2 km/h, rejections count as failures | 2/100 | 100/100 |
| Synthetic validation: mean needle-tip error at 512×512 | 115.84 px | 1.72 px |
| Same-video real diagnostics: within ±2 km/h, rejections count as failures | 0/8 | 7/8 |
| Same-video real diagnostics: mean needle-tip error | 46.62 px | 5.02 px |
| Real readings accepted, including the uncertain-label frame | 3/9 | 8/9 |

The remaining rejection is a dark frame with insufficient dial-registration feature matches, even though its needle prediction improved. One glare-obscured real frame is excluded from precise speed/keypoint metrics because its manual reference is uncertain. Three accepted real readings use the declared endpoint clamp. The seven accepted, labeled real readings have 0.39 km/h mean absolute indicated-speed error; this accepted-only figure excludes the rejection and is not an all-input accuracy score.

The auxiliary direct-speed head remains weak (15.73 km/h real-frame mean absolute error). Use the predicted needle plus calibrated geometry as the primary output. These nine diagnostic frames are a small same-recording development check; this run does not establish 98% performance on unseen videos. See [aggregate_results.json](aggregate_results.json) for exact denominators and before/after values.

## Limits and provenance

The user approved the 20-image visual preview before expansion. One ImageGen edit assisted removal of the old needle in source preparation; generation of these 500 variants uses OpenCV/NumPy and makes no ImageGen calls. The generator and rendering helpers are included for inspection; rerunning source preparation requires the private reviewed source assets, which are intentionally not published.

Calibration and real-frame labels are manual visual estimates of indicated dial speed, not measurements of road speed. Exact agreement between a renderer and its labels does not establish physical accuracy. Results from this recording cannot substantiate 98% real-world accuracy; a separately recorded, labeled video is still required. This project is not affiliated with Triumph.
