# V4 training results

Training completed all **20 epochs on a Tesla T4**. Epoch **17** had the lowest validation selection score and was exported for inference.

The dataset contains **1,600 training images and 400 validation images**, all 512 × 512 grayscale with the corrected LCD mask. The model predicts the needle pivot/tip and ten speed-scale landmarks.

## Synthetic development validation

| Metric | Before training | Selected epoch 17 |
| --- | ---: | ---: |
| Mean needle-tip error | 6.13 px | 1.45 px |
| Mean pivot error | 4.29 px | 0.64 px |
| Mean tick error | 97.72 px | 1.13 px |
| Geometry accepted | 0 / 400 | 400 / 400 |
| All-input speed within ±2 km/h | 0 / 400 | **397 / 400 (99.25%)** |
| Mean accepted speed error | Not available | 0.538 km/h |
| 95th-percentile speed error | Not available | 1.413 km/h |

The baseline is V4 immediately after transferring the working V3 needle weights; its newly added tick branch was still untrained. Two selected-epoch readings used the existing ±2 km/h endpoint-clamp tolerance. Rejected readings count as failures in the all-input metric; there were no rejected validation readings at epoch 17.

**99.25% is a synthetic development result, not measured real-world accuracy.** Training and validation use different source plates and donor groups, but all originate from the same short video. Rendering labels also depend on the manually reviewed calibration. The full-video check below uses that same recording; independently recorded test clips are still required.

## Same-recording video inspection

Epoch 17 processed the complete development clip using image-based stabilization, the fixed LCD mask and the predicted needle/tick landmarks. All **600 distinct source frames** were included. The 30 FPS playback timeline contains **602 model inputs/output frames**, including two repeated source frames; repeated frames are counted once in the visual-reference comparison.

The geometry checks accepted **600 / 600 distinct frames** and **602 / 602 output readings**, with **zero rejections and zero endpoint clamps**. This is reading coverage, not accuracy. No speed smoothing or stale mask/speed fallback was used. The final 1024 × 720 video was decoded and checked: 602 frames, 30 FPS, about 20.07 seconds, with an approximately 0.012-second duration difference from the source.

Eight frames had approximate visual needle annotations prepared before inference. Their mean differences from the predictions were **2.52 px for the pivot**, **2.80 px for the tip**, and **0.49 km/h for indicated speed**. The manual estimates have subjective uncertainty of about 2.5 px at the pivot and 1.5–2 px at the tip; the scale calibration is approximate too. These are same-recording development diagnostics, not precise ground truth, an accuracy percentage or evidence of performance on unseen recordings.

CPU processing took about **233 seconds**, or **2.59 output frames/second**, including decoding, preprocessing, inference and overlay writing, but excluding final H.264 encoding. **30 FPS describes sampling/playback, not demonstrated live processing speed.**

## Verification and runtime

The downloaded results ZIP and selected weight file passed checksum verification. Model architecture, tensor shapes, dataset/preprocessing identity, implementation hashes, epoch selection, pixel errors and speed/rejection accounting were checked. All **400 saved prediction records** were replayed through the geometry decoder and matched the Colab readings. This audit did not rerun neural inference on every validation image.

Use **`opencv-python-headless==4.14.0.94`** (OpenCV runtime 4.14.0); the tested NumPy version is **2.1.3**. An older local OpenCV 4.6.0 produced 56 additional endpoint-consistency rejections from the same saved landmarks. Matching the training runtime restored all 400 readings without changing the model or relaxing its geometry checks. Inference now checks the OpenCV version before processing video.

## Drive checkpoints

The remote audit found **21 full-state checkpoint files, epochs 0–20**, and checked every completion marker and file size. It recomputed hashes and validated complete resumable payloads for **best epoch 17** and **latest epoch 20**, including 72 optimizer-state entries in each. The exported inference weights matched the selected full checkpoint exactly.

The post-training audit did **not** recompute all 21 checkpoint hashes. Full-state files remain in Drive; the compact downloaded inference export contains no optimizer state. The local audit verified the consistency of the included remote Drive audit report, not every Drive file's bytes.

Machine-readable training and same-recording video aggregates are in `results_summary.json`; private model weights, individual prediction records and source videos are not published here.
