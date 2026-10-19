# Dashboard, LCD, speed and RPM training

This version provides one Colab workflow for three sequential training tasks. Dataset preparation is CPU-only; neural model training is restricted to a Colab CUDA runtime. Inference defaults to Colab CUDA and also supports explicitly selected Colab CPU inference. Local neural execution and automatic device fallback remain disabled. The source code does not include private frames, videos, model weights or OAuth credentials. Publication of code alone does not mean a dataset has been uploaded or training has run.

For CPU video tests, set `device='cpu'`, a modest `batch_size` such as 4, and `cpu_threads=2` in the video configuration, then select CPU hardware in Colab. CPU inference uses float32 with unchanged preprocessing, model weights and reading thresholds. CUDA uses float16 autocast, so small numerical differences near decision boundaries are possible. Saved reports identify the device, precision, thread count and actual conditional reader coverage; CPU FPS must not be presented as GPU FPS. The video workflow verifies Drive outputs before releasing the runtime.

## Models and inputs

| Task | Input | Learned outputs |
|---|---|---|
| Localizer | Whole-frame grayscale, letterboxed to 512×512 | Dashboard presence and spatial box corners, precise LCD mask, four physical LCD corners and four distinct RPM crop-region corners |
| Speed | Dashboard crop 512×512, LCD interior filled gray 128 | Needle pivot/tip and ten named scale ticks 0–180 |
| RPM | Wider rectified LCD scale view, 256 wide × 384 high | Seven RPM scale anchors 0, 2, …, 12 and visible fill-bar endpoint |

The exported task manifests declare actual input dimensions and semantic point order; the trainer validates and uses those declarations. By default each task starts with ImageNet MobileNetV3Small features and new task heads. Optional verified weights-only transfer retains the selected encoder and task heads from a compatible completed experiment while creating new optimizer, scheduler, scaler, RNG and history state. The localizer shares one dense decoder between eight semantic point heatmaps, LCD segmentation and four separate dashboard bounding-box corner heatmaps.

The eight semantic points are `lcd_tl, lcd_tr, lcd_br, lcd_bl, rpm_roi_tl, rpm_roi_tr, rpm_roi_br, rpm_roi_bl`. The last four describe the RPM crop rectangle, not physical LCD corners. They can support a direct RPM-region warp when the full LCD extends beyond the frame. Their targets are derived only from credible source geometry and supervised only when the crop corners fall inside the image. Their destination rectangle and pixel-center convention must match the RPM training crop exactly. They do not define the LCD segmentation mask.

New schema-2 datasets distinguish known invisible landmarks from unknown labels using `visibility_known`. Unknown labels contribute neither location nor visibility loss, and are excluded from visibility metrics. Older archives without this field retain their original supervision; they are not silently relabeled. Heatmap supervision uses spatial cross-entropy on visible points; LCD segmentation uses foreground-weighted BCE and Dice. Dashboard-absent examples have an empty LCD mask and known unavailable corners.

Reviewed `rpm_roi_corners` and `rpm_roi_review` can specify RPM crop geometry independently of whether the numbers or fill endpoint are readable. The review must explicitly mark the geometry reviewed and explain its basis. Without reviewed geometry or sufficient existing numeric anchors to justify the legacy derived region, the region's visibility is unknown rather than a negative training target. Direct reviewed crop corners also support RPM training when the far-right LCD corners are outside the image. Numeric labels are never invented to justify a crop. New datasets require the updated trainer and a new run identity.

## Data and evaluation limits

`build_dataset.py` and `preprocess.py` keep image geometry, LCD masks and point labels in the same coordinate system. Exact LCD masking is reapplied after appearance changes. Padded crop context and the precise speed mask are separate: extra RPM context must not enlarge the speed mask across the analog needle or scale.

All descendants of a source frame stay in its original split. Current source time blocks come from two recordings, so results measure development performance within those recordings. Thousands of augmented images do not represent thousands of independent scenes. The private manifest and provenance identify the included examples; review generated crops before training.

Optional `synthetic_rpm.py` examples draw segmented fill inside a reviewed empty training LCD while preserving its scale and numbers. These are marked synthetic and used only for training; their chosen scalar is not a measured engine RPM. Needle positions otherwise remain the observed real states. The manifest records how many procedural examples were included.

Reports include visible-point error in input pixels, per-landmark error, visibility/presence classification, dashboard box IoU and LCD mask IoU. Empty-mask false positives are reported separately. These are **not physical engine-RPM accuracy, speed-km/h accuracy, or a 98% real-world accuracy claim**. Converting predicted anchors to readings and testing on independent recordings remains separate work.

## Colab operation

1. Use the configured private notebook, or configure the public template with the private Drive dataset reference. Keep OAuth JSON only in the existing Colab secret `TRIUMPH_DRIVE_OAUTH`; enable access for the notebook. Never put the secret in a notebook cell or Git.
2. Run CPU preparation first. It verifies archive bytes, dataset hashes, dimensions, labels, splits and Drive write access.
3. Select a T4 runtime and run training. Code is pinned by commit and SHA256; data can be restored after runtime replacement. A disposable smoke check exercises one training step and one validation batch for each task before full training. Its timings include warmup and are not sustained inference FPS.
4. Train the localizer, speed and RPM tasks sequentially. Initial AMP batch sizes are 16, 32, 64 respectively. Each task uses at most 40 epochs, at least 10 before early stopping, and patience 8. Colab smoke must verify that those batch sizes fit the actual runtime.
5. Wait for verified Drive completion. The workflow preserves full epoch checkpoints and exports a smaller archive containing selected weights, reports and history before attempting automatic runtime release. If saving fails, it preserves recovery artifacts first; if those cannot be verified, it retains the runtime and reports the problem.

Both smoke checks and training use guarded AMP updates. The initial loss scale is 256. If a finite loss produces nonfinite gradients, GradScaler skips the optimizer update and the same batch is retried at half the scale, up to eight retries and a minimum scale of 1. RNG state and mutable model buffers are restored before each retry. Only finite unscaled FP32 gradients with a finite clipping norm can update weights. Nonfinite losses or persistent gradient failures stop the run. Smoke and epoch reports record overflow retries and loss scales; this does not silently discard difficult batches or disable numerical checks.

Private output layout is `SpeedometerTraining_API_v1/dashboard_joint_v1/<RUN_ID>/<task>/attempt_<timestamp>/`. The run folder contains an immutable `<attempt>_COMPLETE.json` receipt and results archive. Earlier attempts and their checkpoints remain available.

## Post-training video inspection

`build_video_cell.py` creates a separate inference cell for the same notebook. It pins the inference code and an existing training completion receipt, loads all three selected new checkpoints, and downloads the configured private videos by SHA256 and byte count. It never resumes training. If training has already released its runtime, reconnect a T4 and run this cell alone.

`video_test.py` processes every decoded frame from each video's final 120 seconds. The output preserves the full frame, with dashboard/LCD outlines, speed ticks and needle, RPM region and landmarks, speed and larger RPM text. Missing or inconsistent geometry produces an unavailable reading and a recorded reason; raw landmarks remain inspectable. The speed reader fits named ticks to a reviewed training reference plane before angle interpolation, while RPM uses the predicted scale anchors and bar endpoint. These are indicated-reading estimates, not independent measurements of vehicle speed or engine RPM.

Outputs include H264 MP4, per-frame CSV/JSONL, timings and model/source provenance. Reports distinguish input playback FPS, model-pipeline throughput, and full processing throughput including video preparation and encoding. First-batch startup is included, and no frames are discarded for warmup. Every output and a result archive are checksum verified in Drive before the runtime is released. This inspection uses the source recordings and is not an independent-recording accuracy test.

Prediction-pipeline FPS describes conditional work: localization runs every frame, while speed and RPM readers require usable regions. Reports now include actual per-task execution coverage, initial timestamp-scan time, and separate prefix/selected-window decode times. These decode components are already included in total decode time. Grayscale conversion is shared once per original frame across all three branches; crop geometry, resolution, masks, and rejection thresholds are unchanged.

## Fresh and resumable runs

Each epoch checkpoint contains model and optimizer weights, scheduler, AMP scaler, Python/NumPy/Torch RNG state, loader generator states, history and the best selected weights. Resume checks dataset, preprocessing, architecture, runtime code and training configuration identities.

| Situation | Mode |
|---|---|
| New experiment | New `RUN_ID`, `fresh` for every task |
| Task started but interrupted | Same `RUN_ID`, `latest` |
| Task already completed | Same `RUN_ID`, `latest`; restore its report and cached final test |
| Later task has not started yet | Same `RUN_ID`, `fresh` for that task |
| Continue from a validation-best epoch | Same `RUN_ID`, `best`; create a new attempt and preserve later old epochs |

Validation alone selects weights. The test split is evaluated after selection and cached for a completed branch; restoring a completed task does not repeat its final test. A deliberate best-epoch branch has separate history. Changing data, code, architecture or training configuration requires a new experiment rather than silently restoring incompatible optimizer state.

## Daylight-focused fine-tuning

`TrainConfig.daylight_fraction` optionally sets the expected fraction of daylight training draws. For example, `.70` gives about 70% daylight and 30% night draws over an epoch; it is stochastic, not an exact batch quota. Sources have equal total probability within each lighting group, so creating more augmentations for a source does not make that source dominate. Epoch length stays equal to the training row count, with replacement. The sampler uses the checkpointed training generator. `None` retains ordinary shuffled training. Explicit balancing requires both lighting groups and known lighting metadata.

Validation and test splits remain fixed, sequential and unweighted. Reports include separate day/night losses, visible-point errors, localizer metrics and visibility-head diagnostics using the same predictions. Losses use their actual labeled-point or image denominators, so missing labels and batch boundaries do not distort checkpoint selection. The combined loss uses the same coefficients as training. Aggregate validation loss still selects the checkpoint: night retention must be inspected before adopting the result. Visibility-head availability is not numeric reading accuracy.

Weights-only transfer first evaluates the unchanged starting weights on the current validation labels, before any training update. This saves `initial_validation.json` and embeds the result in initialization provenance, using the same day/night metrics and corrected denominators as the new run. It does not consult the test set. Compare night retention against this baseline rather than incomparable old combined losses.

The unchanged transferred model is also an eligible **epoch 0** candidate. Its full `epoch_0000.pt` checkpoint is verified through the normal callback before training starts, with fresh optimizer/scheduler/scaler state and validation-time RNG/loader states. A trained epoch replaces it only when the same validation selection score improves by `min_improvement`. If none improves, `best_model.pt` retains the exact starting weights and `best_epoch` is `0`; `selection_origin` is `initialization`. Training history still contains only actual epochs 1 onward. A `best` restart from epoch 0 restores the matching fresh optimizer state; a `latest` restart retains its later optimizer state and the separate best weights. The final test runs once on whichever candidate validation selected. Models with newly initialized heads keep the existing trained-epoch selection behavior. This protects aggregate validation loss, not every individual day/night metric: inspect subgroup regressions before adoption. Old reports and checkpoints are unchanged; revised runtime code requires a new experiment.

For a new dataset, use a **new run ID** and optionally pass `initialization_reference` to both `colab_workflow.prepare` and `run_all`. This reference pins the old `run_id`, exact `complete_id`, and `expected_identity` including its dataset manifest hash. Preparation verifies the completed archive and extracts only each task's selected `best_model.pt` plus checksum marker. The trainer strictly checks task, architecture, preprocessing, dimensions and semantic point order, then strictly loads weights. Changed dataset/sample/code hashes are recorded as transfer provenance; full epoch files cannot be used through this path. A changed preprocessing such as RGB requires an explicitly designed new experiment, not silently loading grayscale weights through this compatibility check.

Changing Colab runtime type replaces the VM and can erase `/content`, including prepared variables and downloaded initialization files. Run `prepare(..., initialization_reference=...)` in the current runtime before a fresh transfer run; a completed CPU preflight does not mean its cache survived. Necessary downloads may repeat on the new VM. An all-`latest`/`best` resume uses self-contained epoch checkpoints and needs no original initialization archive. For mixed fresh/resumed tasks, only fresh tasks receive initialization weights. Runtime code and configuration must still exactly match for resumed tasks.

New daylight frames and registration proposals are preparation assets, not automatically training labels. The builder rejects explicitly unreviewed rows before producing files. Review image geometry and every visible/unknown label; unreadable scale numbers must not be invented. Preserve RGB source originals even when the current task crops use grayscale. Geometric, blur and lighting variants remain within the source split.

`build_dataset.py --rpm-train-corner-jitter-fraction 0.025` is an optional preparation experiment. For real-source TRAIN variants after variant 0, it perturbs the same four semantic RPM-region corners used at inference, with per-axis Gaussian sigma 2.5% of the ROI top-edge length and offsets clipped at 5%. Every retained example must keep all reviewed visible RPM landmarks and their bounding context (4 canonical pixels of margin where available) inside the final crop. Invalid proposals retry at most 12 times, then fall back; later geometric transforms that clip this protected context are skipped. The manifest, sample metadata and QA counts record the setting and outcomes. The default remains the historical corner-noise behavior; validation/test rows and pixels, baseline variant 0, and procedural RPM renders are unchanged. This augmentation does not label unknown scale content or replace review of the LCD/RPM crop boundaries.

`build_dataset.py --train-appearance-guard` optionally rejects a TRAIN appearance change when it newly clips or collapses local contrast around supervised physical landmarks or the speed needle. It compares the geometrically warped image before and after appearance changes, then keeps the original warped pixels on failure. It does not remove real glare or infer missing labels. The deterministic heuristic consumes no extra random draws; validation/test pixels and records, geometry, labels and default behavior remain unchanged. Sample metadata and QA counts record fallbacks and thresholds. Review the resulting images: this guard is not a learned readability detector.

## Runtime interfaces and verification

`train_all(dataset_root, output_root, configs=None, resume_modes=None, resume_checkpoints=None, checkpoint_callback=None, progress_callback=None, initialization_weights=None)` trains all three tasks. `initialization_weights` optionally maps fresh tasks to verified selected model files. The synchronous checkpoint callback receives `(task, checkpoint_path, metadata)` after atomic epoch saves and at finalization. `smoke_test(..., initialization_weights=None)` tests the same selected initial weights in disposable models and restores caller RNG states.

`colab_workflow.prepare(...)` verifies private data; `colab_workflow.run_all(...)` restores selected checkpoints, performs the smoke check, trains, saves verified artifacts and releases the runtime. Credentials stay within the existing Drive transport.

Offline checks can run without neural execution:

```text
python -m unittest -v test_pure_contract.py test_workflow.py test_notebook.py test_numerics.py
```

These cover numerical transforms/metrics, label and manifest integrity, resume identities, mocked Drive preservation, completion restoration, runtime-release ordering and bounded AMP retry control flow with NumPy fakes. They do not replace the required Colab model smoke check.
