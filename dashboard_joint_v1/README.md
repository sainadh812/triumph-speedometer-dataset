# Dashboard, LCD, speed and RPM training

This version provides one Colab workflow for three sequential training tasks. Dataset preparation is CPU-only; neural model smoke checks, training and inference are restricted to a Colab CUDA runtime. The source code does not include private frames, videos, model weights or OAuth credentials. Publication of code alone does not mean a dataset has been uploaded or training has run.

## Models and inputs

| Task | Input | Learned outputs |
|---|---|---|
| Localizer | Whole-frame grayscale, letterboxed to 512×512 | Dashboard presence and spatial box corners, precise LCD mask, four physical LCD corners and four distinct RPM crop-region corners |
| Speed | Dashboard crop 512×512, LCD interior filled gray 128 | Needle pivot/tip and ten named scale ticks 0–180 |
| RPM | Wider rectified LCD scale view, 256 wide × 384 high | Seven RPM scale anchors 0, 2, …, 12 and visible fill-bar endpoint |

The exported task manifests declare actual input dimensions and semantic point order; the trainer validates and uses those declarations. Each task starts with ImageNet MobileNetV3Small features and new task heads. It does not load the previous speed or RPM task heads. The localizer shares one dense decoder between eight semantic point heatmaps, LCD segmentation and four separate dashboard bounding-box corner heatmaps.

The eight semantic points are `lcd_tl, lcd_tr, lcd_br, lcd_bl, rpm_roi_tl, rpm_roi_tr, rpm_roi_br, rpm_roi_bl`. The last four describe the RPM crop rectangle, not physical LCD corners. They can support a direct RPM-region warp when the full LCD extends beyond the frame. Their targets are derived only from credible source geometry and supervised only when the crop corners fall inside the image. Their destination rectangle and pixel-center convention must match the RPM training crop exactly. They do not define the LCD segmentation mask.

Unavailable or uncertain landmarks are marked invisible and contribute no location loss. Their visibility labels still train the visibility head. Heatmap supervision uses spatial cross-entropy on visible points; LCD segmentation uses foreground-weighted BCE and Dice. Dashboard-absent examples have an empty LCD mask and unavailable corners.

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

## Runtime interfaces and verification

`train_all(dataset_root, output_root, configs=None, resume_modes=None, resume_checkpoints=None, checkpoint_callback=None, progress_callback=None)` trains all three tasks. The synchronous checkpoint callback receives `(task, checkpoint_path, metadata)` after atomic epoch saves and at finalization. `smoke_test(...)` discards its temporary models and restores caller RNG states.

`colab_workflow.prepare(...)` verifies private data; `colab_workflow.run_all(...)` restores selected checkpoints, performs the smoke check, trains, saves verified artifacts and releases the runtime. Credentials stay within the existing Drive transport.

Offline checks can run without neural execution:

```text
python -m unittest -v test_pure_contract.py test_workflow.py test_notebook.py test_numerics.py
```

These cover numerical transforms/metrics, label and manifest integrity, resume identities, mocked Drive preservation, completion restoration, runtime-release ordering and bounded AMP retry control flow with NumPy fakes. They do not replace the required Colab model smoke check.
