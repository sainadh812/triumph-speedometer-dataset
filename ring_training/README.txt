Triumph dashboard: arc and needle learning check

Open Speedometer_Arc_Needle_Colab.ipynb in Google Colab, select a T4 GPU,
and Run all. Authorize Google Drive when Colab asks.

Scope
- Exactly 20 approved synthetic review images, at 512 x 512 pixels.
- Same images used for training and evaluation: a reconstruction/learning check.
- No held-out accuracy, 98% target, real-camera result, or 500-image run is claimed.
- New outputs: six named tick locations, six visibility scores, visible scale arc.
- Existing needle and speed outputs are retained for before/after comparison.
- Geometric inference uses predicted landmarks and visibility, never label coordinates.

Drive setup
Root: MyDrive/SpeedometerTraining/experiments/
Default experiment: speedometer_arc_v1
Initial source: speedometer_rotation_v2_from47/runs/
  20261017_100006_266720_f43a79f6/best_model.pt
Required source SHA256:
  ba47ae2a3c07fd9a490342eebb8a01ac160dfe5e9dbca24dd13014ed68cac270
The private source weights are read from Drive, not downloaded from GitHub.

Checkpoint modes
- resume_latest (default): restores the newest complete epoch. Only a brand-new
  experiment automatically initializes from the verified epoch-19 source weights.
- resume_best: restores the complete state with the lowest reconstruction loss.
- transfer: explicitly starts a new, unused experiment from epoch-19 weights.
- fresh: creates a unique experiment with random weights and no previous weights.
  Record the printed experiment name and use it for subsequent resumes.
Fresh random runs fine-tune the entire encoder in stage 2; transferred/pretrained
runs fine-tune its final four blocks. Encoder BatchNorm running statistics remain
fixed for these small batches.

The default budget is fixed at 3 new-head epochs plus 30 fine-tuning epochs (33).
The short local run checked software and persistence; it did not establish
successful landmark learning. This bounded run tests learning before scale-up.
Fine-tuning rates: new heads 1e-3, shared decoder/old heads 1e-4, encoder 2e-5.
These settings are part of the saved experiment configuration and identity.
Source epoch 19 is provenance; the expanded model starts at epoch zero.
Every completed epoch retains full model/optimizer/scaler/random-generator/data-
loader/history state. The original epoch-zero baseline remains available.
Saved configuration is reused on resume; rerunning does not add another 33 epochs.
Corrupt candidates are reported and excluded. Recovery requires a verified,
compatible checkpoint; if none remains, there is an error and no fresh fallback.
Exact resume requires matching code, dependencies and device type/model.

Dataset and code
Public archive: ring_training/arc_needle_review20.zip in the GitHub repository.
Archive SHA256:
  651c9052d59f3652bcdff10b04d559f0858e9cfbb067c1eccb372f202b896dee
Labels SHA256:
  a7fb97a37edbb654ff46653d7cc4090b5dbc4e5f7e2e90eb51cc4729bc40c3d5
The notebook embeds and verifies six Python sources while preserving their folder
layout. The dataset ZIP's code is not executed. Images, eight masks per image,
keypoints, mask semantics and pose-group separation are checked before training.

Results
Drive retains before/after metrics, all predictions, plots and all full checkpoints.
The compact downloadable results ZIP includes best weights and source code but
excludes all epoch full-state files. It contains private Drive paths and weights.
Inspect arc overlap, landmark error, speed error, geometry coverage and rejection
reasons. Rejections count against geometric success across all inputs.

Rebuild locally: python ring_training/build_notebook.py
Local verification harness only: set RING_LOCAL_SMOKE=1, RING_WORK_DIR,
RING_DRIVE_ROOT, RING_SOURCE_CHECKPOINT and RING_DATASET_ZIP; optionally set
RING_STOP_AFTER_EPOCHS=0 for a no-update notebook execution. These overrides are
not needed in Colab. Building or verifying the notebook does not publish files.
