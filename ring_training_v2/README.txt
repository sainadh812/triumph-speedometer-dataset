Augmented arc and tick training
==============================

Open Speedometer_Arc_Ticks_Augmented_Colab.ipynb in Google Colab,
select a T4 GPU and Run all. Connect Google Drive when requested.
The notebook downloads the public dataset and verifies its checksum.
The owner's default transfer mode reads private Drive checkpoints;
other users should select fresh and a new experiment name.

Dataset
-------
700 grayscale images, 512 x 512: 500 train / 100 validation / 100 test.
350 independently sampled camera/needle/occlusion pose families, with two
appearance variants each. Split assignment happens before rendering.
All samples share one reconstructed dashboard source, so this is a
synthetic holdout, not evidence of real-camera accuracy.

Transformations include continuous 360-degree camera rotation, translation,
mild shear, perspective, scale, motion/defocus blur, exposure/noise, glare
and controlled occlusion. Mirrored flips are disabled because they reverse
the scale. Image geometry, keypoints and supervision masks transform
together. Intended speed remains unchanged. Hidden landmarks and uncertain
arc regions do not receive reliable-coordinate/mask supervision.

Images are high-quality grayscale JPEG. Masks are lossless PNG. Labels
include speed, needle pivot/tip, named tick points, visibility, homography,
pose family, split, source provenance and image hashes. QA contact sheets
are review overlays only; overlays never enter training images.

The source annotations retain uncertainty around the reconstructed 60 km/h
region. No new claim is made about the underlying photo's ownership/license.
Generation uses OpenCV; there are no per-image AI generation calls.

Model changes
-------------
The arc/direct-speed branch initializes from the previous arc model's best
epoch 33. A dedicated multiscale decoder predicts six named tick heatmaps.
The needle branch separately loads rotation epoch 19 and stays fully frozen
during transfer, including normalization statistics. Fresh mode trains this
branch from random weights. Visibility weighting uses training labels only.

Default schedule: three epochs training new heads, then 37 epochs fine-tuning
the final ring encoder blocks. Batch size eight means 63 training batches per
epoch (the last contains four images) and 13 validation batches. Grayscale is
repeated into three channels and normalized for the MobileNet encoder.
The changed architecture starts a new optimizer once; later resumes restore
the complete V2 training state. The previous experiment is preserved.

Checkpoint selection uses validation tick error plus a penalty for geometric
readings outside +/-2 km/h, counting rejected readings as failures. Geometry
uses predicted tick references, visibility and needle endpoints; the known
camera homography and ground-truth landmarks are never given to inference.

Metrics and test discipline
---------------------------
Tick error in input pixels: lower is better.
Needle pivot/tip error in input pixels: lower is better.
Arc intersection over union: higher is better.
Visibility balanced accuracy: averages visible and hidden tick recall.
Speed MAE: average absolute error in km/h; lower is better.
Within +/-2 km/h: fraction of readings within the chosen tolerance.
Geometry coverage: fraction of inputs producing an accepted geometric reading.
All-input geometric success counts rejected readings as failures.

Inspect validation predictions and finish all model decisions before setting
RUN_FINAL_TEST to true. Run the settings cell, then final-test and export cells.
Test results are cached for the selected tensor state and exact dataset.
Do not repeatedly tune on this final test. Real labeled dashboard footage is
still required to assess the user's 98% within +/-2 km/h target.

Drive storage and restart
-------------------------
Every completed epoch, including epoch zero, has a full immutable checkpoint:
model, optimizer, scaler, random states, loader state, history and provenance.
resume_latest continues the most recent verified complete epoch.
resume_best rewinds to the best complete training state.
transfer initializes a new experiment from the two private source models.
fresh creates a unique experiment with random weights.
An interrupted epoch restarts from the preceding completed epoch.

Saved settings set a fixed total epoch target; running a finished experiment
does not add epochs. Dataset, source or runtime changes are rejected on exact
resume. A corrupt newest checkpoint can recover from a verified earlier one.
Best weights, metrics, predictions and source snapshots are exported to a ZIP;
full epoch checkpoints stay in Drive. Private model weights are not published.

Reproduction
------------
generate_dataset.py uses the included render_source assets.
Run it from the repository root using Python with NumPy, Pillow and OpenCV.
The generator also uses local ring_geometry_pilot validation code. Training
requires PyTorch, torchvision, NumPy, Pillow, OpenCV and matplotlib. The Colab
notebook embeds exact implementation files and validates their hashes.
The dataset archive is rooted at dataset_v2/.
