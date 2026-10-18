# Upload a video and test the existing models

Run the three Colab cells **individually**, using the same Google Drive as the previous model runs:

1. **CPU:** choose your video through the browser. It is saved and verified in private Drive. `START_SECONDS=0` and `MAX_SECONDS=0` select the entire video.
2. **T4 GPU:** reconnect and run cell 2. It reloads the uploaded job, adds a large RPM label and raw speed, saves the verified MP4 to Drive, and disconnects the GPU automatically.
3. **CPU:** reconnect and run cell 3 to download the saved MP4. Keep the CPU session connected until the browser finishes downloading.

This reuses LCD epoch9, speed epoch11 and RPM epoch9. No training occurs. RPM positioning uses the learned LCD outline rather than SIFT; its upright-corner assumption is not reliable for arbitrary camera orientations. The speed model receives the full unmasked frame in this diagnostic despite having been trained with a gray LCD mask. Readings are unvalidated and can be wrong.

Only code and this notebook are public. Input videos, pretrained weights and outputs stay in private Drive. This notebook requires the existing `SpeedometerTraining` assets; it does not provide an anonymous public model download. A missing-asset error means the mounted Drive does not contain the previous runs.

The Drive result folder and its receipt remain available after GPU disconnect. The final download runs on CPU, so releasing the GPU cannot interrupt a runtime-served browser download. Failed runs preserve diagnostic files and are never advertised as completed videos.

For publication, first commit the four runtime Python files, then run `build_notebook.py --code-commit <full-commit-sha>` and publish the generated notebook in a second commit. The builder pins the immutable code commit and the exact SHA256/size of each runtime file. It performs no model or video execution.
