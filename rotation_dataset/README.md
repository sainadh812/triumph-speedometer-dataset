# Reproduce the rotations

Download and extract the v1 release into `bw_dataset_500/`. Install Python packages `numpy`, `opencv-python` and `Pillow`, then run from the repository root:

```sh
python rotation_dataset/generate_rotations.py --source bw_dataset_500 --output bw_dataset_rotation_v2 --archive speedometer_bw_rotation_v2.zip
```

The output must not already exist. The generator preserves all zero-angle inputs byte-for-byte and uses the same affine for pixels, keypoints, masks and homographies. It produces 6,000 main inputs and 1,200 intermediate-angle robustness inputs. Speed, original pose group and split remain unchanged.

Run the geometry/provenance tests after downloading v1 and generating v2:

```sh
python training_pilot/test_rotation_dataset.py
```

Image compression/library versions, preview fonts, timestamps and metadata can affect archive bytes; use the published checksums to identify the exact training release. The generator does not create new independent dashboard scenes or verify physical dial calibration.
