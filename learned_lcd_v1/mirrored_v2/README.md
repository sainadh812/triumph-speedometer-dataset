# Learned LCD dataset: mirrored V2

This version preserves the original 4,426 LCD segmentation samples and adds 1,200 training images: 600 horizontal reflections and 600 vertical reflections. The original dataset and publication remain unchanged.

| Split | Original samples | Added mirrors | Total |
|---|---:|---:|---:|
| Train | 3,600 | 1,200 | 4,800 |
| Validation | 614 | 0 | 614 |
| Test | 212 | 0 | 212 |

The mirrors come from 600 distinct augmented training parents, 200 per source recording. Both reflection axes cover all 32 training source families and all twelve parent rotation bins. Mirrored images make up 25% of training. Validation and test rows and image/mask bytes are unchanged.

Each 512 × 512 grayscale image, LCD mask, valid-sensor mask, and available protected-analog mask is reflected identically. Horizontal reflection uses `x' = 511 - x`; vertical reflection uses `y' = 511 - y`. The recorded transform is `F @ parent_to_sample_h`. LCD polygons, full-frame footprints, and affected truncation-side metadata follow the same reflection. No new crop, ImageGen output, or model-generated label is used.

`mirror_axis`, `mirror_parent_id`, and `mirrored` identify the new rows. `rotation_deg` retains the parent's generating rotation; it does not describe a new physical camera angle. The original perspective augmentations remain separate from the reflection operation. **Mirroring reverses printed digits and is not equivalent to moving a real camera to the other side of the dashboard.** These are LCD segmentation labels; no speed or RPM targets are created or altered.

The runtime preprocessing ID is unchanged. Mirroring is a training augmentation only; the inference input is not reflected. The new manifest and preprocessing metadata declare the revised training dataset identity.

The base manifest SHA-256 is `b5bca7882b2f21671860203e7b16ff44ba0728cc0eaaa8f732bd534e20502835`. `dataset/provenance/mirror_lineage.json` records parent IDs and reflection matrices. The generator checks exact pixel flips and double-flip round trips, matrix/polygon/truncation round trips, mask containment, protected-region separation, original asset hashes, and unchanged original row bytes. Triplet review sheets and static audit reports are in `qa/`. Download hashes and archive parts are in `publication/`.

The independent real source count remains 58. Train and validation share three recordings through separated temporal blocks; test is one held-out recording with 12 real source frames. Augmentations are not additional independent real observations, and these checks do not establish model accuracy. Model training and video testing are performed in Colab.

Use [the V2 Colab notebook](https://colab.research.google.com/github/sainadh812/triumph-speedometer-dataset/blob/main/learned_lcd_v1/mirrored_v2/Triumph_Learned_LCD_V2_Mirrors_Colab.ipynb). It downloads this version from a pinned Git commit and checks the manifest, archive parts, and extracted dataset hashes. The experiment is named `learned_lcd_v2_mirrors`; every completed epoch saves full training state to Google Drive. Start with `imagenet`, or resume a compatible V2 detector checkpoint. The old V1 dataset has a different identity and cannot silently resume into this run.

Only the LCD detector is trained by this notebook. The existing V5 epoch-11 speed weights remain frozen. LCD data covers 360-degree roll, but the frozen speed model was trained around plus/minus 25 degrees; these mirrors do not establish speed performance at arbitrary viewpoints. At inference, the full video frame is neither mirrored nor cropped. A vertical mirror reflects the top and bottom of the layout and is distinct from a 180-degree camera rotation.

The public directory stores `download_manifest.json` and its archive parts side by side. The local build stores those same files under `publication/`. [Preview the original, horizontal, and vertical variants](qa/final_preview.jpg).
