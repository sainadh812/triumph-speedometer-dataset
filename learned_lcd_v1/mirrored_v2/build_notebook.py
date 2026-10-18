"""Build the separate mirrored LCD experiment without rewriting the V1 notebook."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from learned_lcd_v1.build_notebook import build


def build_mirrored(commit, manifest_sha):
    return build(
        commit, manifest_sha,
        output_dir=ROOT,
        notebook_name='Triumph_Learned_LCD_V2_Mirrors_Colab.ipynb',
        experiment_name='learned_lcd_v2_mirrors',
        dataset_subdir='mirrored_v2',
        title='Triumph — learned LCD V2 with mirrored training data',
        introduction='''This separate experiment selects **600 unique training parents** and adds one horizontal mirror and one vertical mirror of each: **600 horizontal + 600 vertical = 1,200 additional training examples**. The resulting dataset has **4,800 training, 614 validation, and 212 test images (5,626 total)**. Mirrors make up 25% of training (12.5% per axis). The validation and test sets remain unmirrored and unchanged; the original dataset and V1 notebook are preserved.

Horizontal mirroring reverses the printed digits left-to-right; vertical mirroring reflects the digits and dashboard layout top-to-bottom. A vertical mirror is **different from a 180° camera rotation**. These are synthetic reflections, **not the views obtained by moving a camera to the right or above the dashboard**. The image and LCD target mask are mirrored together.

Only the LCD dataset changes. The detector architecture, training procedure, validation gates, and frozen V5 epoch-11 speed weights are reused. This experiment starts a separate detector run from ImageNet by default. Resume modes require a checkpoint from this mirrored experiment; a V1 checkpoint has a different dataset identity.''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', default='SET_AFTER_PUBLICATION')
    parser.add_argument('--manifest-sha', default='SET_AFTER_DATASET_PUBLICATION')
    args = parser.parse_args()
    build_mirrored(args.commit, args.manifest_sha)
