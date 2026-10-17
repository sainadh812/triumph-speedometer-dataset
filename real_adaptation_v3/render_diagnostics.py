"""Private visual comparison of actual model predictions, never model inputs."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    for name in ('DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_real_comparison(real_data_dir, before_rows, after_rows, output_path):
    """Render paired baseline/best needle detections on private masked frames.

    Yellow: manually estimated reference needle. Cyan: model predicted needle.
    No predicted tick/ring is drawn because this model detects needle endpoints.
    """
    root = Path(real_data_dir)
    annotations = {row['frame_index']: row for row in json.loads((root / 'labels.json').read_text(encoding='utf-8'))['frames']}
    previous = {row['sample_id']: row for row in before_rows}
    if not after_rows or set(previous) != {row['sample_id'] for row in after_rows}:
        raise ValueError('Need matching nonempty before/after sample IDs')
    columns, pair_width, pair_height, header = 3, 512, 330, 70
    rows = (len(after_rows) + columns - 1) // columns
    canvas = Image.new('RGB', (columns * pair_width, header + rows * pair_height), (20, 23, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), 'Needle detection: before / selected best', fill='white', font=_font(24))
    draw.text((12, 40), 'Cyan = prediction | Yellow = manual visual draft | Same video development check, not an independent test', fill=(225, 225, 225), font=_font(16))
    for index, after in enumerate(after_rows):
        ann = annotations[after['source_frame']]
        before = previous[after['sample_id']]
        x, y = index % columns * pair_width, header + index // columns * pair_height
        for side, result in enumerate((before, after)):
            tile = Image.open(root / ann['image']).convert('RGB')
            ink = ImageDraw.Draw(tile)
            ground = [tuple(ann['pivot_xy']), tuple(ann['tip_xy'])]
            predicted = [tuple(result['predicted_pivot_xy']), tuple(result['predicted_tip_xy'])]
            ink.line(ground, fill=(255, 220, 40), width=3)
            ink.line(predicted, fill=(0, 240, 255), width=3)
            for point in ground:
                px, py = point
                ink.ellipse((px-5, py-5, px+5, py+5), outline=(255, 220, 40), width=2)
            for point in predicted:
                px, py = point
                ink.ellipse((px-4, py-4, px+4, py+4), fill=(0, 240, 255))
            ox = x + side * 256
            canvas.paste(tile.resize((256, 256), Image.Resampling.LANCZOS), (ox, y+24))
            draw.text((ox+5, y+2), f"Frame {after['source_frame']} | {'BEFORE' if side == 0 else 'AFTER'}", fill='white', font=_font(15))
            estimate = result.get('primary_speed_kmh')
            text = f'{estimate:.1f} km/h' if result.get('geometry_accepted') and estimate is not None else 'Reading rejected'
            tip_error = result.get('tip_error_px')
            error = f'Tip error: {tip_error:.1f}px' if tip_error is not None else 'Glare: reference uncertain'
            draw.text((ox+5, y+283), text, fill=(0, 240, 255), font=_font(15))
            draw.text((ox+5, y+305), error, fill=(230, 230, 230), font=_font(14))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return str(destination)
