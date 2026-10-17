"""Image-derived geometry evaluation. Ground-truth ticks never enter the reader."""
from pathlib import Path
from collections import Counter
import csv
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
for folder in (HERE.parent / 'ring_geometry_pilot', HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from geometry_reader import load_template, read_speed, TICK_NAMES


def _summary(errors):
    e = np.asarray(errors, dtype=float)
    if not len(e):
        return {'count': 0, 'mae_kmh': None, 'rmse_kmh': None, 'p95_kmh': None,
                'within_2_kmh_percent': None, 'within_5_kmh_percent': None}
    return {'count': len(e), 'mae_kmh': float(e.mean()),
            'rmse_kmh': float(np.sqrt(np.mean(e * e))), 'p95_kmh': float(np.percentile(e, 95)),
            'within_2_kmh_percent': float((e <= 2).mean() * 100),
            'within_5_kmh_percent': float((e <= 5).mean() * 100)}


@torch.no_grad()
def evaluate(model, dataloader, device, template=None, visibility_threshold=0.5):
    dataset = dataloader.dataset
    if template is None:
        root = Path(dataset.root)
        template = load_template(root / 'source_annotations_snapshot.json', root / 'calibration_snapshot.json')
    model.eval()
    rows, direct_errors, geometry_errors, tick_errors, needle_errors = [], [], [], [], []
    intersections = unions = predicted_pixels = actual_pixels = 0
    vis_correct = vis_total = 0
    visibility_counts = Counter()
    reasons = Counter()
    for batch in dataloader:
        output = model(batch['image'].to(device))
        needle = output['points'].float().cpu().numpy()
        ticks = output['tick_points'].float().cpu().numpy()
        visibility = output['tick_visibility_logits'].float().sigmoid().cpu().numpy()
        direct = output['speed'].float().cpu().numpy()
        arc = output['arc_logits'].float().sigmoid().cpu().numpy()[:, 0] >= 0.5
        gt_arc = batch['arc_mask'].numpy()[:, 0] > 0.5
        valid = batch['arc_valid'].numpy()[:, 0] > 0.5
        for j, raw_index in enumerate(batch['index']):
            index = int(raw_index)
            row = dataset.records[index]
            limits = np.array([row['width'] - 1, row['height'] - 1], float)
            p, t = needle[j] * limits
            predicted_ticks = ticks[j] * limits
            observed = {name: predicted_ticks[k].tolist() for k, name in enumerate(TICK_NAMES)
                        if visibility[j, k] >= visibility_threshold}
            # Only image-derived coordinates and predicted visibility are used.
            reading = read_speed(observed, p.tolist(), t.tolist(), template)
            speed = float(row['speed_kmh'])
            error = abs(float(direct[j]) - speed)
            direct_errors.append(error)
            if reading['accepted']:
                geom_error = abs(reading['speed_kmh'] - speed)
                geometry_errors.append(geom_error)
            else:
                geom_error = None
                reasons[reading['reason']] += 1
            gt_points = batch['points'][j].numpy() * limits
            n_error = np.linalg.norm(np.stack([p, t]) - gt_points, axis=1)
            needle_errors.append(n_error.tolist())
            tick_valid = batch['tick_valid'][j].numpy().astype(bool)
            gt_ticks = batch['tick_points'][j].numpy() * limits
            errors = np.linalg.norm(predicted_ticks - gt_ticks, axis=1)
            tick_errors.extend(errors[tick_valid].tolist())
            vis_correct += int(((visibility[j] >= visibility_threshold) == tick_valid).sum())
            vis_total += len(tick_valid)
            predicted_visible = visibility[j] >= visibility_threshold
            visibility_counts['visible_detected'] += int((predicted_visible & tick_valid).sum())
            visibility_counts['visible_missed'] += int((~predicted_visible & tick_valid).sum())
            visibility_counts['occluded_detected'] += int((~predicted_visible & ~tick_valid).sum())
            visibility_counts['occluded_called_visible'] += int((predicted_visible & ~tick_valid).sum())
            a = arc[j] & valid[j]
            b = gt_arc[j] & valid[j]
            inter, union = int((a & b).sum()), int((a | b).sum())
            intersections += inter
            unions += union
            predicted_pixels += int(a.sum())
            actual_pixels += int(b.sum())
            rows.append({'sample_id': row['sample_id'], 'image': row['image'],
                         'target_speed_kmh': speed, 'direct_speed_kmh': float(direct[j]),
                         'direct_error_kmh': error, 'geometry_speed_kmh': reading['speed_kmh'],
                         'geometry_error_kmh': geom_error, 'geometry_accepted': reading['accepted'],
                         'geometry_reason': reading['reason'], 'geometry_input_kind': 'predicted coordinates and visibility only',
                         'predicted_pivot_xy': p.tolist(), 'predicted_tip_xy': t.tolist(),
                         'predicted_tick_xy': predicted_ticks.tolist(),
                         'predicted_tick_visibility': visibility[j].tolist(),
                         'visible_tick_error_px': float(errors[tick_valid].mean()) if tick_valid.any() else None,
                         'arc_iou': inter / max(union, 1),
                         'camera_roll_degrees': row.get('camera', {}).get('clockwise_rotation_degrees'),
                         'inferred_60_influence': bool(row.get('inferred_60_influence', False))})
    n = len(rows)
    if not n:
        raise ValueError('Evaluation requires nonempty data')
    splits = {r.get('dataset_split') for r in dataset.records}
    scope = 'training_reconstruction_check' if splits == {'review_only'} else 'synthetic_evaluation'
    ne = np.asarray(needle_errors)
    metrics = {
        'scope': scope, 'count': n, 'direct_speed': _summary(direct_errors),
        'geometric_speed': {'accepted': len(geometry_errors), 'rejected': n - len(geometry_errors),
                            'coverage_percent': 100 * len(geometry_errors) / n,
                            'within_2_kmh_all_inputs_percent': 100 * sum(e <= 2 for e in geometry_errors) / n,
                            'accepted_only': _summary(geometry_errors), 'rejection_reasons': dict(reasons)},
        'landmarks': {'visible_tick_mean_error_px': float(np.mean(tick_errors)) if tick_errors else None,
                      'visible_tick_p95_error_px': float(np.percentile(tick_errors, 95)) if tick_errors else None,
                      'needle_pivot_mean_error_px': float(ne[:, 0].mean()),
                      'needle_tip_mean_error_px': float(ne[:, 1].mean()),
                      'tick_visibility_accuracy_percent': 100 * vis_correct / max(vis_total, 1),
                      'visibility_confusion_counts': dict(visibility_counts),
                      'occluded_tick_recall_percent': 100 * visibility_counts['occluded_detected'] / max(
                          visibility_counts['occluded_detected'] + visibility_counts['occluded_called_visible'], 1)},
        'arc': {'valid_pixel_iou': intersections / max(unions, 1),
                'valid_pixel_dice': 2 * intersections / max(predicted_pixels + actual_pixels, 1)},
        'visibility_threshold': visibility_threshold,
        'inference_uses_ground_truth_visibility': False,
        'inference_uses_oracle_camera_transform': False,
        'limitation': 'Review reconstruction is not held-out accuracy. Visibility scores and geometry gates are not calibrated probabilities.',
    }
    return metrics, rows


def save_evaluation(folder, prefix, metrics, rows):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{prefix}_metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (folder / f'{prefix}_predictions.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    fields = ['sample_id', 'target_speed_kmh', 'direct_speed_kmh', 'direct_error_kmh', 'geometry_speed_kmh',
              'geometry_error_kmh', 'geometry_accepted', 'geometry_reason', 'visible_tick_error_px', 'arc_iou']
    with (folder / f'{prefix}_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def comparison_plot(before, after, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [
        ('Visible tick error (pixels)', before['landmarks']['visible_tick_mean_error_px'], after['landmarks']['visible_tick_mean_error_px']),
        ('Arc overlap (IoU, higher is better)', before['arc']['valid_pixel_iou'], after['arc']['valid_pixel_iou']),
        ('Direct speed MAE (km/h)', before['direct_speed']['mae_kmh'], after['direct_speed']['mae_kmh']),
        ('Geometry coverage (%)', before['geometric_speed']['coverage_percent'], after['geometric_speed']['coverage_percent']),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (title, a, b) in zip(axes, panels):
        bars = ax.bar(['Before', 'After'], [a or 0, b or 0], color=['#9babc1', '#199f9a'])
        ax.set_title(title, fontsize=10)
        ax.bar_label(bars, fmt='%.2f', padding=3)
        ax.margins(y=0.2)
    fig.suptitle('20-image learning check — same images used for training and evaluation', fontsize=14)
    fig.text(.5, .02, 'These results show label learning only; they do not establish unseen-image or real-camera accuracy.', ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .07, 1, .91])
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def prediction_sheet(model, dataset, device, path, template=None):
    """Show all 20 model predictions with truth needle endpoints for inspection."""
    from torch.utils.data import DataLoader
    _, rows = evaluate(model, DataLoader(dataset, batch_size=4, shuffle=False), device, template)
    model.eval()
    tile, gap, footer, top = 256, 12, 65, 85
    cols = 4
    nrows = (len(rows) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tile + (cols + 1) * gap, top + nrows * (tile + footer + gap)), '#111d2d')
    draw = ImageDraw.Draw(sheet)
    def font(size):
        for name in ('DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf'):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
        return ImageFont.load_default()
    draw.text((gap, 12), 'MODEL PREDICTIONS | 20-IMAGE LEARNING CHECK', font=font(24), fill='white')
    draw.text((gap, 48), 'Cyan: predicted arc | Yellow: ticks | Pink: needle | Green crosses: true needle | NOT a test set', font=font(13), fill='#b9cbe1')
    for i, row in enumerate(rows):
        item = dataset[i]
        output = model(item['image'].unsqueeze(0).to(device))
        mask = output['arc_logits'][0, 0].sigmoid().cpu().numpy() >= .5
        original = Image.open(Path(dataset.root) / row['image']).convert('RGB')
        rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        rgba[mask] = [40, 217, 232, 90]
        im = Image.alpha_composite(original.convert('RGBA'), Image.fromarray(rgba)).convert('RGB')
        d = ImageDraw.Draw(im)
        d.line([tuple(row['predicted_pivot_xy']), tuple(row['predicted_tip_xy'])], fill='#ff66cc', width=2)
        for xy in (item['points'].numpy() * 511):
            x, y = xy
            d.line([(x - 4, y), (x + 4, y)], fill='#78ff83', width=2)
            d.line([(x, y - 4), (x, y + 4)], fill='#78ff83', width=2)
        for k, (x, y) in enumerate(row['predicted_tick_xy']):
            col = '#ffe066' if row['predicted_tick_visibility'][k] >= .5 else '#ff9944'
            d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=col, width=2)
        x = gap + (i % cols) * (tile + gap)
        y = top + (i // cols) * (tile + footer + gap)
        sheet.paste(im.resize((tile, tile), Image.Resampling.LANCZOS), (x, y))
        geom = f"{row['geometry_speed_kmh']:.1f}" if row['geometry_accepted'] else 'rejected'
        draw.text((x, y + tile + 6), f"#{i+1:02d} target {row['target_speed_kmh']:g} km/h", font=font(15), fill='white')
        draw.text((x, y + tile + 29), f"Direct {row['direct_speed_kmh']:.1f} | geometry {geom}", font=font(13), fill='#b9cbe1')
    sheet.save(path)
