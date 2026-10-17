"""Report separate synthetic splits without using annotations in inference."""
from pathlib import Path
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from ring_training import ring_eval


def evaluate(model, loader, device):
    metrics, rows = ring_eval.evaluate(model, loader, device)
    splits = sorted({r['dataset_split'] for r in loader.dataset.records})
    metrics.update(scope='synthetic_' + '_'.join(splits), dataset_splits=splits,
                   limitation='All samples derive from one reconstructed source. Pose families are split before rendering; these are not independent real-camera tests.')
    confusion = metrics['landmarks']['visibility_confusion_counts']
    visible = confusion.get('visible_detected', 0) + confusion.get('visible_missed', 0)
    hidden = confusion.get('occluded_detected', 0) + confusion.get('occluded_called_visible', 0)
    visible_recall = confusion.get('visible_detected', 0) / visible if visible else None
    hidden_recall = confusion.get('occluded_detected', 0) / hidden if hidden else None
    metrics['landmarks']['visible_tick_recall_percent'] = None if visible_recall is None else visible_recall * 100
    metrics['landmarks']['visibility_balanced_accuracy_percent'] = (
        50 * (visible_recall + hidden_recall) if visible_recall is not None and hidden_recall is not None else None)
    return metrics, rows


save_evaluation = ring_eval.save_evaluation


def comparison_plot(before, after, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [
        ('Tick error: lower is better (px)', before['landmarks']['visible_tick_mean_error_px'], after['landmarks']['visible_tick_mean_error_px']),
        ('Arc overlap: higher is better', before['arc']['valid_pixel_iou'], after['arc']['valid_pixel_iou']),
        ('Direct speed error (km/h)', before['direct_speed']['mae_kmh'], after['direct_speed']['mae_kmh']),
        ('Geometry within ±2, all inputs (%)', before['geometric_speed']['within_2_kmh_all_inputs_percent'], after['geometric_speed']['within_2_kmh_all_inputs_percent']),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (label, a, b) in zip(axes, panels):
        bars = ax.bar(['Previous', 'Improved'], [a or 0, b or 0], color=['#93a4ba', '#149b8e'])
        ax.set_title(label, fontsize=10)
        ax.bar_label(bars, fmt='%.2f', padding=3)
        ax.set_ylim(bottom=0, top=max(a or 0, b or 0, .05) * 1.25)
    split = ', '.join(after.get('dataset_splits', []))
    fig.suptitle(f'Separate synthetic {split} set: {after["count"]} images', fontsize=15)
    fig.text(.5, .03, 'Shared source dashboard; results do not establish real-camera accuracy. Rejected geometry readings count as unsuccessful.', ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .09, 1, .9])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def history_plot(history_path, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = json.loads(Path(history_path).read_text())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    epochs, errors = [], []
    for row in history:
        metrics = row.get('validation_metrics', row.get('validation', row.get('val_metrics', {})))
        value = metrics.get('visible_tick_mean_error_px', metrics.get('tick_mean_error_px'))
        if value is None:
            value = metrics.get('landmarks', {}).get('visible_tick_mean_error_px')
        if value is not None:
            epochs.append(row['epoch']); errors.append(value)
    if epochs:
        ax.plot(epochs, errors, color='#149b8e', marker='.', label='Validation tick error')
        ax.legend()
    ax.set(xlabel='Completed epoch', ylabel='Mean visible tick error (pixels)', title='Tick learning on separate synthetic validation poses')
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output_path, dpi=140); plt.close(fig)


@torch.no_grad()
def prediction_sheet(model, dataset, device, path, limit=20):
    """Inspect a deterministic spread of examples; drawings never enter model input."""
    indices = np.linspace(0, len(dataset)-1, min(limit, len(dataset)), dtype=int)
    tile, gap, footer, top = 256, 12, 56, 75
    cols = 4
    sheet = Image.new('RGB', (cols*tile+(cols+1)*gap, top+int(np.ceil(len(indices)/cols))*(tile+footer+gap)), '#111d2d')
    draw = ImageDraw.Draw(sheet)
    def font(size):
        for name in ('DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf'):
            try: return ImageFont.truetype(name, size)
            except OSError: pass
        return ImageFont.load_default()
    split = dataset.records[0]['dataset_split']
    draw.text((gap, 10), f'PREDICTIONS | SYNTHETIC {split.upper()} SET', font=font(23), fill='white')
    draw.text((gap, 43), 'Cyan: arc | Yellow: predicted ticks | Green: true visible ticks | Pink: predicted needle', font=font(13), fill='#bbccdd')
    from ring_geometry_pilot.geometry_reader import load_template, read_speed, TICK_NAMES
    template = load_template(Path(dataset.root)/'source_annotations_snapshot.json', Path(dataset.root)/'calibration_snapshot.json')
    model.eval()
    for n, index in enumerate(indices):
        item = dataset[int(index)]; row = dataset.records[int(index)]
        output = model(item['image'].unsqueeze(0).to(device))
        needle = output['points'][0].cpu().numpy()*511
        ticks = output['tick_points'][0].cpu().numpy()*511
        visibility = output['tick_visibility_logits'][0].sigmoid().cpu().numpy()
        arc = output['arc_logits'][0, 0].sigmoid().cpu().numpy() >= .5
        im = Image.open(Path(dataset.root)/row['image']).convert('RGBA')
        rgba = np.zeros((512, 512, 4), np.uint8); rgba[arc] = [40, 217, 232, 85]
        im = Image.alpha_composite(im, Image.fromarray(rgba)).convert('RGB'); d = ImageDraw.Draw(im)
        d.line([tuple(p) for p in needle], fill='#ff66cc', width=2)
        for xy, valid in zip(item['tick_points'].numpy()*511, item['tick_valid'].numpy()):
            if valid:
                x,y=xy; d.line([(x-5,y),(x+5,y)],fill='#78ff83',width=2); d.line([(x,y-5),(x,y+5)],fill='#78ff83',width=2)
        for (x,y),v in zip(ticks,visibility):
            d.ellipse([x-4,y-4,x+4,y+4],outline='#ffe066' if v>=.5 else '#ff9944',width=2)
        observed = {name: ticks[k].tolist() for k,name in enumerate(TICK_NAMES) if visibility[k]>=.5}
        reading = read_speed(observed, *needle.tolist(), template)
        x=gap+n%cols*(tile+gap); y=top+n//cols*(tile+footer+gap)
        sheet.paste(im.resize((tile,tile),Image.Resampling.LANCZOS),(x,y))
        geom=f'{reading["speed_kmh"]:.1f}' if reading['accepted'] else 'rejected'
        draw.text((x,y+tile+5),f'Target {row["speed_kmh"]:g} | direct {float(output["speed"][0]):.1f}',font=font(14),fill='white')
        draw.text((x,y+tile+26),f'Geometry: {geom}',font=font(14),fill='#bbccdd')
    sheet.save(path)
