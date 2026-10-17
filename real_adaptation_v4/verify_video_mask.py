"""Audit the published masked reference against every frame of the source clip."""
from pathlib import Path
import argparse
import collections
import json
import cv2
import numpy as np
from real_adaptation_v4.stabilize_mask import FixedViewPreprocessor, prepare_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    cfg = json.loads((args.dataset/'preprocessing_config.json').read_text())
    cal = json.loads((args.dataset/'calibration.json').read_text())
    reference = cv2.imread(str(args.dataset/cfg['reference_image_path']), 0)
    processor = FixedViewPreprocessor(reference, cfg['lcd_polygon_reference_xy'],
        [tick['xy'] for tick in cal['ticks']], cal['pivot_xy'], cfg['lcd_mask_value'])
    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    records, tiles = [], []
    selected = {0, 96, 144, 240, 360, 432, 480, 599}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index = len(records)
        prepared = prepare_frame(frame)
        masked, meta = processor.process(prepared)
        records.append({'frame_index': index, **meta})
        if masked is not None:
            assert np.all(masked[processor.fixed_mask > 0] == 128)
        if index in selected:
            view = prepared if masked is None else masked
            cv2.imwrite(str(args.output/f'masked_{index:04d}.png'), view)
            tile = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
            cv2.putText(tile, f'Frame {index} | {meta["reason"]}', (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (40, 230, 80), 1, cv2.LINE_AA)
            tiles.append(tile)
        if len(records) % 100 == 0:
            print(f'Checked {len(records)} frames', flush=True)
    cap.release()
    report = {'frames': len(records), 'accepted': sum(r['accepted'] for r in records),
        'rejection_reasons': dict(collections.Counter(r['reason'] for r in records if not r['accepted'])),
        'reference_sha256': cfg['reference_sha256'], 'records': records}
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if len(tiles) == 8:
        cv2.imwrite(str(args.output/'contact.jpg'), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
