"""Isolate image registration/calibration error using labeled needle points.

This is a geometry diagnostic, never a model accuracy score or inference shortcut.
Neither the known rendering H nor per-image target ticks enter the speed reader.
"""
from collections import Counter
from pathlib import Path
import json
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from real_adaptation_v3.needle_data import verify_dataset, read_speed


def main():
    root = PROJECT / 'real_adaptation_v3'
    private = root / 'real_review'
    # Public clones can inspect the public synthetic data without private frames.
    real_dir = private if (private / 'labels.json').is_file() else None
    partitions, integrity = verify_dataset(root / 'dataset500', real_dir, real_dir is not None)
    report = {'scope': 'Labeled needle points plus image-only estimated registration; no neural predictions, no claimed model accuracy',
              'uses_rendering_homography': False, 'uses_per_image_target_ticks': False,
              'dataset_integrity': integrity, 'partitions': {}}
    for split, records in partitions.items():
        errors, rejected, accepted, eligible, clamped = [], Counter(), 0, 0, 0
        for row in records:
            reading = read_speed(row['points_xy'], row['registration'], row['calibration'])
            eligible += int(row['speed_valid'])
            accepted += int(reading['accepted'])
            clamped += int(reading['endpoint_clamped'])
            if not reading['accepted']:
                rejected[reading['reason']] += 1
            elif row['speed_valid']:
                errors.append(abs(reading['speed_kmh'] - row['speed_kmh']))
        report['partitions'][split] = {
            'count': len(records), 'geometry_accepted': accepted, 'speed_label_eligible': eligible,
            'endpoint_clamped_count': clamped,
            'image_registration_accepted': sum(row['registration']['accepted'] for row in records),
            'accepted_and_eligible_count': len(errors),
            'mean_calibration_difference_kmh': float(np.mean(errors)) if errors else None,
            'p95_calibration_difference_kmh': float(np.percentile(errors, 95)) if errors else None,
            'max_calibration_difference_kmh': max(errors) if errors else None,
            'within_2_all_eligible_percent': 100 * sum(error <= 2 for error in errors) / max(eligible, 1),
            'rejection_reasons': dict(rejected)}
    (root / 'checks').mkdir(exist_ok=True)
    (root / 'checks/perfect_label_registration_diagnostic.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'counts': integrity['counts'], 'partitions': report['partitions']}, indent=2))


if __name__ == '__main__':
    main()
