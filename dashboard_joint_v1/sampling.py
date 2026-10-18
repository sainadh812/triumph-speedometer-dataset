"""Pure lighting/source sampling policy; no neural or PyTorch dependency."""
from __future__ import annotations

from collections import Counter
import math


def lighting_name(value):
    """Legacy samples without lighting remain usable, but cannot set a mix."""
    return value if isinstance(value, str) and value in ('day', 'night') else 'unknown'


def training_sampling_plan(rows, daylight_fraction=None):
    """Return row probabilities and an auditable plan (None means old shuffle).

    In fraction mode, first choose lighting, then a source uniformly within that
    lighting, then one of that source's rows uniformly. Extra augmentations or
    synthetic states of a source therefore do not increase its total weight.
    Actual epoch fractions fluctuate around the requested expectation; drawing
    len(rows) times with replacement is the sampler's responsibility.
    """
    rows = list(rows)
    if not rows:
        raise ValueError('Cannot sample an empty training split')
    counts = Counter(lighting_name(row.get('lighting')) for row in rows)
    plan = {'mode': 'shuffle_without_replacement', 'epoch_samples': len(rows),
            'daylight_fraction': daylight_fraction, 'row_counts': dict(counts),
            'replacement': False, 'validation_test_sampling': 'unchanged sequential full splits'}
    if daylight_fraction is None:
        return None, plan
    if (isinstance(daylight_fraction, bool) or not isinstance(daylight_fraction, (int, float)) or
            not math.isfinite(daylight_fraction) or not 0 < daylight_fraction < 1):
        raise ValueError('daylight_fraction must be None or a finite number strictly between 0 and 1')
    if counts.get('unknown') or not counts.get('day') or not counts.get('night'):
        raise ValueError('Daylight fraction sampling requires day and night labels on every training row, with both groups present')
    source_counts, source_lighting = Counter(), {}
    for row in rows:
        source = row.get('source_id')
        if not isinstance(source, str) or not source:
            raise ValueError('Lighting/source sampling requires a nonempty source_id on every row')
        light = row['lighting']
        if source_lighting.setdefault(source, light) != light:
            raise ValueError('One source_id cannot belong to different lighting groups')
        source_counts[source] += 1
    sources_per_light = Counter(source_lighting.values())
    mass = {'day': float(daylight_fraction), 'night': 1. - float(daylight_fraction)}
    weights = [mass[row['lighting']] / sources_per_light[row['lighting']] / source_counts[row['source_id']]
               for row in rows]
    plan.update(mode='lighting_then_uniform_source_then_uniform_row', replacement=True,
                source_counts=dict(sources_per_light), expected_lighting_fraction=mass,
                expected_lighting_samples={name: fraction * len(rows) for name, fraction in mass.items()},
                rng='Shared train DataLoader generator, saved in every epoch checkpoint',
                fraction_note='Expected fraction under replacement sampling; realized epoch counts can vary.')
    return weights, plan


def batch_lighting_groups(values, size):
    """Group indices without shuffling evaluation inputs or inferring lighting."""
    if values is None:
        values = ['unknown'] * size
    if len(values) != size:
        raise ValueError('Batch lighting metadata count differs from batch size')
    groups = {}
    for index, value in enumerate(values):
        groups.setdefault(lighting_name(value), []).append(index)
    return groups
