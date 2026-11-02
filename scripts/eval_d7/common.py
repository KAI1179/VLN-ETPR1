"""CPU-only paired statistics for complete R2R val_unseen evaluations."""

import gzip
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REPO = Path('/home/xukai/code/ETP-R1-snapshot/ETP-R1')
RUNS = REPO / 'data/logs/checkpoints'
METRICS = ('success', 'spl', 'oracle_success', 'ndtw')


def load_run(path):
    files = sorted((Path(path) / 'eval_results').glob('stats_ep_*_val_unseen_r*_w*.json'))
    if not files:
        raise FileNotFoundError(f'No episode results: {path}/eval_results')
    data = {}
    for file in files:
        part = json.loads(file.read_text())
        assert not (data.keys() & part.keys()), f'Duplicate episode IDs: {file}'
        data.update(part)
    assert len(data) == 1839, (path, len(data))
    return data


def scenes():
    path = REPO / 'data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz'
    with gzip.open(path, 'rt') as stream:
        episodes = json.load(stream)['episodes']
    return {str(ep['episode_id']): ep['scene_id'] for ep in episodes}


def compare(a, b, scene_ids):
    assert a.keys() == b.keys() == scene_ids.keys(), 'Episode sets differ'
    ids = sorted(a, key=int)
    x = np.array([a[i]['success'] > 0 for i in ids])
    y = np.array([b[i]['success'] > 0 for i in ids])
    only_a = int((x & ~y).sum())
    only_b = int((~x & y).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    groups = {}
    for index, eid in enumerate(ids):
        groups.setdefault(scene_ids[eid], []).append(index)
    indices = [np.array(v) for _, v in sorted(groups.items())]
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(2000):
        sample = np.concatenate([indices[j] for j in rng.integers(0, len(indices), len(indices))])
        draws.append(float((x[sample].mean() - y[sample].mean()) * 100))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float((x.mean() - y.mean()) * 100), float(lo), float(hi), only_a, only_b, float(p)


def metrics_row(label, data):
    values = [np.mean([v[k] for v in data.values()]) * 100 for k in METRICS]
    return '| ' + label + ' | ' + ' | '.join(f'{v:.2f}' for v in values) + f' | {len(data)} |'


def paired_row(label, a, b, scene_ids):
    delta, lo, hi, only_a, only_b, p = compare(a, b, scene_ids)
    return f'| {label} | {delta:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {only_a} / {only_b} | {p:.4g} |'
