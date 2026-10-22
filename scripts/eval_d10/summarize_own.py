"""Summarize Z's own-policy metrics, never the expert S rollout metrics."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = {}
    sources = []
    for rank in range(4):
        directory = ROOT / f'data/logs/checkpoints/d10_zero_O_rank{rank}/eval_results'
        paths = list(directory.glob('stats_ep_ckpt_28000_val_unseen_r0_w1.json'))
        if len(paths) != 1:
            raise ValueError(f'Expected one logical-rank result: {directory}')
        path = paths[0]
        part = json.loads(path.read_text())
        if rows.keys() & part.keys():
            raise ValueError(f'Duplicate episode IDs: {path}')
        rows.update(part)
        sources.append(str(path))
    if len(rows) != 1839:
        raise ValueError(f'Expected 1839 own episodes, got {len(rows)}')
    fields = ('success', 'spl', 'oracle_success', 'ndtw', 'distance_to_goal',
              'path_length', 'steps_taken')
    metrics = {key: sum(row[key] for row in rows.values()) / len(rows)
               for key in fields}
    missing = sum(row.get('llm_cache_missing', 0) for row in rows.values())
    result = {
        'protocol': 'own; no expert action replay', 'episodes': len(rows),
        'metrics': metrics, 'reference': {'SR': 63.30, 'SPL': 53.20},
        'logical_world_size': 4, 'environments_per_rank': 4, 'sources': sources,
        'cache_missing_episodes': missing,
    }
    output = ROOT / 'reports/d10/own_metrics.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"Z own: SR {100*metrics['success']:.2f}, SPL {100*metrics['spl']:.2f}; archived 63.30/53.20")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    main()
