"""Prepare/render each split's existing top20 list, one case at a time."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from tap import Tap

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.analysis.paper_cases import materials


class Args(Tap):
    render: bool = False


def main():
    args = Args().parse_args()
    inventory = []
    for split, table, output in (
        ('val_unseen', ROOT / 'reports/paper_cases/candidates.json', ROOT / 'vis'),
        ('val_seen', ROOT / 'reports/paper_cases/val_seen/candidates.json', ROOT / 'vis/val_seen'),
    ):
        selected = json.loads(table.read_text())['candidates'][:20]
        ids = [str(row['episode_id']) for row in selected]
        if len(set(ids)) != 20:
            raise ValueError(f'Expected 20 unique candidate episodes: {table}')
        missing = []
        for eid in ids:
            path = output / eid / 'meta.json'
            if not path.exists():
                missing.append(eid)
            elif json.loads(path.read_text()).get('split', 'val_unseen') != split:
                raise ValueError(f'Manifest split mismatch: {path}')
        if missing:
            materials.prepare(SimpleNamespace(split=split, episodes=missing, output=str(output)))
        if args.render:
            materials.render(SimpleNamespace(split=split, episodes=ids, output=str(output), gpu_device_id=0))
        for rank, eid in enumerate(ids, 1):
            path = output / eid / 'meta.json'
            meta = json.loads(path.read_text())
            inventory.append({'split': split, 'rank': rank, 'episode_id': eid,
                              'source_table': str(table), 'directory': str(path.parent),
                              'render_status': meta['render_status'],
                              'frames': {arm: len(model['frames']) for arm, model in meta['models'].items()}})
    materials.write_json(ROOT / 'vis/top20_manifest.json', inventory)
    completed = sum(row['render_status'] == 'complete' for row in inventory)
    print(f'Top20 inventory: {len(inventory)} cases, {completed} rendered; {ROOT / "vis/top20_manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
