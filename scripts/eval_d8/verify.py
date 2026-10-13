"""CPU checks for hook equivalence and exact own-rollout reproduction."""

import json
from pathlib import Path

from tap import Tap

ROOT = Path('/home/xukai/code/ETP-R1-snapshot/ETP-R1')


class Args(Tap):
    reference: str
    candidate: str
    output: str
    expected: int = 1839


def main():
    args = Args().parse_args()
    def read(directory):
        rows = {}
        for path in sorted(Path(directory).glob('stats_ep_*_val_unseen_r*_w*.json')):
            data = json.loads(path.read_text())
            if rows.keys() & data.keys():
                raise ValueError(f'Duplicate episode IDs: {path}')
            rows.update(data)
        if len(rows) != args.expected:
            raise ValueError(f'Expected {args.expected} episodes at {directory}, found {len(rows)}')
        return rows
    reference, candidate = read(args.reference), read(args.candidate)
    if reference.keys() != candidate.keys():
        raise ValueError('Episode sets differ')
    keys = ['success', 'oracle_success', 'spl', 'steps_taken', 'distance_to_goal']
    changes = [{ 'episode': eid, 'field': key, 'reference': reference[eid][key], 'candidate': candidate[eid][key] }
               for eid in reference for key in keys if reference[eid][key] != candidate[eid][key]]
    result = {'n': len(reference), 'fields': keys, 'changes': changes,
              'reference': args.reference, 'candidate': args.candidate}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    if changes:
        raise ValueError(f'Hook/reproduction mismatch: {len(changes)} values; see {args.output}')
    print(f'PASS: {len(reference)} episodes, all five fields identical', flush=True)


if __name__ == '__main__':
    main()
