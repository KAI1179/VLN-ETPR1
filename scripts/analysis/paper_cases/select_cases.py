"""Select paired paper cases using existing CPU map-quality metrics."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import List

from tap import Tap

ROOT = Path(__file__).resolve().parents[3]
TURN_CUE = re.compile(
    r"\b(?:turn|bear|veer)\s+(?:to\s+the\s+)?(?:left|right)\b"
    r"|\b(?:take|make)\s+(?:a\s+|the\s+)?(?:left|right)(?:\s+turn)?\b",
    re.IGNORECASE,
)


class Args(Tap):
    exclude: List[str]
    """Episode IDs confirmed by the user to appear in the existing attention figure."""
    ours: str = str(ROOT / 'results/ours_val_unseen_episodes.jsonl')
    zeromap: str = str(ROOT / 'results/zeromap_val_unseen_episodes.jsonl')
    features: str = str(ROOT / 'data/logs/analysis/features_val_unseen.json')
    quality_dir: str = str(ROOT / 'reports/map_quality')
    output: str = str(ROOT / 'reports/paper_cases')


def read_rows(path, condition=None):
    rows = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if condition is not None and row['condition'] != condition:
                continue
            eid = str(row['episode_id'])
            if eid in rows:
                raise ValueError(f'Duplicate episode {eid}: {path}')
            rows[eid] = row
    return rows


def mentioned_quality(metrics):
    counts = {
        suffix: sum(metrics[f'mentioned_{group}_category_{suffix}']
                    for group in ('object', 'region'))
        for suffix in ('true_positive_count', 'predicted_count', 'target_count')
    }
    denominator = counts['predicted_count'] + counts['target_count']
    return {
        'pooled_mentioned_category_f1': 2 * counts['true_positive_count'] / denominator
        if denominator else 0.,
        'mentioned_object_f1': metrics['mentioned_object_category_f1'],
        'mentioned_region_f1': metrics['mentioned_region_category_f1'],
        'pooled_counts': counts,
    }


def screen(ours, zeromap, features, validity, presence, excluded):
    if ours.keys() != zeromap.keys() or len(ours) != 1839:
        raise ValueError('Screening requires two complete, matching 1839-episode val_unseen tables')
    candidates = []
    counts = {'paired_episodes': len(ours), 'ours_success_zero_failure': 0,
              'after_explicit_exclusions': 0, 'valid_json_and_schema': 0,
              'GT_turns_ge2_and_GT_regions_ge2': 0, 'ours_high_steps_8_to15': 0}
    for eid in sorted(ours, key=int):
        a, b = ours[eid], zeromap[eid]
        if a['success'] != 1 or b['success'] != 0:
            continue
        counts['ours_success_zero_failure'] += 1
        if eid in excluded:
            continue
        counts['after_explicit_exclusions'] += 1
        valid = validity[eid]
        if valid['json_valid'] is not True or valid['schema_valid'] is not True:
            continue
        counts['valid_json_and_schema'] += 1
        feat = features[eid]
        if feat['n_turns'] is None or feat['n_rooms'] is None:
            raise ValueError(f'Missing GT route features: {eid}')
        if feat['n_turns'] < 2 or feat['n_rooms'] < 2:
            continue
        counts['GT_turns_ge2_and_GT_regions_ge2'] += 1
        if not 8 <= a['high_level_steps'] <= 15:
            continue
        counts['ours_high_steps_8_to15'] += 1
        instruction = a['instruction']
        if instruction != b['instruction']:
            raise ValueError(f'Instruction mismatch: {eid}')
        cues = [match.group(0) for match in TURN_CUE.finditer(instruction)]
        candidates.append({
            'episode_id': eid, 'scene': feat['scene'], 'instruction': instruction,
            'quality': {**mentioned_quality(presence[eid]['metrics']),
                        'json_valid': True, 'schema_valid': True,
                        'source_zero_filled': presence[eid]['source_zero_filled']},
            'GT_route_features': {key: feat[key] for key in
                                  ('n_turns', 'n_rooms', 'n_room_changes', 'path_len_m')},
            'lexical_turn_cue_count': len(cues), 'lexical_turn_cues': cues,
            'instruction_layout_review': 'pending human review; GT complexity is not a text-semantic label',
            'ours': {key: a[key] for key in
                     ('success', 'SPL', 'NE', 'OSR', 'nDTW', 'steps', 'high_level_steps', 'source_eval')},
            'zeromap': {key: b[key] for key in
                        ('success', 'SPL', 'NE', 'OSR', 'nDTW', 'steps', 'high_level_steps', 'source_eval')},
        })
    candidates.sort(key=lambda row: (
        -row['quality']['pooled_mentioned_category_f1'],
        -row['GT_route_features']['n_turns'], -row['GT_route_features']['n_rooms'],
        abs(row['ours']['high_level_steps'] - 10), int(row['episode_id'])))
    return candidates[:20], counts


def write_report(result, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'candidates.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Paper qualitative-case shortlist', '',
             'CPU screening only. No model inference, simulator or GPU is used.', '',
             '## Definitions and provenance', '',
             *[f'- **{key}:** {value}' for key, value in result['definitions'].items()], '',
             '```json', json.dumps(result['sources'], indent=2), '```', '',
             '## Filter counts', '', '```json', json.dumps(result['counts'], indent=2), '```', '',
             '## Top candidates', '',
             '| Episode | Mentioned F1 (pooled / object / region) | JSON / schema | GT turns / distinct regions / crossings | Lexical turn cues | High-level steps (ours / zero) | Low-level steps (ours / zero) | Ours SR / SPL / NE | Zero SR / SPL / NE |',
             '|---|---|---|---|---|---|---|---|---|']
    for row in result['candidates']:
        q, f, a, b = row['quality'], row['GT_route_features'], row['ours'], row['zeromap']
        lines.append(
            f"| {row['episode_id']} | {q['pooled_mentioned_category_f1']:.4f} / {q['mentioned_object_f1']:.4f} / {q['mentioned_region_f1']:.4f} | valid / valid | "
            f"{f['n_turns']} / {f['n_rooms']} / {f['n_room_changes']} | {row['lexical_turn_cue_count']} | "
            f"{a['high_level_steps']} / {b['high_level_steps']} | {a['steps']} / {b['steps']} | "
            f"{a['success']:.0f} / {a['SPL']:.4f} / {a['NE']:.3f}m | {b['success']:.0f} / {b['SPL']:.4f} / {b['NE']:.3f}m |")
    if len(result['candidates']) < 20:
        lines += ['', f"Only {len(result['candidates'])} candidates satisfy every implemented condition; no thresholds were relaxed."]
    lines += ['', '## Full instructions for layout review', '']
    for row in result['candidates']:
        lines += [f"### Episode {row['episode_id']}", '', row['instruction'].strip(), '',
                  f"Lexical turn cues: {row['lexical_turn_cues']}. Text-semantic review remains pending.", '']
    lines += ['## Next action', '',
              'Choose two episodes after reviewing the instructions and trajectories. Rerun only those episodes for panoramic/topological assets; this CPU screen does not render them.', '']
    (directory / 'candidates.md').write_text('\n'.join(lines))


def main():
    args = Args(underscores_to_dashes=True).parse_args()
    ours, zeromap = read_rows(args.ours), read_rows(args.zeromap)
    features = json.loads(Path(args.features).read_text())
    quality_dir = Path(args.quality_dir)
    validity = read_rows(quality_dir / 't0_val_unseen_episodes.jsonl')
    presence = read_rows(quality_dir / 't1_val_unseen_episodes.jsonl', 'LLM')
    excluded = set(map(str, args.exclude))
    if len(excluded) != 2 or not excluded <= ours.keys():
        raise ValueError('Supply the two confirmed val_unseen episode IDs used in attn_case.pdf')
    selected, counts = screen(ours, zeromap, features, validity, presence, excluded)
    result = {
        'definitions': {
            'paired_outcome': 'ours success=1, zeromap success=0',
            'excluded_episodes': sorted(excluded, key=int),
            'exclusion_source': 'explicit --exclude input; no automatic guess of attn_case.pdf IDs',
            'map_quality': 'JSON and schema both valid; rank pooled mentioned-category F1 descending, no arbitrary F1 cutoff',
            'mentioned_F1': '2*(objectTP+regionTP)/(objectPred+regionPred+objectTarget+regionTarget); empty denominator=0',
            'mentions': 'original GT mention flags and category-presence metrics, not LLM mentioned flags',
            'route_complexity': 'GT n_turns>=2 AND distinct MP3D GT region instances n_rooms>=2; not an instruction-semantic assertion',
            'turn_feature': 'dense GT direction cosine<0.8, consecutive triggers merged; original episode_features.py',
            'lexical_cues': TURN_CUE.pattern,
            'length': '8<=ours high_level_steps<=15; not low-level steps or trajectory coordinate count',
            'ranking': 'mentioned pooled F1, GT turns, GT rooms, closeness to10 high-level steps, then episode ID',
            'metric_units': 'SR and SPL fractions; NE metres; source low-level and high-level step counts kept separately',
        },
        'sources': {'ours': str(Path(args.ours).resolve()), 'zeromap': str(Path(args.zeromap).resolve()),
                    'features': str(Path(args.features).resolve()),
                    'validity': str((quality_dir / 't0_val_unseen_episodes.jsonl').resolve()),
                    'presence': str((quality_dir / 't1_val_unseen_episodes.jsonl').resolve()),
                    'feature_script': str(ROOT / 'scripts/analysis/episode_features.py'),
                    'quality_script': str(ROOT / 'scripts/analysis/map_quality/t1_presence.py')},
        'counts': counts, 'candidates': selected,
    }
    write_report(result, Path(args.output))
    print(json.dumps({'counts': counts, 'selected_episodes': [row['episode_id'] for row in selected],
                      'output': args.output}, indent=2))


if __name__ == '__main__':
    main()
