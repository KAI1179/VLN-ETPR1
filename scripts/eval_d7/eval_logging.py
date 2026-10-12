"""Eval-only snapshots; no model, environment, or RNG mutations."""

import json
from pathlib import Path

import numpy as np


class EvalDecisionLog:
    def __init__(self):
        self.episodes = {}

    def decision(self, episodes, step, positions, graphs, current_ids, previous_ids,
                 candidate_ids, actions, no_candidates, max_steps):
        for i, episode in enumerate(episodes):
            graph = graphs[i]
            action = int(actions[i])
            budget = step == max_steps - 1
            empty = bool(no_candidates[i])
            stop = action == 0
            # Priority for simultaneous triggers: forced budget, exhausted graph, STOP.
            reason = None
            if budget:
                reason = 'budget'
            elif empty:
                reason = 'no_candidate'
            elif stop:
                reason = 'stop'
            chosen = None if reason is not None else candidate_ids[i][action]
            record = {
                'step_idx': int(step),
                'current_position': np.asarray(positions[i]).tolist(),
                'current_node_id': current_ids[i],
                'current_node_position': np.asarray(graph.node_pos[current_ids[i]]).tolist(),
                'incoming_position': None if previous_ids[i] is None else np.asarray(graph.node_pos[previous_ids[i]]).tolist(),
                'selected_candidate_id': chosen,
                'selected_candidate_position': None if chosen is None else np.asarray(graph.ghost_aug_pos[chosen]).tolist(),
                'raw_argmax_id': candidate_ids[i][action],
                'is_stop': reason is not None,
                'candidate_count': sum(node is not None and node.startswith('g') for node in candidate_ids[i]),
                'stop_requested': stop, 'budget_reached': budget, 'no_candidate': empty,
                'termination_reason': reason,
            }
            self.episodes.setdefault(str(episode.episode_id), []).append(record)

    def termination(self, episode_id):
        reason = self.episodes[str(episode_id)][-1]['termination_reason']
        if reason not in {'stop', 'budget', 'no_candidate'}:
            raise RuntimeError(f'Episode {episode_id} ended outside the high-level stop branch')
        return reason

    def write(self, directory, checkpoint, split, rank, world):
        path = Path(directory) / f'traj_ep_{checkpoint}_{split}_r{rank}_w{world}.json'
        path.write_text(json.dumps(self.episodes, indent=2, allow_nan=False) + '\n')
