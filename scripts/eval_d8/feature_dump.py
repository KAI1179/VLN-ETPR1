"""Eval-only STOP-head snapshots and replay of the existing expert actions."""

import json
from pathlib import Path

import numpy as np
import torch


class DecisionFeatureDump:
    def __init__(self, network, directory, shared, replay_directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shared = shared
        self.replay_directory = Path(replay_directory) if replay_directory else None
        self.records = {}
        self.references = {}
        self.feature = None
        self.handle = network.vln_bert.global_sap_head.register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        stop_feature = inputs[0][:, 0, :]
        if stop_feature.shape[1] != 1536:
            raise ValueError(f'STOP-head input dimension: {stop_feature.shape}')
        self.feature = stop_feature.detach().to(device='cpu', dtype=torch.float16).numpy().copy()

    def replay_actions(self, episodes, step, positions, candidate_ids, graphs, device):
        actions = []
        for i, episode in enumerate(episodes):
            eid = str(episode.episode_id)
            if eid not in self.references:
                with np.load(self.replay_directory / f'{eid}.npz', allow_pickle=False) as data:
                    self.references[eid] = {key: data[key].copy() for key in
                                            ['step', 'pos', 'sel_id', 'candidate_ids', 'candidate_positions']}
            reference = self.references[eid]
            ids = [node for node in candidate_ids[i] if node is not None and node.startswith('g')]
            coordinates = [np.asarray(graphs[i].ghost_aug_pos[node]).tolist() for node in ids]
            if step >= len(reference['step']) or int(reference['step'][step]) != step:
                raise ValueError(f'{eid}: shared replay step mismatch at {step}')
            if not np.array_equal(np.asarray(positions[i]), reference['pos'][step]):
                raise ValueError(f'{eid}/{step}: shared replay position differs')
            if ids != json.loads(str(reference['candidate_ids'][step])):
                raise ValueError(f'{eid}/{step}: shared candidate IDs differ')
            if coordinates != json.loads(str(reference['candidate_positions'][step])):
                raise ValueError(f'{eid}/{step}: shared candidate coordinates differ')
            chosen = str(reference['sel_id'][step])
            actions.append(0 if chosen == '-1' else candidate_ids[i].index(chosen))
        return torch.tensor(actions, device=device)

    def decision(self, episodes, step, positions, rotations, graphs, candidate_ids,
                 probabilities, actions, no_candidates, max_steps):
        from vlnce_baselines.models.graph_utils import heading_from_quaternion

        probabilities = probabilities.detach().cpu().numpy()
        if self.feature is None or len(self.feature) != len(episodes):
            raise RuntimeError('Missing STOP-head forward snapshot')
        for i, episode in enumerate(episodes):
            eid = str(episode.episode_id)
            ids = [node for node in candidate_ids[i] if node is not None and node.startswith('g')]
            stop = int(actions[i]) == 0 or step == max_steps - 1 or bool(no_candidates[i])
            row = {
                'feat': self.feature[i], 'pos': np.asarray(positions[i]).copy(),
                'heading': float(heading_from_quaternion(rotations[i])),
                'stop_prob': float(probabilities[i, 0]), 'step': int(step),
                'sel_id': '-1' if stop else candidate_ids[i][int(actions[i])],
                'candidate_count': len(ids), 'candidate_ids': json.dumps(ids),
                'candidate_positions': json.dumps([np.asarray(graphs[i].ghost_aug_pos[node]).tolist() for node in ids]),
            }
            self.records.setdefault(eid, {'scene_id': Path(episode.scene_id).stem, 'steps': []})['steps'].append(row)
        self.feature = None

    def finish(self, episode_id):
        eid = str(episode_id)
        record = self.records.pop(eid)
        path = self.directory / f'{eid}.npz'
        if path.exists():
            raise FileExistsError(path)
        steps = record['steps']
        arrays = {key: np.asarray([row[key] for row in steps]) for key in steps[0]}
        np.savez_compressed(path, **arrays, episode_id=np.asarray(eid),
                            scene_id=np.asarray(record['scene_id']), source=np.asarray('S' if self.shared else 'O'))
        self.references.pop(eid, None)

    def close(self):
        self.handle.remove()
