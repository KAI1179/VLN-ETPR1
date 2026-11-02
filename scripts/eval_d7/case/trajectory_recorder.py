"""Optional eval-only decision records; no model or environment mutations."""

import json
from pathlib import Path

import numpy as np


class TrajectoryRecorder:
    def __init__(self, output_dir, episodes):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records = {
            str(ep.episode_id): {
                "episode_id": str(ep.episode_id),
                "scene": Path(ep.scene_id).stem,
                "instruction": ep.instruction.instruction_text,
                "steps": [],
            }
            for ep in episodes
        }

    def decision(self, episodes, step, positions, rotations, graphs, vp_ids,
                 probabilities, actions, no_vp_left, max_len):
        # Snapshot before consume_ghost removes the selected node from the graph.
        from vlnce_baselines.models.graph_utils import heading_from_quaternion

        probs = probabilities.detach().cpu().numpy()
        for i, ep in enumerate(episodes):
            selected = int(actions[i])
            stop = selected == 0 or step == max_len - 1 or bool(no_vp_left[i])
            candidates = [
                {"id": node, "position": np.asarray(graphs[i].ghost_aug_pos[node]).tolist(),
                 "probability": float(probs[i, index])}
                for index, node in enumerate(vp_ids[i])
                if node is not None and node.startswith("g")
            ]
            self.records[str(ep.episode_id)]["steps"].append({
                "step": step,
                "position": np.asarray(positions[i]).tolist(),
                "rotation_xyzw": np.asarray(rotations[i]).tolist(),
                "heading": float(heading_from_quaternion(rotations[i])),
                "candidates": candidates,
                "stop_probability": float(probs[i, 0]),
                "argmax_id": vp_ids[i][selected],
                "selected_id": None if stop else vp_ids[i][selected],
                "stop": stop,
                "forced_stop": stop and selected != 0,
            })

    def executed(self, episodes, env_actions):
        for ep, entry in zip(episodes, env_actions):
            action = entry["action"]
            record = self.records[str(ep.episode_id)]["steps"][-1]
            record["executed_action"] = int(action["act"])
            target = action["stop_pos"] if action["act"] == 0 else action["ghost_pos"]
            record["executed_target_position"] = np.asarray(target).tolist()

    def finish(self, episode_id, metric, path):
        record = self.records.pop(str(episode_id))
        record["trajectory"] = np.asarray(path).tolist()
        record["final"] = {
            key: float(value) for key, value in metric.items()
        }
        (self.output_dir / f"{episode_id}.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
