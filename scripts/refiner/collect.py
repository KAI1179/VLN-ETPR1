#!/usr/bin/env python3
"""Collect teacher and perturbed trajectories for cognitive-map refinement."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.utils.common import batch_obs
from tap import Tap

import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401
from vlnce_baselines.config.default import get_config
from vlnce_baselines.models.graph_utils import GraphMap
from vlnce_baselines.models.etp_llm.navigation import (
    llm_navigation_cognitive_map_raster_path,
)
from vlnce_baselines.models.etp_prior_gt.map_utils import cognitive_map_cache_path
from vlnce_baselines.ss_trainer_ETP_LLM import RLTrainer


class Arguments(Tap):
    exp_config: Path = Path("run_r2r/iter_train.yaml")
    output_dir: Path = Path("data/refiner")
    report_dir: Path = Path("reports/refiner")
    split: str = "train"
    suffix: Optional[str] = None
    limit: int = 100
    num_environments: int = 4
    gpu_device: int = 0
    llm_cache_model_key: str = "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree"
    llm_cache_dir: str = ""
    gt_namespace: str = "gt.legacy.r1p5.direction5.v1"
    pretrained_path: str = (
        "pretrained/r2r_rxr_ce/llm_grid_try5/store2/model_step_460000.pt"
    )
    kinds: Tuple[str, ...] = ("teacher", "perturbed")


def _config(args: Arguments):
    config = get_config(str(args.exp_config))
    config.defrost()
    config.local_rank = 0
    config.GPU_NUMBERS = 1
    config.NUM_ENVIRONMENTS = args.num_environments
    config.SIMULATOR_GPU_IDS = [args.gpu_device]
    config.TORCH_GPU_IDS = [args.gpu_device]
    config.TORCH_GPU_ID = args.gpu_device
    config.TRAINER_NAME = "SS-ETP-LLM"
    config.MODEL.policy_name = "LLMGridTry5Policy"
    config.MODEL.task_type = "r2r"
    config.MODEL.pretrained_path = args.pretrained_path
    config.MODEL.MAP_ENCODER.enabled = True
    config.MODEL.MAP_ENCODER.architecture = "try5"
    config.MODEL.MAP_ENCODER.source = "llm_grid"
    config.MODEL.MAP_ENCODER.llm_cache_model_key = args.llm_cache_model_key
    config.MODEL.MAP_ENCODER.llm_cache_dir = args.llm_cache_dir
    config.TASK_CONFIG.DATASET.SPLIT = args.split
    config.TASK_CONFIG.DATASET.SUFFIX = (
        "_90" if args.suffix is None and args.split == "train" else args.suffix or ""
    )
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.IL.waypoint_aug = False
    RLTrainer._add_refiner_semantic_sensors(config.TASK_CONFIG)
    config.SENSORS = config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
    config.freeze()
    return config


def _map_paths(args: Arguments, trainer: RLTrainer, episode):
    cache_id = trainer._cognitive_map_cache_id(episode)
    p0_path = llm_navigation_cognitive_map_raster_path(
        episode.scene_id,
        cache_id,
        "r2r",
        args.split,
        cache_dir=args.llm_cache_dir or None,
        model_key=args.llm_cache_model_key,
    )
    gt_path = cognitive_map_cache_path(
        episode.scene_id,
        cache_id,
        namespace=args.gt_namespace,
    )
    return cache_id, p0_path, gt_path


def _snapshot(evidence, pose):
    return {
        "pose": np.asarray(pose, dtype=np.float32),
        "sem": np.packbits(evidence.sem),
        "observed": np.packbits(evidence.observed),
        "free": np.packbits(evidence.free),
        "blocked": np.packbits(evidence.blocked),
    }


def _save_trajectory(
    args: Arguments,
    trainer: RLTrainer,
    episode,
    kind: str,
    evidence,
    steps,
) -> Path:
    cache_id, p0_path, gt_path = _map_paths(args, trainer, episode)
    scene = Path(episode.scene_id).stem
    output_path = args.output_dir / args.split / scene / f"{episode.episode_id}_{kind}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        scene_id=np.asarray(episode.scene_id),
        episode_id=np.asarray(episode.episode_id),
        cache_id=np.asarray(cache_id),
        kind=np.asarray(kind),
        origin_xz=np.asarray(evidence.origin_xz, dtype=np.float32),
        range_y=np.asarray(evidence.range_y, dtype=np.float32),
        p0_path=np.asarray(str(p0_path)),
        gt_path=np.asarray(str(gt_path)),
        steps=np.asarray(steps, dtype=object),
    )
    return output_path


def _choose_actions(trainer, kind, teacher_actions, nav_inputs, no_vp_left, rngs, dev_left):
    actions = []
    for index, teacher_tensor in enumerate(teacher_actions):
        teacher = int(teacher_tensor.item())
        if teacher == -100 or no_vp_left[index]:
            actions.append(0)
            continue
        if kind == "teacher" or (teacher == 0 and dev_left[index] == 0):
            actions.append(teacher)
            continue
        alternative = [
            action
            for action, vp in enumerate(nav_inputs["gmap_vp_ids"][index])
            if action != teacher and vp in trainer.gmaps[index].ghost_aug_pos
        ]
        if dev_left[index] > 0:
            dev_left[index] -= 1
            actions.append(int(rngs[index].choice(alternative)) if alternative else teacher)
        elif rngs[index].random() < 0.25 and alternative:
            actions.append(int(rngs[index].choice(alternative)))
            dev_left[index] = int(rngs[index].integers(1, 5))
        else:
            actions.append(teacher)
    return np.asarray(actions, dtype=np.int64)


def _env_actions(trainer, actions, nav_inputs, no_vp_left, cur_vp, step):
    result = []
    use_tryout = (
        trainer.config.IL.tryout
        and not trainer.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING
    )
    for index, gmap in enumerate(trainer.gmaps):
        if actions[index] == 0 or step == trainer.max_len - 1 or no_vp_left[index]:
            stop_vp = cur_vp[index]
            stop_pos = gmap.node_pos[stop_vp]
            result.append(
                {
                    "action": {
                        "act": 0,
                        "cur_vp": cur_vp[index],
                        "stop_vp": stop_vp,
                        "stop_pos": stop_pos,
                        "back_path": None,
                        "tryout": use_tryout,
                    },
                    "vis_info": None,
                }
            )
            continue
        ghost_vp = nav_inputs["gmap_vp_ids"][index][actions[index]]
        ghost_pos = gmap.ghost_aug_pos[ghost_vp]
        _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
        front_pos = gmap.node_pos[front_vp]
        if trainer.config.IL.back_algo == "control":
            back_path = [
                (vp, gmap.node_pos[vp])
                for vp in gmap.shortest_path[cur_vp[index]][front_vp]
            ][1:]
        else:
            back_path = None
        result.append(
            {
                "action": {
                    "act": 4,
                    "cur_vp": cur_vp[index],
                    "front_vp": front_vp,
                    "front_pos": front_pos,
                    "ghost_vp": ghost_vp,
                    "ghost_pos": ghost_pos,
                    "back_path": back_path,
                    "tryout": use_tryout,
                },
                "vis_info": None,
            }
        )
        trainer._collect_prev_vp[index] = front_vp
        if trainer.config.MODEL.consume_ghost:
            gmap.delete_ghost(ghost_vp)
    return result


@torch.no_grad()
def _collect_rollout(
    args: Arguments,
    trainer: RLTrainer,
    kind: str,
    remaining: int,
) -> List[Path]:
    trainer.envs.resume_all()
    observations = trainer.envs.reset()
    episodes = list(trainer.envs.current_episodes())
    cognitive_maps = trainer._build_cognitive_maps(random_rotation_augmentation=False)
    trainer._initialize_refiner_state(cognitive_maps)
    trainer.gmaps = [
        GraphMap(
            True,
            trainer.config.IL.loc_noise,
            trainer.config.MODEL.merge_ghost,
            0,
        )
        for _ in range(trainer.envs.num_envs)
    ]
    trainer._collect_prev_vp = [None] * trainer.envs.num_envs
    records = [[] for _ in range(trainer.envs.num_envs)]
    rngs = [np.random.default_rng(int(ep.episode_id)) for ep in episodes]
    dev_left = [0] * trainer.envs.num_envs
    written: List[Path] = []

    for step in range(15):
        evidence_states = trainer._update_online_evidence(observations)
        for index, state in enumerate(evidence_states):
            records[index].append(_snapshot(trainer.evidence[index], state["pose"]))

        transformed = apply_obs_transforms_batch(
            batch_obs(observations, trainer.device), trainer.obs_transforms
        )
        wp_outputs = trainer.policy.net(
            mode="waypoint",
            waypoint_predictor=trainer.waypoint_predictor,
            observations=transformed,
            in_train=False,
        )
        vp_inputs = trainer._vp_feature_variable(wp_outputs)
        vp_inputs["mode"] = "panorama"
        pano_embeds, pano_masks = trainer.policy.net(**vp_inputs)
        avg_pano_embeds = torch.sum(
            pano_embeds * pano_masks.unsqueeze(2), 1
        ) / torch.sum(pano_masks, 1, keepdim=True)
        cur_pos, cur_ori = trainer.get_pos_ori()
        cur_vp, cand_vp, cand_pos = [], [], []
        for index in range(trainer.envs.num_envs):
            current, candidates, positions = trainer.gmaps[index].identify_node(
                cur_pos[index],
                cur_ori[index],
                wp_outputs["cand_angles"][index],
                wp_outputs["cand_distances"][index],
            )
            cur_vp.append(current)
            cand_vp.append(candidates)
            cand_pos.append(positions)
        cand_real_pos = [
            [
                trainer.envs.call_at(
                    index,
                    "get_cand_real_pos",
                    {"angle": angle, "forward": distance},
                )
                for angle, distance in zip(
                    wp_outputs["cand_angles"][index],
                    wp_outputs["cand_distances"][index],
                )
            ]
            for index in range(trainer.envs.num_envs)
        ]
        for index in range(trainer.envs.num_envs):
            trainer.gmaps[index].update_graph(
                trainer._collect_prev_vp[index],
                step + 1,
                cur_vp[index],
                cur_pos[index],
                avg_pano_embeds[index],
                cand_vp[index],
                cand_pos[index],
                pano_embeds[index][vp_inputs["nav_types"][index] == 1],
                cand_real_pos[index],
            )
        nav_inputs = trainer._nav_gmap_variable(cur_vp, cur_pos, cur_ori, 1)
        no_vp_left = nav_inputs.pop("no_vp_left")
        teacher_actions = trainer._teacher_action_new(
            nav_inputs["gmap_vp_ids"], no_vp_left, args.split == "train"
        )
        actions = _choose_actions(
            trainer, kind, teacher_actions, nav_inputs, no_vp_left, rngs, dev_left
        )
        outputs = trainer.envs.step(
            _env_actions(trainer, actions, nav_inputs, no_vp_left, cur_vp, step)
        )
        observations, _, dones, _ = [list(items) for items in zip(*outputs)]

        for index in reversed(range(trainer.envs.num_envs)):
            if not dones[index]:
                continue
            if len(written) < remaining:
                written.append(
                    _save_trajectory(
                        args,
                        trainer,
                        episodes[index],
                        kind,
                        trainer.evidence[index],
                        records[index],
                    )
                )
            trainer.envs.pause_at(index)
            observations.pop(index)
            episodes.pop(index)
            trainer.gmaps.pop(index)
            trainer._collect_prev_vp.pop(index)
            cognitive_maps.pop(index)
            trainer.evidence.pop(index)
            trainer.refiner_p0.pop(index)
            trainer.refiner_semantic_luts.pop(index)
            records.pop(index)
            rngs.pop(index)
            dev_left.pop(index)
        if trainer.envs.num_envs == 0:
            break
    return written


def _collect_kind(args: Arguments, kind: str) -> List[Path]:
    trainer = RLTrainer(_config(args))
    trainer._set_config()
    observation_space, action_space = trainer._init_envs()
    trainer._initialize_policy(
        trainer.config,
        False,
        observation_space=observation_space,
        action_space=action_space,
    )
    trainer.policy.eval()
    trainer.waypoint_predictor.eval()
    available = sum(trainer.envs.number_of_episodes)
    target = available if args.limit < 0 else min(args.limit, available)
    output_paths: List[Path] = []
    started = time.perf_counter()
    while len(output_paths) < target:
        output_paths.extend(
            _collect_rollout(args, trainer, kind, target - len(output_paths))
        )
    elapsed = time.perf_counter() - started
    trainer.envs.close()
    print(
        f"kind={kind} episodes={len(output_paths)} seconds={elapsed:.2f} "
        f"episodes_per_second={len(output_paths) / elapsed:.4f}",
        flush=True,
    )
    return output_paths


def _write_reports(paths: List[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for trajectory_path in paths[:3]:
        with np.load(trajectory_path, allow_pickle=True) as trajectory:
            gt_path = Path(str(trajectory["gt_path"].item()))
            step = trajectory["steps"][-1]
        with np.load(gt_path) as target:
            gt_object = target["grid"][:27].max(axis=0)
        sem = np.unpackbits(
            step["sem"], count=27 * 100 * 100
        ).reshape(27, 100, 100)
        observed = np.unpackbits(
            step["observed"], count=100 * 100
        ).reshape(100, 100)
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, image, title in zip(
            axes,
            (gt_object, sem.max(axis=0), observed),
            ("GT objects", "visual semantics", "observed"),
        ):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_dir / f"{trajectory_path.stem}.png")
        plt.close(figure)


def main() -> None:
    args = Arguments(underscores_to_dashes=True).parse_args()
    torch.cuda.set_device(args.gpu_device)
    all_paths: List[Path] = []
    kinds = args.kinds if args.split == "train" else ("teacher",)
    for kind in kinds:
        all_paths.extend(_collect_kind(args, kind))
    _write_reports(all_paths, args.report_dir)


if __name__ == "__main__":
    main()
