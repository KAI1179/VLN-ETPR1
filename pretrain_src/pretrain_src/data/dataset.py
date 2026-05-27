"""
Instruction and trajectory (view and object features) dataset
"""

import json
import random
import jsonlines
import numpy as np
import h5py
import math
from typing import Any, Dict, List

import torch

from .common import (
    load_nav_graphs,
    get_angle_fts,
    get_view_rel_angles,
    calculate_vp_rel_pos_fts,
    softmax,
)
from prior import DATA_DIR
from prior.grid_map import CognitiveGridMap
from vlnce_baselines.models.etp_prior_gt.map_utils import cognitive_map_to_tensors

MAX_DIST = 30  # normalize
MAX_STEP = 10  # normalize
TRAIN_MAX_STEP = 20
PRETRAIN_COGNITIVE_MAP_DIR = DATA_DIR / "cognitive_maps_etp_r1"


def _rotate_points_by_right_angle(
    points: torch.Tensor,
    turns: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    row = points[..., 0]
    col = points[..., 1]
    turns %= 4
    if turns == 0:
        return points
    if turns == 1:
        return torch.stack((col.new_tensor(float(cols)) - col, row), dim=-1)
    if turns == 2:
        return torch.stack(
            (row.new_tensor(float(rows)) - row, col.new_tensor(float(cols)) - col),
            dim=-1,
        )
    return torch.stack((col, row.new_tensor(float(rows)) - row), dim=-1)


def _rotate_direction_by_right_angle(vector: torch.Tensor, turns: int) -> torch.Tensor:
    x = vector[..., 0]
    z = vector[..., 1]
    turns %= 4
    if turns == 0:
        return vector
    if turns == 1:
        return torch.stack((-z, x), dim=-1)
    if turns == 2:
        return torch.stack((-x, -z), dim=-1)
    return torch.stack((z, -x), dim=-1)


def _rotate_cognitive_map_tensors_by_right_angle(
    tensors: Dict[str, torch.Tensor], turns: int
) -> Dict[str, torch.Tensor]:
    turns %= 4
    if turns == 0:
        return dict(tensors)

    grid = torch.rot90(tensors["grid"], turns, dims=(-2, -1)).contiguous()
    rows = int(tensors["grid"].shape[-2])
    cols = int(tensors["grid"].shape[-1])
    rotated = dict(tensors)
    rotated["grid"] = grid
    rotated["reference_paths"] = _rotate_points_by_right_angle(
        tensors["reference_paths"], turns, rows, cols
    )
    rotated["start_position"] = _rotate_points_by_right_angle(
        tensors["start_position"], turns, rows, cols
    )
    rotated["start_direction_vector"] = _rotate_direction_by_right_angle(
        tensors["start_direction_vector"], turns
    )
    return rotated


class ReverieTextPathData(object):
    def __init__(
        self,
        anno_files,
        img_ft_file,
        dep_ft_file,
        obj_ft_file,
        scanvp_cands_file,
        connectivity_dir,
        image_feat_size=2048,
        image_prob_size=1000,
        depth_feat_size=128,
        angle_feat_size=4,
        obj_feat_size=None,
        obj_prob_size=None,
        max_objects=20,
        max_txt_len=100,
        in_memory=True,
        act_visited_node=False,
        val_sample_num=None,
        use_prior_gt=False,
        random_rotation_augmentation=False,
    ):
        self.use_prior_gt = use_prior_gt
        self.random_rotation_augmentation = random_rotation_augmentation
        self.connectivity_dir = connectivity_dir
        self.img_ft_file = img_ft_file
        self.dep_ft_file = dep_ft_file
        self.obj_ft_file = obj_ft_file

        self.image_feat_size = image_feat_size
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.depth_feat_size = depth_feat_size
        self.obj_feat_size = obj_feat_size
        self.obj_prob_size = obj_prob_size

        self.obj_image_h = 480
        self.obj_image_w = 640
        self.obj_image_size = 480 * 640

        self.max_txt_len = max_txt_len
        self.max_objects = max_objects
        self.act_visited_node = act_visited_node

        self.in_memory = in_memory
        if self.in_memory:
            self._feature_store = {}
            self._feature_store_depth = {}

        self.scanvp_cands = json.load(open(scanvp_cands_file))

        self.graphs, self.shortest_distances, self.shortest_paths = load_nav_graphs(
            connectivity_dir
        )
        self.all_point_rel_angles = [
            get_view_rel_angles(baseViewId=i) for i in range(36)
        ]
        self.all_point_angle_fts = [
            get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size)
            for x in self.all_point_rel_angles
        ]

        self.data: List[Dict[str, Any]] = []

        for anno_file in anno_files:
            with jsonlines.open(anno_file, "r") as f:
                for item in f:
                    self.data.append(item)

        if val_sample_num:
            # cannot evaluate all the samples as it takes too much time
            sel_idxs = np.random.permutation(len(self.data))[:val_sample_num]
            self.data = [self.data[sidx] for sidx in sel_idxs]

    def __len__(self):
        return len(self.data)

    def _load_pretrain_cognitive_map(self, item: Dict[str, Any]):
        map_path = PRETRAIN_COGNITIVE_MAP_DIR / item["scan"] / f"{item['instr_id']}.npz"
        if not map_path.is_file():
            raise FileNotFoundError(f"Missing pretrain cognitive map: {map_path}")
        cognitive_map = CognitiveGridMap.load(map_path)
        tensors = cognitive_map_to_tensors(cognitive_map)
        if getattr(self, "random_rotation_augmentation", False):
            tensors = _rotate_cognitive_map_tensors_by_right_angle(
                tensors, random.randrange(4)
            )
        return {
            "cognitive_maps": tensors["grid"],
            "reference_paths": tensors["reference_paths"],
            "start_direction_vectors": tensors["start_direction_vector"],
            "start_positions": tensors["start_position"],
        }

    def get_scanvp_feature(self, scan, viewpoint):
        key = "%s_%s" % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            view_fts, obj_fts, obj_attrs = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, "r") as f:
                view_fts = f[key][...].astype(np.float32)

            obj_attrs = {}
            obj_feature_size = (self.obj_feat_size or 0) + (self.obj_prob_size or 0)
            obj_fts = np.zeros((0, obj_feature_size), dtype=np.float32)
            if self.obj_ft_file is not None:
                if self.obj_feat_size is None or self.obj_prob_size is None:
                    raise ValueError(
                        "obj feature sizes are required when obj_ft_file is set"
                    )
                with h5py.File(self.obj_ft_file, "r") as f:
                    if key in f:
                        obj_fts = f[key][...].astype(np.float32)
                        obj_fts = obj_fts[: self.max_objects]
                        for attr_key, attr_value in f[key].attrs.items():
                            if attr_key in ["directions", "sizes", "bboxes", "obj_ids"]:
                                obj_attrs[attr_key] = attr_value[: self.max_objects]
            if self.in_memory:
                self._feature_store[key] = (view_fts, obj_fts, obj_attrs)

        return view_fts, obj_fts, obj_attrs

    def get_obj_label(self, item, last_vp_objids):
        gt_obj_id = item["instr_id"].split("_")[1]
        for k, obj_id in enumerate(last_vp_objids):
            if obj_id == gt_obj_id:
                obj_label = k
                break
        else:
            obj_label = -100  # ignore
        return obj_label

    def get_act_labels(self, end_vp, *args):
        item, gmap_vpids, gmap_visited_masks, traj_cand_vpids = args
        scan = item["scan"]
        pos_vps = item["pos_vps"]
        if end_vp in pos_vps:
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            cand_min_dist = float("inf")
            for k, cand_vp in enumerate(gmap_vpids):
                if (k > 0) and (not gmap_visited_masks[k]):
                    min_dist = min(
                        [
                            self.shortest_distances[scan][end_vp][cand_vp]
                            + self.shortest_distances[scan][cand_vp][pos_vp]
                            for pos_vp in pos_vps
                        ]
                    )
                    if min_dist < cand_min_dist:
                        cand_min_dist = min_dist
                        global_act_label = k  # [stop] is 0
            # local:
            cand_min_dist = float("inf")
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                min_dist = min(
                    [
                        self.shortest_distances[scan][end_vp][cand_vp]
                        + self.shortest_distances[scan][cand_vp][pos_vp]
                        for pos_vp in pos_vps
                    ]
                )
                if min_dist < cand_min_dist:
                    cand_min_dist = min_dist
                    local_act_label = k + 1  # [stop] is 0
        return global_act_label, local_act_label

    def get_input(
        self,
        idx: int,
        end_vp_type,
        return_img_probs=False,
        return_act_label=False,
        return_obj_label=False,
        end_vp=None,
    ):
        item = self.data[idx]
        scan = item["scan"]
        start_vp = item["path"][0]
        start_heading = item.get("heading", 0)
        pos_vps = item["pos_vps"]
        gt_path = item["path"]

        if end_vp is None:
            if end_vp_type == "pos":
                end_vp = pos_vps[np.random.randint(len(pos_vps))]
            elif end_vp_type == "neg_in_gt_path":
                end_vps = [vp for vp in gt_path if vp not in pos_vps]
                if len(end_vps) == 0:
                    end_vps = gt_path
                end_vp = end_vps[np.random.randint(len(end_vps))]
            elif end_vp_type == "neg_others":
                noneg_vp_set = set(pos_vps + gt_path)
                end_vps = [
                    vp
                    for vp in self.graphs[scan].nodes.keys()
                    if vp not in noneg_vp_set
                ]
                end_vp = end_vps[np.random.randint(len(end_vps))]

        gt_path = self.shortest_paths[scan][start_vp][end_vp]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]

        (
            traj_view_img_fts,
            traj_obj_img_fts,
            traj_loc_fts,
            traj_nav_types,
            traj_cand_vpids,
            last_vp_angles,
            last_vp_objids,
        ) = self.get_traj_pano_fts(scan, gt_path)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = (
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)
        )

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(
            scan,
            start_vp,
            end_vp,
            traj_cand_vpids[-1],
            cur_heading,
            cur_elevation,
            len(traj_nav_types[-1]),
        )

        outs = {
            "instr_id": item["instr_id"],
            "instr_encoding": item["instr_encoding"][: self.max_txt_len],
            "traj_view_img_fts": [
                x[:, : self.image_feat_size] for x in traj_view_img_fts
            ],
            "traj_obj_img_fts": [x[:, : self.obj_feat_size] for x in traj_obj_img_fts],
            "traj_loc_fts": traj_loc_fts,
            "traj_nav_types": traj_nav_types,
            "traj_cand_vpids": traj_cand_vpids,
            "traj_vpids": gt_path,
            "gmap_vpids": gmap_vpids,
            "gmap_step_ids": gmap_step_ids,
            "gmap_visited_masks": gmap_visited_masks,
            "gmap_pos_fts": gmap_pos_fts,
            "gmap_pair_dists": gmap_pair_dists,
            "vp_pos_fts": vp_pos_fts,
            "vp_angles": last_vp_angles,
        }

        if return_obj_label:
            outs["obj_labels"] = self.get_obj_label(item, last_vp_objids)

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids
            )
            outs["global_act_labels"] = global_act_label
            outs["local_act_labels"] = local_act_label

        if return_img_probs:
            outs["vp_view_probs"] = softmax(
                traj_view_img_fts[-1][:, self.image_feat_size :], dim=1
            )
            outs["vp_obj_probs"] = softmax(
                traj_obj_img_fts[-1][:, self.obj_feat_size :], dim=1
            )

        return outs

    def get_cur_angle(self, scan, path, start_heading):
        if len(path) < 2:
            heading = start_heading
            elevation = 0
        else:
            prev_vp = path[-2]
            cur_vp = path[-1]
            viewidx = self.scanvp_cands["%s_%s" % (scan, prev_vp)][cur_vp][0]
            heading = (viewidx % 12) * math.radians(30)
            elevation = (viewidx // 12 - 1) * math.radians(30)
        return heading, elevation

    def get_traj_pano_fts(self, scan, path):
        """
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        """
        (
            traj_view_img_fts,
            traj_obj_img_fts,
            traj_loc_fts,
            traj_nav_types,
            traj_cand_vpids,
        ) = [], [], [], [], []

        for vp in path:
            view_fts, obj_img_fts, obj_attrs = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_angles, cand_vpids = [], [], []
            # cand views
            nav_cands = self.scanvp_cands["%s_%s" % (scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
            # non cand views
            view_img_fts.extend(
                [view_fts[idx] for idx in range(36) if idx not in used_viewidxs]
            )
            view_angles.extend(
                [
                    self.all_point_rel_angles[12][idx]
                    for idx in range(36)
                    if idx not in used_viewidxs
                ]
            )
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(
                view_angles[:, 0], view_angles[:, 1], self.angle_feat_size
            )
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)

            # object features
            num_objs = obj_img_fts.shape[0]
            obj_angles = np.zeros((num_objs, 2), dtype=np.float32)
            obj_ang_fts = np.zeros((num_objs, self.angle_feat_size), dtype=np.float32)
            obj_box_fts = np.zeros((num_objs, 3), dtype=np.float32)
            if num_objs > 0:
                for k, (w, h) in enumerate(obj_attrs["sizes"]):
                    obj_angles[k] = obj_attrs["directions"][k]
                    obj_box_fts[k] = [
                        h / self.obj_image_h,
                        w / self.obj_image_w,
                        (h * w) / self.obj_image_size,
                    ]
                obj_ang_fts = get_angle_fts(
                    obj_angles[:, 0], obj_angles[:, 1], self.angle_feat_size
                )

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_obj_img_fts.append(obj_img_fts)
            traj_loc_fts.append(
                np.concatenate(
                    [
                        np.concatenate([view_ang_fts, view_box_fts], 1),
                        np.concatenate([obj_ang_fts, obj_box_fts], 1),
                    ],
                    axis=0,
                )
            )
            traj_nav_types.append(
                [1] * len(cand_vpids)
                + [0] * (36 - len(used_viewidxs))
                + [2] * len(obj_img_fts)
            )
            traj_cand_vpids.append(cand_vpids)

            last_vp_objids = obj_attrs.get("obj_ids", [])
            last_vp_angles = np.concatenate([view_angles, obj_angles], 0)

        return (
            traj_view_img_fts,
            traj_obj_img_fts,
            traj_loc_fts,
            traj_nav_types,
            traj_cand_vpids,
            last_vp_angles,
            last_vp_objids,
        )

    def get_gmap_inputs(self, scan, path, cur_heading, cur_elevation):
        cur_vp = path[-1]

        visited_vpids, unvisited_vpids = {}, {}
        for t, vp in enumerate(path):
            visited_vpids[vp] = t + 1
            if vp in unvisited_vpids:
                del unvisited_vpids[vp]
            for next_vp in self.scanvp_cands["%s_%s" % (scan, vp)].keys():
                if next_vp not in visited_vpids:
                    unvisited_vpids[next_vp] = 0
        # add [stop] token
        gmap_vpids = [None] + list(visited_vpids.keys()) + list(unvisited_vpids.keys())
        gmap_step_ids = (
            [0] + list(visited_vpids.values()) + list(unvisited_vpids.values())
        )
        if self.act_visited_node:
            gmap_visited_masks = [0]
            for vp in gmap_vpids[1:]:
                if vp == path[-1]:
                    gmap_visited_masks.append(1)
                else:
                    gmap_visited_masks.append(0)
        else:
            gmap_visited_masks = (
                [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            )

        gmap_pos_fts = self.get_gmap_pos_fts(
            scan, cur_vp, gmap_vpids, cur_heading, cur_elevation
        )

        gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
        for i in range(1, len(gmap_vpids)):
            for j in range(i + 1, len(gmap_vpids)):
                gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = (
                    self.shortest_distances[scan][gmap_vpids[i]][gmap_vpids[j]]
                    / MAX_DIST
                )
        return (
            gmap_vpids,
            gmap_step_ids,
            gmap_visited_masks,
            gmap_pos_fts,
            gmap_pair_dists,
        )

    def get_gmap_pos_fts(self, scan, cur_vp, gmap_vpids, cur_heading, cur_elevation):
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.graphs[scan].nodes[cur_vp]["position"],
                    self.graphs[scan].nodes[vp]["position"],
                    base_heading=cur_heading,
                    base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [
                        rel_dist / MAX_DIST,
                        self.shortest_distances[scan][cur_vp][vp] / MAX_DIST,
                        (len(self.shortest_paths[scan][cur_vp][vp]) - 1) / MAX_STEP,
                    ]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(
            rel_angles[:, 0], rel_angles[:, 1], self.angle_feat_size
        )
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    def get_vp_pos_fts(
        self, scan, start_vp, cur_vp, cand_vpids, cur_heading, cur_elevation, vp_ft_len
    ):
        cur_cand_pos_fts = self.get_gmap_pos_fts(
            scan, cur_vp, cand_vpids, cur_heading, cur_elevation
        )
        cur_start_pos_fts = self.get_gmap_pos_fts(
            scan, cur_vp, [start_vp], cur_heading, cur_elevation
        )

        # add [stop] token at beginning
        vp_pos_fts = np.zeros((vp_ft_len + 1, 14), dtype=np.float32)
        vp_pos_fts[:, :7] = cur_start_pos_fts
        vp_pos_fts[1 : len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts

        return vp_pos_fts


class R2RTextPathData(ReverieTextPathData):
    def __init__(
        self,
        anno_files,
        img_ft_file,
        dep_ft_file,
        scanvp_cands_file,
        connectivity_dir,
        image_feat_size=2048,
        image_prob_size=1000,
        depth_feat_size=128,
        angle_feat_size=4,
        max_txt_len=100,
        in_memory=True,
        act_visited_node=False,
        val_sample_num=None,
        start_vp_file=None,
        use_prior_gt=False,
        random_rotation_augmentation=False,
    ):
        super().__init__(
            anno_files,
            img_ft_file,
            dep_ft_file,
            None,
            scanvp_cands_file,
            connectivity_dir,
            image_feat_size=image_feat_size,
            image_prob_size=image_prob_size,
            depth_feat_size=depth_feat_size,
            angle_feat_size=angle_feat_size,
            obj_feat_size=0,
            obj_prob_size=0,
            max_objects=0,
            max_txt_len=max_txt_len,
            in_memory=in_memory,
            act_visited_node=act_visited_node,
            val_sample_num=val_sample_num,
            use_prior_gt=use_prior_gt,
            random_rotation_augmentation=random_rotation_augmentation,
        )

    def get_scanvp_feature(self, scan, viewpoint):
        key = "%s_%s" % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            view_fts = self._feature_store[key]
            dep_fts = self._feature_store_depth[key]
        else:
            with h5py.File(self.img_ft_file, "r") as f:
                view_fts = f[key][...].astype(np.float32)
            with h5py.File(self.dep_ft_file, "r") as f:
                dep_fts = f[key][...].astype(np.float32)
            if self.in_memory:
                self._feature_store[key] = view_fts
                self._feature_store_depth[key] = dep_fts
        return view_fts, dep_fts

    def get_act_labels(self, end_vp, *args):
        end_idx, item, gmap_vpids, traj_cand_vpids = args
        if end_vp == item["path"][-1]:
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            gt_next_vp = item["path"][end_idx + 1]
            for k, cand_vp in enumerate(gmap_vpids):
                if cand_vp == gt_next_vp:
                    global_act_label = k
                    break
            # local:
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                if cand_vp == gt_next_vp:
                    local_act_label = k + 1  # [stop] is 0
                    break
        return global_act_label, local_act_label

    def get_input(
        self,
        idx,
        end_vp_type,
        return_img_probs=False,
        return_act_label=False,
        return_obj_label=False,
        end_vp=None,
    ):
        item = self.data[idx]
        scan = item["scan"]
        start_vp = item["path"][0]
        start_heading = item["heading"]
        gt_path = item["path"]

        if end_vp is None:
            if end_vp_type == "pos":
                # name convention with REVERIE (last vp)
                end_idx = len(gt_path) - 1
                end_vp = gt_path[-1]
            elif end_vp_type in ["neg_in_gt_path", "neg_others"]:
                # name convention with REVERIE (mid vps in the path)
                end_vps = gt_path[:-1]
                end_idx = np.random.randint(len(end_vps))
                end_vp = end_vps[end_idx]
        else:
            assert end_vp in gt_path
            end_idx = gt_path.index(end_vp)

        gt_path = gt_path[: end_idx + 1]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]

        (
            traj_view_img_fts,
            traj_view_dep_fts,
            traj_loc_fts,
            traj_nav_types,
            traj_cand_vpids,
            last_vp_angles,
        ) = self.get_traj_pano_fts(scan, gt_path)

        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = (
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)
        )

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(
            scan,
            start_vp,
            end_vp,
            traj_cand_vpids[-1],
            cur_heading,
            cur_elevation,
            len(traj_nav_types[-1]),
        )

        outs = {
            "instr_id": item["instr_id"],
            "instr_encoding": item["instr_encoding"][: self.max_txt_len],  # ID
            "task_type_encoding": item["task_type_encoding"],
            "traj_view_img_fts": [
                x[:, : self.image_feat_size] for x in traj_view_img_fts
            ],
            "traj_view_dep_fts": [
                x[:, : self.depth_feat_size] for x in traj_view_dep_fts
            ],
            "traj_loc_fts": traj_loc_fts,
            "traj_nav_types": traj_nav_types,
            "traj_cand_vpids": traj_cand_vpids,
            "traj_vpids": gt_path,
            "gmap_vpids": gmap_vpids,
            "gmap_step_ids": gmap_step_ids,
            "gmap_visited_masks": gmap_visited_masks,
            "gmap_pos_fts": gmap_pos_fts,
            "gmap_pair_dists": gmap_pair_dists,
            "vp_pos_fts": vp_pos_fts,
        }

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, end_idx, item, gmap_vpids, traj_cand_vpids
            )
            outs["global_act_labels"] = global_act_label
            outs["local_act_labels"] = local_act_label

        if return_img_probs:
            outs["vp_view_probs"] = softmax(
                traj_view_img_fts[-1][:, self.image_feat_size :], dim=1
            )

        if self.use_prior_gt:
            outs.update(self._load_pretrain_cognitive_map(item))

        return outs

    def get_traj_pano_fts(self, scan, path):
        """
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        """
        (
            traj_view_img_fts,
            traj_view_dep_fts,
            traj_loc_fts,
            traj_nav_types,
            traj_cand_vpids,
        ) = [], [], [], [], []

        for vp in path:
            view_fts, dep_fts = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_dep_fts, view_angles, cand_vpids = [], [], [], []
            # cand views
            nav_cands = self.scanvp_cands["%s_%s" % (scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                view_dep_fts.append(dep_fts[v[0]])
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
            # non cand views
            view_img_fts.extend(
                [view_fts[idx] for idx in range(36) if idx not in used_viewidxs]
            )
            view_dep_fts.extend(
                [dep_fts[idx] for idx in range(36) if idx not in used_viewidxs]
            )
            view_angles.extend(
                [
                    self.all_point_rel_angles[12][idx]
                    for idx in range(36)
                    if idx not in used_viewidxs
                ]
            )

            view_img_fts = np.stack(view_img_fts, 0)
            view_dep_fts = np.stack(view_dep_fts, 0)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(
                view_angles[:, 0], view_angles[:, 1], self.angle_feat_size
            )

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_view_dep_fts.append(view_dep_fts)
            traj_loc_fts.append(view_ang_fts)
            traj_nav_types.append(
                [1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs))
            )
            traj_cand_vpids.append(cand_vpids)

            last_vp_angles = view_angles

        return (
            traj_view_img_fts,
            traj_view_dep_fts,
            traj_loc_fts,
            traj_nav_types,
            traj_cand_vpids,
            last_vp_angles,
        )
