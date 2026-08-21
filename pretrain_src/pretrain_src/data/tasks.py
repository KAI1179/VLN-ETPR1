import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .common import pad_tensors


COGNITIVE_MAP_TENSOR_KEYS = (
    "cognitive_maps",
    "trajectory_keypoints",
    "map_trajectory_metadata",
    "start_direction_vectors",
    "start_positions",
)


def _copy_cognitive_map_inputs(inputs, output):
    present_keys = [key for key in COGNITIVE_MAP_TENSOR_KEYS if key in inputs]
    if not present_keys:
        if "cognitive_map_box_targets" in inputs:
            raise ValueError("cognitive_map_box_targets require cognitive_maps")
        return
    if len(present_keys) != len(COGNITIVE_MAP_TENSOR_KEYS):
        missing_keys = sorted(set(COGNITIVE_MAP_TENSOR_KEYS) - set(present_keys))
        raise ValueError(f"Incomplete cognitive-map inputs; missing {missing_keys}")
    output.update({key: inputs[key] for key in COGNITIVE_MAP_TENSOR_KEYS})
    if "cognitive_map_box_targets" in inputs:
        output["cognitive_map_box_targets"] = inputs["cognitive_map_box_targets"]


def _copy_pose_gated_map_inputs(inputs, output):
    keys = (
        "traj_spatial_view_fts",
        "traj_spatial_dep_fts",
        "spatial_semantic_targets",
        "spatial_coverage_targets",
        "traj_positions",
        "traj_rotations",
        "target_cognitive_maps",
        "route_negative_spatial_view_fts",
        "route_negative_spatial_dep_fts",
    )
    if "traj_spatial_view_fts" in inputs:
        output.update({key: inputs[key] for key in keys})


############### Masked Language Modeling ###############
def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_tokens, output_label = [], []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_tokens.append(mask)

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_tokens.append(random.choice(list(range(*vocab_range))))

            # -> rest 10% randomly keep current token
            else:
                output_tokens.append(token)

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            output_tokens.append(token)
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        output_tokens[0] = mask

    return output_tokens, output_label


class MlmDataset(Dataset):
    def __init__(self, nav_db, tok):
        self.nav_db = nav_db
        self.tok = tok

        self.vocab_range = [4, 250000]
        self.cls_token_id = self.tok.cls_token_id  # 0
        self.sep_token_id = self.tok.sep_token_id  # 2
        self.mask_token_id = self.tok.mask_token_id  # 250001
        self.pad_token_id = self.tok.pad_token_id  # 1
        print(
            f"Special tok id check: \
              self.cls_token_id:{self.cls_token_id}, self.sep_token_id:{self.sep_token_id}, self.mask_token_id:{self.mask_token_id}, self.pad_token_id:{self.pad_token_id}"
        )

    def __len__(self):
        return len(self.nav_db)

    def __getitem__(self, idx):
        inputs = self.nav_db.get_input(idx, "pos")

        output = {}

        txt_ids, txt_labels = random_word(
            inputs["instr_encoding"], self.vocab_range, self.mask_token_id
        )
        output["txt_ids"] = torch.LongTensor(txt_ids)
        output["txt_labels"] = torch.LongTensor(txt_labels)

        output["traj_view_img_fts"] = [
            torch.from_numpy(x) for x in inputs["traj_view_img_fts"]
        ]
        output["traj_view_dep_fts"] = [
            torch.from_numpy(x) for x in inputs["traj_view_dep_fts"]
        ]
        if "traj_obj_img_fts" in inputs:
            output["traj_obj_img_fts"] = [
                torch.from_numpy(x) for x in inputs["traj_obj_img_fts"]
            ]
        output["traj_loc_fts"] = [torch.from_numpy(x) for x in inputs["traj_loc_fts"]]
        output["traj_nav_types"] = [
            torch.LongTensor(x) for x in inputs["traj_nav_types"]
        ]
        output["traj_cand_vpids"] = inputs["traj_cand_vpids"]
        output["traj_vpids"] = inputs["traj_vpids"]

        output["gmap_vpids"] = inputs["gmap_vpids"]
        output["gmap_step_ids"] = torch.LongTensor(inputs["gmap_step_ids"])
        output["gmap_visited_masks"] = torch.BoolTensor(inputs["gmap_visited_masks"])
        output["gmap_pos_fts"] = torch.from_numpy(inputs["gmap_pos_fts"])
        output["gmap_pair_dists"] = torch.from_numpy(inputs["gmap_pair_dists"])

        task_type_encoding = inputs["task_type_encoding"]
        if task_type_encoding == 1:
            output["txt_task_encoding"] = torch.full_like(output["txt_ids"], 1)
            output["gmap_task_embeddings"] = torch.full_like(output["gmap_step_ids"], 1)
        elif task_type_encoding == 2:
            output["txt_task_encoding"] = torch.full_like(output["txt_ids"], 2)
            output["gmap_task_embeddings"] = torch.full_like(output["gmap_step_ids"], 2)
        elif task_type_encoding == 3:
            output["txt_task_encoding"] = torch.full_like(output["txt_ids"], 3)
            output["gmap_task_embeddings"] = torch.full_like(output["gmap_step_ids"], 1)
        else:
            print(
                "*************************** task_type_encoding ERROR ***************************"
            )
            output["txt_task_encoding"] = None
            output["gmap_task_embeddings"] = None

        _copy_cognitive_map_inputs(inputs, output)
        _copy_pose_gated_map_inputs(inputs, output)
        return output


def mlm_collate(inputs):
    batch = {k: [x[k] for x in inputs] for k in inputs[0].keys()}

    if "cognitive_maps" in batch:
        batch["cognitive_maps"] = torch.stack(batch["cognitive_maps"])
        batch["trajectory_keypoints"] = torch.stack(batch["trajectory_keypoints"])
        batch["map_trajectory_metadata"] = torch.stack(batch["map_trajectory_metadata"])
        batch["start_direction_vectors"] = torch.stack(batch["start_direction_vectors"])
        batch["start_positions"] = torch.stack(batch["start_positions"])
    if "traj_spatial_view_fts" in batch:
        batch["traj_spatial_view_fts"] = pad_tensors(sum(batch["traj_spatial_view_fts"], []))
        batch["traj_spatial_dep_fts"] = pad_tensors(sum(batch["traj_spatial_dep_fts"], []))
        batch["spatial_semantic_targets"] = pad_tensors(sum(batch["spatial_semantic_targets"], []))
        batch["spatial_coverage_targets"] = pad_tensors(sum(batch["spatial_coverage_targets"], []))
        batch["traj_positions"] = pad_tensors(sum(batch["traj_positions"], []))
        batch["traj_rotations"] = pad_tensors(sum(batch["traj_rotations"], []))
        batch["target_cognitive_maps"] = torch.stack(batch["target_cognitive_maps"])
        batch["route_negative_spatial_view_fts"] = torch.stack(
            batch["route_negative_spatial_view_fts"]
        )
        batch["route_negative_spatial_dep_fts"] = torch.stack(
            batch["route_negative_spatial_dep_fts"]
        )

    batch["txt_lens"] = torch.LongTensor([len(x) for x in batch["txt_ids"]])
    batch["txt_ids"] = pad_sequence(batch["txt_ids"], batch_first=True, padding_value=1)
    batch["txt_labels"] = pad_sequence(
        batch["txt_labels"], batch_first=True, padding_value=-1
    )
    batch["txt_task_encoding"] = pad_sequence(
        batch["txt_task_encoding"], batch_first=True, padding_value=0
    )
    batch["gmap_task_embeddings"] = pad_sequence(
        batch["gmap_task_embeddings"], batch_first=True, padding_value=0
    )

    batch["traj_step_lens"] = [len(x) for x in batch["traj_view_img_fts"]]
    batch["traj_vp_view_lens"] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch["traj_view_img_fts"]], [])
    )

    batch["traj_view_img_fts"] = pad_tensors(sum(batch["traj_view_img_fts"], []))
    batch["traj_view_dep_fts"] = pad_tensors(sum(batch["traj_view_dep_fts"], []))
    if "traj_obj_img_fts" in batch:
        batch["traj_vp_obj_lens"] = torch.LongTensor(
            sum([[len(y) for y in x] for x in batch["traj_obj_img_fts"]], [])
        )
        batch["traj_obj_img_fts"] = pad_tensors(sum(batch["traj_obj_img_fts"], []))
    batch["traj_loc_fts"] = pad_tensors(sum(batch["traj_loc_fts"], []))
    batch["traj_nav_types"] = pad_sequence(
        sum(batch["traj_nav_types"], []), batch_first=True, padding_value=0
    )

    batch["gmap_lens"] = torch.LongTensor([len(x) for x in batch["gmap_step_ids"]])
    batch["gmap_step_ids"] = pad_sequence(
        batch["gmap_step_ids"], batch_first=True, padding_value=0
    )
    batch["gmap_visited_masks"] = pad_sequence(
        batch["gmap_visited_masks"], batch_first=True, padding_value=0
    )
    batch["gmap_pos_fts"] = pad_tensors(batch["gmap_pos_fts"])
    max_gmap_len = max(batch["gmap_lens"])
    batch_size = len(batch["gmap_lens"])
    gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
    for i in range(batch_size):
        gmap_pair_dists[i, : batch["gmap_lens"][i], : batch["gmap_lens"][i]] = batch[
            "gmap_pair_dists"
        ][i]
    batch["gmap_pair_dists"] = gmap_pair_dists

    return batch


############### Single-step Action Prediction ###############
class SapDataset(Dataset):
    def __init__(self, nav_db, tok, end_vp_pos_ratio=0.2):
        """Instruction Trajectory Matching"""
        self.nav_db = nav_db
        self.tok = tok

        self.cls_token_id = self.tok.cls_token_id  # 101
        self.sep_token_id = self.tok.sep_token_id  # 102
        self.pad_token_id = self.tok.pad_token_id  # 0

        self.end_vp_pos_ratio = end_vp_pos_ratio

    def __len__(self):
        return len(self.nav_db.data)

    def __getitem__(self, idx):
        r = np.random.rand()
        if r < self.end_vp_pos_ratio:
            end_vp_type = "pos"
        elif r < 0.6:
            end_vp_type = "neg_in_gt_path"
        else:
            end_vp_type = "neg_others"
        inputs = self.nav_db.get_input(idx, end_vp_type, return_act_label=True)

        output = {}

        output["txt_ids"] = torch.LongTensor(inputs["instr_encoding"])

        output["traj_view_img_fts"] = [
            torch.from_numpy(x) for x in inputs["traj_view_img_fts"]
        ]
        output["traj_view_dep_fts"] = [
            torch.from_numpy(x) for x in inputs["traj_view_dep_fts"]
        ]
        if "traj_obj_img_fts" in inputs:
            output["traj_obj_img_fts"] = [
                torch.from_numpy(x) for x in inputs["traj_obj_img_fts"]
            ]
        output["traj_loc_fts"] = [torch.from_numpy(x) for x in inputs["traj_loc_fts"]]
        output["traj_nav_types"] = [
            torch.LongTensor(x) for x in inputs["traj_nav_types"]
        ]
        output["traj_cand_vpids"] = inputs["traj_cand_vpids"]
        output["traj_vpids"] = inputs["traj_vpids"]

        output["gmap_vpids"] = inputs["gmap_vpids"]
        output["gmap_step_ids"] = torch.LongTensor(inputs["gmap_step_ids"])
        output["gmap_visited_masks"] = torch.BoolTensor(inputs["gmap_visited_masks"])
        output["gmap_pos_fts"] = torch.from_numpy(inputs["gmap_pos_fts"])
        output["gmap_pair_dists"] = torch.from_numpy(inputs["gmap_pair_dists"])

        task_type_encoding = inputs["task_type_encoding"]
        if task_type_encoding == 1:
            output["txt_task_encoding"] = torch.full_like(output["txt_ids"], 1)
            output["gmap_task_embeddings"] = torch.full_like(output["gmap_step_ids"], 1)
        elif task_type_encoding == 2:
            output["txt_task_encoding"] = torch.full_like(output["txt_ids"], 2)
            output["gmap_task_embeddings"] = torch.full_like(output["gmap_step_ids"], 2)
        elif task_type_encoding == 3:
            output["txt_task_encoding"] = torch.full_like(output["txt_ids"], 3)
            output["gmap_task_embeddings"] = torch.full_like(output["gmap_step_ids"], 1)
        else:
            print(
                "*************************** task_type_encoding ERROR ***************************"
            )
            output["txt_task_encoding"] = None
            output["gmap_task_embeddings"] = None

        output["local_act_labels"] = inputs["local_act_labels"]
        output["global_act_labels"] = inputs["global_act_labels"]
        _copy_cognitive_map_inputs(inputs, output)
        _copy_pose_gated_map_inputs(inputs, output)
        return output


def sap_collate(inputs):
    batch = {k: [x[k] for x in inputs] for k in inputs[0].keys()}

    if "cognitive_maps" in batch:
        batch["cognitive_maps"] = torch.stack(batch["cognitive_maps"])
        batch["trajectory_keypoints"] = torch.stack(batch["trajectory_keypoints"])
        batch["map_trajectory_metadata"] = torch.stack(batch["map_trajectory_metadata"])
        batch["start_direction_vectors"] = torch.stack(batch["start_direction_vectors"])
        batch["start_positions"] = torch.stack(batch["start_positions"])
    if "traj_spatial_view_fts" in batch:
        batch["traj_spatial_view_fts"] = pad_tensors(sum(batch["traj_spatial_view_fts"], []))
        batch["traj_spatial_dep_fts"] = pad_tensors(sum(batch["traj_spatial_dep_fts"], []))
        batch["spatial_semantic_targets"] = pad_tensors(sum(batch["spatial_semantic_targets"], []))
        batch["spatial_coverage_targets"] = pad_tensors(sum(batch["spatial_coverage_targets"], []))
        batch["traj_positions"] = pad_tensors(sum(batch["traj_positions"], []))
        batch["traj_rotations"] = pad_tensors(sum(batch["traj_rotations"], []))
        batch["target_cognitive_maps"] = torch.stack(batch["target_cognitive_maps"])
        batch["route_negative_spatial_view_fts"] = torch.stack(
            batch["route_negative_spatial_view_fts"]
        )
        batch["route_negative_spatial_dep_fts"] = torch.stack(
            batch["route_negative_spatial_dep_fts"]
        )

    batch["txt_lens"] = torch.LongTensor([len(x) for x in batch["txt_ids"]])
    batch["txt_ids"] = pad_sequence(batch["txt_ids"], batch_first=True, padding_value=1)
    batch["txt_task_encoding"] = pad_sequence(
        batch["txt_task_encoding"], batch_first=True, padding_value=0
    )
    batch["gmap_task_embeddings"] = pad_sequence(
        batch["gmap_task_embeddings"], batch_first=True, padding_value=0
    )

    batch["traj_step_lens"] = [len(x) for x in batch["traj_view_img_fts"]]
    batch["traj_vp_view_lens"] = torch.LongTensor(
        sum([[len(y) for y in x] for x in batch["traj_view_img_fts"]], [])
    )
    batch["traj_view_img_fts"] = pad_tensors(sum(batch["traj_view_img_fts"], []))
    batch["traj_view_dep_fts"] = pad_tensors(sum(batch["traj_view_dep_fts"], []))
    if "traj_obj_img_fts" in batch:
        batch["traj_vp_obj_lens"] = torch.LongTensor(
            sum([[len(y) for y in x] for x in batch["traj_obj_img_fts"]], [])
        )
        batch["traj_obj_img_fts"] = pad_tensors(sum(batch["traj_obj_img_fts"], []))
    batch["traj_loc_fts"] = pad_tensors(sum(batch["traj_loc_fts"], []))
    batch["traj_nav_types"] = pad_sequence(
        sum(batch["traj_nav_types"], []), batch_first=True, padding_value=0
    )

    batch["gmap_lens"] = torch.LongTensor([len(x) for x in batch["gmap_step_ids"]])
    batch["gmap_step_ids"] = pad_sequence(
        batch["gmap_step_ids"], batch_first=True, padding_value=0
    )
    batch["gmap_visited_masks"] = pad_sequence(
        batch["gmap_visited_masks"], batch_first=True, padding_value=0
    )
    batch["gmap_pos_fts"] = pad_tensors(batch["gmap_pos_fts"])
    max_gmap_len = max(batch["gmap_lens"])
    batch_size = len(batch["gmap_lens"])
    gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
    for i in range(batch_size):
        gmap_pair_dists[i, : batch["gmap_lens"][i], : batch["gmap_lens"][i]] = batch[
            "gmap_pair_dists"
        ][i]
    batch["gmap_pair_dists"] = gmap_pair_dists

    batch["local_act_labels"] = torch.LongTensor(batch["local_act_labels"])
    batch["global_act_labels"] = torch.LongTensor(batch["global_act_labels"])

    return batch
