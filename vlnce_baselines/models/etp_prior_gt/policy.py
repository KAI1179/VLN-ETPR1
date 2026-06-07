"""PriorGTPolicy and ETP_PriorGT — ETP-R1 variant with embedding grid map support.

This module is a self-contained copy of vlnce_baselines/models/R1Policy.py with
two additions:
  1. ETP_PriorGT.forward() has a new 'map_encoding' mode that calls EmbeddingGridMapEncoder.
  2. ETP_PriorGT.forward() passes map_tokens/map_token_masks to forward_navigation().

The author's R1Policy.py and etp/ directory are not modified.
"""

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.models.etp_prior_gt.vlnbert_init import get_vlnbert_models
from vlnce_baselines.models.etp_prior_gt.map_encoder import EmbeddingGridMapEncoder
from vlnce_baselines.models.encoders.resnet_encoders import (
    VlnResnetDepthEncoder,
    CLIPEncoder,
)
from vlnce_baselines.models.policy import ILPolicy

from vlnce_baselines.waypoint_pred.utils import nms
from vlnce_baselines.models.utils import angle_feature_torch
import math


@baseline_registry.register_policy
class PriorGTPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
        dropout_rate=0.1,
    ):
        super().__init__(
            ETP_PriorGT(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
                dropout_rate=dropout_rate,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        observation_space: Space,
        action_space: Space,
        dropout_rate=0.1,
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
            dropout_rate=dropout_rate,
        )


class Critic(nn.Module):
    def __init__(self, drop_ratio):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()


class ETP_PriorGT(Net):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions, dropout_rate
    ):
        super().__init__()

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        print("\nInitalizing the ETP_PriorGT model ...")
        self.vln_bert = get_vlnbert_models(
            config=model_config, dropout_rate=dropout_rate
        )
        self.drop_env = nn.Dropout(p=0.4)

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in ["VlnResnetDepthEncoder"], (
            "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        )
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=model_config.spatial_output,
        )
        self.space_pool_depth = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=2)
        )

        self.rgb_encoder = CLIPEncoder(self.device)
        self.space_pool_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=2)
        )

        self.pano_img_idxes = np.arange(0, 12, dtype=np.int64)
        pano_angle_rad_c = (1 - self.pano_img_idxes / 12) * 2 * math.pi
        self.pano_angle_fts = angle_feature_torch(torch.from_numpy(pano_angle_rad_c))

        # Map encoder (optional — enabled by config)
        map_cfg = getattr(model_config, "MAP_ENCODER", None)
        self.map_encoder_enabled = map_cfg is not None and getattr(
            map_cfg, "enabled", False
        )
        if self.map_encoder_enabled:
            map_hidden_size = self.vln_bert.config.hidden_size
            self.map_encoder = EmbeddingGridMapEncoder(
                hidden_size=map_hidden_size,
            )
            print(
                f"  Map encoder enabled: CLIP 37-category init -> map tokens (101, {map_hidden_size})"
            )

    @property
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return 1

    def forward(
        self,
        mode=None,
        txt_ids=None,
        txt_task_encoding=None,
        txt_masks=None,
        txt_embeds=None,
        waypoint_predictor=None,
        observations=None,
        in_train=True,
        rgb_fts=None,
        dep_fts=None,
        loc_fts=None,
        nav_types=None,
        view_lens=None,
        gmap_vp_ids=None,
        gmap_step_ids=None,
        gmap_img_fts=None,
        gmap_pos_fts=None,
        gmap_masks=None,
        gmap_visited_masks=None,
        gmap_pair_dists=None,
        gmap_task_embeddings=None,
        cognitive_crops=None,
        trajectory_keypoints=None,
        start_direction_vectors=None,
        start_positions=None,
        map_tokens=None,
        map_token_masks=None,
    ):

        if mode == "language":
            encoded_sentence = self.vln_bert.forward_txt(
                txt_ids,
                txt_task_encoding,
                txt_masks,
            )
            return encoded_sentence

        elif mode == "waypoint":
            batch_size = observations["rgb"].shape[0]
            NUM_ANGLES = 120
            NUM_IMGS = 12
            NUM_CLASSES = 12
            depth_batch = torch.zeros_like(observations["depth"]).repeat(
                NUM_IMGS, 1, 1, 1
            )
            rgb_batch = torch.zeros_like(observations["rgb"]).repeat(NUM_IMGS, 1, 1, 1)

            a_count = 0
            for i, (k, v) in enumerate(observations.items()):
                if "depth" in k:
                    for bi in range(v.size(0)):
                        ra_count = (NUM_IMGS - a_count) % NUM_IMGS
                        depth_batch[ra_count + bi * NUM_IMGS] = v[bi]
                        rgb_batch[ra_count + bi * NUM_IMGS] = observations[
                            k.replace("depth", "rgb")
                        ][bi]
                    a_count += 1
            obs_view12 = {}
            obs_view12["depth"] = depth_batch
            obs_view12["rgb"] = rgb_batch
            depth_embedding = self.depth_encoder(obs_view12)
            rgb_embedding = self.rgb_encoder(obs_view12)

            waypoint_heatmap_logits = waypoint_predictor(rgb_embedding, depth_embedding)

            rgb_embed_reshape = rgb_embedding.reshape(batch_size, NUM_IMGS, 512, 1, 1)
            depth_embed_reshape = depth_embedding.reshape(
                batch_size, NUM_IMGS, 128, 4, 4
            )
            rgb_feats = torch.cat(
                (
                    rgb_embed_reshape[:, 0:1, :],
                    torch.flip(rgb_embed_reshape[:, 1:, :], [1]),
                ),
                dim=1,
            )
            depth_feats = torch.cat(
                (
                    depth_embed_reshape[:, 0:1, :],
                    torch.flip(depth_embed_reshape[:, 1:, :], [1]),
                ),
                dim=1,
            )

            batch_x_norm = torch.softmax(
                waypoint_heatmap_logits.reshape(
                    batch_size,
                    NUM_ANGLES * NUM_CLASSES,
                ),
                dim=1,
            )
            batch_x_norm = batch_x_norm.reshape(
                batch_size,
                NUM_ANGLES,
                NUM_CLASSES,
            )
            batch_x_norm_wrap = torch.cat(
                (batch_x_norm[:, -1:, :], batch_x_norm, batch_x_norm[:, :1, :]), dim=1
            )
            batch_output_map = nms(
                batch_x_norm_wrap.unsqueeze(1), max_predictions=5, sigma=(7.0, 5.0)
            )

            batch_output_map = batch_output_map.squeeze(1)[:, 1:-1, :]

            if in_train:
                HEATMAP_OFFSET = 5
                batch_way_heats_regional = torch.cat(
                    (
                        waypoint_heatmap_logits[:, -HEATMAP_OFFSET:, :],
                        waypoint_heatmap_logits[:, :-HEATMAP_OFFSET, :],
                    ),
                    dim=1,
                )
                batch_way_heats_regional = batch_way_heats_regional.reshape(
                    batch_size, 12, 10, 12
                )
                batch_sample_angle_idxes = []
                batch_sample_distance_idxes = []
                for j in range(batch_size):
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    img_idxes = (angle_idxes.cpu().numpy() + 5) // 10
                    img_idxes[img_idxes == 12] = 0
                    way_heats_regional = batch_way_heats_regional[j][img_idxes].view(
                        img_idxes.size, -1
                    )
                    way_heats_probs = F.softmax(way_heats_regional, 1)
                    probs_c = torch.distributions.Categorical(way_heats_probs)
                    way_heats_act = probs_c.sample().detach()
                    sample_angle_idxes = []
                    sample_distance_idxes = []
                    for k, way_act in enumerate(way_heats_act):
                        if img_idxes[k] != 0:
                            angle_pointer = (img_idxes[k] - 1) * 10 + 5
                        else:
                            angle_pointer = 0
                        sample_angle_idxes.append(way_act // 12 + angle_pointer)
                        sample_distance_idxes.append(way_act % 12)
                    batch_sample_angle_idxes.append(sample_angle_idxes)
                    batch_sample_distance_idxes.append(sample_distance_idxes)
            else:
                None

            rgb_feats = self.space_pool_rgb(rgb_feats)
            depth_feats = self.space_pool_depth(depth_feats)

            cand_rgb = []
            cand_depth = []
            cand_angle_fts = []
            cand_img_idxes = []
            cand_angles = []
            cand_distances = []
            for j in range(batch_size):
                if in_train:
                    angle_idxes = torch.tensor(batch_sample_angle_idxes[j])
                    distance_idxes = torch.tensor(batch_sample_distance_idxes[j])
                else:
                    angle_idxes = batch_output_map[j].nonzero()[:, 0]
                    distance_idxes = batch_output_map[j].nonzero()[:, 1]
                angle_rad_c = angle_idxes.cpu().float() / 120 * 2 * math.pi
                angle_rad_cc = 2 * math.pi - angle_idxes.float() / 120 * 2 * math.pi
                cand_angle_fts.append(angle_feature_torch(angle_rad_c))
                cand_angles.append(angle_rad_cc.tolist())
                cand_distances.append(((distance_idxes + 1) * 0.25).tolist())
                img_idxes = 12 - (angle_idxes.cpu().numpy() + 5) // 10
                img_idxes[img_idxes == 12] = 0
                cand_img_idxes.append(img_idxes)
                cand_rgb.append(rgb_feats[j, img_idxes, ...])
                cand_depth.append(depth_feats[j, img_idxes, ...])

            pano_rgb = rgb_feats
            pano_depth = depth_feats
            pano_angle_fts = deepcopy(self.pano_angle_fts)
            pano_img_idxes = deepcopy(self.pano_img_idxes)

            outputs = {
                "cand_rgb": cand_rgb,
                "cand_depth": cand_depth,
                "cand_angle_fts": cand_angle_fts,
                "cand_img_idxes": cand_img_idxes,
                "cand_angles": cand_angles,
                "cand_distances": cand_distances,
                "pano_rgb": pano_rgb,
                "pano_depth": pano_depth,
                "pano_angle_fts": pano_angle_fts,
                "pano_img_idxes": pano_img_idxes,
            }

            return outputs

        elif mode == "panorama":
            rgb_fts = self.drop_env(rgb_fts)
            outs = self.vln_bert.forward_panorama(
                rgb_fts,
                dep_fts,
                loc_fts,
                nav_types,
                view_lens,
            )
            return outs

        elif mode == "map_encoding":
            # cognitive_crops: (B, CATEGORIES, H, W) -> tokens (B, 101, H), masks (B, 101)
            assert self.map_encoder_enabled, (
                "map_encoding mode requires MAP_ENCODER.enabled=True"
            )
            return self.map_encoder(
                cognitive_crops,
                trajectory_keypoints,
                start_direction_vectors,
                start_positions,
            )

        elif mode == "navigation":
            outs = self.vln_bert.forward_navigation(
                txt_embeds,
                txt_masks,
                gmap_vp_ids,
                gmap_step_ids,
                gmap_img_fts,
                gmap_pos_fts,
                gmap_masks,
                gmap_visited_masks,
                gmap_pair_dists,
                gmap_task_embeddings,
                map_tokens=map_tokens,
                map_token_masks=map_token_masks,
            )
            return outs
