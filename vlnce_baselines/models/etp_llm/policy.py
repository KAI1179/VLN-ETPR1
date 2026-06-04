"""LLMPolicy scaffold for the LLM-Navigation milestone."""

from typing import Any

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.models.etp_prior_gt.policy import ETP_PriorGT
from vlnce_baselines.models.policy import ILPolicy


@baseline_registry.register_policy
class LLMPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Any,
        model_config: Config,
        dropout_rate=0.1,
    ):
        super().__init__(
            ETP_LLMNavigation(
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
        action_space: Any,
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


class ETP_LLMNavigation(ETP_PriorGT):
    """Navigation scaffold for LLM-derived cognitive maps."""

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
        reference_paths=None,
        start_direction_vectors=None,
        start_positions=None,
        map_tokens=None,
        map_token_masks=None,
    ):
        return super().forward(
            mode=mode,
            txt_ids=txt_ids,
            txt_task_encoding=txt_task_encoding,
            txt_masks=txt_masks,
            txt_embeds=txt_embeds,
            waypoint_predictor=waypoint_predictor,
            observations=observations,
            in_train=in_train,
            rgb_fts=rgb_fts,
            dep_fts=dep_fts,
            loc_fts=loc_fts,
            nav_types=nav_types,
            view_lens=view_lens,
            gmap_vp_ids=gmap_vp_ids,
            gmap_step_ids=gmap_step_ids,
            gmap_img_fts=gmap_img_fts,
            gmap_pos_fts=gmap_pos_fts,
            gmap_masks=gmap_masks,
            gmap_visited_masks=gmap_visited_masks,
            gmap_pair_dists=gmap_pair_dists,
            gmap_task_embeddings=gmap_task_embeddings,
            cognitive_crops=cognitive_crops,
            reference_paths=reference_paths,
            start_direction_vectors=start_direction_vectors,
            start_positions=start_positions,
            map_tokens=map_tokens,
            map_token_masks=map_token_masks,
        )
