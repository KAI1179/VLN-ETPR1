"""ImaginedPolicy — ETP variant that predicts cognitive maps from instruction."""

import torch

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.models.etp_prior_gt.map_encoder import EmbeddingGridMapEncoder
from vlnce_baselines.models.etp_prior_gt.map_utils import (
    DIRECTION_VECTOR_CNT,
    NUM_MAP_CATEGORIES,
    SIZE,
)
from vlnce_baselines.models.etp_prior_gt.policy import ETP_PriorGT
from vlnce_baselines.models.etp_imagined.instruction_map_predictor import (
    InstructionCognitiveMapPredictor,
)
from vlnce_baselines.models.policy import ILPolicy


@baseline_registry.register_policy
class ImaginedPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
        dropout_rate=0.1,
    ):
        super().__init__(
            ETP_Imagined(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
                dropout_rate=dropout_rate,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space, dropout_rate=0.1
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


class ETP_Imagined(ETP_PriorGT):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions, dropout_rate
    ):
        super().__init__(observation_space, model_config, num_actions, dropout_rate)
        hidden_size = self.vln_bert.config.hidden_size
        if not hasattr(self, "map_encoder"):
            self.map_encoder = EmbeddingGridMapEncoder(hidden_size=hidden_size)
            self.map_encoder_enabled = True
        self.map_predictor = InstructionCognitiveMapPredictor(
            hidden_size=hidden_size,
            num_heads=self.vln_bert.config.num_attention_heads,
            num_layers=2,
            dropout=0.0,
        )
        print(f"  Imagined map predictor enabled: text -> (37, {SIZE}, {SIZE})")

    def forward(self, mode=None, **kwargs):
        if mode == "predict_cognitive_map":
            return self.forward_predict_cognitive_map(
                kwargs["txt_embeds"],
                kwargs["txt_masks"],
            )
        if mode == "imagined_map_encoding":
            return self.forward_imagined_map_encoding(
                kwargs["txt_embeds"],
                kwargs["txt_masks"],
            )
        return super().forward(mode=mode, **kwargs)

    def forward_predict_cognitive_map(self, txt_embeds, txt_masks):
        return self.map_predictor(txt_embeds, txt_masks)

    def forward_imagined_map_encoding(self, txt_embeds, txt_masks):
        map_logits = self.forward_predict_cognitive_map(txt_embeds, txt_masks)
        map_probs = torch.sigmoid(map_logits)
        batch_size = map_probs.shape[0]
        direction_vectors = map_probs.new_zeros(batch_size, DIRECTION_VECTOR_CNT, 2)
        start_direction_vectors = map_probs.new_zeros(batch_size, 2)
        start_positions = map_probs.new_zeros(batch_size, 2)
        map_tokens, map_token_masks = self.forward(
            mode="map_encoding",
            cognitive_crops=map_probs,
            direction_vectors=direction_vectors,
            start_direction_vectors=start_direction_vectors,
            start_positions=start_positions,
        )
        return map_logits, map_tokens, map_token_masks
