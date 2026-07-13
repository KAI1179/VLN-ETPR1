"""ImaginedPolicy — ETP variant that predicts cognitive maps from instruction."""

from pathlib import Path

import torch

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.models.cognitive_map_candidate import CognitiveMapCandidate
from vlnce_baselines.models.etp_prior_gt.map_encoder import EmbeddingGridMapEncoder
from vlnce_baselines.models.etp_prior_gt.map_utils import (
    NUM_MAP_CATEGORIES,
    SIZE,
)
from vlnce_baselines.models.etp_prior_gt.policy import ETP_PriorGT
from vlnce_baselines.models.etp_imagined.instruction_map_predictor import (
    InstructionCognitiveMapPredictor,
)
from vlnce_baselines.models.etp_imagined.checkpoint import load_complete_state_dict
from vlnce_baselines.models.policy import ILPolicy


def _extract_module_state_dict(state_dict, module_name):
    prefixes = (
        f"{module_name}.",
        f"net.{module_name}.",
        f"net.module.{module_name}.",
        f"module.{module_name}.",
        f"module.net.{module_name}.",
        f"module.net.module.{module_name}.",
    )
    extracted = {}
    for key, value in state_dict.items():
        normalized_key = key
        for prefix in prefixes:
            if normalized_key.startswith(prefix):
                extracted[normalized_key[len(prefix) :]] = value
                break
    return extracted


def _load_optional_module_checkpoint(module, module_name, checkpoint_path):
    if not checkpoint_path:
        return False
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if module_name == "map_predictor" and "map_predictor" in checkpoint:
        module_state_dict = checkpoint["map_predictor"]
    else:
        module_state_dict = _extract_module_state_dict(
            checkpoint.get("state_dict", checkpoint),
            module_name,
        )
    if not module_state_dict:
        return False
    load_complete_state_dict(
        module,
        module_state_dict,
        checkpoint_path,
        module_name,
    )
    print(f"  Loaded {module_name} weights from {checkpoint_path}")
    return True


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
        cls,
        config: Config,
        observation_space: Space,
        action_space: Space,
        dropout_rate=0.1,
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        candidate = CognitiveMapCandidate.parse(
            config.MODEL.MAP_ENCODER.architecture,
            config.MODEL.MAP_ENCODER.source,
        )
        expected = CognitiveMapCandidate.parse("current", "imagined")
        if candidate != expected:
            raise ValueError(
                "ImaginedPolicy requires architecture=current, source=imagined"
            )
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
        self._load_map_module_weights(model_config)
        print(
            f"  Imagined map predictor enabled: text + start pose -> "
            f"({NUM_MAP_CATEGORIES}, {SIZE}, {SIZE}) + trajectory keypoints"
        )

    def _load_map_module_weights(self, model_config):
        map_cfg = getattr(model_config, "MAP_ENCODER", None)
        pretrained_path = getattr(model_config, "pretrained_path", "")
        predictor_checkpoint = (
            getattr(map_cfg, "predictor_checkpoint", "") if map_cfg is not None else ""
        )

        _load_optional_module_checkpoint(
            self.map_encoder,
            "map_encoder",
            pretrained_path,
        )
        loaded_from_pretrain = _load_optional_module_checkpoint(
            self.map_predictor,
            "map_predictor",
            pretrained_path,
        )
        if predictor_checkpoint:
            loaded_from_predictor = _load_optional_module_checkpoint(
                self.map_predictor,
                "map_predictor",
                predictor_checkpoint,
            )
            if not loaded_from_predictor and not loaded_from_pretrain:
                raise ValueError(
                    f"No map_predictor weights found in {predictor_checkpoint}"
                )

    def forward(self, mode=None, **kwargs):
        if mode == "predict_cognitive_map":
            return self.forward_predict_cognitive_map(
                kwargs["txt_embeds"],
                kwargs["txt_masks"],
                kwargs.get("start_direction_vectors"),
                kwargs.get("start_positions"),
            )
        if mode == "imagined_map_encoding":
            return self.forward_imagined_map_encoding(
                kwargs["txt_embeds"],
                kwargs["txt_masks"],
                kwargs.get("start_direction_vectors"),
                kwargs.get("start_positions"),
            )
        return super().forward(mode=mode, **kwargs)

    def forward_predict_cognitive_map(
        self,
        txt_embeds,
        txt_masks,
        start_direction_vectors=None,
        start_positions=None,
    ):
        return self.map_predictor(
            txt_embeds,
            txt_masks,
            start_direction_vectors=start_direction_vectors,
            start_positions=start_positions,
        )

    def forward_imagined_map_encoding(
        self,
        txt_embeds,
        txt_masks,
        start_direction_vectors=None,
        start_positions=None,
    ):
        map_logits, trajectory_keypoints = self.forward_predict_cognitive_map(
            txt_embeds,
            txt_masks,
            start_direction_vectors=start_direction_vectors,
            start_positions=start_positions,
        )
        map_probs = torch.sigmoid(map_logits)
        batch_size = map_probs.shape[0]
        if start_direction_vectors is None:
            start_direction_vectors = map_probs.new_zeros(batch_size, 2)
        if start_positions is None:
            start_positions = map_probs.new_zeros(batch_size, 2)
        map_tokens, map_token_masks = self.forward(
            mode="map_encoding",
            cognitive_crops=map_probs,
            trajectory_keypoints=trajectory_keypoints,
            start_direction_vectors=start_direction_vectors,
            start_positions=start_positions,
        )
        return map_logits, trajectory_keypoints, map_tokens, map_token_masks
