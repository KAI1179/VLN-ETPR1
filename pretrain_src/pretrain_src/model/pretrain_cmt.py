from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from vlnce_baselines.models.etp_imagined.checkpoint import load_complete_state_dict

from .vilmodel import (
    BertLayerNorm,
    BertOnlyMLMHead,
    GlocalTextPathCMT,
    gelu,
    BertOutAttention,
)
from .ops import gen_seq_masks, extend_neg_masks


class RegionClassification(nn.Module):
    "for MRC(-kl)"

    def __init__(self, config, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            BertLayerNorm(hidden_size, eps=config.layer_norm_eps),
            nn.Linear(hidden_size, label_dim),
        )

    def forward(self, input_):
        output = self.net(input_)
        return output


class ClsPrediction(nn.Module):
    def __init__(self, config, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            BertLayerNorm(hidden_size, eps=config.layer_norm_eps),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


class ResidualTransformBlock(nn.Module):
    def __init__(self, config, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = gelu
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = x + residual
        x = self.LayerNorm(x)
        return x


class NextActionPrediction(nn.Module):
    def __init__(self, config, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            BertLayerNorm(hidden_size * 2, eps=config.layer_norm_eps),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class GlocalTextPathCMTPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = GlocalTextPathCMT(config)
        self.use_imagined = getattr(config, "use_imagined", False)
        self.use_prior_gt = getattr(config, "use_prior_gt", False)
        self.use_llm = getattr(config, "use_llm", False)
        self.map_loss_weight = getattr(config, "map_loss_weight", 0.1)
        self.trajectory_keypoint_loss_weight = getattr(
            config, "trajectory_keypoint_loss_weight", 0.001
        )
        self.map_encoder = None
        self.map_decoder = None
        self.map_box_criterion = None
        self.map_predictor = None

        if "mlm" in config.pretrain_tasks:
            self.mlm_head = BertOnlyMLMHead(self.config)
        if "sap" in config.pretrain_tasks:
            self.graph_query_text = BertOutAttention(config)
            self.graph_attentioned_txt_embeds_transform = ResidualTransformBlock(
                self.config, self.config.hidden_size, self.config.hidden_dropout_prob
            )
            self.global_sap_head = NextActionPrediction(
                self.config, self.config.hidden_size, self.config.pred_head_dropout_prob
            )

        map_inputs_enabled = self.use_imagined or self.use_prior_gt or self.use_llm
        if map_inputs_enabled:
            try:
                from vlnce_baselines.models.etp_prior_gt.map_encoder import (
                    EmbeddingGridMapEncoder,
                )
                from vlnce_baselines.models.etp_prior_gt.map_decoder import (
                    CognitiveMapDecoder,
                    CognitiveMapSetCriterion,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Map-enabled pretraining requires etp_prior_gt map modules "
                    "to be importable from PYTHONPATH."
                ) from exc

            print(
                "Successfully imported EmbeddingGridMapEncoder, initializing map encoder..."
            )
            self.map_encoder = EmbeddingGridMapEncoder(
                hidden_size=self.config.hidden_size
            )
            self.map_decoder = CognitiveMapDecoder(hidden_size=self.config.hidden_size)
            self.map_box_criterion = CognitiveMapSetCriterion()
            if self.use_imagined:
                from vlnce_baselines.models.etp_imagined.instruction_map_predictor import (
                    InstructionCognitiveMapPredictor,
                )

                self.map_predictor = InstructionCognitiveMapPredictor(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    num_layers=2,
                    dropout=0.0,
                )
                print(
                    "Successfully initialized InstructionCognitiveMapPredictor for imagined pretraining"
                )

        self.init_weights()
        self.tie_weights()
        self._load_map_predictor_checkpoint()

    def _load_map_predictor_checkpoint(self):
        checkpoint_path = getattr(self.config, "map_predictor_checkpoint", "")
        if not checkpoint_path:
            return
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        if (
            not getattr(self, "use_imagined", False)
            or self.map_predictor is None
        ):
            raise ValueError("--map_predictor_checkpoint requires --use_imagined")
        map_predictor = self.map_predictor

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "map_predictor" in checkpoint:
            state_dict = checkpoint["map_predictor"]
        else:
            raw_state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = {}
            for key, value in raw_state_dict.items():
                normalized_key = key
                if normalized_key.startswith("module."):
                    normalized_key = normalized_key[len("module.") :]
                if normalized_key.startswith("map_predictor."):
                    normalized_key = normalized_key[len("map_predictor.") :]
                elif normalized_key.startswith("predictor."):
                    normalized_key = normalized_key[len("predictor.") :]
                else:
                    continue
                state_dict[normalized_key] = value
        if not state_dict:
            raise ValueError(f"No map predictor weights found in {checkpoint_path}")
        load_complete_state_dict(
            map_predictor,
            state_dict,
            checkpoint_path,
            "map_predictor",
        )
        print(f"Loaded map predictor checkpoint: {checkpoint_path}")

    def tie_weights(self):
        if "mlm" in self.config.pretrain_tasks:
            self._tie_or_clone_weights(
                self.mlm_head.predictions.decoder, self.bert.embeddings.word_embeddings
            )

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)

        map_tokens, map_token_masks, map_loss = self._prepare_map_inputs(
            batch, compute_loss
        )

        if task.startswith("mlm"):
            losses = self.forward_mlm(
                batch["txt_ids"],
                batch["txt_lens"],
                batch["txt_task_encoding"],
                batch["traj_view_img_fts"],
                batch["traj_view_dep_fts"],
                batch["traj_obj_img_fts"],
                batch["traj_loc_fts"],
                batch["traj_nav_types"],
                batch["traj_step_lens"],
                batch["traj_vp_view_lens"],
                batch["traj_vp_obj_lens"],
                batch["traj_vpids"],
                batch["traj_cand_vpids"],
                batch["gmap_lens"],
                batch["gmap_step_ids"],
                batch["gmap_task_embeddings"],
                batch["gmap_pos_fts"],
                batch["gmap_pair_dists"],
                batch["gmap_vpids"],
                batch["txt_labels"],
                compute_loss,
                cognitive_maps=batch["cognitive_maps"],
                cognitive_map_box_targets=batch["cognitive_map_box_targets"],
                map_tokens=map_tokens,
                map_token_masks=map_token_masks,
            )
            return (
                losses + map_loss if compute_loss and map_loss is not None else losses
            )
        elif task.startswith("sap"):
            losses = self.forward_sap(
                batch["txt_ids"],
                batch["txt_lens"],
                batch["txt_task_encoding"],
                batch["traj_view_img_fts"],
                batch["traj_view_dep_fts"],
                batch["traj_obj_img_fts"],
                batch["traj_loc_fts"],
                batch["traj_nav_types"],
                batch["traj_step_lens"],
                batch["traj_vp_view_lens"],
                batch["traj_vp_obj_lens"],
                batch["traj_vpids"],
                batch["traj_cand_vpids"],
                batch["gmap_lens"],
                batch["gmap_step_ids"],
                batch["gmap_task_embeddings"],
                batch["gmap_pos_fts"],
                batch["gmap_pair_dists"],
                batch["gmap_vpids"],
                batch["gmap_visited_masks"],
                batch["global_act_labels"],
                batch["local_act_labels"],
                compute_loss,
                cognitive_maps=batch["cognitive_maps"],
                cognitive_map_box_targets=batch["cognitive_map_box_targets"],
                map_tokens=map_tokens,
                map_token_masks=map_token_masks,
            )
            return (
                losses + map_loss if compute_loss and map_loss is not None else losses
            )
        else:
            raise ValueError("invalid task")

    def _prepare_map_inputs(self, batch, compute_loss=True):
        if (
            "cognitive_maps" not in batch
            or batch["cognitive_maps"] is None
            or self.map_encoder is None
        ):
            return None, None, None
        map_encoder = self.map_encoder

        if self.use_imagined:
            if self.map_predictor is None:
                raise RuntimeError("use_imagined requires map_predictor")
            map_predictor = self.map_predictor
            txt_token_type_ids = torch.zeros_like(batch["txt_ids"])
            txt_embeds = self.bert.embeddings(
                batch["txt_ids"],
                batch["txt_task_encoding"],
                token_type_ids=txt_token_type_ids,
            )
            txt_masks = gen_seq_masks(batch["txt_lens"])
            txt_embeds = self.bert.lang_encoder(txt_embeds, txt_masks)
            map_logits, pred_trajectory_keypoints = map_predictor(
                txt_embeds,
                txt_masks,
                start_direction_vectors=batch["start_direction_vectors"],
                start_positions=batch["start_positions"],
            )
            map_tokens, map_token_masks = map_encoder(
                torch.sigmoid(map_logits),
                pred_trajectory_keypoints,
                batch["start_direction_vectors"],
                batch["start_positions"],
            )
            map_loss = None
            if compute_loss:
                map_loss = F.binary_cross_entropy_with_logits(
                    map_logits,
                    batch["cognitive_maps"],
                    reduction="mean",
                )
                trajectory_keypoint_loss = F.smooth_l1_loss(
                    pred_trajectory_keypoints,
                    batch["trajectory_keypoints"],
                    beta=5.0,
                )
                map_loss = self.map_loss_weight * (
                    map_loss
                    + self.trajectory_keypoint_loss_weight * trajectory_keypoint_loss
                )
            return map_tokens, map_token_masks, map_loss

        if self.use_prior_gt or self.use_llm:
            # Pretraining collate stacks cached map tensors to:
            # cognitive_maps=(B, 37, 100, 100), keypoints=(B, 5, 2),
            # direction=(B, 2), start=(B, 2). The map encoder returns
            # map_tokens=(B, 101, hidden_size), map_token_masks=(B, 101).
            map_tokens, map_token_masks = map_encoder(
                batch["cognitive_maps"],
                batch["trajectory_keypoints"],
                batch["start_direction_vectors"],
                batch["start_positions"],
            )
            return map_tokens, map_token_masks, None

        return None, None, None

    def _compute_updated_cognitive_map_loss(
        self,
        updated_map_tokens,
        cognitive_map_box_targets,
        compute_loss,
    ):
        if not compute_loss or updated_map_tokens is None:
            return None
        if cognitive_map_box_targets is None:
            raise ValueError(
                "cognitive_map_box_targets are required to supervise updated_map_tokens"
            )
        if self.map_decoder is None:
            raise RuntimeError(
                "updated_map_tokens were produced but map_decoder is not initialized"
            )
        if self.map_box_criterion is None:
            raise RuntimeError(
                "updated_map_tokens were produced but map_box_criterion is not initialized"
            )
        decoder_output = self.map_decoder(updated_map_tokens)
        return self.map_loss_weight * self.map_box_criterion(
            decoder_output,
            cognitive_map_box_targets,
        )

    def forward_mlm(
        self,
        txt_ids,
        txt_lens,
        txt_task_encoding,
        traj_view_img_fts,
        traj_view_dep_fts,
        traj_obj_img_fts,
        traj_loc_fts,
        traj_nav_types,
        traj_step_lens,
        traj_vp_view_lens,
        traj_vp_obj_lens,
        traj_vpids,
        traj_cand_vpids,
        gmap_lens,
        gmap_step_ids,
        gmap_task_embeddings,
        gmap_pos_fts,
        gmap_pair_dists,
        gmap_vpids,
        txt_labels,
        compute_loss,
        cognitive_maps=None,
        cognitive_map_box_targets=None,
        map_tokens=None,
        map_token_masks=None,
    ):
        navigation_output = self.bert(
            txt_ids,
            txt_lens,
            txt_task_encoding,
            traj_view_img_fts,
            traj_view_dep_fts,
            traj_obj_img_fts,
            traj_loc_fts,
            traj_nav_types,
            traj_step_lens,
            traj_vp_view_lens,
            traj_vp_obj_lens,
            traj_vpids,
            traj_cand_vpids,
            gmap_lens,
            gmap_step_ids,
            gmap_task_embeddings,
            gmap_pos_fts,
            gmap_pair_dists,
            gmap_vpids,
            map_tokens=map_tokens,
            map_token_masks=map_token_masks,
        )
        txt_embeds = navigation_output.txt_embeds
        updated_map_loss = self._compute_updated_cognitive_map_loss(
            navigation_output.updated_map_tokens,
            cognitive_map_box_targets,
            compute_loss,
        )

        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)

        if compute_loss:
            mask_loss = F.cross_entropy(
                prediction_scores, txt_labels[txt_labels != -1], reduction="none"
            )
            if updated_map_loss is not None:
                mask_loss = mask_loss + updated_map_loss
            return mask_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        """get only the masked region (don't compute unnecessary hiddens)"""
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_sap(
        self,
        txt_ids,
        txt_lens,
        txt_task_encoding,
        traj_view_img_fts,
        traj_view_dep_fts,
        traj_obj_img_fts,
        traj_loc_fts,
        traj_nav_types,
        traj_step_lens,
        traj_vp_view_lens,
        traj_vp_obj_lens,
        traj_vpids,
        traj_cand_vpids,
        gmap_lens,
        gmap_step_ids,
        gmap_task_embeddings,
        gmap_pos_fts,
        gmap_pair_dists,
        gmap_vpids,
        gmap_visited_masks,
        global_act_labels,
        local_act_labels,
        compute_loss,
        cognitive_maps=None,
        cognitive_map_box_targets=None,
        map_tokens=None,
        map_token_masks=None,
    ):
        navigation_output = self.bert(
            txt_ids,
            txt_lens,
            txt_task_encoding,
            traj_view_img_fts,
            traj_view_dep_fts,
            traj_obj_img_fts,
            traj_loc_fts,
            traj_nav_types,
            traj_step_lens,
            traj_vp_view_lens,
            traj_vp_obj_lens,
            traj_vpids,
            traj_cand_vpids,
            gmap_lens,
            gmap_step_ids,
            gmap_task_embeddings,
            gmap_pos_fts,
            gmap_pair_dists,
            gmap_vpids,
            map_tokens=map_tokens,
            map_token_masks=map_token_masks,
        )
        txt_embeds = navigation_output.txt_embeds
        gmap_embeds = navigation_output.gmap_embeds
        updated_map_loss = self._compute_updated_cognitive_map_loss(
            navigation_output.updated_map_tokens,
            cognitive_map_box_targets,
            compute_loss,
        )

        txt_masks = gen_seq_masks(txt_lens)
        extended_txt_masks = extend_neg_masks(txt_masks)
        graph_attentioned_txt_embeds, _ = self.graph_query_text(
            gmap_embeds, txt_embeds, attention_mask=extended_txt_masks
        )
        graph_attentioned_txt_embeds = self.graph_attentioned_txt_embeds_transform(
            graph_attentioned_txt_embeds
        )
        fusion_input = torch.cat([gmap_embeds, graph_attentioned_txt_embeds], dim=-1)
        global_logits = self.global_sap_head(fusion_input).squeeze(2)

        global_logits.masked_fill_(gmap_visited_masks, -float("inf"))
        global_logits.masked_fill_(
            gen_seq_masks(gmap_lens).logical_not(), -float("inf")
        )

        if compute_loss:
            global_losses = F.cross_entropy(
                global_logits, global_act_labels, reduction="none"
            )
            losses = global_losses
            if updated_map_loss is not None:
                losses = losses + updated_map_loss
            return losses
        else:
            return global_logits, global_act_labels
