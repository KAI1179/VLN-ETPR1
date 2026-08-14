"""Shared cognitive-map and graph-token fusion modules."""

from dataclasses import dataclass
from math import isqrt
from typing import Iterator, Optional

import torch
import torch.nn as nn

from .map_utils import MAP_SPATIAL_TOKEN_COUNT, NUM_MAP_CATEGORIES, SIZE


@dataclass(frozen=True)
class OnlineMapFusionOutput:
    """Rich online-fusion output with legacy two-item tuple behavior.

    Existing pretraining and navigation code can continue to use
    ``updated_graph, updated_map = output``. New call sites should use named
    fields so auxiliary predictions and recurrent state are not discarded.
    """

    updated_gmap_embeds: torch.Tensor
    updated_map_tokens: torch.Tensor
    visual_logits: torch.Tensor
    dense_grid_logits: Optional[torch.Tensor]
    phase_logits: torch.Tensor
    route_state_logits: torch.Tensor
    remaining: torch.Tensor
    recovery_logits: torch.Tensor
    action_logit_residuals: torch.Tensor
    ghost_logit_residuals: torch.Tensor
    stop_logit_residuals: torch.Tensor
    map_write_gate: torch.Tensor
    map_write_residual: torch.Tensor
    map_state_residual: torch.Tensor
    ghost_masks: torch.Tensor

    @property
    def map_state(self) -> torch.Tensor:
        """State to pass back as ``previous_map_state`` on the next step."""

        return self.updated_map_tokens

    def legacy_tuple(self):
        return self.updated_gmap_embeds, self.updated_map_tokens

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.legacy_tuple())

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.legacy_tuple()[index]


class GraphMapCrossAttention(nn.Module):
    """Try5 one-way map-token fusion: graph nodes attend to fixed map tokens."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.residual_projection = nn.Linear(hidden_size, hidden_size)
        self._zero_residual_projection()

    def _zero_residual_projection(self):
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def _map_key_padding_mask(self, map_token_masks):
        if map_token_masks is None:
            return None
        if map_token_masks.any(dim=1).logical_not().any():
            map_token_masks = map_token_masks.clone()
            map_token_masks[map_token_masks.any(dim=1).logical_not(), 0] = True
        return map_token_masks.logical_not()

    def forward(self, gmap_embeds, gmap_masks, map_tokens, map_token_masks):
        if map_tokens is None:
            return gmap_embeds, None

        map_context, _ = self.attention(
            gmap_embeds,
            map_tokens,
            map_tokens,
            key_padding_mask=self._map_key_padding_mask(map_token_masks),
            need_weights=False,
        )
        return gmap_embeds + self.residual_projection(map_context), None


class BidirectionalMapTokenFusion(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.map_from_graph_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.graph_from_map_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.map_residual_projection = nn.Linear(hidden_size, hidden_size)
        self.graph_residual_projection = nn.Linear(hidden_size, hidden_size)
        self._zero_residual_projection()

    def _zero_residual_projection(self):
        nn.init.zeros_(self.map_residual_projection.weight)
        nn.init.zeros_(self.map_residual_projection.bias)
        nn.init.zeros_(self.graph_residual_projection.weight)
        nn.init.zeros_(self.graph_residual_projection.bias)

    def _map_key_padding_mask(self, map_token_masks):
        if map_token_masks is None:
            return None
        if map_token_masks.any(dim=1).logical_not().any():
            # MultiheadAttention returns NaNs for rows with all keys masked.
            map_token_masks = map_token_masks.clone()
            map_token_masks[map_token_masks.any(dim=1).logical_not(), 0] = True
        return map_token_masks.logical_not()

    def _graph_key_padding_mask(self, gmap_embeds, gmap_masks):
        if gmap_masks is None:
            graph_key_padding_mask = torch.zeros(
                gmap_embeds.shape[:2],
                dtype=torch.bool,
                device=gmap_embeds.device,
            )
        else:
            graph_key_padding_mask = gmap_masks.logical_not().clone()
        if graph_key_padding_mask.size(1) > 0:
            graph_key_padding_mask[:, 0] = True
        fully_masked = graph_key_padding_mask.all(dim=1)
        if fully_masked.any():
            # Numeric fallback for degenerate rows; normal rollouts have a real graph node.
            graph_key_padding_mask[fully_masked, 0] = False
        return graph_key_padding_mask

    def forward(self, gmap_embeds, gmap_masks, map_tokens, map_token_masks):
        if map_tokens is None:
            return gmap_embeds, None

        graph_key_padding_mask = self._graph_key_padding_mask(gmap_embeds, gmap_masks)
        map_context, _ = self.map_from_graph_attention(
            map_tokens,
            gmap_embeds,
            gmap_embeds,
            key_padding_mask=graph_key_padding_mask,
            need_weights=False,
        )
        updated_map_tokens = map_tokens + self.map_residual_projection(map_context)

        map_key_padding_mask = self._map_key_padding_mask(map_token_masks)
        graph_context, _ = self.graph_from_map_attention(
            gmap_embeds,
            updated_map_tokens,
            updated_map_tokens,
            key_padding_mask=map_key_padding_mask,
            need_weights=False,
        )
        updated_gmap_embeds = gmap_embeds + self.graph_residual_projection(
            graph_context
        )
        return updated_gmap_embeds, updated_map_tokens


class OnlineMapGraphFusion(nn.Module):
    """Recurrently update map tokens from visited graph evidence.

    The module deliberately has no graph-to-grid coordinate correspondence. All
    spatial map tokens query the complete visited graph, then a gated write and
    map self-attention propagate the innovation across the imagined map. The
    updated map is returned to the caller as the recurrent episode state before
    graph entries query it for action prediction.

    ``gmap_visited_masks`` is required because STOP and ghost entries must never
    write the cognitive map. ``new_evidence_mask`` is optional for static
    pretraining batches, where every visited node is treated as new evidence.
    Online rollouts should pass it explicitly so an unchanged observation does
    not update the recurrent state twice. Dense grid decoding is opt-in because
    it is a supervision/inspection path rather than an inference requirement.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        spatial_token_count: int = MAP_SPATIAL_TOKEN_COUNT,
        dense_grid_size: int = SIZE,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError(
                "num_heads must be positive and divide hidden_size, "
                f"got hidden_size={hidden_size}, num_heads={num_heads}"
            )
        if spatial_token_count <= 0:
            raise ValueError(
                f"spatial_token_count must be positive, got {spatial_token_count}"
            )
        token_grid_size = isqrt(spatial_token_count)
        if token_grid_size * token_grid_size != spatial_token_count:
            raise ValueError(
                "spatial_token_count must form a square token lattice, "
                f"got {spatial_token_count}"
            )
        if dense_grid_size <= 0 or dense_grid_size % token_grid_size != 0:
            raise ValueError(
                "dense_grid_size must be positive and divisible by the token-grid "
                f"side, got dense_grid_size={dense_grid_size}, "
                f"token_grid_size={token_grid_size}"
            )

        self.hidden_size = hidden_size
        self.spatial_token_count = spatial_token_count
        self.token_grid_size = token_grid_size
        self.dense_grid_size = dense_grid_size
        self.map_position_embeddings = nn.Embedding(spatial_token_count, hidden_size)
        self.map_from_graph_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.map_from_text_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.map_self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.graph_from_map_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.graph_text_query_projection = nn.Linear(hidden_size, hidden_size)
        self.graph_progress_query_projection = nn.Linear(hidden_size, hidden_size)
        self.graph_query_norm = nn.LayerNorm(hidden_size)

        self.visual_evidence_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, NUM_MAP_CATEGORIES),
        )
        self.visual_category_projection = nn.Linear(
            NUM_MAP_CATEGORIES, hidden_size, bias=False
        )
        self.step_encoder = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
        )
        self.new_evidence_embedding = nn.Embedding(2, hidden_size)

        innovation_size = hidden_size * 5
        self.innovation_norm = nn.LayerNorm(innovation_size)
        self.map_residual_projection = nn.Linear(innovation_size, hidden_size)
        self.write_gate = nn.Linear(innovation_size, hidden_size)
        self.map_self_residual_projection = nn.Linear(hidden_size, hidden_size)
        self.graph_residual_projection = nn.Linear(hidden_size, hidden_size)

        progress_input_size = hidden_size * 4
        self.progress_encoder = nn.Sequential(
            nn.LayerNorm(progress_input_size),
            nn.Linear(progress_input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.phase_head = nn.Linear(hidden_size, 5)
        self.route_state_head = nn.Linear(hidden_size, 3)
        self.remaining_head = nn.Linear(hidden_size, 1)
        candidate_input_size = hidden_size * 3
        self.recovery_head = nn.Sequential(
            nn.LayerNorm(candidate_input_size),
            nn.Linear(candidate_input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.ghost_residual_head = nn.Sequential(
            nn.LayerNorm(candidate_input_size),
            nn.Linear(candidate_input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.stop_residual_head = nn.Sequential(
            nn.LayerNorm(candidate_input_size),
            nn.Linear(candidate_input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        dense_scale = dense_grid_size // token_grid_size
        self.dense_grid_decoder = nn.ConvTranspose2d(
            hidden_size,
            NUM_MAP_CATEGORIES,
            kernel_size=dense_scale,
            stride=dense_scale,
        )
        self._zero_residual_projection()

    def _zero_residual_projection(self):
        """Initialize the new path as an exact identity over a loaded model."""

        nn.init.zeros_(self.map_residual_projection.weight)
        nn.init.zeros_(self.map_residual_projection.bias)
        nn.init.zeros_(self.map_self_residual_projection.weight)
        nn.init.zeros_(self.map_self_residual_projection.bias)
        nn.init.zeros_(self.graph_residual_projection.weight)
        nn.init.zeros_(self.graph_residual_projection.bias)
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, -2.0)
        for head in (self.ghost_residual_head, self.stop_residual_head):
            final = head[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @staticmethod
    def _validate_mask(
        name: str,
        mask: Optional[torch.Tensor],
        expected_shape,
        device: torch.device,
        *,
        required: bool,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            if required:
                raise ValueError(f"{name} is required for online_fusion")
            return None
        if tuple(mask.shape) != tuple(expected_shape):
            raise ValueError(
                f"{name} must have shape {tuple(expected_shape)}, "
                f"got {tuple(mask.shape)}"
            )
        if mask.dtype is not torch.bool:
            raise TypeError(f"{name} must have dtype torch.bool, got {mask.dtype}")
        if mask.device != device:
            raise ValueError(f"{name} must be on device {device}, got {mask.device}")
        return mask

    def _validate_inputs(
        self,
        gmap_embeds: torch.Tensor,
        visual_node_embeds: Optional[torch.Tensor],
        gmap_masks: Optional[torch.Tensor],
        map_tokens: torch.Tensor,
        map_token_masks: Optional[torch.Tensor],
        gmap_visited_masks: Optional[torch.Tensor],
        gmap_step_ids: Optional[torch.Tensor],
        txt_embeds: Optional[torch.Tensor],
        txt_masks: Optional[torch.Tensor],
        new_evidence_mask: Optional[torch.Tensor],
        previous_map_state: Optional[torch.Tensor],
    ):
        if gmap_embeds.dim() != 3:
            raise ValueError(
                "gmap_embeds must have shape (B, G, hidden_size), "
                f"got {tuple(gmap_embeds.shape)}"
            )
        if map_tokens.dim() != 3:
            raise ValueError(
                "map_tokens must have shape (B, spatial_tokens + 1, hidden_size), "
                f"got {tuple(map_tokens.shape)}"
            )
        batch_size, graph_size, graph_hidden_size = gmap_embeds.shape
        if graph_size == 0:
            raise ValueError("gmap_embeds must include at least the STOP entry")
        expected_map_shape = (
            batch_size,
            self.spatial_token_count + 1,
            self.hidden_size,
        )
        if graph_hidden_size != self.hidden_size:
            raise ValueError(
                f"gmap_embeds last dimension must be {self.hidden_size}, "
                f"got {graph_hidden_size}"
            )
        if tuple(map_tokens.shape) != expected_map_shape:
            raise ValueError(
                f"map_tokens must have shape {expected_map_shape}, "
                f"got {tuple(map_tokens.shape)}"
            )
        if map_tokens.device != gmap_embeds.device:
            raise ValueError(
                "gmap_embeds and map_tokens must be on the same device, "
                f"got {gmap_embeds.device} and {map_tokens.device}"
            )
        if map_tokens.dtype != gmap_embeds.dtype:
            raise TypeError(
                "gmap_embeds and map_tokens must have the same dtype, "
                f"got {gmap_embeds.dtype} and {map_tokens.dtype}"
            )
        if visual_node_embeds is None:
            raise ValueError("visual_node_embeds is required for online_fusion")
        if tuple(visual_node_embeds.shape) != tuple(gmap_embeds.shape):
            raise ValueError(
                "visual_node_embeds must have the same shape as gmap_embeds, "
                f"got {tuple(visual_node_embeds.shape)} and "
                f"{tuple(gmap_embeds.shape)}"
            )
        graph_shape = (batch_size, graph_size)
        gmap_masks = self._validate_mask(
            "gmap_masks",
            gmap_masks,
            graph_shape,
            gmap_embeds.device,
            required=False,
        )
        if gmap_masks is None:
            gmap_masks = torch.ones(
                graph_shape, dtype=torch.bool, device=gmap_embeds.device
            )
        gmap_visited_masks = self._validate_mask(
            "gmap_visited_masks",
            gmap_visited_masks,
            graph_shape,
            gmap_embeds.device,
            required=True,
        )
        new_evidence_mask = self._validate_mask(
            "new_evidence_mask",
            new_evidence_mask,
            graph_shape,
            gmap_embeds.device,
            required=False,
        )
        if new_evidence_mask is None:
            new_evidence_mask = gmap_visited_masks

        if gmap_step_ids is None:
            gmap_step_ids = torch.zeros(
                graph_shape, dtype=torch.long, device=gmap_embeds.device
            )
        else:
            if tuple(gmap_step_ids.shape) != graph_shape:
                raise ValueError(
                    f"gmap_step_ids must have shape {graph_shape}, "
                    f"got {tuple(gmap_step_ids.shape)}"
                )
            if gmap_step_ids.device != gmap_embeds.device:
                raise ValueError(
                    "gmap_step_ids and gmap_embeds must be on the same device, "
                    f"got {gmap_step_ids.device} and {gmap_embeds.device}"
                )
            if gmap_step_ids.is_floating_point() or gmap_step_ids.dtype is torch.bool:
                raise TypeError(
                    "gmap_step_ids must have an integer dtype, "
                    f"got {gmap_step_ids.dtype}"
                )
            if (gmap_step_ids < 0).any():
                raise ValueError("gmap_step_ids must be nonnegative")

        map_shape = (batch_size, self.spatial_token_count + 1)
        map_token_masks = self._validate_mask(
            "map_token_masks",
            map_token_masks,
            map_shape,
            map_tokens.device,
            required=False,
        )
        if map_token_masks is None:
            map_token_masks = torch.ones(
                map_shape, dtype=torch.bool, device=map_tokens.device
            )

        if previous_map_state is not None:
            if tuple(previous_map_state.shape) != expected_map_shape:
                raise ValueError(
                    "previous_map_state must have shape "
                    f"{expected_map_shape}, got {tuple(previous_map_state.shape)}"
                )
            if previous_map_state.device != map_tokens.device:
                raise ValueError(
                    "previous_map_state and map_tokens must be on the same device, "
                    f"got {previous_map_state.device} and {map_tokens.device}"
                )
            if previous_map_state.dtype != map_tokens.dtype:
                raise TypeError(
                    "previous_map_state and map_tokens must have the same dtype, "
                    f"got {previous_map_state.dtype} and {map_tokens.dtype}"
                )
            if not torch.equal(
                previous_map_state[:, self.spatial_token_count :],
                map_tokens[:, self.spatial_token_count :],
            ):
                raise ValueError(
                    "previous_map_state must preserve the immutable metadata token"
                )

        if txt_embeds is None:
            if txt_masks is not None:
                raise ValueError("txt_masks cannot be provided without txt_embeds")
        else:
            if txt_embeds.dim() != 3:
                raise ValueError(
                    "txt_embeds must have shape (B, T, hidden_size), "
                    f"got {tuple(txt_embeds.shape)}"
                )
            if (
                txt_embeds.shape[0] != batch_size
                or txt_embeds.shape[2] != self.hidden_size
                or txt_embeds.shape[1] == 0
            ):
                raise ValueError(
                    "txt_embeds must have shape "
                    f"(B, T>0, {self.hidden_size}), got {tuple(txt_embeds.shape)}"
                )
            if txt_embeds.device != map_tokens.device:
                raise ValueError(
                    "txt_embeds and map_tokens must be on the same device, "
                    f"got {txt_embeds.device} and {map_tokens.device}"
                )
            if txt_embeds.dtype != map_tokens.dtype:
                raise TypeError(
                    "txt_embeds and map_tokens must have the same dtype, "
                    f"got {txt_embeds.dtype} and {map_tokens.dtype}"
                )
            txt_masks = self._validate_mask(
                "txt_masks",
                txt_masks,
                txt_embeds.shape[:2],
                txt_embeds.device,
                required=False,
            )
            if txt_masks is None:
                txt_masks = torch.ones(
                    txt_embeds.shape[:2],
                    dtype=torch.bool,
                    device=txt_embeds.device,
                )

        valid_visited = gmap_masks & gmap_visited_masks
        invalid_new = new_evidence_mask & ~valid_visited
        if invalid_new.any():
            raise ValueError(
                "new_evidence_mask must be a subset of valid visited graph entries"
            )
        if graph_size > 0 and gmap_visited_masks[:, 0].any():
            raise ValueError("STOP at graph index 0 must not be marked visited")

        return (
            gmap_masks,
            map_token_masks,
            valid_visited,
            gmap_step_ids,
            txt_masks,
            new_evidence_mask,
        )

    @staticmethod
    def _safe_key_padding_mask(valid_mask: torch.Tensor) -> torch.Tensor:
        """Build an MHA padding mask without leaving a fully masked row."""

        padding_mask = valid_mask.logical_not().clone()
        fully_masked = padding_mask.all(dim=1)
        if fully_masked.any():
            padding_mask[fully_masked, 0] = False
        return padding_mask

    @staticmethod
    def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.unsqueeze(-1).to(dtype=values.dtype)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (values * weights).sum(dim=1) / denominator

    def forward(
        self,
        gmap_embeds: torch.Tensor,
        gmap_masks: Optional[torch.Tensor],
        map_tokens: Optional[torch.Tensor],
        map_token_masks: Optional[torch.Tensor],
        *,
        visual_node_embeds: Optional[torch.Tensor] = None,
        gmap_visited_masks: Optional[torch.Tensor] = None,
        gmap_step_ids: Optional[torch.Tensor] = None,
        txt_embeds: Optional[torch.Tensor] = None,
        txt_masks: Optional[torch.Tensor] = None,
        previous_map_state: Optional[torch.Tensor] = None,
        new_evidence_mask: Optional[torch.Tensor] = None,
        decode_dense_grid: bool = False,
        # Temporary source-compatibility aliases; new call sites should use the
        # singular/state names above.
        gmap_new_evidence_masks: Optional[torch.Tensor] = None,
        previous_map_tokens: Optional[torch.Tensor] = None,
    ):
        if new_evidence_mask is not None and gmap_new_evidence_masks is not None:
            raise ValueError(
                "provide only one of new_evidence_mask and gmap_new_evidence_masks"
            )
        if previous_map_state is not None and previous_map_tokens is not None:
            raise ValueError(
                "provide only one of previous_map_state and previous_map_tokens"
            )
        if new_evidence_mask is None:
            new_evidence_mask = gmap_new_evidence_masks
        if previous_map_state is None:
            previous_map_state = previous_map_tokens
        if not isinstance(decode_dense_grid, bool):
            raise TypeError(
                "decode_dense_grid must be bool, "
                f"got {type(decode_dense_grid).__name__}"
            )
        if map_tokens is None:
            if previous_map_state is not None:
                raise ValueError(
                    "previous_map_state cannot be provided when map_tokens is None"
                )
            return gmap_embeds, None

        (
            gmap_masks,
            map_token_masks,
            valid_visited,
            gmap_step_ids,
            txt_masks,
            new_evidence_mask,
        ) = self._validate_inputs(
            gmap_embeds,
            visual_node_embeds,
            gmap_masks,
            map_tokens,
            map_token_masks,
            gmap_visited_masks,
            gmap_step_ids,
            txt_embeds,
            txt_masks,
            new_evidence_mask,
            previous_map_state,
        )

        recurrent_tokens = (
            map_tokens if previous_map_state is None else previous_map_state
        )
        spatial_tokens = recurrent_tokens[:, : self.spatial_token_count]
        spatial_masks = map_token_masks[:, : self.spatial_token_count]
        position_ids = torch.arange(self.spatial_token_count, device=map_tokens.device)
        position_embeds = self.map_position_embeddings(position_ids).unsqueeze(0)

        if txt_embeds is None:
            text_context = torch.zeros_like(spatial_tokens)
            text_pool = torch.zeros(
                spatial_tokens.shape[0],
                self.hidden_size,
                dtype=spatial_tokens.dtype,
                device=spatial_tokens.device,
            )
        else:
            text_key_padding_mask = self._safe_key_padding_mask(txt_masks)
            text_context, _ = self.map_from_text_attention(
                spatial_tokens + position_embeds,
                txt_embeds,
                txt_embeds,
                key_padding_mask=text_key_padding_mask,
                need_weights=False,
            )
            has_valid_text = txt_masks.any(dim=1, keepdim=True).unsqueeze(-1)
            text_context = text_context * has_valid_text
            text_pool = self._masked_mean(txt_embeds, txt_masks)

        # Keep graph step/task/global-position embeddings out of the semantic
        # auxiliary head; it reads the existing panorama node representation.
        visual_logits = self.visual_evidence_head(visual_node_embeds)
        visual_context = self.visual_category_projection(torch.sigmoid(visual_logits))
        normalized_steps = torch.log1p(
            gmap_step_ids.to(dtype=gmap_embeds.dtype)
        ).unsqueeze(-1)
        step_context = self.step_encoder(normalized_steps)
        new_evidence_context = self.new_evidence_embedding(
            new_evidence_mask.to(dtype=torch.long)
        )
        graph_evidence_embeds = (
            gmap_embeds
            + visual_context
            + step_context
            + new_evidence_context
        )

        visited_pool = self._masked_mean(graph_evidence_embeds, valid_visited)
        latest_steps = gmap_step_ids.masked_fill(~valid_visited, -1).max(
            dim=1, keepdim=True
        ).values
        current_masks = valid_visited & (gmap_step_ids == latest_steps)
        current_pool = self._masked_mean(graph_evidence_embeds, current_masks)
        previous_map_pool = self._masked_mean(spatial_tokens, spatial_masks)
        update_progress_hidden = self.progress_encoder(
            torch.cat(
                [visited_pool, previous_map_pool, current_pool, text_pool],
                dim=-1,
            )
        )
        update_progress_context = update_progress_hidden.unsqueeze(1)

        graph_key_padding_mask = self._safe_key_padding_mask(valid_visited)
        graph_context, _ = self.map_from_graph_attention(
            spatial_tokens + position_embeds + text_context,
            graph_evidence_embeds,
            graph_evidence_embeds,
            key_padding_mask=graph_key_padding_mask,
            need_weights=False,
        )
        innovation_features = torch.cat(
            [
                spatial_tokens,
                graph_context,
                spatial_tokens - graph_context,
                spatial_tokens * graph_context,
                text_context,
            ],
            dim=-1,
        )
        innovation_features = self.innovation_norm(innovation_features)
        candidate_residual = torch.tanh(
            self.map_residual_projection(innovation_features)
        )
        write_gate = torch.sigmoid(self.write_gate(innovation_features))
        has_new_evidence = new_evidence_mask.any(dim=1, keepdim=True)
        write_mask = spatial_masks & has_new_evidence
        base_spatial_tokens = map_tokens[:, : self.spatial_token_count]
        previous_residual = spatial_tokens - base_spatial_tokens
        local_write_residual = (
            write_mask.unsqueeze(-1)
            * write_gate
            * (candidate_residual - previous_residual)
        )
        next_residual = previous_residual + local_write_residual
        updated_spatial_tokens = base_spatial_tokens + next_residual

        spatial_key_padding_mask = self._safe_key_padding_mask(spatial_masks)
        map_context, _ = self.map_self_attention(
            updated_spatial_tokens
            + position_embeds
            + text_context
            + update_progress_context,
            updated_spatial_tokens
            + position_embeds
            + text_context
            + update_progress_context,
            updated_spatial_tokens,
            key_padding_mask=spatial_key_padding_mask,
            need_weights=False,
        )
        propagated_write_residual = write_mask.unsqueeze(
            -1
        ) * self.map_self_residual_projection(map_context)
        updated_spatial_tokens = updated_spatial_tokens + propagated_write_residual
        map_write_residual = local_write_residual + propagated_write_residual
        evidence_rows = has_new_evidence.unsqueeze(-1)
        updated_spatial_tokens = torch.where(
            evidence_rows,
            updated_spatial_tokens,
            spatial_tokens,
        )
        map_write_residual = torch.where(
            evidence_rows,
            map_write_residual,
            torch.zeros_like(map_write_residual),
        )
        map_state_residual = updated_spatial_tokens - base_spatial_tokens

        # Metadata is immutable episode context; only the spatial state recurs.
        updated_map_tokens = torch.cat(
            [updated_spatial_tokens, map_tokens[:, self.spatial_token_count :]],
            dim=1,
        )

        map_pool = self._masked_mean(updated_spatial_tokens, spatial_masks)
        progress_hidden = self.progress_encoder(
            torch.cat([visited_pool, map_pool, current_pool, text_pool], dim=-1)
        )
        graph_queries = self.graph_query_norm(
            gmap_embeds
            + self.graph_text_query_projection(text_pool).unsqueeze(1)
            + self.graph_progress_query_projection(progress_hidden).unsqueeze(1)
        )
        graph_context, _ = self.graph_from_map_attention(
            graph_queries,
            updated_spatial_tokens,
            updated_spatial_tokens,
            key_padding_mask=spatial_key_padding_mask,
            need_weights=False,
        )
        has_valid_map = spatial_masks.any(dim=1, keepdim=True).unsqueeze(-1)
        graph_context = graph_context * has_valid_map
        graph_indices = torch.arange(
            gmap_embeds.shape[1], device=gmap_embeds.device
        ).unsqueeze(0)
        ghost_masks = gmap_masks & valid_visited.logical_not() & (graph_indices > 0)
        stop_masks = gmap_masks & (graph_indices == 0)
        # Only actionable frontier entries receive the map-conditioned graph
        # residual. Visited nodes and padding remain on the established ETP-R1
        # path; STOP is adjusted separately by its dedicated residual head.
        graph_write_mask = ghost_masks.unsqueeze(-1)
        updated_gmap_embeds = gmap_embeds + (
            graph_write_mask
            * has_valid_map
            * self.graph_residual_projection(graph_context)
        )

        phase_logits = self.phase_head(progress_hidden)
        route_state_logits = self.route_state_head(progress_hidden)
        remaining = torch.sigmoid(self.remaining_head(progress_hidden)).squeeze(-1)

        progress_per_node = progress_hidden.unsqueeze(1).expand(
            -1, gmap_embeds.shape[1], -1
        )
        candidate_features = torch.cat(
            [gmap_embeds, graph_context, progress_per_node], dim=-1
        )
        recovery_logits = (
            self.recovery_head(candidate_features).squeeze(-1) * ghost_masks
        )
        ghost_logit_residuals = (
            self.ghost_residual_head(candidate_features).squeeze(-1) * ghost_masks
        )
        stop_logit_residuals = (
            self.stop_residual_head(candidate_features).squeeze(-1) * stop_masks
        )
        action_logit_residuals = ghost_logit_residuals + stop_logit_residuals

        dense_grid_logits = None
        if decode_dense_grid:
            spatial_grid = updated_spatial_tokens.transpose(1, 2).reshape(
                updated_spatial_tokens.shape[0],
                self.hidden_size,
                self.token_grid_size,
                self.token_grid_size,
            )
            dense_grid_logits = self.dense_grid_decoder(spatial_grid)
            expected_dense_shape = (
                updated_spatial_tokens.shape[0],
                NUM_MAP_CATEGORIES,
                self.dense_grid_size,
                self.dense_grid_size,
            )
            if tuple(dense_grid_logits.shape) != expected_dense_shape:
                raise RuntimeError(
                    "dense grid decoder returned an unexpected shape: "
                    f"expected {expected_dense_shape}, "
                    f"got {tuple(dense_grid_logits.shape)}"
                )

        return OnlineMapFusionOutput(
            updated_gmap_embeds=updated_gmap_embeds,
            updated_map_tokens=updated_map_tokens,
            visual_logits=visual_logits,
            dense_grid_logits=dense_grid_logits,
            phase_logits=phase_logits,
            route_state_logits=route_state_logits,
            remaining=remaining,
            recovery_logits=recovery_logits,
            action_logit_residuals=action_logit_residuals,
            ghost_logit_residuals=ghost_logit_residuals,
            stop_logit_residuals=stop_logit_residuals,
            map_write_gate=write_mask.unsqueeze(-1) * write_gate,
            map_write_residual=map_write_residual,
            map_state_residual=map_state_residual,
            ghost_masks=ghost_masks,
        )


def build_map_token_fusion(
    architecture: str,
    hidden_size: int,
    num_heads: int,
    dropout: float = 0.1,
) -> nn.Module:
    if architecture == "current":
        return BidirectionalMapTokenFusion(hidden_size, num_heads, dropout)
    if architecture == "try5":
        return GraphMapCrossAttention(hidden_size, num_heads, dropout)
    if architecture == "online_fusion":
        return OnlineMapGraphFusion(hidden_size, num_heads, dropout)
    raise ValueError(f"Unknown navigation architecture: {architecture}")
