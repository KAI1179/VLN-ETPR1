"""Read-only eval observers for decision-query text and map attention."""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.eval_d8.feature_dump import DecisionFeatureDump


def map_attention_probabilities(module, inputs):
    """Observe QK probabilities without changing MHA's original forward path."""
    graph, _, tokens, masks = inputs[:4]
    if tokens is None:
        return None
    attention = module.attention
    with torch.autocast(device_type=graph.device.type, enabled=False):
        graph, tokens = graph.float(), tokens.float()
        width = attention.embed_dim
        bias = attention.in_proj_bias
        query = F.linear(graph, attention.in_proj_weight[:width],
                         None if bias is None else bias[:width])
        key = F.linear(tokens, attention.in_proj_weight[width:2 * width],
                       None if bias is None else bias[width:2 * width])
        batch, nodes, _ = query.shape
        heads = attention.num_heads
        head_width = width // heads
        query = query.view(batch, nodes, heads, head_width).transpose(1, 2)
        key = key.view(batch, tokens.size(1), heads, head_width).transpose(1, 2)
        scores = torch.matmul(query / math.sqrt(head_width), key.transpose(-1, -2))
        if module.coordinate_fusion:
            # D9 runs disable coordinate fusion; reject a silently wrong observer.
            raise ValueError("D9 observer requires coordinate_fusion=False")
        padding = module._map_key_padding_mask(masks)
        if padding is not None:
            scores.masked_fill_(padding[:, None, None, :], -float("inf"))
        return scores.softmax(dim=-1)


class AttentionDump(DecisionFeatureDump):
    """Implements the D8 replay interface, but observes attention, not features."""

    def __init__(self, network, directory, shared, replay_directory, map_config=None):
        from transformers import XLMRobertaTokenizerFast

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shared = shared
        self.replay_directory = Path(replay_directory) if replay_directory else None
        self.records = {}
        self.references = {}
        self.text = None
        self.maps = None
        self.map_config = map_config
        tokenizer_path = Path(__file__).resolve().parents[2] / "bert_config/xlm-roberta-base"
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            str(tokenizer_path), local_files_only=True
        )
        self.handles = [network.vln_bert.graph_query_text.register_forward_hook(self._text)]
        fusion = getattr(network.vln_bert, "graph_map_attention", None)
        if fusion is not None:
            self.handles.append(fusion.register_forward_pre_hook(self._map))

    def _text(self, module, inputs, output):
        self.text = output[1].softmax(dim=-1).detach().cpu()

    def _map(self, module, inputs):
        probabilities = map_attention_probabilities(module, inputs)
        self.maps = None if probabilities is None else probabilities.detach().cpu()

    def _language(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        word_index, words = [], []
        special_ids = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                       self.tokenizer.pad_token_id}
        for token, token_id in zip(tokens, token_ids):
            if token_id in special_ids:
                word_index.append(-1)
            else:
                if token.startswith("▁") or not words:
                    words.append(token.lstrip("▁"))
                else:
                    words[-1] += token
                word_index.append(len(words) - 1)
        return {
            "token_ids": np.asarray(token_ids), "tokens": np.asarray(tokens),
            "word_index": np.asarray(word_index), "words": np.asarray(words),
        }

    def _special(self, attention, token_ids):
        ids = [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
               self.tokenizer.pad_token_id]
        return np.stack([attention[:, token_ids == token].sum(axis=-1)
                         for token in ids], axis=-1)

    def decision(self, episodes, step, positions, rotations, graphs, candidate_ids,
                 probabilities, actions, no_candidates, max_steps, txt_ids,
                 cognitive_maps=None):
        from vlnce_baselines.models.graph_utils import heading_from_quaternion

        probabilities = probabilities.detach().cpu().numpy()
        token_ids = txt_ids.detach().cpu().numpy()
        if self.text is None or self.text.size(0) != len(episodes):
            raise RuntimeError("Missing graph_query_text attention snapshot")
        for index, episode in enumerate(episodes):
            eid = str(episode.episode_id)
            selected = int(probabilities[index].argmax())
            ids = [node for node in candidate_ids[index]
                   if node is not None and node.startswith("g")]
            stopped = int(actions[index]) == 0 or step == max_steps - 1 or bool(no_candidates[index])
            text = self.text[index, :, selected].numpy()
            stop_text = self.text[index, :, 0].numpy()
            row = {
                "step": int(step), "pos": np.asarray(positions[index]),
                "heading": float(heading_from_quaternion(rotations[index])),
                "stop_prob": float(probabilities[index, 0]),
                "policy_argmax_id": "-1" if selected == 0 else candidate_ids[index][selected],
                "executed_id": "-1" if stopped else candidate_ids[index][int(actions[index])],
                "candidate_ids": json.dumps(ids),
                "candidate_positions": json.dumps([
                    np.asarray(graphs[index].ghost_aug_pos[node]).tolist() for node in ids
                ]),
                "txt_attn": text.astype(np.float16),
                "stop_txt": stop_text.astype(np.float16),
                "txt_special": self._special(text, token_ids[index]).astype(np.float16),
                "stop_txt_special": self._special(stop_text, token_ids[index]).astype(np.float16),
            }
            if self.maps is not None:
                row["map_attn"] = self.maps[index, :, selected].numpy().astype(np.float16)
                row["stop_map"] = self.maps[index, :, 0].numpy().astype(np.float16)
            if eid not in self.records:
                record = {
                    "scene_id": Path(episode.scene_id).stem,
                    "episode_id": eid, "source": "S" if self.shared else "O",
                    **self._language(token_ids[index].tolist()),
                }
                if cognitive_maps is not None:
                    start = cognitive_maps[index]["start_position"].detach().cpu().numpy()
                    scale = float(self.map_config.start_position_scale)
                    unit = float(self.map_config.start_position_meters_per_unit)
                    start = start * scale
                    record.update(
                        map_start_position=start,
                        map_origin_xz=np.asarray(episode.start_position)[[0, 2]] - start * unit,
                        meters_per_cell=0.5,
                        registration="assumed registration",
                    )
                self.records[eid] = {"metadata": record, "steps": []}
            self.records[eid]["steps"].append(row)
        self.text = None
        self.maps = None

    def finish(self, episode_id):
        eid = str(episode_id)
        record = self.records.pop(eid)
        path = self.directory / f"{eid}.npz"
        if path.exists():
            raise FileExistsError(path)
        steps = record["steps"]
        arrays = {key: np.asarray([row[key] for row in steps]) for key in steps[0]}
        np.savez_compressed(path, **arrays, **record["metadata"])
        self.references.pop(eid, None)

    def close(self):
        for handle in self.handles:
            handle.remove()
