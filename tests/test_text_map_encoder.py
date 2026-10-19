"""input_form text: serialisation, tokeniser and TextMapEncoder shapes/masks."""

import numpy as np
import torch
import torch.nn as nn

from vlnce_baselines.models.etp_prior_gt.text_map_encoder import (
    MapTextTokenizer,
    TextMapEncoder,
    map_text_entities,
)


class _Embeddings(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.word = nn.Embedding(30522, hidden)

    def forward(self, input_ids, txt_task_encoding, token_type_ids=None):
        return self.word(input_ids) + txt_task_encoding.unsqueeze(-1).float()


class _LangEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, embeds, masks):
        return self.lin(embeds) * masks.unsqueeze(-1)


class _VlnBert(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.embeddings = _Embeddings(hidden)
        self.lang_encoder = _LangEncoder(hidden)


def _grid():
    grid = np.zeros((37, 100, 100), dtype=np.float32)
    grid[2, 10:14, 20:26] = 1.0  # object
    grid[30, 30:50, 40:60] = 0.6  # region (channels >= 27)
    return grid


def test_entities_sentences():
    directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 1]], dtype=np.float32)
    sentences = map_text_entities(_grid(), directions, scale=2)
    assert len(sentences) == 3
    assert sentences[0].startswith("object ") and "6 cells, center (6,11), extent 2x3" in sentences[0]
    assert sentences[1].startswith("region ") and "100 cells, center (20,24), extent 10x10" in sentences[1]
    assert sentences[2].startswith("heading: (0.00,1.00)")


def test_tokenizer_and_encoder():
    tokenizer = MapTextTokenizer("bert_config/bert-base-uncased", max_length=32)
    ids = tokenizer(["object chair: 3 cells, center (12,30), extent 2x2", "heading: (0.00,1.00)"])
    assert ids.shape == (2, 32) and ids[0, 0].item() == 101 and (ids == 1).any()

    hidden = 16
    bert = _VlnBert(hidden)
    encoder = TextMapEncoder(bert, hidden)
    encoder.train()
    assert not encoder.embeddings.training and not encoder.lang_encoder.training
    assert all(not p.requires_grad for p in encoder.lang_encoder.parameters())
    trainable = [n for n, p in encoder.named_parameters() if p.requires_grad]
    assert all(n.startswith(("entity_projection", "metadata_encoder", "output_norm")) for n in trainable)

    map_text_ids = torch.full((2, 3, 32), 1, dtype=torch.long)
    map_text_ids[0, :2] = ids
    map_text_ids[1, :3] = torch.cat([ids, ids[:1]])
    masks = torch.tensor([[True, True, False], [True, True, True]])
    tokens, token_masks = encoder(map_text_ids, masks, torch.zeros(2, 2), torch.ones(2, 2))
    assert tokens.shape == (2, 4, hidden) and token_masks.shape == (2, 4)
    assert token_masks.tolist() == [[True, True, False, True], [True, True, True, True]]
    assert torch.allclose(tokens[0, 2], encoder.output_norm(torch.zeros(hidden)))
    tokens.sum().backward()
    assert encoder.entity_projection[0].weight.grad is not None
