import pytest
import torch

from vlnce_baselines.models.etp_prior_gt.map_decoder import CognitiveMapDecoder
from vlnce_baselines.models.etp_prior_gt.map_utils import NUM_MAP_CATEGORIES, SIZE


def test_cognitive_map_decoder_predicts_dense_logits_from_updated_tokens():
    decoder = CognitiveMapDecoder(hidden_size=32)
    updated_map_tokens = torch.randn(2, 101, 32)

    logits = decoder(updated_map_tokens)

    assert logits.shape == (2, NUM_MAP_CATEGORIES, SIZE, SIZE)
    assert torch.isfinite(logits).all()


def test_cognitive_map_decoder_rejects_missing_metadata_token():
    decoder = CognitiveMapDecoder(hidden_size=32)
    updated_map_tokens = torch.randn(2, 100, 32)

    with pytest.raises(ValueError, match="updated_map_tokens"):
        decoder(updated_map_tokens)
