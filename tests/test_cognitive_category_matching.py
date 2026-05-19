import importlib
import sys
import types

import torch


def _install_fake_clip(monkeypatch):
    fake_clip = types.ModuleType("clip")
    prompt_store = {}

    def _feature_for_prompt(prompt):
        normalized = prompt.lower().replace("_", " ").replace("-", " ")
        feature = torch.zeros(512, dtype=torch.float32)
        if "sofa" in normalized or normalized == "couch":
            feature[0] = 1.0
        elif "chair" in normalized:
            feature[1] = 1.0
        elif "bedroom" in normalized or normalized == "bed":
            feature[2] = 1.0
        else:
            feature[511] = 1.0
        return feature

    class FakeClipModel(torch.nn.Module):
        def eval(self):
            return self

        def encode_text(self, tokens):
            rows = []
            for token_id in tokens[:, 0].tolist():
                rows.append(_feature_for_prompt(prompt_store[token_id]))
            return torch.stack(rows, dim=0)

    def load(name, device="cpu"):
        return FakeClipModel(), None

    def tokenize(prompts):
        start = len(prompt_store)
        for offset, prompt in enumerate(prompts):
            prompt_store[start + offset] = prompt
        tokens = torch.zeros(len(prompts), 77, dtype=torch.long)
        tokens[:, 0] = torch.arange(start, start + len(prompts))
        return tokens

    fake_clip.load = load
    fake_clip.tokenize = tokenize
    monkeypatch.setitem(sys.modules, "clip", fake_clip)


def _load_cognitive(monkeypatch):
    _install_fake_clip(monkeypatch)
    sys.modules.pop("prior.grid_map._cognitive", None)
    return importlib.import_module("prior.grid_map._cognitive")


def test_spacy_noun_extraction_filters_spatial_terms(monkeypatch):
    cognitive = _load_cognitive(monkeypatch)

    lemmas = [token.lemma_.lower() for token in cognitive.extract_nouns("Turn left by the side table.")]

    assert "left" not in lemmas
    assert "side" not in lemmas
    assert "table" in lemmas


def test_exact_matching_uses_spacy_lemma_for_plural_nouns(monkeypatch):
    cognitive = _load_cognitive(monkeypatch)

    object_categories, _ = cognitive.extract_categories("Walk past the chairs.")

    assert cognitive.MAPPED_OBJECT_NAMES.index("chair") in object_categories


def test_clip_similarity_fallback_maps_synonyms(monkeypatch):
    cognitive = _load_cognitive(monkeypatch)

    object_categories, _ = cognitive.extract_categories("Walk past the couch.")

    assert cognitive.MAPPED_OBJECT_NAMES.index("sofa") in object_categories


def test_exact_region_match_suppresses_object_similarity_fallback(monkeypatch):
    cognitive = _load_cognitive(monkeypatch)

    object_categories, region_categories = cognitive.extract_categories("Enter the bedroom.")

    assert cognitive.MAPPED_REGION_NAMES.index("private room") in region_categories
    assert cognitive.MAPPED_OBJECT_NAMES.index("bed") not in object_categories
