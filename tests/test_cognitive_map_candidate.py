import pytest

from vlnce_baselines.models.cognitive_map_candidate import (
    CognitiveMapCandidate,
    CognitiveMapSource,
    NavigationArchitecture,
)


@pytest.mark.parametrize(
    ("architecture", "source", "metadata_schema", "requires_box_targets"),
    [
        ("current", "imagined", "path5", True),
        ("current", "llm_boxes", "path5", True),
        ("current", "prior_gt", "path5", True),
        ("try5", "llm_grid", "direction5", False),
        ("try5", "prior_gt", "direction5", False),
    ],
)
def test_candidate_contract(
    architecture, source, metadata_schema, requires_box_targets
):
    candidate = CognitiveMapCandidate.parse(architecture, source)

    assert candidate.metadata_schema == metadata_schema
    assert candidate.requires_box_targets is requires_box_targets
    assert candidate.uses_llm_cache is source.startswith("llm_")


@pytest.mark.parametrize(
    ("architecture", "source"),
    [
        (NavigationArchitecture.CURRENT, CognitiveMapSource.LLM_GRID),
        (NavigationArchitecture.TRY5, CognitiveMapSource.IMAGINED),
        (NavigationArchitecture.TRY5, CognitiveMapSource.LLM_BOXES),
    ],
)
def test_candidate_rejects_unsupported_combinations(architecture, source):
    with pytest.raises(ValueError, match="Unsupported cognitive-map candidate"):
        CognitiveMapCandidate(architecture, source)


def test_candidate_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unknown navigation architecture"):
        CognitiveMapCandidate.parse("future", "prior_gt")
    with pytest.raises(ValueError, match="Unknown cognitive-map source"):
        CognitiveMapCandidate.parse("try5", "oracle")
