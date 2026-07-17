from vlnce_baselines.models.etp_llm.sft import (
    LengthFilterResult,
    SourceLoadStats,
    fixed_corpus_metrics,
    rendered_token_counts,
)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


def test_rendered_token_counts_uses_completion_delta():
    counts = rendered_token_counts(
        _Tokenizer(),
        prompt_text="system user assistant-start",
        completion_text="system user assistant-start answer eos",
    )

    assert counts.prompt_tokens == 3
    assert counts.completion_tokens == 2
    assert counts.sequence_tokens == 5


def test_rendered_token_counts_rejects_non_prefix_rendering():
    try:
        rendered_token_counts(
            _Tokenizer(),
            prompt_text="one two three",
            completion_text="one two",
        )
    except ValueError as error:
        assert "completion rendering is shorter than prompt rendering" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_fixed_corpus_metrics_reports_asymmetric_sources():
    items = (
        {"example_id": "r2r-keep", "dataset": "R2R"},
        {"example_id": "r2r-prompt", "dataset": "R2R"},
        {"example_id": "rxr-both", "dataset": "RxR"},
    )
    metrics = fixed_corpus_metrics(
        {
            "R2R": SourceLoadStats(4, 2, ("r2r-missing-a", "r2r-missing-b")),
            "RxR": SourceLoadStats(3, 1, ("rxr-missing",)),
        },
        items,
        LengthFilterResult(
            kept=(items[0],),
            dropped_prompt_example_ids=("r2r-prompt", "rxr-both"),
            dropped_completion_example_ids=("rxr-both",),
        ),
    )

    assert metrics == {
        "combined_discovered": 7.0,
        "combined_loaded": 3.0,
        "combined_missing_cache": 3.0,
        "combined_prompt_dropped": 2.0,
        "combined_completion_dropped": 1.0,
        "combined_retained": 1.0,
        "r2r_discovered": 4.0,
        "r2r_loaded": 2.0,
        "r2r_missing_cache": 2.0,
        "r2r_prompt_dropped": 1.0,
        "r2r_completion_dropped": 0.0,
        "r2r_retained": 1.0,
        "rxr_discovered": 3.0,
        "rxr_loaded": 1.0,
        "rxr_missing_cache": 1.0,
        "rxr_prompt_dropped": 1.0,
        "rxr_completion_dropped": 1.0,
        "rxr_retained": 0.0,
    }
