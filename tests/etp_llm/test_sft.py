from vlnce_baselines.models.etp_llm.sft import rendered_token_counts


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
