import json

from vlnce_baselines.models.etp_llm import analyze_tokens


class _Tokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
    ):
        assert tokenize is False
        rendered = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "\nassistant:"
        return rendered

    def encode(self, text, add_special_tokens=False):
        return text.split()


class _AutoTokenizer:
    calls = []

    @staticmethod
    def from_pretrained(model_name_or_path):
        _AutoTokenizer.calls.append(model_name_or_path)
        return _Tokenizer()


def test_analyze_llm_boxes_tokens_reports_distribution(monkeypatch):
    items = [
        {
            "input_text": "short input",
            "target_text": "one two",
            "example_id": "a",
        },
        {
            "input_text": "longer input text",
            "target_text": "one two three four",
            "example_id": "b",
        },
    ]
    monkeypatch.setattr(analyze_tokens, "AutoTokenizer", _AutoTokenizer)
    monkeypatch.setattr(
        analyze_tokens,
        "load_llm_boxes_examples",
        lambda *args, **kwargs: ["example-a", "example-b"],
    )
    monkeypatch.setattr(analyze_tokens, "LLMBoxesDataset", lambda examples: items)
    monkeypatch.setattr(analyze_tokens, "load_system_prompt", lambda: "system prompt")

    args = analyze_tokens.TokenAnalysisArgs().parse_args(
        [
            "--model-name-or-path",
            "tiny-tokenizer",
            "--splits",
            "train,val_seen",
            "--max-input-length",
            "4",
            "--max-new-tokens",
            "3",
            "--budgets",
            "2,4",
            "--quiet",
        ]
    )

    report = analyze_tokens.analyze_llm_boxes_tokens(args)

    assert _AutoTokenizer.calls == ["tiny-tokenizer"]
    assert report["dataset"] == "R2R"
    assert report["splits"] == ["train", "val_seen"]
    assert report["example_count"] == 2
    assert report["target_tokens"]["p50"] == 3.0
    assert report["target_tokens"]["max"] == 4.0
    assert report["over_budget"]["2"] == {"count": 1, "rate": 0.5}
    assert report["over_budget"]["4"] == {"count": 0, "rate": 0.0}
    assert report["configured_budget"]["max_input_length"] == 4
    assert report["configured_budget"]["max_new_tokens"] == 3
    assert report["configured_budget"]["target_over_budget_count"] == 1


def test_analyze_tokens_main_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(
        analyze_tokens,
        "analyze_llm_boxes_tokens",
        lambda args: {"max_new_tokens": args.max_new_tokens},
    )

    analyze_tokens.main(["--max-new-tokens", "2048"])

    assert json.loads(capsys.readouterr().out) == {"max_new_tokens": 2048}
