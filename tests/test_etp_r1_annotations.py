import json

from prior import etp_r1


def _write_mixed_annotations(path):
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "instr_id": instr_id,
                    "scan": "scene",
                    "path": ["vp0", "vp1"],
                    "heading": 0.0,
                    "instr_encoding": [encoding],
                    "task_type_encoding": 2,
                }
            )
            for instr_id, encoding in [
                ("english", 1),
                ("hindi", 2),
                ("english_curly", 3),
            ]
        )
        + "\n"
    )


def _patch_decode_tokens(monkeypatch):
    monkeypatch.setattr(
        etp_r1,
        "decode_tokens",
        lambda tokens: {
            1: "Go straight to the chair.",
            2: "भोजनकक्ष के कोने में दाएं चलिये।",
            3: "You’re facing the hallway; go forward.",
        }[tokens[0]],
    )


def test_annotation_entry_iter_from_defaults_to_unfiltered(tmp_path, monkeypatch):
    _write_mixed_annotations(tmp_path / "mixed.jsonl")
    monkeypatch.setattr(etp_r1, "ANNOTATION_DIR", tmp_path)
    _patch_decode_tokens(monkeypatch)

    entries = list(etp_r1.AnnotationEntry.iter_from("mixed.jsonl"))

    assert [entry.instr_id for entry in entries] == [
        "english",
        "hindi",
        "english_curly",
    ]


def test_annotation_entry_iter_from_can_filter_english_like(
    tmp_path,
    monkeypatch,
):
    _write_mixed_annotations(tmp_path / "mixed.jsonl")
    monkeypatch.setattr(etp_r1, "ANNOTATION_DIR", tmp_path)
    _patch_decode_tokens(monkeypatch)

    entries = list(etp_r1.AnnotationEntry.iter_from("mixed.jsonl", english_only=True))

    assert [entry.instr_id for entry in entries] == ["english", "english_curly"]


def test_is_english_like_instruction_rejects_supported_non_latin_scripts():
    assert etp_r1.is_english_like_instruction("Turn left, then go forward.")
    assert etp_r1.is_english_like_instruction("You’re facing the hallway.")
    assert not etp_r1.is_english_like_instruction("भोजनकक्ष के कोने में दाएं चलिये।")
    assert not etp_r1.is_english_like_instruction("మీరు నిల్చున్న ప్రదేశము నుంచి నేరుగా వెళ్లండి.")
