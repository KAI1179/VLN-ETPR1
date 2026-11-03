"""CPU contracts for graph SFT exports, budgets, and the shared trainer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from prior.bbox import AABB2D, OBB2D
from prior.cogmap_v2 import llm_data
from prior.cogmap_v2.graph import (
    CognitiveGraph,
    GraphEdge,
    GraphGT,
    GraphMeta,
    GraphNode,
    GraphObject,
    NodeGT,
)
from prior.cogmap_v2.serialize import graph_to_text
from vlnce_baselines.models.etp_llm import llm_graph_train, llm_grid_train, sft
from vlnce_baselines.models.etp_llm.llm_boxes_train import (
    _render_chat_completion,
    _render_chat_prompt,
)


class ChatTokenizer:
    """Small deterministic tokenizer, optionally merging across a chat boundary."""

    pad_token_id = 0
    eos_token_id = 3

    def __init__(self, *, merge_boundary: bool = False) -> None:
        self.merge_boundary = merge_boundary

    def __len__(self) -> int:
        return 512

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=False
    ) -> str:
        assert tokenize is False
        result = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        if add_generation_prompt:
            result += "<assistant>"
        elif messages[-1]["role"] == "assistant":
            result += "<eot>"
        return result

    def encode(self, text: str, *, add_special_tokens=False):
        assert add_special_tokens is False
        if self.merge_boundary:
            text = text.replace("<assistant>N", "\x01")
        text = text.replace("<assistant>", "\x02").replace("<eot>", "\x03")
        return [ord(character) + 4 for character in text]

    def __call__(
        self,
        texts,
        *,
        max_length,
        padding,
        truncation,
        return_tensors,
        add_special_tokens=False,
    ):
        assert padding is True and return_tensors == "pt"
        rows = [
            self.encode(text, add_special_tokens=add_special_tokens) for text in texts
        ]
        assert not truncation or all(len(row) <= max_length for row in rows)
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor([
                row + [self.pad_token_id] * (width - len(row)) for row in rows
            ]),
            "attention_mask": torch.tensor([
                [1] * len(row) + [0] * (width - len(row)) for row in rows
            ]),
        }


def _graph(episode_id=999991) -> CognitiveGraph:
    box = AABB2D(min=(813.25, 819.75), max=(814.25, 820.75))
    return CognitiveGraph(
        nodes=[
            GraphNode(
                id=0,
                region_category="circulation",
                role="start",
                anchor_kind="region_entry",
                order=0,
                mentioned=False,
                objects=[
                    GraphObject(
                        category="chair",
                        mentioned=True,
                        offset_bin="left",
                        gt=OBB2D(center=(723.25, 729.75), half_extents=(0.3, 0.3)),
                    )
                ],
                gt=NodeGT(
                    center_level_xz=(1.25, 2.75),
                    aabb_level=box,
                    path_index_range=(0, 0),
                    region_id="PRIVATE_REGION_712",
                ),
            ),
            GraphNode(
                id=1,
                region_category="bathroom/sanitary",
                role="goal",
                anchor_kind="turn",
                order=1,
                mentioned=False,
                gt=NodeGT(
                    center_level_xz=(2.25, 2.75),
                    aabb_level=box,
                    path_index_range=(1, 1),
                    region_id="PRIVATE_REGION_713",
                ),
            ),
        ],
        edges=[
            GraphEdge(
                src=0,
                dst=1,
                distance_bin="short",
                passage="door",
                turn="right",
                vertical="none",
            )
        ],
        route=[0, 1],
        meta=GraphMeta(
            scene="PRIVATE_SCENE_718",
            episode_id=episode_id,
            level_idx=918,
            instruction="Go past the chair.",
            start_position_level_xz=(1.25, 2.75),
            start_direction_vector=(0.0, 1.0),
        ),
        gt=GraphGT(
            path_level_xz=[(1.25, 2.75), (2.25, 2.75)],
            path_world_indices=[0, 1],
            trajectory_world=[(92818.25, 0.0, 42819.75), (92819.25, 0.0, 42819.75)],
            range_y=[None, 4.0],
            level_origin=(92817.0, 42817.0),
            direction_vectors=[(0.0, 1.0)] + [(0.0, 0.0)] * 4,
        ),
    )


@pytest.fixture
def export_tree(tmp_path, monkeypatch):
    """Use temporary V2 output roots, leaving the real generated corpus alone."""
    monkeypatch.setattr(llm_data, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(llm_data, "output_path", lambda path: Path(path))
    system = tmp_path / "system.md"
    system.write_text("Predict the graph.")
    monkeypatch.setattr(llm_data, "SYSTEM_PROMPT_PATH", system)
    graph_dir = tmp_path / "graphs"
    for split in ("train", "val_seen", "val_unseen"):
        _graph().save(graph_dir / split / "PRIVATE_SCENE_718/999991.json")
    return graph_dir, tmp_path / "data_v2/llm_graph", system.read_text()


def test_export_has_only_instruction_pose_prompt_and_semantic_target(export_tree):
    graph_dir, output, _ = export_tree
    report = llm_data.export_data(
        graph_dir, output, ChatTokenizer(), tokenizer_source="fixture"
    )
    assert report["datasets"] == ["R2R"]
    for split in ("train", "val_seen", "val_unseen"):
        row = json.loads((output / f"{split}.jsonl").read_text())
        assert set(row) == {"example_id", "dataset", "input_text", "target_text"}
        assert row["example_id"] == f"R2R_{split}_999991"
        assert row["input_text"] == (
            "Instruction: Go past the chair.\n"
            "Start pose (level-local meters): position=(1.2, 2.8); "
            "direction=(0.0000, 1.0000)"
        )
        assert row["target_text"] == graph_to_text(_graph())
        combined = row["input_text"] + row["target_text"]
        for poison in (
            "PRIVATE_SCENE",
            "PRIVATE_REGION",
            "999991",
            "92817",
            "42817",
            "813.25",
            "723.25",
            "918",
            '"gt"',
        ):
            assert poison not in combined
        assert "circulation" in row["target_text"]  # Unmentioned context survives.
        assert "R2R" not in row["input_text"]


@pytest.mark.parametrize("merge_boundary", [False, True])
def test_export_completion_budget_uses_full_chat_prefix_delta(
    export_tree, merge_boundary
):
    graph_dir, output, system = export_tree
    tokenizer = ChatTokenizer(merge_boundary=merge_boundary)
    report = llm_data.export_data(
        graph_dir, output, tokenizer, tokenizer_source="fixture"
    )
    row = json.loads((output / "train.jsonl").read_text())
    prompt = _render_chat_prompt(tokenizer, system, row["input_text"])
    full = _render_chat_completion(
        tokenizer, system, row["input_text"], row["target_text"]
    )
    expected = sft.rendered_token_counts(tokenizer, prompt, full)
    assert report["rendered_prompt_tokens"]["max"] == expected.prompt_tokens
    assert report["rendered_completion_tokens"]["max"] == expected.completion_tokens
    assert report["rendered_sequence_tokens"]["max"] == expected.sequence_tokens
    assert report["raw_target_tokens"]["max"] == len(
        tokenizer.encode(row["target_text"])
    )


@pytest.mark.parametrize("examples,budget,dropped", [(200, 128, 1), (199, 256, 0)])
def test_budget_bucket_retains_at_least_995_per_mille(
    export_tree, monkeypatch, examples, budget, dropped
):
    graph_dir, output, _ = export_tree
    for path in (graph_dir / "train").glob("*/*.json"):
        path.unlink()
    for index in range(examples):
        (graph_dir / "train/PRIVATE_SCENE_718" / f"{index}.json").touch()
    monkeypatch.setattr(
        llm_data.CognitiveGraph, "load", lambda path: _graph(int(path.stem))
    )
    monkeypatch.setattr(
        llm_data,
        "graph_to_text",
        lambda graph: "x" * (129 if graph.meta.episode_id == 0 else 1),
    )
    report = llm_data.export_data(
        graph_dir, output, ChatTokenizer(), tokenizer_source="fixture"
    )
    assert report["max_new_tokens"] == budget
    assert report["dropped_completions"] == dropped
    assert report["dropped_fraction"] <= 0.005


def _rows(tmp_path, *, examples=2, long_target=False):
    rows = [
        {
            "example_id": f"R2R_train_{index}",
            "dataset": "R2R",
            "input_text": "Instruction: Turn right.\nStart pose: (0, 0) (0, 1)",
            "target_text": graph_to_text(_graph()) + ("\n" if index else ""),
        }
        for index in range(examples)
    ]
    if long_target:
        rows[-1]["target_text"] += " extra" * 20
    source = tmp_path / "train.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return source, rows


def _args(source):
    return cast(
        Any,
        SimpleNamespace(
            training_data=source,
            max_input_length=1024,
            max_new_tokens=1024,
            max_sequence_length=2048,
            max_dropped_fraction=0.005,
        ),
    )


def test_manifest_round_trip_and_existing_grid_collator_accept_graph_text(tmp_path):
    source, rows = _rows(tmp_path)
    tokenizer = ChatTokenizer()
    manifest = llm_graph_train.build_graph_training_manifest(
        _args(source), tokenizer, "Graph"
    )
    path = tmp_path / "artifacts/training_manifest.jsonl"
    sft._write_training_manifest(path, manifest)
    loaded = sft._read_training_manifest(path)
    assert loaded == manifest
    assert loaded.metadata["candidate"] == "llm_graph"
    assert loaded.metadata["skipped_over_budget_fraction"] == 0
    items = [
        llm_grid_train._grid_training_item_from_manifest(item) for item in loaded.items
    ]
    dataset = llm_grid_train.LLMGridItemsDataset(items)
    batch = [dataset[sft.TrainingIndex(index)] for index in range(len(items))]
    collated = llm_grid_train.collate_llm_grid_batch(
        batch, tokenizer, "Graph", 1024, 1024, 2048
    )
    assert collated["example_ids"] == [row["example_id"] for row in rows]
    assert torch.all(collated["labels"][collated["attention_mask"] == 0] == -100)
    for index, item in enumerate(loaded.items):
        prompt_length = item["prompt_tokens"]
        assert torch.all(collated["labels"][index, :prompt_length] == -100)
        assert (
            int((collated["labels"][index] != -100).sum()) == item["completion_tokens"]
        )
        assert (
            item["sequence_tokens"] == item["prompt_tokens"] + item["completion_tokens"]
        )


def test_local_llama_tokenizer_export_manifest_and_collator_agree(export_tree):
    from transformers import AutoTokenizer

    tokenizer_path = (
        Path(__file__).resolve().parents[1] / "data_v2/tokenizers/llama-3.1-8b"
    )
    if not tokenizer_path.is_dir():
        # The generated tokenizer artifact is optional in fresh checkouts.
        pytest.skip()
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    graph_dir, output, system = export_tree
    report = llm_data.export_data(
        graph_dir, output, tokenizer, tokenizer_source=str(tokenizer_path)
    )
    manifest = llm_graph_train.build_graph_training_manifest(
        _args(output / "train.jsonl"), tokenizer, system
    )
    row = manifest.items[0]
    prompt = _render_chat_prompt(tokenizer, system, row["input_text"])
    full = _render_chat_completion(
        tokenizer, system, row["input_text"], row["target_text"]
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    assert full_ids[: len(prompt_ids)] == prompt_ids
    assert report["rendered_prompt_tokens"]["max"] == row["prompt_tokens"]
    assert report["rendered_completion_tokens"]["max"] == row["completion_tokens"]
    assert report["rendered_sequence_tokens"]["max"] == row["sequence_tokens"]
    dataset = llm_grid_train.LLMGridItemsDataset([
        llm_grid_train._grid_training_item_from_manifest(row)
    ])
    batch = llm_grid_train.collate_llm_grid_batch(
        [dataset[sft.TrainingIndex(0)]], tokenizer, system, 1024, 1024, 2048
    )
    assert int((batch["labels"] != -100).sum()) == row["completion_tokens"]


def test_manifest_drop_rate_allows_exactly_half_percent_and_rejects_more(tmp_path):
    source, rows = _rows(tmp_path, examples=200, long_target=True)
    tokenizer = ChatTokenizer()
    short = rows[0]
    prompt = _render_chat_prompt(tokenizer, "Graph", short["input_text"])
    full = _render_chat_completion(
        tokenizer, "Graph", short["input_text"], short["target_text"]
    )
    limit = sft.rendered_token_counts(tokenizer, prompt, full).completion_tokens + 1
    args = _args(source)
    args.max_new_tokens = limit
    manifest = llm_graph_train.build_graph_training_manifest(args, tokenizer, "Graph")
    assert len(manifest.items) == 199
    assert manifest.metadata["skipped_over_budget_fraction"] == 0.005
    assert manifest.metadata["dropped_example_ids"] == ["R2R_train_199"]
    args.max_dropped_fraction = 0.004
    with pytest.raises(ValueError, match="Graph token budgets drop"):
        llm_graph_train.build_graph_training_manifest(args, tokenizer, "Graph")


@pytest.mark.parametrize(
    "bad_field,bad_value",
    [
        ("example_id", 123),
        ("dataset", "RxR"),
        ("input_text", ""),
    ],
)
def test_manifest_rejects_malformed_rows_with_value_error(
    tmp_path, bad_field, bad_value
):
    source, rows = _rows(tmp_path)
    rows[0][bad_field] = bad_value
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        llm_graph_train.build_graph_training_manifest(
            _args(source), ChatTokenizer(), "Graph"
        )


def test_graph_manifest_injection_reaches_shared_loop_before_model_allocation(
    tmp_path, monkeypatch
):
    source, _ = _rows(tmp_path)
    tokenizer = ChatTokenizer()
    args = _args(source)
    args.finetune_method = "lora"
    args.gradient_accumulation_steps = 1
    args.device_map = "none"
    args.seed = 42
    args.per_device_batch_size = 1
    args.output_dir = str(tmp_path / "run")
    args.model_name_or_path = "fixture-tokenizer"
    accelerator = SimpleNamespace(
        is_main_process=True, num_processes=1, wait_for_everyone=lambda: None
    )
    monkeypatch.setattr(
        llm_grid_train, "_validate_llm_grid_training_args", lambda args: None
    )
    monkeypatch.setattr(llm_grid_train, "make_sft_accelerator", lambda _: accelerator)
    monkeypatch.setattr(
        llm_grid_train, "validate_distributed_device_map", lambda *args: None
    )
    monkeypatch.setattr(llm_grid_train, "_set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_grid_train, "distributed_batch_metrics", lambda *args: None)
    monkeypatch.setattr(
        llm_grid_train, "_load_grid_training_tokenizer", lambda _: tokenizer
    )
    monkeypatch.setattr(sft, "broadcast_object_list", lambda _: None)

    def reject_grid_builder(*args):
        raise AssertionError("Graph training must not read raster targets")

    monkeypatch.setattr(
        llm_grid_train, "_build_grid_training_manifest", reject_grid_builder
    )

    class BeforeModelAllocation(Exception):
        pass

    def stop_before_weights(*args, **kwargs):
        raise BeforeModelAllocation

    monkeypatch.setattr(
        llm_grid_train, "_load_grid_training_model", stop_before_weights
    )
    prompts = []

    def graph_builder(actual_tokenizer, system):
        assert actual_tokenizer is tokenizer
        prompts.append(system)
        return llm_graph_train.build_graph_training_manifest(args, tokenizer, system)

    with pytest.raises(BeforeModelAllocation):
        llm_grid_train.train_model(
            args, manifest_builder=graph_builder, system_prompt_override="Graph only"
        )
    assert prompts == ["Graph only"]
    artifact = Path(args.output_dir) / "artifacts"
    assert (artifact / "system_prompt.md").read_text().strip() == "Graph only"
    assert (
        len(sft._read_training_manifest(artifact / "training_manifest.jsonl").items)
        == 2
    )
