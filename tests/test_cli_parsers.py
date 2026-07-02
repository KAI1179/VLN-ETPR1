from functools import partial
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_run_parser_returns_typed_args_with_passthrough_opts():
    import run

    args = run.parse_args(
        [
            "--exp_name",
            "debug",
            "--run-type",
            "eval",
            "--exp-config",
            "run_r2r/iter_train.yaml",
            "--local-rank",
            "3",
            "MODEL.hidden_size",
            "768",
        ]
    )

    assert isinstance(args, run.RunArgs)
    assert args.exp_name == "debug"
    assert args.run_type == "eval"
    assert args.exp_config == "run_r2r/iter_train.yaml"
    assert args.local_rank == 3
    assert args.opts == ["MODEL.hidden_size", "768"]


def test_train_map_predictor_parser_returns_typed_args():
    from vlnce_baselines.models.etp_imagined import train_map_predictor

    args = train_map_predictor.parse_args(
        [
            "--exp-config",
            "run_r2r/iter_train.yaml",
            "--dataset",
            "r2r",
            "--train-splits",
            "train",
            "val_seen",
            "--val-splits",
            "val_unseen",
            "--output",
            "predictor.pt",
            "--batch-size",
            "4",
            "--epochs",
            "2",
            "--lr",
            "0.001",
            "--loss",
            "focal",
            "--max-pos-weight",
            "7.5",
            "--focal-gamma",
            "1.5",
            "--trajectory-keypoint-loss-weight",
            "0.25",
            "--init-positive-prob",
            "0.01",
            "--thresholds",
            "0.1,0.2",
            "--max-text-len",
            "16",
            "--num-workers",
            "0",
            "--seed",
            "9",
            "--limit",
            "10",
            "--val-limit",
            "3",
            "--log-every",
            "2",
            "--device",
            "cpu",
            "--opts",
            "MODEL.hidden_size",
            "768",
        ]
    )

    assert isinstance(args, train_map_predictor.TrainMapPredictorArgs)
    assert args.exp_config == "run_r2r/iter_train.yaml"
    assert args.dataset == "r2r"
    assert args.train_splits == ["train", "val_seen"]
    assert args.val_splits == ["val_unseen"]
    assert args.output == Path("predictor.pt")
    assert args.batch_size == 4
    assert args.epochs == 2
    assert args.lr == 0.001
    assert args.loss == "focal"
    assert args.max_pos_weight == 7.5
    assert args.focal_gamma == 1.5
    assert args.trajectory_keypoint_loss_weight == 0.25
    assert args.init_positive_prob == 0.01
    assert args.thresholds == "0.1,0.2"
    assert args.max_text_len == 16
    assert args.num_workers == 0
    assert args.seed == 9
    assert args.limit == 10
    assert args.val_limit == 3
    assert args.log_every == 2
    assert args.device == "cpu"
    assert args.opts == ["MODEL.hidden_size", "768"]


def test_train_map_predictor_parser_preserves_defaults():
    from vlnce_baselines.models.etp_imagined import train_map_predictor

    args = train_map_predictor.parse_args([])

    assert args.exp_config == "run_r2r/iter_train.yaml"
    assert args.dataset == "r2r"
    assert args.train_splits == ["train"]
    assert args.val_splits == ["val_unseen"]
    assert args.batch_size == 8
    assert args.max_pos_weight == 20.0
    assert args.focal_gamma == 2.0
    assert args.trajectory_keypoint_loss_weight == 0.001
    assert args.init_positive_prob == 0.002
    assert args.max_text_len is None
    assert args.num_workers == 2
    assert args.val_limit is None
    assert args.log_every == 1


def test_default_config_exposes_llm_navigation_cache_settings():
    from vlnce_baselines.config.default import get_config

    config = get_config()

    assert config.MODEL.MAP_ENCODER.llm_cache_dir == ""
    assert config.MODEL.MAP_ENCODER.llm_cache_model_key == "llama-3.1-8b-instruct"


def test_bbox_parser_accepts_scenes_and_optional_episode_selector():
    from prior.bbox import __main__ as bbox_main

    args = bbox_main.parse_args(
        [
            "17DRP5sb8fy",
            "--dataset",
            "r2r",
            "--episode-id",
            "123",
            "--split",
            "val_unseen",
            "--output",
            "boxes",
        ]
    )

    assert isinstance(args, bbox_main.BoundingBoxArgs)
    assert args.scenes == ["17DRP5sb8fy"]
    assert args.dataset == "r2r"
    assert args.episode_id == 123
    assert args.split == "val_unseen"
    assert args.output == Path("boxes")


def test_train_map_predictor_uses_spawn_context_for_workers(monkeypatch):
    from vlnce_baselines.models.etp_imagined import train_map_predictor

    monkeypatch.setattr(
        train_map_predictor,
        "load_predictor_examples",
        lambda *args, **kwargs: [object()],
    )

    loader = train_map_predictor._build_dataloader(
        dataset_name="r2r",
        splits=["train"],
        max_text_len=16,
        batch_size=1,
        num_workers=2,
        limit=1,
        shuffle=False,
    )

    assert loader.num_workers == 2
    assert loader.multiprocessing_context.get_start_method() == "spawn"
    assert isinstance(loader.collate_fn, partial)
    assert loader.collate_fn.func is train_map_predictor.collate_predictor_batch


def test_pretrain_prior_map_loads_cached_map(tmp_path, monkeypatch):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    pretrain_dataset = importlib.import_module("data.dataset")

    captured = {}

    def fake_cached_cognitive_map_to_tensors(
        scene_id,
        cache_id,
        cache_dir,
        random_rotation_augmentation,
    ):
        captured["scene_id"] = scene_id
        captured["cache_id"] = cache_id
        captured["cache_dir"] = cache_dir
        captured["random_rotation_augmentation"] = random_rotation_augmentation
        return {
            "grid": "loaded-grid",
            "trajectory_keypoints": "trajectory_keypoints",
            "start_direction_vector": "direction",
            "start_position": "position",
        }

    monkeypatch.setattr(pretrain_dataset, "PRETRAIN_COGNITIVE_MAP_DIR", tmp_path)
    monkeypatch.setattr(
        pretrain_dataset,
        "cached_cognitive_map_to_tensors",
        fake_cached_cognitive_map_to_tensors,
    )

    nav_db = pretrain_dataset.ReverieTextPathData.__new__(
        pretrain_dataset.ReverieTextPathData
    )
    outputs = nav_db._load_pretrain_cognitive_map(
        {
            "instr_id": "42_0",
            "scan": "scene",
        }
    )

    assert captured == {
        "scene_id": "scene",
        "cache_id": "42_0",
        "cache_dir": tmp_path,
        "random_rotation_augmentation": False,
    }
    assert outputs == {
        "cognitive_maps": "loaded-grid",
        "trajectory_keypoints": "trajectory_keypoints",
        "start_direction_vectors": "direction",
        "start_positions": "position",
    }


def test_pretrain_llm_map_loads_precomputed_cache(monkeypatch):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    pretrain_dataset = importlib.import_module("data.dataset")
    captured = {}

    def fake_llm_cached_cognitive_map_to_tensors(
        scene_id,
        cache_id,
        dataset,
        split,
        cache_dir=None,
        model_key="llama-3.1-8b-instruct",
        random_rotation_augmentation=False,
    ):
        captured["scene_id"] = scene_id
        captured["cache_id"] = cache_id
        captured["dataset"] = dataset
        captured["split"] = split
        captured["cache_dir"] = cache_dir
        captured["model_key"] = model_key
        captured["random_rotation_augmentation"] = random_rotation_augmentation
        return {
            "grid": "llm-grid",
            "trajectory_keypoints": "llm-trajectory-keypoints",
            "start_direction_vector": "llm-direction",
            "start_position": "llm-start",
        }

    monkeypatch.setattr(
        pretrain_dataset,
        "llm_cached_cognitive_map_to_tensors",
        fake_llm_cached_cognitive_map_to_tensors,
    )

    nav_db = pretrain_dataset.ReverieTextPathData.__new__(
        pretrain_dataset.ReverieTextPathData
    )
    nav_db.random_rotation_augmentation = True

    outputs = nav_db._load_llm_cognitive_map({"instr_id": "42_0", "scan": "scene"})

    assert captured == {
        "scene_id": "scene",
        "cache_id": "42_0",
        "dataset": "pretrain",
        "split": "mixed",
        "cache_dir": None,
        "model_key": "llama-3.1-8b-instruct",
        "random_rotation_augmentation": True,
    }
    assert outputs == {
        "cognitive_maps": "llm-grid",
        "trajectory_keypoints": "llm-trajectory-keypoints",
        "start_direction_vectors": "llm-direction",
        "start_positions": "llm-start",
    }


def test_pretrain_llm_map_requires_precomputed_cache(tmp_path):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    pretrain_dataset = importlib.import_module("data.dataset")
    nav_db = pretrain_dataset.ReverieTextPathData.__new__(
        pretrain_dataset.ReverieTextPathData
    )
    nav_db.llm_cache_dir = tmp_path
    nav_db.llm_cache_model_key = "test-model"

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Missing LLM-Navigation raster cognitive map cache: .*"
            "generate_navigation_cache"
        ),
    ):
        nav_db._load_llm_cognitive_map({"instr_id": "42_0", "scan": "scene"})


def test_pretrain_parser_accepts_llm_mode_and_rejects_mixed_map_modes(monkeypatch):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    pretrain_parser = importlib.import_module("parser")
    parser = pretrain_parser.load_parser()
    base_args = [
        "prog",
        "--vlnbert",
        "cmt",
        "--model_config",
        "model.json",
        "--output_dir",
        "out",
        "--config",
        "cfg.json",
    ]
    monkeypatch.setattr(pretrain_parser.json, "load", lambda _: {})
    monkeypatch.setattr(
        pretrain_parser, "open", lambda *_args, **_kwargs: None, raising=False
    )

    monkeypatch.setattr(sys, "argv", [*base_args, "--use_llm"])
    args = pretrain_parser.parse_with_config(parser)
    assert args.use_llm is True

    parser = pretrain_parser.load_parser()
    monkeypatch.setattr(sys, "argv", [*base_args, "--use_llm", "--use_imagined"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        pretrain_parser.parse_with_config(parser)


def test_pretrain_prior_map_requires_cached_map(tmp_path, monkeypatch):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    pretrain_dataset = importlib.import_module("data.dataset")

    monkeypatch.setattr(pretrain_dataset, "PRETRAIN_COGNITIVE_MAP_DIR", tmp_path)
    nav_db = pretrain_dataset.ReverieTextPathData.__new__(
        pretrain_dataset.ReverieTextPathData
    )

    with pytest.raises(FileNotFoundError, match="Missing cached cognitive map"):
        nav_db._load_pretrain_cognitive_map({"instr_id": "42_0", "scan": "scene"})


def test_pretrain_prior_map_filter_skips_missing_entries(
    tmp_path, monkeypatch, capsys
):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    pretrain_dataset = importlib.import_module("data.dataset")
    monkeypatch.setattr(pretrain_dataset, "PRETRAIN_COGNITIVE_MAP_DIR", tmp_path)
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    (scene_dir / "good.npz").touch()
    items = [
        {"instr_id": "good", "scan": "scene"},
        {"instr_id": "missing", "scan": "scene"},
    ]

    filtered = pretrain_dataset._filter_missing_pretrain_cognitive_maps(items)

    assert filtered == [{"instr_id": "good", "scan": "scene"}]
    assert (
        "pretrain_cognitive_maps: available=1 skipped_missing=1"
        in capsys.readouterr().out
    )


def test_pretrain_prior_map_filter_rejects_entirely_missing_cache(
    tmp_path, monkeypatch
):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    pretrain_dataset = importlib.import_module("data.dataset")
    monkeypatch.setattr(pretrain_dataset, "PRETRAIN_COGNITIVE_MAP_DIR", tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="No PriorGT pretraining cognitive-map caches were found",
    ):
        pretrain_dataset._filter_missing_pretrain_cognitive_maps(
            [{"instr_id": "missing", "scan": "scene"}]
        )


def test_pretrain_llm_map_filter_skips_missing_entries(
    tmp_path, monkeypatch, capsys
):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    pretrain_dataset = importlib.import_module("data.dataset")
    monkeypatch.setattr(pretrain_dataset, "PRETRAIN_LLM_COGNITIVE_MAP_DIR", tmp_path)
    raster_path = pretrain_dataset.llm_navigation_cognitive_map_raster_path(
        "scene",
        "good",
        "pretrain",
        "mixed",
        cache_dir=tmp_path,
        model_key="llama-3.1-8b-instruct",
    )
    boxes_path = pretrain_dataset.llm_navigation_cognitive_map_boxes_path(
        "scene",
        "good",
        "pretrain",
        "mixed",
        cache_dir=tmp_path,
        model_key="llama-3.1-8b-instruct",
    )
    raster_path.parent.mkdir(parents=True)
    boxes_path.parent.mkdir(parents=True)
    raster_path.touch()
    boxes_path.touch()
    items = [
        {"instr_id": "good", "scan": "scene"},
        {"instr_id": "missing", "scan": "scene"},
    ]

    filtered = pretrain_dataset._filter_missing_pretrain_llm_cognitive_maps(items)

    assert filtered == [{"instr_id": "good", "scan": "scene"}]
    assert (
        "pretrain_llm_navigation_maps: available=1 skipped_missing=1"
        in capsys.readouterr().out
    )


def test_pretrain_llm_map_filter_rejects_entirely_missing_cache(
    tmp_path, monkeypatch
):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    pretrain_dataset = importlib.import_module("data.dataset")
    monkeypatch.setattr(pretrain_dataset, "PRETRAIN_LLM_COGNITIVE_MAP_DIR", tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="No LLM-Navigation pretraining cognitive-map caches were found",
    ):
        pretrain_dataset._filter_missing_pretrain_llm_cognitive_maps(
            [{"instr_id": "missing", "scan": "scene"}]
        )


def test_pretrain_llm_language_filter_keeps_raw_dict_shape(monkeypatch, capsys):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    pretrain_dataset = importlib.import_module("data.dataset")
    monkeypatch.setattr(
        pretrain_dataset,
        "is_english_like_pretrain_record",
        lambda item: item["instr_id"] != "non-english",
    )
    items = [
        {"instr_id": "english", "scan": "scene", "instr_encoding": [1]},
        {"instr_id": "non-english", "scan": "scene", "instr_encoding": [2]},
    ]

    filtered = pretrain_dataset._filter_non_english_pretrain_records(items)

    assert filtered == [{"instr_id": "english", "scan": "scene", "instr_encoding": [1]}]
    assert filtered[0] is items[0]
    assert (
        "pretrain_language_filter: available=1 skipped_non_english=1"
        in capsys.readouterr().out
    )


def test_pretrain_llm_dataset_init_filters_non_english_records(
    tmp_path,
    monkeypatch,
):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    pretrain_dataset = importlib.import_module("data.dataset")
    records = [
        {"instr_id": "english", "scan": "scene", "instr_encoding": [1]},
        {"instr_id": "non-english", "scan": "scene", "instr_encoding": [2]},
    ]

    class FakeJsonLines:
        def __enter__(self):
            return iter(records)

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        pretrain_dataset.jsonlines,
        "open",
        lambda *_args, **_kwargs: FakeJsonLines(),
    )
    monkeypatch.setattr(
        pretrain_dataset,
        "is_english_like_pretrain_record",
        lambda item: item["instr_id"] != "non-english",
    )
    monkeypatch.setattr(
        pretrain_dataset,
        "_filter_missing_pretrain_llm_cognitive_maps",
        lambda items: items,
    )
    monkeypatch.setattr(pretrain_dataset, "load_nav_graphs", lambda *_args: ({}, {}, {}))
    monkeypatch.setattr(
        pretrain_dataset,
        "get_view_rel_angles",
        lambda baseViewId: np.zeros((1, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        pretrain_dataset,
        "get_angle_fts",
        lambda headings, elevations, angle_feat_size: np.zeros(
            (len(headings), angle_feat_size), dtype=np.float32
        ),
    )
    scanvp_cands_file = tmp_path / "scanvp_cands.json"
    scanvp_cands_file.write_text("{}")

    nav_db = pretrain_dataset.ReverieTextPathData(
        anno_files=["ignored.jsonl"],
        img_ft_file="ignored.hdf5",
        dep_ft_file="ignored.hdf5",
        obj_ft_file=None,
        scanvp_cands_file=scanvp_cands_file,
        connectivity_dir="ignored",
        use_llm=True,
        in_memory=False,
    )

    assert nav_db.data == [{"instr_id": "english", "scan": "scene", "instr_encoding": [1]}]
    assert nav_db.data[0] is records[0]


@pytest.mark.parametrize(
    ("turns", "expected_paths", "expected_start", "expected_direction"),
    [
        (0, [[10.0, 20.0], [30.0, 40.0]], [10.0, 20.0], [1.0, 2.0]),
        (1, [[79.0, 10.0], [59.0, 30.0]], [79.0, 10.0], [2.0, -1.0]),
        (2, [[89.0, 79.0], [69.0, 59.0]], [89.0, 79.0], [-1.0, -2.0]),
        (3, [[20.0, 89.0], [40.0, 69.0]], [20.0, 89.0], [-2.0, 1.0]),
    ],
)
def test_pretrain_prior_map_rotates_tensor_bundle(
    turns, expected_paths, expected_start, expected_direction
):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    grid = torch.arange(100 * 100, dtype=torch.float32).reshape(1, 100, 100)
    tensors = {
        "grid": grid,
        "trajectory_keypoints": torch.tensor(
            [[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32
        ),
        "start_direction_vector": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "start_position": torch.tensor([10.0, 20.0], dtype=torch.float32),
    }

    from vlnce_baselines.models.etp_prior_gt.map_utils import (
        rotate_cognitive_map_tensors_by_right_angle,
    )

    rotated = rotate_cognitive_map_tensors_by_right_angle(tensors, turns)

    assert torch.equal(rotated["grid"], torch.rot90(grid, turns % 4, dims=(-2, -1)))
    assert torch.allclose(rotated["trajectory_keypoints"], torch.tensor(expected_paths))
    assert torch.allclose(rotated["start_position"], torch.tensor(expected_start))
    assert torch.allclose(
        rotated["start_direction_vector"], torch.tensor(expected_direction)
    )


def test_pretrain_prior_map_applies_random_rotation(tmp_path, monkeypatch):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    pretrain_dataset = importlib.import_module("data.dataset")
    captured = {}

    def fake_cached_cognitive_map_to_tensors(
        scene_id,
        cache_id,
        cache_dir,
        random_rotation_augmentation,
    ):
        captured["scene_id"] = scene_id
        captured["cache_id"] = cache_id
        captured["cache_dir"] = cache_dir
        captured["random_rotation_augmentation"] = random_rotation_augmentation
        return {
            "grid": torch.arange(100 * 100, dtype=torch.float32).reshape(1, 100, 100),
            "trajectory_keypoints": torch.tensor([[10.0, 20.0]], dtype=torch.float32),
            "start_direction_vector": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "start_position": torch.tensor([10.0, 20.0], dtype=torch.float32),
        }

    monkeypatch.setattr(pretrain_dataset, "PRETRAIN_COGNITIVE_MAP_DIR", tmp_path)
    monkeypatch.setattr(
        pretrain_dataset,
        "cached_cognitive_map_to_tensors",
        fake_cached_cognitive_map_to_tensors,
    )

    nav_db = pretrain_dataset.ReverieTextPathData.__new__(
        pretrain_dataset.ReverieTextPathData
    )
    nav_db.random_rotation_augmentation = True

    outputs = nav_db._load_pretrain_cognitive_map(
        {
            "instr_id": "42_0",
            "scan": "scene",
        }
    )

    assert captured == {
        "scene_id": "scene",
        "cache_id": "42_0",
        "cache_dir": tmp_path,
        "random_rotation_augmentation": True,
    }
    assert outputs["trajectory_keypoints"].tolist() == [[10.0, 20.0]]
    assert outputs["start_positions"].tolist() == [10.0, 20.0]
    assert outputs["start_direction_vectors"].tolist() == [1.0, 2.0]
