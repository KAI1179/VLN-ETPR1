from functools import partial
import sys
from pathlib import Path

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
            "--local_rank",
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
            "--reference-path-loss-weight",
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
    assert args.reference_path_loss_weight == 0.25
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
    assert args.reference_path_loss_weight == 0.1
    assert args.init_positive_prob == 0.002
    assert args.max_text_len is None
    assert args.num_workers == 2
    assert args.val_limit is None
    assert args.log_every == 1


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


def test_pretrain_loader_uses_spawn_context_for_workers(monkeypatch):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    from data.loader import build_dataloader

    class TinyDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return index

    opts = type(
        "Opts",
        (),
        {
            "train_batch_size": 1,
            "val_batch_size": 1,
            "local_rank": -1,
            "n_workers": 2,
            "pin_mem": False,
        },
    )()
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    loader, _ = build_dataloader(
        task="mlm",
        dataset=TinyDataset(),
        collate_fn=lambda batch: batch,
        is_train=True,
        opts=opts,
    )

    assert loader.num_workers == 2
    assert loader.multiprocessing_context.get_start_method() == "spawn"


def test_pretrain_prior_map_uses_cached_categories(monkeypatch):
    pretrain_src = ROOT / "pretrain_src" / "pretrain_src"
    if str(pretrain_src) not in sys.path:
        sys.path.insert(0, str(pretrain_src))

    from data import dataset as pretrain_dataset

    nav_db = pretrain_dataset.ReverieTextPathData.__new__(
        pretrain_dataset.ReverieTextPathData
    )
    nav_db.connectivity_dir = "connectivity"
    nav_db._prior_category_by_instr_id = {"42_0": ({3}, {7})}
    captured = {}

    def fake_build(annotation, connectivity_dir=None, category_extractor=None):
        captured["connectivity_dir"] = connectivity_dir
        captured["categories"] = category_extractor("do not decode in worker")
        return object()

    monkeypatch.setattr(
        pretrain_dataset,
        "build_cognitive_map_for_annotation",
        fake_build,
    )
    monkeypatch.setattr(
        pretrain_dataset,
        "cognitive_map_to_tensors",
        lambda cognitive_map: {
            "grid": "grid",
            "reference_paths": "reference_paths",
            "start_direction_vector": "direction",
            "start_position": "position",
        },
    )

    outputs = nav_db._load_pretrain_cognitive_map(
        {
            "instr_id": "42_0",
            "scan": "scene",
            "path": ["a"],
            "heading": 0.0,
            "instr_encoding": [2],
            "task_type_encoding": 1,
        }
    )

    assert captured == {
        "connectivity_dir": "connectivity",
        "categories": ({3}, {7}),
    }
    assert outputs["cognitive_maps"] == "grid"
    assert outputs["direction_vectors"] == "reference_paths"
