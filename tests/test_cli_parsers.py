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
            "--train-dataset",
            "train.json.gz",
            "extra.json.gz",
            "--val-dataset",
            "val.json.gz",
            "--dataset",
            "r2r",
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
            "--direction-loss-weight",
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
    assert args.train_dataset == [Path("train.json.gz"), Path("extra.json.gz")]
    assert args.val_dataset == [Path("val.json.gz")]
    assert args.dataset == "r2r"
    assert args.output == Path("predictor.pt")
    assert args.batch_size == 4
    assert args.epochs == 2
    assert args.lr == 0.001
    assert args.loss == "focal"
    assert args.max_pos_weight == 7.5
    assert args.focal_gamma == 1.5
    assert args.direction_loss_weight == 0.25
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
    assert args.train_dataset == [
        Path("data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/train/train_90.json.gz")
    ]
    assert args.val_dataset == [
        Path("data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz")
    ]
    assert args.batch_size == 8
    assert args.max_pos_weight == 20.0
    assert args.focal_gamma == 2.0
    assert args.direction_loss_weight == 0.1
    assert args.init_positive_prob == 0.002
    assert args.max_text_len is None
    assert args.num_workers == 2
    assert args.val_limit is None
    assert args.log_every == 1
