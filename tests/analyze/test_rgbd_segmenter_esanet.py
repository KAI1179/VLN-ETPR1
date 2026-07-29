from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from prior.analyze.d2026_07_29 import rgbd_segmenter_esanet as esanet
from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    PreparedHostBatch,
    SpatialTransform,
    transfer_prepared_host_batch,
)


def test_parse_args_uses_dashed_tap_argument() -> None:
    assert esanet.parse_args(["--device", "cpu"]).device == "cpu"
    assert esanet.parse_args(["--sealed-batch"]).sealed_batch is True


def test_preprocess_matches_frozen_rgb_and_raw_depth_contract() -> None:
    rgb = torch.tensor((0, 127, 255), dtype=torch.uint8).view(
        1, 1, 1, 3
    ).expand(1, 256, 256, 3)
    depth = torch.zeros((1, 256, 256), dtype=torch.float32)
    depth[:, 0, 0] = 1.2349
    actual_rgb, actual_depth = esanet._preprocess_tensors(rgb, depth)

    expected_rgb = rgb.permute(0, 3, 1, 2).float().div(255.0)
    expected_rgb = expected_rgb.sub(
        torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    ).div(torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
    expected_mm = (depth.numpy() * np.float32(1000.0)).astype(np.uint16)
    expected_depth = torch.from_numpy(expected_mm.astype(np.float32)).unsqueeze(1)
    expected_depth = expected_depth.sub(2841.94941272766).div(
        1417.2594281672277
    )
    expected_depth.masked_fill_(
        torch.from_numpy(expected_mm == 0).unsqueeze(1), 0.0
    )

    assert torch.equal(actual_rgb, expected_rgb)
    assert torch.equal(actual_depth, expected_depth)
    assert actual_depth[0, 0, 0, 0].item() == pytest.approx(
        (1234.0 - 2841.94941272766) / 1417.2594281672277
    )
    assert actual_depth[0, 0, 1, 1].item() == 0.0


@pytest.mark.parametrize(
    ("rgb", "depth"),
    (
        (
            torch.zeros((1, 256, 256, 3), dtype=torch.float32),
            torch.zeros((1, 256, 256), dtype=torch.float32),
        ),
        (
            torch.zeros((1, 3, 256, 256), dtype=torch.uint8),
            torch.zeros((1, 256, 256), dtype=torch.float32),
        ),
        (
            torch.zeros((1, 256, 256, 3), dtype=torch.uint8),
            torch.zeros((1, 1, 256, 256), dtype=torch.float32),
        ),
    ),
)
def test_preprocess_rejects_schema_drift(
    rgb: torch.Tensor, depth: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="schema differs"):
        esanet._preprocess_tensors(rgb, depth)


def test_activate_rejects_preloaded_generic_src(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "src", object())
    with pytest.raises(ValueError, match="already loaded"):
        esanet.activate_candidate_imports(esanet.ESANetPaths.default())


def test_technical_smoke_result_rejects_semantic_contract_drift() -> None:
    with pytest.raises(ValueError, match="result is invalid"):
        esanet.TechnicalSmokeResult(
            checkpoint_state_sha256="0" * 64,
            model_parameters=46_955_840,
            output_shape=(1, 40, 256, 256),
            output_dtype="torch.float32",
            output_finite=True,
            repeated_label_sha256="1" * 64,
        )


def test_adapter_rejects_output_contract_drift() -> None:
    class WrongShapeModel(torch.nn.Module):
        def forward(
            self, rgb: torch.Tensor, depth: torch.Tensor
        ) -> torch.Tensor:
            del depth
            return torch.zeros(
                (12, 39, 256, 256), dtype=torch.float32, device=rgb.device
            )

    model = WrongShapeModel().eval()
    adapter = esanet.ESANetSegmenterAdapter(model)
    transform = SpatialTransform.from_sizes(
        raw_height=256,
        raw_width=256,
        model_height=256,
        model_width=256,
    )
    batch = transfer_prepared_host_batch(
        PreparedHostBatch(
            tensors=(
                torch.zeros((12, 3, 256, 256), dtype=torch.float32),
                torch.zeros((12, 1, 256, 256), dtype=torch.float32),
            ),
            spatial_transform=transform,
        ),
        device=torch.device("cpu"),
        non_blocking=False,
    )
    with pytest.raises(ValueError, match="output differs"):
        adapter.infer(batch)


@pytest.mark.skipif(
    not esanet.ESANetPaths.default().checkpoint.is_file(),
    reason="ignored official ESANet asset is not installed",
)
def test_local_candidate_semantic_blind_smoke_in_fresh_process() -> None:
    paths = esanet.ESANetPaths.default()
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "prior.analyze.d2026_07_29.rgbd_segmenter_esanet",
            "--device",
            "cpu",
        ),
        cwd=paths.repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["checkpoint_state_sha256"] == (
        "52d436ab959e79c552617b04aab513a681ed5086d67add5773d260c6603776bb"
    )
    assert record["model_parameters"] == 46_955_840
    assert record["output_shape"] == [1, 40, 256, 256]
    assert record["output_dtype"] == "torch.float32"
    assert record["output_finite"] is True
    assert record["semantic_endpoints_exposed"] is False
    assert record["smoke_kind"] == "synthetic-one-view"
    assert len(record["repeated_label_sha256"]) == 64


@pytest.mark.skipif(
    (
        not esanet.ESANetPaths.default().checkpoint.is_file()
        or torch.cuda.device_count() < 5
    ),
    reason="local official asset and authorized physical GPU 4 are required",
)
def test_local_sealed_twelve_view_smoke_in_fresh_process() -> None:
    paths = esanet.ESANetPaths.default()
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "4"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "prior.analyze.d2026_07_29.rgbd_segmenter_esanet",
            "--device",
            "cuda:0",
            "--sealed-batch",
        ),
        cwd=paths.repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["smoke_kind"] == "sealed-twelve-view"
    assert record["sealed_input_sha256"] == (
        "1ca2f07e10eb29ae162db81b29fb662b79e68acd6bd31d9a0025bbb88d8c99cf"
    )
    assert record["output_shape"] == [12, 40, 256, 256]
    assert record["output_dtype"] == "torch.float32"
    assert record["output_finite"] is True
    assert record["repeated_label_sha256"] == (
        "80c0f5f37d3318549c1d034c095e4a9b4506ea0b9e8e707fa526e6a0df476192"
    )
    assert record["peak_allocated_bytes"] > 0
    assert record["semantic_endpoints_exposed"] is False


def test_requirements_path_is_tracked_under_dated_analysis() -> None:
    paths = esanet.ESANetPaths.default()
    assert paths.requirements.relative_to(paths.repository_root) == Path(
        "prior/analyze/d2026_07_29/rgbd_segmenter_esanet_requirements.txt"
    )
