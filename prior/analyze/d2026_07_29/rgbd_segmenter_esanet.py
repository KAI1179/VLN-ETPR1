"""Pinned ESANet candidate adapter and semantic-blind technical smoke."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import stat
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from tap import Tap

from prior.analyze.d2026_07_29.rgbd_segmenter_benchmark_contract import (
    DeviceBatch,
    PreparedHostBatch,
    SegmenterInput,
    SpatialTransform,
    capture_environment_sha256,
    transfer_prepared_host_batch,
)
from prior.analyze.d2026_07_29.rgbd_segmenter_raw_frame_package import (
    parse_raw_frame_npz_bytes,
)

_REVISION = "820c5bb633e49e69dcd075d4330165bb540a0cc9"
_REPOSITORY_URL = "https://github.com/TUI-NICR/ESANet.git"
_ARCHIVE_SHA256 = (
    "ffe69568e107471d18a31493ef4aae2aa7aa1572ead51d072c4f02d8b4aef47d"
)
_CHECKPOINT_SHA256 = (
    "6b84f77dee42739fd3c5dd9e6b278450fa14e6977e2ec1609060eb6eb05cf456"
)
_STATE_FINGERPRINT_SHA256 = (
    "52d436ab959e79c552617b04aab513a681ed5086d67add5773d260c6603776bb"
)
_OVERLAY_TREE_SHA256 = (
    "1673bdeeb522bd437eba149b83a2ad11d39f9e59fcacad43223cde05f2de3995"
)
_COMPLETE_ENVIRONMENT_SHA256 = (
    "2a02828d8a98ae9336a9a94b7e885a357d09489d01eb489a91506f94ee6fdecd"
)
_REQUIREMENTS_SHA256 = (
    "400cf8bbe845b1e677c772cc6d3238fbbfb1e39103e5dbfc50c2150017ffb6a3"
)
_PYPROJECT_SHA256 = (
    "3b411d4c0c3322006a5d6500292a604c34276bd9c224746ed78b924afba5348e"
)
_UV_LOCK_SHA256 = (
    "142cbd37a450be3e46cb585335972f87e2ef9cd14b81586e6b0d4609c9f5791e"
)
_ARCHIVE_BYTES = 174_833_589
_CHECKPOINT_BYTES = 188_218_765
_STATE_KEYS = 898
_MODEL_PARAMETERS = 46_955_840
_DEPTH_MEAN = 2841.94941272766
_DEPTH_STD = 1417.2594281672277
_RGB_MEAN = (0.485, 0.456, 0.406)
_RGB_STD = (0.229, 0.224, 0.225)
_MODEL_SIZE = (256, 256)
_SEALED_ARTIFACT_PATH = (
    "data/rgbd_segmenter_benchmark/r2r-val-unseen-50-raw-v1/"
    "observations/2azQ1b91cZZ/00-bd81990a0f88c2c285ac.npz"
)
_SEALED_ARTIFACT_SHA256 = (
    "1ca2f07e10eb29ae162db81b29fb662b79e68acd6bd31d9a0025bbb88d8c99cf"
)
_OVERLAY_ROOTS = {
    "pandas",
    "pandas-2.0.3.dist-info",
    "pandas.libs",
    "pytz",
    "pytz-2025.2.dist-info",
    "tzdata",
    "tzdata-2025.3.dist-info",
}
_BASE_DISTRIBUTIONS = {
    "numpy": "1.24.3",
    "python-dateutil": "2.9.0.post0",
    "torch": "2.1.2+cu118",
    "torchvision": "0.16.2+cu118",
}
_OVERLAY_DISTRIBUTIONS = {
    "pandas": "2.0.3",
    "pytz": "2025.2",
    "tzdata": "2025.3",
}

__all__ = (
    "ESANetArgs",
    "ESANetPaths",
    "ESANetSegmenterAdapter",
    "SealedBatchSmokeResult",
    "TechnicalSmokeResult",
    "activate_candidate_imports",
    "audit_candidate_assets",
    "build_esanet",
    "load_checkpoint_strict",
    "main",
    "parse_args",
    "run_technical_smoke",
    "run_sealed_batch_smoke",
    "verify_upstream_preprocessing",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ESANetPaths:
    repository_root: Path
    candidate_root: Path
    source_root: Path
    archive: Path
    checkpoint: Path
    overlay: Path
    requirements: Path

    @classmethod
    def default(cls) -> "ESANetPaths":
        root = Path(__file__).resolve(strict=True).parents[3]
        candidate = (
            root
            / "data/rgbd_segmenter_benchmark/candidates/"
            "esanet-r34-nbt1d-scenenet"
        )
        return cls(
            repository_root=root,
            candidate_root=candidate,
            source_root=candidate / "source/ESANet",
            archive=candidate / "archives/nyuv2_r34_NBt1D_scenenet.tar.gz",
            checkpoint=(
                candidate / "checkpoints/nyuv2/r34_NBt1D_scenenet.pth"
            ),
            overlay=candidate / "environment/pandas-overlay-v1",
            requirements=(
                root
                / "prior/analyze/d2026_07_29/"
                "rgbd_segmenter_esanet_requirements.txt"
            ),
        )


@dataclass(frozen=True)
class TechnicalSmokeResult:
    checkpoint_state_sha256: str
    model_parameters: int
    output_shape: Tuple[int, ...]
    output_dtype: str
    output_finite: bool
    repeated_label_sha256: str

    def __post_init__(self) -> None:
        if (
            self.checkpoint_state_sha256 != _STATE_FINGERPRINT_SHA256
            or self.model_parameters != _MODEL_PARAMETERS
            or self.output_shape != (1, 40, 256, 256)
            or self.output_dtype != "torch.float32"
            or not self.output_finite
            or len(self.repeated_label_sha256) != 64
        ):
            raise ValueError("ESANet technical smoke result is invalid")


@dataclass(frozen=True)
class SealedBatchSmokeResult:
    sealed_input_sha256: str
    output_shape: Tuple[int, ...]
    output_dtype: str
    output_finite: bool
    repeated_label_sha256: str
    peak_allocated_bytes: int

    def __post_init__(self) -> None:
        if (
            self.sealed_input_sha256 != _SEALED_ARTIFACT_SHA256
            or self.output_shape != (12, 40, 256, 256)
            or self.output_dtype != "torch.float32"
            or not self.output_finite
            or len(self.repeated_label_sha256) != 64
            or type(self.peak_allocated_bytes) is not int
            or self.peak_allocated_bytes < 1
        ):
            raise ValueError("ESANet sealed batch smoke result is invalid")


def _require_regular_file(
    path: Path, *, expected_bytes: int, expected_sha256: str
) -> None:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & (stat.S_IWGRP | stat.S_IWOTH)
        or info.st_size != expected_bytes
        or _sha256(path) != expected_sha256
    ):
        raise ValueError(f"candidate artifact failed authority: {path}")


def _overlay_tree_sha256(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("ESANet overlay root is not a real directory")
    if {path.name for path in root.iterdir()} != _OVERLAY_ROOTS:
        raise ValueError("ESANet overlay roots differ from the lock")
    digest = hashlib.sha256()
    files = 0
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        mode = stat.S_IMODE(info.st_mode)
        if (
            path.is_symlink()
            or not (path.is_file() or path.is_dir())
            or mode
            & (
                stat.S_ISUID
                | stat.S_ISGID
                | stat.S_ISVTX
                | stat.S_IWGRP
                | stat.S_IWOTH
            )
        ):
            raise ValueError(f"unsafe ESANet overlay entry: {relative}")
        if (
            path.is_dir()
            and next(path.iterdir(), None) is None
            and relative != "pandas.libs"
        ):
            raise ValueError(f"empty ESANet overlay directory: {relative}")
        if path.is_file():
            if info.st_nlink != 1 or path.suffix == ".pth":
                raise ValueError(f"unsafe ESANet overlay file: {relative}")
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(str(mode).encode("ascii") + b"\0")
            digest.update(bytes.fromhex(_sha256(path)))
            files += 1
    if files != 2813:
        raise ValueError("ESANet overlay file count differs from the lock")
    return digest.hexdigest()


def _verify_overlay_records(root: Path) -> None:
    for record in sorted(root.glob("*.dist-info/RECORD")):
        with record.open("r", encoding="utf-8", newline="") as stream:
            rows = tuple(csv.reader(stream))
        for relative, encoded_digest, encoded_size in rows:
            path = root / relative
            if (
                root.resolve(strict=True) not in path.resolve(strict=True).parents
                or path.is_symlink()
                or not path.is_file()
            ):
                raise ValueError(f"invalid overlay RECORD path: {relative}")
            if encoded_digest:
                algorithm, expected = encoded_digest.split("=", 1)
                if algorithm != "sha256":
                    raise ValueError("overlay RECORD uses a non-SHA256 digest")
                actual = (
                    base64.urlsafe_b64encode(bytes.fromhex(_sha256(path)))
                    .decode("ascii")
                    .rstrip("=")
                )
                if actual != expected:
                    raise ValueError(f"overlay RECORD digest mismatch: {relative}")
            if encoded_size and path.stat().st_size != int(encoded_size):
                raise ValueError(f"overlay RECORD size mismatch: {relative}")


def _git_output(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(source_root), *arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0 or result.stderr:
        raise ValueError("ESANet source Git authority failed")
    return result.stdout.strip()


def _require_source_authority(paths: ESANetPaths) -> None:
    if (
        paths.source_root.resolve(strict=True) != paths.source_root
        or _git_output(paths.source_root, "rev-parse", "HEAD") != _REVISION
        or _git_output(paths.source_root, "remote", "get-url", "origin")
        != _REPOSITORY_URL
        or _git_output(
            paths.source_root, "status", "--porcelain", "--untracked-files=all"
        )
    ):
        raise ValueError("ESANet source checkout differs from authority")


def audit_candidate_assets(paths: ESANetPaths) -> None:
    if not isinstance(paths, ESANetPaths):
        raise TypeError("ESANet paths are required")
    root = paths.repository_root.resolve(strict=True)
    expected_paths = (
        paths.candidate_root,
        paths.source_root,
        paths.archive,
        paths.checkpoint,
        paths.overlay,
        paths.requirements,
    )
    if any(
        root not in path.resolve(strict=True).parents for path in expected_paths
    ):
        raise ValueError("ESANet path escapes the repository")
    _require_regular_file(
        paths.archive,
        expected_bytes=_ARCHIVE_BYTES,
        expected_sha256=_ARCHIVE_SHA256,
    )
    _require_regular_file(
        paths.checkpoint,
        expected_bytes=_CHECKPOINT_BYTES,
        expected_sha256=_CHECKPOINT_SHA256,
    )
    if _sha256(paths.requirements) != _REQUIREMENTS_SHA256:
        raise ValueError("ESANet overlay requirements differ from the lock")
    if _sha256(root / "pyproject.toml") != _PYPROJECT_SHA256:
        raise ValueError("ESANet base pyproject differs from the lock")
    if _sha256(root / "uv.lock") != _UV_LOCK_SHA256:
        raise ValueError("ESANet base uv.lock differs from the lock")
    if (
        sys.version_info[:3] != (3, 8, 19)
        or platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or {
            name: importlib.metadata.version(name)
            for name in _BASE_DISTRIBUTIONS
        }
        != _BASE_DISTRIBUTIONS
    ):
        raise ValueError("ESANet base runtime differs from the lock")
    overlay_distributions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions(
            path=[str(paths.overlay)]
        )
    }
    if overlay_distributions != _OVERLAY_DISTRIBUTIONS:
        raise ValueError("ESANet overlay distributions differ from the lock")
    if _overlay_tree_sha256(paths.overlay) != _OVERLAY_TREE_SHA256:
        raise ValueError("ESANet overlay tree differs from the lock")
    _verify_overlay_records(paths.overlay)
    _require_source_authority(paths)


def activate_candidate_imports(paths: ESANetPaths) -> None:
    if any(name == "src" or name.startswith("src.") for name in sys.modules):
        raise ValueError("generic src package is already loaded")
    if importlib.util.find_spec("src") is not None:
        raise ValueError("a competing generic src package is importable")
    if any(name in sys.modules for name in ("pandas", "pytz", "tzdata")):
        raise ValueError("ESANet overlay packages are already loaded")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ValueError("ESANet child requires PYTHONNOUSERSITE=1")
    source = str(paths.source_root)
    overlay = str(paths.overlay)
    if source in sys.path or overlay in sys.path:
        raise ValueError("ESANet import roots were already active")
    sys.path[:0] = [source, overlay]
    importlib.invalidate_caches()


def _upstream_model_module() -> ModuleType:
    module = importlib.import_module("src.models.model")
    location = Path(cast(str, module.__file__)).resolve(strict=True)
    expected = ESANetPaths.default().source_root.resolve(strict=True)
    if expected not in location.parents:
        raise ValueError("ESANet model resolved outside pinned source")
    return module


def _require_loaded_source_origins(paths: ESANetPaths) -> None:
    source = paths.source_root.resolve(strict=True)
    loaded = 0
    for name, module in sys.modules.items():
        if name != "src" and not name.startswith("src."):
            continue
        if not isinstance(module, ModuleType):
            raise ValueError(f"loaded ESANet module has no real origin: {name}")
        if module.__file__ is None:
            spec = module.__spec__
            locations = (
                ()
                if spec is None or spec.submodule_search_locations is None
                else tuple(spec.submodule_search_locations)
            )
            if not locations or any(
                source not in Path(location).resolve(strict=True).parents
                and Path(location).resolve(strict=True) != source
                for location in locations
            ):
                raise ValueError(
                    f"loaded ESANet namespace escaped pinned source: {name}"
                )
            loaded += 1
            continue
        origin = Path(module.__file__)
        if origin.is_symlink() or source not in origin.resolve(strict=True).parents:
            raise ValueError(f"loaded ESANet module escaped pinned source: {name}")
        loaded += 1
    if loaded < 2:
        raise ValueError("pinned ESANet modules were not loaded")


def build_esanet(*, device: torch.device) -> torch.nn.Module:
    if not isinstance(device, torch.device):
        raise TypeError("ESANet device is required")
    module = _upstream_model_module()
    model_class = module.ESANet
    model = model_class(
        height=256,
        width=256,
        num_classes=40,
        encoder_rgb="resnet34",
        encoder_depth="resnet34",
        encoder_block="NonBottleneck1D",
        channels_decoder=[512, 256, 128],
        pretrained_on_imagenet=False,
        activation="relu",
        encoder_decoder_fusion="add",
        context_module="ppm",
        nr_decoder_blocks=[3, 3, 3],
        fuse_depth_in_rgb_encoder="SE-add",
        upsampling="learned-3x3-zeropad",
    )
    if not isinstance(model, torch.nn.Module):
        raise ValueError("upstream ESANet constructor returned the wrong type")
    return model.eval().to(device)


def _state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    dtype_counts: Counter[str] = Counter()
    for key, tensor in state.items():
        if (
            not isinstance(key, str)
            or not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or tensor.is_floating_point()
            and not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError("ESANet checkpoint tensor schema is unsafe")
        raw = tensor.detach().contiguous().numpy().tobytes(order="C")
        descriptor = (
            f"{key}\0{tuple(tensor.shape)}\0{tensor.dtype}\0{len(raw)}\0"
        ).encode("utf-8")
        digest.update(descriptor)
        digest.update(hashlib.sha256(raw).digest())
        dtype_counts[str(tensor.dtype)] += 1
    if (
        len(state) != _STATE_KEYS
        or dtype_counts != Counter({"torch.float32": 799, "torch.int64": 99})
    ):
        raise ValueError("ESANet checkpoint key or dtype counts differ")
    return digest.hexdigest()


def load_checkpoint_strict(
    model: torch.nn.Module, checkpoint: Path
) -> str:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("ESANet model is required")
    value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or tuple(value) != ("state_dict",):
        raise ValueError("ESANet checkpoint outer schema differs")
    raw_state = value["state_dict"]
    if not isinstance(raw_state, dict) or not raw_state:
        raise ValueError("ESANet checkpoint state_dict differs")
    if any(
        not isinstance(key, str) or not isinstance(tensor, torch.Tensor)
        for key, tensor in raw_state.items()
    ):
        raise ValueError("ESANet checkpoint state_dict is not tensor-only")
    state = cast(Mapping[str, torch.Tensor], raw_state)
    fingerprint = _state_fingerprint(state)
    if fingerprint != _STATE_FINGERPRINT_SHA256:
        raise ValueError("ESANet checkpoint state fingerprint differs")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("ESANet strict state load was not exact")
    return fingerprint


def _preprocess_tensors(
    rgb_nhwc: torch.Tensor, depth_m: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (
        rgb_nhwc.dtype is not torch.uint8
        or rgb_nhwc.ndim != 4
        or rgb_nhwc.shape[1:] != (256, 256, 3)
        or depth_m.dtype is not torch.float32
        or depth_m.shape != rgb_nhwc.shape[:3]
        or rgb_nhwc.device.type != "cpu"
        or depth_m.device.type != "cpu"
    ):
        raise ValueError("ESANet preprocessing input schema differs")
    rgb = rgb_nhwc.permute(0, 3, 1, 2).to(dtype=torch.float32).div(255.0)
    mean = torch.tensor(_RGB_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(_RGB_STD, dtype=torch.float32).view(1, 3, 1, 1)
    rgb = rgb.sub(mean).div(std)
    depth_mm_numpy = (
        depth_m.detach().contiguous().numpy() * np.float32(1000.0)
    ).astype(np.uint16)
    invalid = depth_mm_numpy == 0
    depth = torch.from_numpy(depth_mm_numpy.astype(np.float32)).unsqueeze(1)
    depth = depth.sub(_DEPTH_MEAN).div(_DEPTH_STD)
    depth.masked_fill_(torch.from_numpy(invalid).unsqueeze(1), 0.0)
    return rgb.contiguous(), depth.contiguous()


def verify_upstream_preprocessing() -> None:
    upstream = importlib.import_module("src.preprocessing")
    preprocessor = upstream.get_preprocessor(
        depth_mean=_DEPTH_MEAN,
        depth_std=_DEPTH_STD,
        depth_mode="raw",
        height=None,
        width=None,
        phase="test",
    )
    y = np.arange(256, dtype=np.uint16)[:, None]
    x = np.arange(256, dtype=np.uint16)[None, :]
    rgb_numpy = np.stack(
        (
            np.broadcast_to((x % 256).astype(np.uint8), (256, 256)),
            np.broadcast_to((y % 256).astype(np.uint8), (256, 256)),
            ((x + y) % 256).astype(np.uint8),
        ),
        axis=2,
    )
    depth_m_numpy = ((x * 31 + y * 17) % 10_001).astype(np.float32) / 1000.0
    depth_m_numpy[0, 0] = 0.0009
    depth_mm_numpy = (
        depth_m_numpy * np.float32(1000.0)
    ).astype(np.uint16)
    expected = preprocessor(
        {"image": rgb_numpy.copy(), "depth": depth_mm_numpy.copy()}
    )
    actual_rgb, actual_depth = _preprocess_tensors(
        torch.from_numpy(rgb_numpy).unsqueeze(0),
        torch.from_numpy(depth_m_numpy).unsqueeze(0),
    )
    if not torch.equal(actual_rgb[0], expected["image"]) or not torch.equal(
        actual_depth[0], expected["depth"]
    ):
        raise ValueError("ESANet preprocessing differs from pinned upstream")


class ESANetSegmenterAdapter:
    def __init__(self, model: torch.nn.Module) -> None:
        if not isinstance(model, torch.nn.Module) or model.training:
            raise ValueError("ESANet adapter requires an eval-mode model")
        self._model = model

    def preprocess_host(self, value: SegmenterInput) -> PreparedHostBatch:
        if not isinstance(value, SegmenterInput):
            raise ValueError("ESANet adapter requires SegmenterInput")
        rgb, depth = _preprocess_tensors(value.rgb, value.depth_m)
        return PreparedHostBatch(
            tensors=(rgb, depth),
            spatial_transform=SpatialTransform.from_sizes(
                raw_height=256,
                raw_width=256,
                model_height=256,
                model_width=256,
            ),
        )

    def infer(self, value: DeviceBatch) -> torch.Tensor:
        if not isinstance(value, DeviceBatch) or len(value.tensors) != 2:
            raise ValueError("ESANet adapter requires the runner device batch")
        rgb, depth = value.tensors
        if (
            rgb.shape != (12, 3, 256, 256)
            or depth.shape != (12, 1, 256, 256)
            or rgb.dtype is not torch.float32
            or depth.dtype is not torch.float32
        ):
            raise ValueError("ESANet prepared tensors differ from commitment")
        logits = self._model(rgb, depth)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.shape != (12, 40, 256, 256)
            or logits.dtype is not torch.float32
            or logits.device != rgb.device
            or not bool(torch.isfinite(logits).all())
        ):
            raise ValueError("ESANet output differs from commitment")
        return logits


def run_technical_smoke(
    paths: ESANetPaths, *, device: torch.device
) -> TechnicalSmokeResult:
    audit_candidate_assets(paths)
    activate_candidate_imports(paths)
    if capture_environment_sha256() != _COMPLETE_ENVIRONMENT_SHA256:
        raise ValueError("complete ESANet runtime environment differs")
    verify_upstream_preprocessing()
    model = build_esanet(device=device)
    fingerprint = load_checkpoint_strict(model, paths.checkpoint)
    if sum(parameter.numel() for parameter in model.parameters()) != _MODEL_PARAMETERS:
        raise ValueError("ESANet model parameter count differs")
    rgb = torch.zeros((1, 3, *_MODEL_SIZE), dtype=torch.float32, device=device)
    depth = torch.zeros((1, 1, *_MODEL_SIZE), dtype=torch.float32, device=device)
    with torch.inference_mode():
        first = model(rgb, depth)
        second = model(rgb, depth)
    if (
        first.shape != (1, 40, 256, 256)
        or first.dtype is not torch.float32
        or not bool(torch.isfinite(first).all())
    ):
        raise ValueError("ESANet technical output contract failed")
    first_labels = first.argmax(dim=1).to("cpu").contiguous().numpy().tobytes()
    second_labels = second.argmax(dim=1).to("cpu").contiguous().numpy().tobytes()
    if first_labels != second_labels:
        raise ValueError("ESANet repeated technical predictions differ")
    _require_loaded_source_origins(paths)
    _require_source_authority(paths)
    return TechnicalSmokeResult(
        checkpoint_state_sha256=fingerprint,
        model_parameters=_MODEL_PARAMETERS,
        output_shape=tuple(first.shape),
        output_dtype=str(first.dtype),
        output_finite=True,
        repeated_label_sha256=hashlib.sha256(first_labels).hexdigest(),
    )


def run_sealed_batch_smoke(
    paths: ESANetPaths, *, device: torch.device
) -> SealedBatchSmokeResult:
    if device.type != "cuda" or torch.cuda.device_count() != 1:
        raise ValueError("sealed ESANet smoke requires one visible CUDA device")
    audit_candidate_assets(paths)
    activate_candidate_imports(paths)
    if capture_environment_sha256() != _COMPLETE_ENVIRONMENT_SHA256:
        raise ValueError("complete ESANet runtime environment differs")
    verify_upstream_preprocessing()
    model = build_esanet(device=device)
    load_checkpoint_strict(model, paths.checkpoint)
    adapter = ESANetSegmenterAdapter(model)
    artifact = paths.repository_root / _SEALED_ARTIFACT_PATH
    data = artifact.read_bytes()
    if hashlib.sha256(data).hexdigest() != _SEALED_ARTIFACT_SHA256:
        raise ValueError("sealed ESANet smoke input differs")
    arrays = parse_raw_frame_npz_bytes(data).arrays
    value = SegmenterInput(
        rgb=torch.from_numpy(arrays.rgb.copy()).pin_memory(),
        depth_m=torch.from_numpy(arrays.depth_m.copy()).pin_memory(),
    )
    prepared = adapter.preprocess_host(value)
    batch = transfer_prepared_host_batch(
        prepared, device=device, non_blocking=True
    )
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        first = adapter.infer(batch)
        second = adapter.infer(batch)
    torch.cuda.synchronize(device)
    first_labels = first.argmax(dim=1).to("cpu").contiguous().numpy().tobytes()
    second_labels = second.argmax(dim=1).to("cpu").contiguous().numpy().tobytes()
    if first_labels != second_labels:
        raise ValueError("sealed ESANet repeated predictions differ")
    result = SealedBatchSmokeResult(
        sealed_input_sha256=_SEALED_ARTIFACT_SHA256,
        output_shape=tuple(first.shape),
        output_dtype=str(first.dtype),
        output_finite=True,
        repeated_label_sha256=hashlib.sha256(first_labels).hexdigest(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
    )
    _require_loaded_source_origins(paths)
    _require_source_authority(paths)
    return result


class ESANetArgs(Tap):
    device: str = "cpu"
    sealed_batch: bool = False


def parse_args(argv: Optional[Sequence[str]] = None) -> ESANetArgs:
    return ESANetArgs(underscores_to_dashes=True).parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.device != "cpu" and args.device != "cuda:0":
        raise ValueError("technical smoke device must be cpu or cuda:0")
    device = torch.device(args.device)
    if args.sealed_batch:
        sealed = run_sealed_batch_smoke(ESANetPaths.default(), device=device)
        record = {
            "output_dtype": sealed.output_dtype,
            "output_finite": sealed.output_finite,
            "output_shape": sealed.output_shape,
            "peak_allocated_bytes": sealed.peak_allocated_bytes,
            "repeated_label_sha256": sealed.repeated_label_sha256,
            "sealed_input_sha256": sealed.sealed_input_sha256,
            "semantic_endpoints_exposed": False,
            "smoke_kind": "sealed-twelve-view",
        }
    else:
        result = run_technical_smoke(ESANetPaths.default(), device=device)
        record = {
            "checkpoint_state_sha256": result.checkpoint_state_sha256,
            "model_parameters": result.model_parameters,
            "output_shape": result.output_shape,
            "output_dtype": result.output_dtype,
            "output_finite": result.output_finite,
            "repeated_label_sha256": result.repeated_label_sha256,
            "semantic_endpoints_exposed": False,
            "smoke_kind": "synthetic-one-view",
        }
    print(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
