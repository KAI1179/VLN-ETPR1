"""Render fixed paired R2R-only and R2R+RxR-EN LLM-Grid examples."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont
from tap import Tap

from prior.analyze.batch_vis import render_prediction_paths


_EXPECTED_EXAMPLE_COUNT = 17
_EPOCHS = (1, 2, 5, 10)
_CACHE_KEYS = {
    "r2r_only": {
        epoch: f"llm-grid-r2r-legacy-r1p5-direction5-scale2-epoch-{epoch}"
        for epoch in _EPOCHS
    },
    "r2r_rxr_en": {
        1: "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-1",
        2: "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2",
        5: "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-5",
        10: "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree",
    },
}
_CORPUS_TITLES = {
    "r2r_only": "R2R only",
    "r2r_rxr_en": "R2R + RxR-EN",
}
_FONT_PATH = findfont(FontProperties(family="DejaVu Sans"))


class CorpusVisualizationArgs(Tap):
    """Fixed inputs and destinations for paired qualitative sheets."""

    source_manifest: Path = Path(
        "outputs/llm_grid_analysis/r2r_rxr_sanity/"
        "visualizations/qualitative_examples.json"
    )
    navigation_root: Path = Path("data/llm_navigation")
    ground_truth_root: Path = Path(
        "data/cognitive_maps/gt.legacy.r1p5.direction5.blurred.v1"
    )
    output_root: Path = Path(
        "outputs/llm_grid_analysis/r2r_only_vs_r2r_rxr_sanity/visualizations"
    )
    docs_image_root: Path = Path(
        "docs/images/llm_grid_r2r_only_vs_r2r_rxr_sanity/comparisons"
    )


@dataclass(frozen=True, order=True)
class ExampleIdentity:
    scene_id: str
    example_id: str


@dataclass(frozen=True)
class PanelSource:
    corpus: str
    epoch: int
    cache_key: str
    raster_path: Path
    status_path: Path

    @property
    def title(self) -> str:
        return f"{_CORPUS_TITLES[self.corpus]} — epoch {self.epoch}"

    @classmethod
    def for_example(
        cls,
        *,
        corpus: str,
        epoch: int,
        navigation_root: Path,
        example: ExampleIdentity,
    ) -> PanelSource:
        cache_key = _CACHE_KEYS[corpus][epoch]
        split_root = navigation_root / cache_key / "r2r" / "val_unseen"
        relative_path = Path(example.scene_id) / example.example_id
        return cls(
            corpus=corpus,
            epoch=epoch,
            cache_key=cache_key,
            raster_path=(
                split_root / "cognitive_maps" / "raster" / relative_path
            ).with_suffix(".npz"),
            status_path=(split_root / "status" / relative_path).with_suffix(".json"),
        )


@dataclass(frozen=True)
class PanelStatus:
    source: PanelSource
    strict_valid: bool
    status: str
    error: Optional[str]

    @classmethod
    def load(cls, source: PanelSource, example: ExampleIdentity) -> PanelStatus:
        if not source.status_path.is_file():
            raise FileNotFoundError(f"Missing cache status: {source.status_path}")
        raw = json.loads(source.status_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Cache status must be an object: {source.status_path}")
        if raw.get("scene_id") != example.scene_id:
            raise ValueError(f"Cache status scene mismatch: {source.status_path}")
        if raw.get("example_id") != example.example_id:
            raise ValueError(f"Cache status example mismatch: {source.status_path}")
        strict_valid = raw.get("strict_valid")
        status = raw.get("status")
        error = raw.get("error")
        if not isinstance(strict_valid, bool):
            raise ValueError(
                f"Cache status strict_valid must be boolean: {source.status_path}"
            )
        if not isinstance(status, str) or not status:
            raise ValueError(f"Cache status is missing status: {source.status_path}")
        if error is not None and not isinstance(error, str):
            raise ValueError(f"Cache status error must be text: {source.status_path}")
        if strict_valid and not source.raster_path.is_file():
            raise FileNotFoundError(
                f"Strict-valid cache entry is missing raster: {source.raster_path}"
            )
        if not strict_valid and source.raster_path.exists():
            raise ValueError(
                f"Schema-invalid cache entry unexpectedly has raster: "
                f"{source.raster_path}"
            )
        if not strict_valid and error is None:
            raise ValueError(
                f"Schema-invalid cache status is missing error: {source.status_path}"
            )
        return cls(
            source=source,
            strict_valid=strict_valid,
            status=status,
            error=error,
        )


def _load_examples(
    path: Path,
    *,
    expected_count: int,
) -> tuple[ExampleIdentity, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"examples"}:
        raise ValueError("qualitative source manifest must contain only 'examples'")
    entries = raw["examples"]
    if not isinstance(entries, list):
        raise ValueError("qualitative source manifest examples must be a list")

    examples: list[ExampleIdentity] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"qualitative example {index} must be an object")
        fields: dict[str, object] = {}
        for key, value in entry.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"qualitative example {index} field names must be strings"
                )
            fields[key] = value
        scene_id = fields.get("scene_id")
        example_id = fields.get("example_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"qualitative example {index} has invalid scene_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"qualitative example {index} has invalid example_id")
        examples.append(ExampleIdentity(scene_id=scene_id, example_id=example_id))

    if len(examples) != expected_count:
        raise ValueError(
            f"expected {expected_count} fixed examples, got {len(examples)}"
        )
    if len(set(examples)) != len(examples):
        raise ValueError("qualitative source manifest has duplicate examples")
    if tuple(sorted(examples)) != tuple(examples):
        raise ValueError("qualitative source manifest examples must be sorted")
    return tuple(examples)


def _panel_sources(
    example: ExampleIdentity,
    navigation_root: Path,
) -> tuple[PanelSource, ...]:
    return tuple(
        PanelSource.for_example(
            corpus=corpus,
            epoch=epoch,
            navigation_root=navigation_root,
            example=example,
        )
        for epoch in _EPOCHS
        for corpus in ("r2r_only", "r2r_rxr_en")
    )


def _write_invalid_placeholder(
    status: PanelStatus,
    output_path: Path,
) -> None:
    image = Image.new("RGB", (3200, 1800), color=(248, 248, 248))
    drawer = ImageDraw.Draw(image)
    font = ImageFont.truetype(_FONT_PATH, 100)
    lines = [
        "No raster: schema-invalid prediction",
        f"status: {status.status}",
        f"error: {status.error}",
    ]
    y_offset = 470
    for line in lines:
        for wrapped_line in textwrap.wrap(line, width=48) or [""]:
            drawer.text((160, y_offset), wrapped_line, fill="black", font=font)
            y_offset += 130
        y_offset += 60
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def _write_comparison_sheet(
    panels: Sequence[tuple[str, Path]],
    output_path: Path,
) -> None:
    if not panels:
        raise ValueError("panels must not be empty")
    images: list[tuple[str, Image.Image]] = []
    for title, path in panels:
        if not path.is_file():
            raise FileNotFoundError(f"Missing comparison panel {title!r}: {path}")
        with Image.open(path) as source:
            images.append((title, source.convert("RGB")))

    column_width = max(image.width for _, image in images)
    image_height = max(image.height for _, image in images)
    title_height = 180
    column_count = min(2, len(images))
    row_count = (len(images) + column_count - 1) // column_count
    sheet = Image.new(
        "RGB",
        (column_width * column_count, row_count * (title_height + image_height)),
        color="white",
    )
    drawer = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(_FONT_PATH, 120)
    for index, (title, panel) in enumerate(images):
        row, column = divmod(index, column_count)
        x_offset = column * column_width
        y_offset = row * (title_height + image_height)
        title_bounds = drawer.textbbox((0, 0), title, font=font)
        title_width = title_bounds[2] - title_bounds[0]
        drawer.text(
            (x_offset + (column_width - title_width) // 2, y_offset + 22),
            title,
            fill="black",
            font=font,
        )
        sheet.paste(
            panel,
            (
                x_offset + (column_width - panel.width) // 2,
                y_offset + title_height + (image_height - panel.height) // 2,
            ),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")


def _paths_overlap(first: Path, second: Path) -> bool:
    resolved_first = first.resolve()
    resolved_second = second.resolve()
    return (
        resolved_first == resolved_second
        or resolved_first in resolved_second.parents
        or resolved_second in resolved_first.parents
    )


def _refuse_input_overwrite(args: CorpusVisualizationArgs) -> None:
    inputs = (
        args.source_manifest,
        args.navigation_root,
        args.ground_truth_root,
    )
    for input_path in inputs:
        for output_root in (args.output_root, args.docs_image_root):
            if _paths_overlap(input_path, output_root):
                raise ValueError(
                    f"visualization input overlaps output: "
                    f"{input_path} and {output_root}"
                )


def _run_visualization(
    args: CorpusVisualizationArgs,
    *,
    expected_count: int,
) -> dict[str, object]:
    _refuse_input_overwrite(args)
    examples = _load_examples(args.source_manifest, expected_count=expected_count)
    panel_statuses = {
        example: tuple(
            PanelStatus.load(source, example)
            for source in _panel_sources(example, args.navigation_root)
        )
        for example in examples
    }

    rendered_paths: dict[tuple[ExampleIdentity, str, int], Path] = {}
    for corpus in ("r2r_only", "r2r_rxr_en"):
        for epoch in _EPOCHS:
            valid_items = [
                (example, status)
                for example, statuses in panel_statuses.items()
                for status in statuses
                if status.source.corpus == corpus
                and status.source.epoch == epoch
                and status.strict_valid
            ]
            rendered = render_prediction_paths(
                [status.source.raster_path for _, status in valid_items],
                args.ground_truth_root,
                args.output_root / "rendered" / corpus / f"epoch-{epoch}",
            )
            if len(rendered) != len(valid_items):
                raise ValueError(
                    f"renderer returned {len(rendered)} paths for "
                    f"{len(valid_items)} inputs"
                )
            for (example, _), rendered_path in zip(valid_items, rendered):
                rendered_paths[(example, corpus, epoch)] = rendered_path

    records = []
    for example in examples:
        panels = []
        columns: list[tuple[str, Path]] = []
        for status in panel_statuses[example]:
            key = (example, status.source.corpus, status.source.epoch)
            if status.strict_valid:
                rendered_path = rendered_paths[key]
            else:
                rendered_path = (
                    args.output_root
                    / "rendered"
                    / status.source.corpus
                    / f"epoch-{status.source.epoch}"
                    / example.scene_id
                    / f"{example.example_id}.png"
                )
                _write_invalid_placeholder(status, rendered_path)
            columns.append((status.source.title, rendered_path))
            panels.append({
                "cache_key": status.source.cache_key,
                "corpus": status.source.corpus,
                "epoch": status.source.epoch,
                "error": status.error,
                "raster_path": str(status.source.raster_path),
                "rendered_path": str(rendered_path),
                "status": status.status,
                "status_path": str(status.source.status_path),
                "strict_valid": status.strict_valid,
            })

        sheet_path = (
            args.docs_image_root / example.scene_id / f"{example.example_id}.png"
        )
        _write_comparison_sheet(columns, sheet_path)
        records.append({
            "scene_id": example.scene_id,
            "example_id": example.example_id,
            "panels": panels,
            "comparison_sheet_path": str(sheet_path),
        })

    manifest = {
        "example_count": len(records),
        "examples": records,
        "source_manifest": str(args.source_manifest),
    }
    manifest_path = args.output_root / "qualitative_comparison.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"examples": records, "manifest_path": manifest_path}


def run_visualization(args: CorpusVisualizationArgs) -> dict[str, object]:
    """Render every fixed example without dropping schema-invalid panels."""
    return _run_visualization(args, expected_count=_EXPECTED_EXAMPLE_COUNT)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = CorpusVisualizationArgs(underscores_to_dashes=True).parse_args(argv)
    result = run_visualization(args)
    print(
        f"Wrote paired LLM-Grid comparison sheets to {args.docs_image_root}; "
        f"manifest: {result['manifest_path']}"
    )


if __name__ == "__main__":
    main()
