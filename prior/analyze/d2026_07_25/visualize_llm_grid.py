"""Render fixed qualitative LLM-Grid checkpoint comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw
from tap import Tap

from prior.analyze.batch_vis import render_prediction_paths


EPOCH_CACHE_KEYS = {
    1: "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-1",
    2: "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-2",
    5: "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree-epoch-5",
    10: "llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree",
}

_HISTORICAL_TITLE = "Historical R2R-only (checkpoint unknown)"
_DEFAULT_COMMON_EXAMPLE_COUNT = 17


@dataclass(frozen=True)
class QualitativeExample:
    """One val_unseen episode available in historical and every epoch raster."""

    scene_id: str
    example_id: str
    historical_path: Path
    raster_paths: Tuple[Tuple[int, Path], ...]

    def raster_path_for(self, epoch: int) -> Path:
        for known_epoch, path in self.raster_paths:
            if known_epoch == epoch:
                return path
        raise KeyError(f"No raster path for epoch {epoch}")


class VisualizationArgs(Tap):
    navigation_root: Path = Path("data/llm_navigation")
    ground_truth_root: Path = Path(
        "data/cognitive_maps/gt.legacy.r1p5.direction5.blurred.v1"
    )
    historical_root: Path = Path(
        "docs/images/llm-grid-s2.legacy.r1p5.direction5.comparison"
    )
    output_root: Path = Path(
        "outputs/llm_grid_analysis/r2r_rxr_sanity/visualizations"
    )
    docs_image_root: Path = Path(
        "docs/images/llm_grid_r2r_rxr_sanity/comparisons"
    )


def select_common_examples(
    historical_root: Path,
    raster_roots: Mapping[int, Path],
    expected_count: Optional[int] = None,
) -> Tuple[QualitativeExample, ...]:
    """Select sorted historical val_unseen examples present in every raster root."""
    if not historical_root.is_dir():
        raise FileNotFoundError(f"Missing historical root: {historical_root}")
    if expected_count is not None and expected_count < 0:
        raise ValueError(f"expected_count must be non-negative, got {expected_count}")
    for epoch, raster_root in raster_roots.items():
        if not raster_root.is_dir():
            raise FileNotFoundError(f"Missing raster root for epoch {epoch}: {raster_root}")

    historical_paths: dict[tuple[str, str], Path] = {}
    for historical_path in sorted(historical_root.rglob("R2R_val_unseen_*.png")):
        identity = (historical_path.parent.name, historical_path.stem)
        if identity in historical_paths:
            raise ValueError(f"duplicate historical identity: {identity}")
        historical_paths[identity] = historical_path

    examples = []
    for (scene_id, example_id), historical_path in sorted(historical_paths.items()):
        raster_paths = tuple(
            (epoch, raster_root / scene_id / f"{example_id}.npz")
            for epoch, raster_root in sorted(raster_roots.items())
        )
        if all(path.is_file() for _, path in raster_paths):
            examples.append(
                QualitativeExample(
                    scene_id=scene_id,
                    example_id=example_id,
                    historical_path=historical_path,
                    raster_paths=raster_paths,
                )
            )
    selected = tuple(examples)
    if expected_count is not None and len(selected) != expected_count:
        raise ValueError(
            f"expected {expected_count} common examples, got {len(selected)}"
        )
    return selected


def write_comparison_sheet(
    columns: Sequence[tuple[str, Path]], output_path: Path
) -> None:
    """Write a titled two-column image sheet from the supplied PNG columns."""
    if not columns:
        raise ValueError("columns must not be empty")
    images = []
    for title, image_path in columns:
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing comparison column {title!r}: {image_path}")
        with Image.open(image_path) as source:
            images.append((title, source.convert("RGB")))

    column_width = max(image.width for _, image in images)
    image_height = max(image.height for _, image in images)
    title_height = 30
    grid_columns = min(2, len(images))
    grid_rows = (len(images) + grid_columns - 1) // grid_columns
    sheet_width = column_width * grid_columns
    sheet = Image.new(
        "RGB",
        (sheet_width, grid_rows * (title_height + image_height)),
        color="white",
    )
    drawer = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate(images):
        row, column = divmod(index, grid_columns)
        final_unpaired_image = index == len(images) - 1 and len(images) % 2
        tile_width = sheet_width if final_unpaired_image else column_width
        x_offset = 0 if final_unpaired_image else column * column_width
        y_offset = row * (title_height + image_height)
        title_bounds = drawer.textbbox((0, 0), title)
        title_width = title_bounds[2] - title_bounds[0]
        drawer.text(
            (x_offset + (tile_width - title_width) // 2, y_offset + 8),
            title,
            fill="black",
        )
        image_x = x_offset + (tile_width - image.width) // 2
        image_y = y_offset + title_height + (image_height - image.height) // 2
        sheet.paste(image, (image_x, image_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")


def _epoch_raster_roots(navigation_root: Path) -> dict[int, Path]:
    return {
        epoch: navigation_root
        / cache_key
        / "r2r"
        / "val_unseen"
        / "cognitive_maps"
        / "raster"
        for epoch, cache_key in EPOCH_CACHE_KEYS.items()
    }


def run_visualization(args: VisualizationArgs) -> dict[str, object]:
    """Render all fixed epoch comparisons and their five-column sheet manifest."""
    raster_roots = _epoch_raster_roots(args.navigation_root)
    examples = select_common_examples(
        args.historical_root,
        raster_roots,
        expected_count=_DEFAULT_COMMON_EXAMPLE_COUNT,
    )
    rendered_by_epoch = {
        epoch: render_prediction_paths(
            [example.raster_path_for(epoch) for example in examples],
            args.ground_truth_root,
            args.output_root / f"epoch-{epoch}",
        )
        for epoch in sorted(raster_roots)
    }

    records = []
    for example_index, example in enumerate(examples):
        rendered_paths = {
            epoch: rendered_by_epoch[epoch][example_index]
            for epoch in sorted(rendered_by_epoch)
        }
        sheet_path = (
            args.docs_image_root / example.scene_id / f"{example.example_id}.png"
        )
        write_comparison_sheet(
            [(_HISTORICAL_TITLE, example.historical_path)]
            + [
                (f"Epoch {epoch}", rendered_paths[epoch])
                for epoch in sorted(rendered_paths)
            ],
            sheet_path,
        )
        records.append(
            {
                "scene_id": example.scene_id,
                "example_id": example.example_id,
                "historical_path": str(example.historical_path),
                "raster_paths": {
                    str(epoch): str(example.raster_path_for(epoch))
                    for epoch in sorted(raster_roots)
                },
                "rendered_paths": {
                    str(epoch): str(rendered_paths[epoch])
                    for epoch in sorted(rendered_paths)
                },
                "comparison_sheet_path": str(sheet_path),
            }
        )

    manifest_path = args.output_root / "qualitative_examples.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"examples": records}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"examples": records, "manifest_path": manifest_path}


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = VisualizationArgs(underscores_to_dashes=True).parse_args(argv)
    result = run_visualization(args)
    print(
        f"Wrote qualitative comparison sheets to {args.docs_image_root}; "
        f"manifest: {result['manifest_path']}"
    )


if __name__ == "__main__":
    main()
