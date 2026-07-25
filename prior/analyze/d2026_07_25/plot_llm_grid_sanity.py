"""Write the fixed quantitative LLM-Grid sanity-analysis artifacts."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Mapping, Optional, Sequence

from tap import Tap

from prior.analyze.llm_grid_eval_plots import (
    plot_category_counts,
    plot_category_f1,
    plot_iou_histograms,
    plot_iou_survival,
)
from prior.analyze.llm_grid_eval_records import (
    EpisodeRecord,
    category_count_histogram_rows,
    category_f1_rows,
    f1_summary,
    iou_histogram_rows,
    iou_survival_rows,
    load_episode_records,
    validate_epoch_populations,
    write_csv_rows,
)


_EXPECTED_COUNTS = {"val_seen": 778, "val_unseen": 1839}
_IOU_SURVIVAL_THRESHOLDS = tuple(index / 20 for index in range(21))
_R2R_ONLY_CACHE = (
    "data/llm_navigation/llm-grid-r2r-legacy-r1p5-direction5-scale2"
)
_R2R_ONLY_EVALUATOR_OUTPUTS = (
    "outputs/llm_grid_eval/r2r-only-checkpoint-sweep-r1p5/runs/*/episodes.csv"
)


class SanityPlotArgs(Tap):
    """Fixed inputs and destinations for today's LLM-Grid sanity report."""

    sweep_root: Path = Path("outputs/llm_grid_eval/checkpoint-sweep-r1p5")
    output_root: Path = Path("outputs/llm_grid_analysis/r2r_rxr_sanity")
    docs_image_root: Path = Path("docs/images/llm_grid_r2r_rxr_sanity")


def epoch_episode_paths(sweep_root: Path) -> dict[int, Path]:
    """Return the evaluator artifacts for the four selected checkpoints."""
    return {
        1: sweep_root / "runs/000/episodes.csv",
        2: sweep_root / "runs/001/episodes.csv",
        5: sweep_root / "runs/004/episodes.csv",
        10: sweep_root / "runs/009/episodes.csv",
    }


def _contains_path(root: Path, path: Path) -> bool:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _refuse_input_overwrite(
    input_paths: Mapping[int, Path], args: SanityPlotArgs
) -> None:
    for root in (args.output_root, args.docs_image_root):
        for input_path in input_paths.values():
            if _contains_path(root, input_path):
                raise ValueError(
                    f"output root {root} contains evaluator input {input_path}"
                )


def _invalid_counts(
    records_by_epoch: Mapping[int, Sequence[EpisodeRecord]],
) -> dict[str, dict[str, int]]:
    return {
        str(epoch): {
            split: sum(
                not record.schema_valid
                for record in records
                if record.split == split
            )
            for split in ("val_seen", "val_unseen")
        }
        for epoch, records in sorted(records_by_epoch.items())
    }


def _population_counts(
    records_by_epoch: Mapping[int, Sequence[EpisodeRecord]],
) -> dict[str, dict[str, int]]:
    return {
        str(epoch): dict(sorted(Counter(record.split for record in records).items()))
        for epoch, records in sorted(records_by_epoch.items())
    }


def _write_figures(
    records_by_epoch: Mapping[int, Sequence[EpisodeRecord]], output_root: Path
) -> dict[str, Path]:
    figure_paths = {
        "iou_histograms": output_root / "iou_histograms.png",
        "iou_survival": output_root / "iou_survival.png",
        "category_f1": output_root / "category_f1.png",
        "object_category_counts": output_root / "object_category_counts.png",
        "region_category_counts": output_root / "region_category_counts.png",
    }
    plot_iou_histograms(records_by_epoch, figure_paths["iou_histograms"])
    plot_iou_survival(
        records_by_epoch,
        figure_paths["iou_survival"],
        thresholds=_IOU_SURVIVAL_THRESHOLDS,
    )
    plot_category_f1(records_by_epoch, figure_paths["category_f1"])
    plot_category_counts(
        records_by_epoch,
        figure_paths["object_category_counts"],
        kind="object",
    )
    plot_category_counts(
        records_by_epoch,
        figure_paths["region_category_counts"],
        kind="region",
    )
    return figure_paths


def _copy_figures(figure_paths: Mapping[str, Path], docs_image_root: Path) -> dict[str, Path]:
    docs_image_root.mkdir(parents=True, exist_ok=True)
    copied_paths = {
        name: docs_image_root / path.name for name, path in figure_paths.items()
    }
    for name, source_path in figure_paths.items():
        shutil.copyfile(source_path, copied_paths[name])
    return copied_paths


def _run_analysis(
    args: SanityPlotArgs, expected_counts: Mapping[str, int]
) -> dict[str, object]:
    """Generate report artifacts with an injectable population for tests only."""
    input_paths = epoch_episode_paths(args.sweep_root)
    _refuse_input_overwrite(input_paths, args)
    records_by_epoch = {
        epoch: load_episode_records(epoch, path)
        for epoch, path in input_paths.items()
    }
    validate_epoch_populations(records_by_epoch, expected_counts)

    args.output_root.mkdir(parents=True, exist_ok=True)
    histogram_rows = iou_histogram_rows(records_by_epoch)
    survival_rows = iou_survival_rows(
        records_by_epoch, thresholds=_IOU_SURVIVAL_THRESHOLDS
    )
    category_rows = category_f1_rows(records_by_epoch)
    count_rows = category_count_histogram_rows(records_by_epoch)
    write_csv_rows(args.output_root / "iou_histograms.csv", histogram_rows)
    write_csv_rows(args.output_root / "iou_survival.csv", survival_rows)
    write_csv_rows(args.output_root / "category_f1.csv", category_rows)
    write_csv_rows(args.output_root / "category_count_histograms.csv", count_rows)

    figure_paths = _write_figures(records_by_epoch, args.output_root)
    copied_figure_paths = _copy_figures(figure_paths, args.docs_image_root)
    summary = {
        "episode_macro_category_f1": f1_summary(category_rows),
        "expected_counts": dict(expected_counts),
        "figure_paths": {
            name: {
                "analysis": str(figure_paths[name]),
                "docs": str(copied_figure_paths[name]),
            }
            for name in sorted(figure_paths)
        },
        "input_paths": {str(epoch): str(path) for epoch, path in input_paths.items()},
        "invalid_counts": _invalid_counts(records_by_epoch),
        "population_counts": _population_counts(records_by_epoch),
        "r2r_only_comparison": {
            "missing_cache": _R2R_ONLY_CACHE,
            "missing_evaluator_outputs": _R2R_ONLY_EVALUATOR_OUTPUTS,
            "status": "postponed",
        },
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"summary": summary, "summary_path": summary_path}


def run_analysis(args: SanityPlotArgs) -> dict[str, object]:
    """Generate the report using the required full R2R evaluation population."""
    return _run_analysis(args, expected_counts=_EXPECTED_COUNTS)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = SanityPlotArgs(underscores_to_dashes=True).parse_args(argv)
    result = run_analysis(args)
    print(f"Wrote LLM-Grid sanity analysis to {result['summary_path']}")


if __name__ == "__main__":
    main()
