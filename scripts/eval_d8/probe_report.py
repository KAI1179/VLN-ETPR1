"""Markdown and plots for the prespecified D8 goal-proximity experiment."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ("base", "try5", "kd")


def _percent(value):
    return "NA" if value is None else f"{100 * value:.2f}"


def _number(value):
    return "NA" if value is None else f"{value:.4f}"


def _ci(row):
    return f"[{_percent(row['ci'][0])}, {_percent(row['ci'][1])}]"


def _status(row):
    return "final" if row["converged"] and row["step_converged"] else "provisional"


def _table(headers, rows):
    return ["|" + "|".join(headers) + "|", "|" + "|".join("---" for _ in headers) + "|"] + [
        "|" + "|".join(map(str, row)) + "|" for row in rows
    ]


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n")


def _main_table(results, source):
    rows = []
    for model in MODELS:
        row = results["probes"][f"{model}/{source}"]
        for kind in ("model", "step_only", "majority"):
            metric = row[kind]
            rows.append([model, kind, _percent(metric["accuracy"]),
                         _percent(metric["balanced_accuracy"]), _number(metric["spearman"]),
                         _status(row)])
    return _table(["Model", "Input", "Accuracy (%)", "Balanced accuracy (%)", "Spearman rho", "Status"], rows)


def _paired_table(results, source):
    rows = []
    for name, row in results["paired"].items():
        if name.endswith("/" + source):
            for metric in ("balanced_accuracy", "accuracy"):
                value = row[metric]
                rows.append([name, metric, _percent(value["difference"]), _ci(value),
                             "provisional" if row["provisional"] else "final"])
    return _table(["Comparison", "Metric", "Difference (pp)", "95% scene-bootstrap CI (pp)", "Status"], rows)


def write_source_report(results, output, source):
    """Write one source after its coordinator has completed the needed stages."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    role = "Shared states: primary" if source == "S" else "Own rollouts: secondary; visited-state difficulty differs across models"
    lines = [f"# D8 goal-proximity probe — {source}", "", role, "", *_main_table(results, source),
             "", "## Definitions", "", "```json", json.dumps(results["definitions"], indent=2), "```",
             "", "## Frame accounting", "", "```json",
             json.dumps({f"{m}/{source}": results["accounting"][f"{m}/{source}"] for m in MODELS}, indent=2),
             "```", "", "## Held-out scene metrics and convergence", ""]
    fold_rows = []
    for model in MODELS:
        row = results["probes"][f"{model}/{source}"]
        for kind, key in (("model", "folds"), ("step_only", "step_folds")):
            for fold in row[key]:
                fold_rows.append([model, kind, fold["scene"], fold["n"], fold["n_iter"],
                                  fold["converged"], _percent(fold["accuracy"]),
                                  _percent(fold["balanced_accuracy"]), _number(fold["spearman"])])
    lines += _table(["Model", "Input", "Held-out scene", "Frames", "n_iter", "Converged", "Accuracy (%)", "BA (%)", "Spearman rho"], fold_rows)
    lines += ["", "## Confusion matrices", "", "Rows are true bins; columns are predicted bins (0–4)."]
    for model in MODELS:
        row = results["probes"][f"{model}/{source}"]
        for kind in ("model", "step_only", "majority"):
            lines += ["", f"### {model}: pooled {kind}", ""]
            lines += _table(["True bin", "0", "1", "2", "3", "4"], [[i, *values] for i, values in enumerate(row[kind]["confusion"])])
        for kind, key in (("model", "folds"), ("step_only", "step_folds")):
            for fold in row[key]:
                lines += ["", f"### {model}: {kind}, held-out {fold['scene']}", ""]
                lines += _table(["True bin", "0", "1", "2", "3", "4"], [[i, *values] for i, values in enumerate(fold["confusion"])])
    _write(output / "progress_probe.md", lines)
    _write(output / "paired_differences.md", [f"# Paired differences — {source}", "", role, "",
           "2000 scene-cluster bootstrap draws, seed 0, fixed out-of-fold predictions; no refitting within bootstrap.",
           "", *_paired_table(results, source)])
    permutation_rows = []
    draw_rows = []
    for model in MODELS:
        row = results["probes"][f"{model}/{source}"]
        draws = row["permutations"]
        ba = np.asarray([draw["balanced_accuracy"] for draw in draws])
        acc = np.asarray([draw["accuracy"] for draw in draws])
        permutation_rows.append([model, f"{len(draws)}/20" + (" pending" if len(draws) < 20 else ""), _percent(row["model"]["balanced_accuracy"]),
                                 f"{_percent(ba.mean())} ± {_percent(ba.std())}" if len(draws) else "pending",
                                 f"{_percent(acc.mean())} ± {_percent(acc.std())}" if len(draws) else "pending",
                                 sum(draw["converged"] for draw in draws), _status(row)])
        for index, draw in enumerate(draws):
            folds = draw["folds"]
            draw_rows.append([model, index + 1, _percent(draw["accuracy"]),
                              _percent(draw["balanced_accuracy"]), draw["converged"],
                              f"{sum(fold['converged'] for fold in folds)}/{len(folds)}",
                              "; ".join(f"{fold['scene']}: {fold['n_iter']}" for fold in folds)])
    _write(output / "permutation_control.md", [f"# Within-scene label-permutation control — {source}", "",
           "BA is the primary metric. SD is the population standard deviation across draws; unconverged draws remain visible.", "",
           *_table(["Model", "Draws", "Real BA (%)", "Permuted BA mean ± SD (%)", "Permuted accuracy mean ± SD (%)", "Fully converged draws", "Real-label status"], permutation_rows),
           "", "## Per-draw convergence", "",
           *_table(["Model", "Draw", "Accuracy (%)", "BA (%)", "Converged", "Converged folds", "Scene: n_iter"], draw_rows)])
    curve_rows = []
    figure, axis = plt.subplots(figsize=(7, 4))
    for model in MODELS:
        rows = results["curves"][f"{model}/{source}"]
        for row in rows:
            curve_rows.append([model, row["bin"], row["n"], _percent(row["mean"]),
                               "NA" if row["ci"] is None else _ci(row)])
        plotted = [row for row in rows if row["bin"] < 4 and row["mean"] is not None]
        axis.plot([r["bin"] for r in plotted], [r["mean"] for r in plotted], marker="o", label=model)
        axis.fill_between([r["bin"] for r in plotted], [r["ci"][0] for r in plotted], [r["ci"][1] for r in plotted], alpha=.15)
    axis.set(xlabel="Geodesic goal-proximity bin", ylabel="Mean STOP probability", title=f"{source}: {role}", xticks=[0, 1, 2, 3])
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / f"stop_curve_{source}.png", dpi=180)
    plt.close(figure)
    _write(output / "stop_prob_by_progress.md", [f"# STOP probability — {source}", "", role, "",
           "Only frames with geodesic distance to goal >3 m; O additionally requires distance to reference polyline <=1 m.",
           "95% episode-bootstrap CIs, 2000 draws, seed 0. Bin 4 is tabulated but never plotted.", "",
           *_table(["Model", "Bin", "Frames", "Mean STOP (%)", "95% CI (%)"], curve_rows)])


def write_final_report(results, outputroot):
    """Apply the independent S comparisons only after all results exist."""
    outputroot = Path(outputroot)
    lines = ["# D8 goal-proximity representation results", "",
             "Baseline has no map; Try5 and KD use their normal with-map inference inputs.",
             "S shared decision states are primary. O own rollouts are secondary and do not isolate representation from state difficulty.",
             "", "## S raw results", "", *_main_table(results, "S"), "", *_paired_table(results, "S"),
             "", "## O raw results (secondary)", "", *_main_table(results, "O"),
             "", "## Detailed outputs", "",
             "[S probe](probe_s/progress_probe.md) · [S paired differences](probe_s/paired_differences.md) · [S permutations](probe_s/permutation_control.md) · [S STOP curves](probe_s/stop_prob_by_progress.md)",
             "", "[O probe](probe_o/progress_probe.md) · [O paired differences](probe_o/paired_differences.md) · [O permutations](probe_o/permutation_control.md) · [O STOP curves](probe_o/stop_prob_by_progress.md)",
             "", "## Independent prespecified readings", ""]
    claims = {"try5-base": "the Try5 representation decodes goal proximity more accurately than the baseline's",
              "kd-base": "the KD representation decodes goal proximity more accurately than the baseline's",
              "kd-try5": "KD adds a further improvement over the map alone"}
    for comparison, claim in claims.items():
        pair = results["paired"][comparison + "/S"]
        left, right = comparison.split("-")
        rows = [results["probes"][f"{m}/S"] for m in (left, right)]
        if pair["provisional"] or any(not row["converged"] or not row["step_converged"] for row in rows):
            reading = "Provisional: a feature or step-only probe did not fully converge; excluded from final interpretation."
        elif pair["balanced_accuracy"]["ci"][0] <= 0:
            reading = "No positive difference detected for this comparison; this says nothing about the other comparisons."
        elif results["paired"][f"{left}-step/S"]["balanced_accuracy"]["ci"][0] <= 0:
            reading = "Positive model difference, but the left model does not beat its step-only control with a positive CI lower bound; no absolute representation claim."
        else:
            reading = "Results support: " + claim + "."
        lines += [f"- **{comparison}**: {reading}"]
    lines += ["", "The readout concerns geodesic goal proximity, not instruction-stage completion. These experiments do not separate training-time map assistance from test-time map use, nor establish a map-specific causal mechanism.",
              "", "## Execution provenance", "", "```json", json.dumps(results["definitions"], indent=2), "```"]
    _write(outputroot / "README.md", lines)
