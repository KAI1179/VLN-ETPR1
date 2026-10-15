"""Render the prespecified D9 tables; no mechanistic inference from attention."""

import json


def number(value):
    return "NA" if value is None else f"{value:.6f}"


def estimate(item):
    ci = item.get("ci")
    return number(item["value"]), "NA" if ci is None else f"[{number(ci[0])}, {number(ci[1])}]"


def table(lines, header, rows):
    lines += ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    lines.append("")


def write_report(result, directory):
    stats = result["statistics"]
    lines = ["# D9 attention statistics", "", "Raw tables precede any reading. All attention is descriptive, not causal.",
             "Map attention is upstream of graph–text interaction; world overlays use **assumed registration**.", "",
             "Undefined statistics are NA. Selected-node word statistics use attention averaged across heads before word renormalisation; head spread is retained separately.", "",
             "## A1–A2: instruction attention", ""]
    rows = []
    for model, subsets in stats["text"].items():
        for subset, metrics in subsets.items():
            for metric, item in metrics.items():
                rows.append([model, subset, metric, item["n"], *estimate(item), number(metrics["special"]["value"])])
    table(lines, ["Model", "Special flag subset", "Metric", "Frames", "Mean", "Scene CI", "Special mass"], rows)
    rows = []
    for model, scenes in stats["A1"].items():
        for scene, values in scenes.items():
            rows.append([model, scene, values["n"], number(values["rho"]), number(values["slope"]), number(values["special"])])
    table(lines, ["Model", "Scene", "Unflagged frames", "Spearman rho", "Slope", "Special mass"], rows)
    rows = []
    for contrast, metrics in stats["paired_text"].items():
        for metric, item in metrics.items():
            rows.append([contrast, metric, item["n"], *estimate(item), *estimate(metrics["special"])])
    table(lines, ["Paired contrast", "Metric", "Common frames", "Difference", "Scene CI", "Special difference (all frames)", "Special CI"], rows)
    rows = []
    for contrast, values in stats["paired_A1"].items():
        for metric in ("rho", "slope"):
            item = values[metric]
            rows.append([contrast, metric, values["common_frames"], item["n"], *estimate(item), *estimate(values["special"])])
    table(lines, ["Paired contrast", "Trend", "Common frames", "Defined scenes", "Difference", "Scene CI", "Special difference", "Special CI"], rows)
    lines += ["## A3: upstream map attention", "", "Regions overlap and are reported independently. Other uses the raster-level union before pooling into patches. Candidate mass is NA for STOP; paired candidate contrasts require both models to choose non-STOP.", ""]
    rows = []
    for model, bins in stats["A3"].items():
        for label, metrics in bins.items():
            for metric, item in metrics.items():
                rows.append([model, label, metric, item["n"], *estimate(item), number(metrics["text_special"]["value"])])
    table(lines, ["Model", "Proximity bin", "Metric", "Defined frames", "Mean", "Scene CI", "Text special mass (all bin frames)"], rows)
    rows = []
    for label, metrics in stats["paired_A3"].items():
        for metric, item in metrics.items():
            rows.append([label, metric, item["n"], *estimate(item), *estimate(metrics["text_special"])])
    table(lines, ["KD − Try5: bin", "Metric", "Common defined frames", "Difference", "Scene CI", "Special difference (all bin frames)", "Special CI"], rows)
    lines += ["## A4: high versus low STOP tendency", "", "High stop probability >0.5 minus low <0.1, actual geodesic distance >3m. These are between-group contrasts, not paired-frame contrasts. Candidate-region comparison is omitted because high-tendency argmax is STOP. Other then uses agent ∪ route. Flagged STOP-row word statistics are kept separate.", ""]
    rows, counts = [], []
    for model, metrics in stats["A4"].items():
        for metric, item in metrics.items():
            rows.append([model, metric, item["high_n"], item["low_n"], *estimate(item), number(item.get("high_special")), number(item.get("low_special"))])
            for scene, values in item["per_scene"].items():
                counts.append([model, metric, scene, values["high"], values["low"]])
    table(lines, ["Model", "STOP-row metric", "High n", "Low n", "High − low", "Scene CI", "High special mass", "Low special mass"], rows)
    table(lines, ["Model", "STOP-row metric", "Scene", "High n", "Low n"], counts)
    lines += ["## Reading boundary", "", "A1–A3 are paired contrasts; A4 is a between-group scene-cluster contrast. The tables describe what the decision attends to. They do not establish causal use, reliance, or training-time versus test-time map contributions.", "",
              "## Provenance", "", "```json", json.dumps(result["definitions"], indent=2), "```", ""]
    (directory / "stats.md").write_text("\n".join(lines))
    readme = ["# D9 attention analysis", "", "See [stats.md](stats.md) for all prespecified A1–A4 tables and [stats.json](stats.json) for machine-readable values and scene counts.",
              "", "Raw per-frame scalar summaries: frames_base.npz, frames_try5.npz, frames_kd.npz. Original attention dumps remain in the provenance directory.",
              "", "Attention is descriptive, not causal. Map attention is upstream of graph–text interaction. No claim about training-time versus test-time assistance is made.",
              "", "## Case assets", ""]
    readme += [f"- `{path}`" for path in result["case_assets"]]
    readme += ["", "### Requested own-rollout frames not recorded", "", "No absent frame is fabricated or substituted.", "", "```json", json.dumps(result.get("missing_case_frames", []), indent=2), "```", ""]
    readme += ["", "## Provenance", "", "```json", json.dumps(result["definitions"], indent=2), "```", ""]
    (directory / "README.md").write_text("\n".join(readme))
