#!/usr/bin/env python3
"""Compose complete offline D7-case2 manifests into CPU-only paper drafts."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch
import numpy as np
from PIL import Image
from tap import Tap

REPO = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
COLORS = {"baseline": "#d62728", "D": "#1f77b4"}


class Args(Tap):
    root: str = str(REPO / "reports/figs/d7_case2")
    episodes: str = ""
    selection: str = str(REPO / "reports/figs/d7_case2/selection.json")


def asset(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def selected_candidate(frame: dict) -> dict | None:
    if frame["stop"]:
        return None
    return next(c for c in frame["candidates"] if c["id"] == frame["selected_id"])


def action_caption(frame: dict) -> str:
    if frame["stop"]:
        reason = " [forced]" if frame.get("forced_stop", False) else ""
        return f"STOP p={frame['stop_probability']:.3f}{reason}"
    candidate = selected_candidate(frame)
    return f"Select {candidate['id']}  p={candidate['probability']:.3f}"


def rgb_panel(axis, frame: dict, root: Path, color: str) -> None:
    image = np.asarray(Image.open(asset(root, frame["fpv_path"])).convert("RGB"))
    height, width = image.shape[:2]
    axis.imshow(image)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(color)
        spine.set_linewidth(2)
    axis.text(0.02, 0.96, f"Step {frame['step']}", transform=axis.transAxes,
              va="top", color="white", fontsize=11,
              bbox={"facecolor": "black", "alpha": 0.65, "pad": 3})
    axis.set_xlabel(action_caption(frame), color=color, fontsize=10)
    candidate = selected_candidate(frame)
    if candidate is None:
        return
    bearing = float(candidate["relative_bearing_rad"])
    if abs(bearing) < np.pi / 4:
        # Exact perspective projection for HFOV=90 degrees, bearing positive right.
        x = (width - 1) / 2 + width / 2 * np.tan(bearing)
        axis.annotate(str(candidate["id"]), xy=(x, height * 0.56),
                      xytext=(width / 2, height * 0.80), color="white",
                      ha="center", fontsize=11,
                      arrowprops={"arrowstyle": "->", "color": color, "lw": 3},
                      bbox={"facecolor": color, "alpha": 0.8, "pad": 3})
    else:
        panorama = np.asarray(Image.open(asset(root, frame["pano_path"])).convert("RGB"))
        inset = axis.inset_axes([0.03, 0.04, 0.94, 0.23])
        inset.imshow(panorama)
        inset.set_xticks([])
        inset.set_yticks([])
        # Twelve perspective views, tile order 0,30,...,330 degrees clockwise.
        # This is not an equirectangular panorama: project into the nearest tile.
        positive = np.rad2deg(bearing) % 360
        tile = int(np.floor(positive / 30 + 0.5)) % 12
        residual = (positive - tile * 30 + 180) % 360 - 180
        tile_width = panorama.shape[1] / 12
        x = tile * tile_width + (tile_width - 1) / 2 + tile_width / 2 * np.tan(np.deg2rad(residual))
        inset.axvline(x, color=color, lw=3)
        inset.set_title(f"Selected direction outside front view: {np.rad2deg(bearing):+.0f}°",
                        fontsize=7, color="white", backgroundcolor="black")


def compose(manifest_path: Path) -> tuple[Path, Path]:
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    episode = str(manifest["episode_id"])
    instruction = manifest["instruction"]
    fig = plt.figure(figsize=(18, 15))
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 2.2],
                            left=0.045, right=0.98, bottom=0.045, top=0.87,
                            hspace=0.34, wspace=0.05)
    summaries = []
    image_axes = {}
    for row, (key, label) in enumerate((("baseline", "Baseline"), ("D", "Ours"))):
        model = manifest["models"][key]
        if len(model["keyframes"]) != 4:
            raise ValueError(f"{episode}/{key}: expected exactly four keyframes")
        metrics = model["final"]
        summaries.append(f"{label}: {model['failure_class']}; PL={metrics['path_length']:.1f} m; "
                         f"steps={int(metrics['steps_taken'])}")
        for column, frame in enumerate(model["keyframes"]):
            axis = fig.add_subplot(grid[row, column])
            image_axes[key, column] = axis
            rgb_panel(axis, frame, root, COLORS[key])
            if column == 0:
                axis.set_ylabel(label, color=COLORS[key], fontsize=13)

    fig.suptitle(f"Episode {episode}\n" + "\n".join(textwrap.wrap(instruction, 150)),
                 y=0.988, fontsize=13, ha="center")
    fig.text(0.975, 0.905, "\n".join(summaries), ha="right", fontsize=11)
    axis = fig.add_subplot(grid[2, :])
    topdown = manifest["topdown"]
    image = np.asarray(Image.open(asset(root, topdown["image_path"])))
    lower, upper = np.asarray(topdown["bounds_lower"]), np.asarray(topdown["bounds_upper"])
    resolution = float(topdown["meters_per_pixel"])
    # Habitat row/column convention: row is increasing z, column increasing x.
    axis.imshow(image, origin="upper", extent=(lower[0], upper[0], upper[2], lower[2]))
    gt = np.asarray(manifest["gt_path"], dtype=float)[:, [0, 2]]
    goal = np.asarray(manifest["goal_position"], dtype=float)[[0, 2]]
    axis.plot(*gt.T, "--", color="0.3", lw=2.2, label="GT path")
    axis.scatter(*gt[0], c="green", marker="o", s=75, label="Start", zorder=6)
    axis.scatter(*goal, c="black", marker="*", s=170, label="Goal", zorder=6)
    axis.add_patch(Circle(goal, 3, fill=False, color="black", ls=":", lw=1.5))
    all_positions = [gt, goal[None]]
    for key, label in (("baseline", "Baseline"), ("D", "Ours")):
        model = manifest["models"][key]
        points = np.asarray(model["trajectory"], dtype=float)[:, [0, 2]]
        all_positions.append(points)
        axis.plot(*points.T, color=COLORS[key], lw=2.3, label=label, zorder=3)
        for column, frame in enumerate(model["keyframes"]):
            position = np.asarray(frame["position"], dtype=float)[[0, 2]]
            # Number distinguishes image slots even when repeated keyframes share a pose.
            axis.scatter(*position, c=COLORS[key], edgecolors="white", s=160, zorder=7)
            axis.text(*position, str(column + 1), color="white", ha="center", va="center", zorder=8)
            x, y, z, w = frame["rotation_xyzw"]
            forward = np.array([-2 * (x * z + w * y), -(1 - 2 * (x * x + y * y))])
            axis.arrow(*position, *(forward * 0.8), color=COLORS[key],
                       head_width=0.18, length_includes_head=True, zorder=6)
            fig.add_artist(ConnectionPatch(
                xyA=(0.5, 0), coordsA="axes fraction", axesA=image_axes[key, column],
                xyB=position, coordsB="data", axesB=axis,
                color=COLORS[key], lw=0.6, ls="--", alpha=0.35, zorder=1,
            ))
    points = np.concatenate(all_positions)
    minimum, maximum = points.min(axis=0) - 4, points.max(axis=0) + 4
    axis.set_xlim(max(lower[0], minimum[0]), min(upper[0], maximum[0]))
    axis.set_ylim(min(upper[2], maximum[1]), max(lower[2], minimum[1]))
    axis.set_aspect("equal")
    axis.set_xlabel("Habitat x (m)")
    axis.set_ylabel("Habitat z (m)")
    axis.legend(loc="best")
    axis.set_title(f"Traversability raster ({resolution:g} m/pixel); goal radius = 3 m. "
                   "Candidate probabilities are from each model's own graph state.", fontsize=10)
    outputs = root / f"{episode}.png", root / f"{episode}.pdf"
    for path in outputs:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    return outputs


def main() -> None:
    args = Args().parse_args()
    root = Path(args.root)
    if args.episodes:
        episodes = args.episodes.split(",")
    else:
        selection = json.loads(Path(args.selection).read_text())
        episodes = [str(episode) for episode in selection["confirmed_episode_ids"]]
    lines = ["# D7-case2 layout drafts", "", "Figures are assembled from saved RGB, poses and candidate probabilities.", ""]
    for episode in episodes:
        png, pdf = compose(root / episode / "manifest.json")
        manifest = json.loads((root / episode / "manifest.json").read_text())
        lines += [f"## Episode {episode}", "", manifest["instruction"], "",
                  f"[PDF]({pdf.relative_to(root)})", "", f"![{episode}]({png.relative_to(root)})", ""]
        print(png)
    (root / "index.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
