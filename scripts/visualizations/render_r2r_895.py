"""Actual MP3D navmesh layout and annotated R2R path 3627, CPU only."""

import gzip
import json
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
from PIL import Image
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class Args(Tap):
    output: str = str(ROOT / "reports/figs/r2r_episode895")
    meters_per_pixel: float = 0.02


def xy(points):
    points = np.asarray(points)
    return points[..., [0, 2]] * [1, -1]


def main():
    args = Args(underscores_to_dashes=True).parse_args()
    # PathFinder and SemanticScene load existing assets without a Simulator/GPU.
    import habitat_sim
    from habitat_sim.scene import SemanticScene
    from prior.constants import HABITAT_MP3D_ROTATION_VECTOR

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen"
    episode_file = dataset / "val_unseen_raw.json.gz"
    gt_file = dataset / "val_unseen_gt.json.gz"
    with gzip.open(episode_file, "rt") as stream:
        episode = next(e for e in json.load(stream)["episodes"] if str(e["episode_id"]) == "895")
    with gzip.open(gt_file, "rt") as stream:
        gt = json.load(stream)["895"]
    route = np.asarray(gt["locations"], dtype=np.float64)
    reference = np.asarray(episode["reference_path"], dtype=np.float64)
    goal = np.asarray(episode["goals"][0]["position"])
    scene_id = Path(episode["scene_id"]).stem
    scene_dir = ROOT / "data/scene_datasets/mp3d" / scene_id
    finder = habitat_sim.PathFinder()
    if not finder.load_nav_mesh(str(scene_dir / f"{scene_id}.navmesh")):
        raise RuntimeError("Cannot load the scene's actual navmesh")
    heights = np.linspace(route[:, 1].min(), route[:, 1].max(),
                          int(np.ceil(np.ptp(route[:, 1]) / 0.1)) + 1)
    mask = finder.get_topdown_view(args.meters_per_pixel, float(heights[0])).astype(bool)
    for height in heights[1:]:
        mask |= finder.get_topdown_view(args.meters_per_pixel, float(height)).astype(bool)
    lower, upper = map(np.asarray, finder.get_bounds())
    rows, cols = mask.shape
    gx, gz = float((upper[0] - lower[0]) / cols), float((upper[2] - lower[2]) / rows)
    image = np.full((*mask.shape, 3), [70, 76, 84], dtype=np.uint8)
    image[mask] = [244, 243, 236]
    Image.fromarray(image).save(output / "navmesh_layout.png")

    semantic_scene = SemanticScene()
    SemanticScene.load_mp3d_house(str(scene_dir / f"{scene_id}.house"),
                                 semantic_scene, HABITAT_MP3D_ROTATION_VECTOR)
    level = next(level for level in semantic_scene.levels if int(level.id) == 1)
    regions = []
    for region in level.regions:
        center, size = np.asarray(region.aabb.center), np.asarray(region.aabb.sizes)
        regions.append(dict(id=region.id, category=region.category.name(),
                            center_xyz=center.tolist(), min_xyz=(center-size/2).tolist(),
                            max_xyz=(center+size/2).tolist()))

    path_xy, ref_xy, goal_xy = xy(route), xy(reference), xy(goal)
    extent = [float(lower[0]), float(upper[0]), float(-upper[2]), float(-lower[2])]
    fig, axes = plt.subplots(1, 2, figsize=(17, 9), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax in axes:
        ax.imshow(image, origin="upper", extent=extent, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_xlabel("World x (m)")
        ax.set_ylabel("World -z (m)")
        ax.tick_params(labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Labels indicate semantic annotations, not invented wall polygons.
    short = {"familyroom/lounge": "Family / lounge", "living room": "Living room",
             "dining room": "Dining room", "other room": "Other room"}
    for region in regions:
        if region["id"] in ("1_15", "1_26"):
            continue
        px, py = xy(region["center_xyz"])
        name = short.get(region["category"], region["category"].capitalize())
        if region["id"] == "1_12":
            name = "Art room*\n(MP3D: bar)"
        axes[0].text(px, py, name, ha="center", va="center", fontsize=9,
                     bbox=dict(boxstyle="round,pad=.22", fc="white", ec="none", alpha=.85))
    for ax in axes:
        ax.plot(path_xy[:, 0], path_xy[:, 1], color="#d84036", lw=2.7,
                solid_capstyle="round", zorder=4, label="Dense GT trajectory")
        ax.scatter(*path_xy[0], s=90, c="#21a179", ec="white", lw=1.2, zorder=8, label="Start")
        ax.scatter(*path_xy[-1], s=55, marker="s", c="#d84036", ec="white", lw=.9,
                   zorder=8, label="Recorded GT endpoint")
        ax.scatter(*goal_xy, s=200, marker="*", c="#ffbf00", ec="#563d00", lw=.8,
                   zorder=9, label="Dataset goal")
        for index in range(6, len(route)-2, 11):
            a, b = path_xy[index-1], path_xy[index+1]
            if np.linalg.norm(b-a) > 0:
                ax.annotate("", xy=b, xytext=a, zorder=6,
                            arrowprops=dict(arrowstyle="-|>", color="#a52923", lw=1.6,
                                            mutation_scale=13))
        ax.annotate("Stairs", xy=(0.01, 3.63), xytext=(-.65, 5.55), fontsize=10,
                    ha="center", bbox=dict(boxstyle="round,pad=.2", fc="#e8eef7", ec="none"),
                    arrowprops=dict(arrowstyle="-", color="#53677f", lw=1))
        ax.text(4.1, 4.8, "Hallway", fontsize=10, ha="center",
                bbox=dict(boxstyle="round,pad=.2", fc="#e8eef7", ec="none"))
    axes[0].set_title("Actual scene layout — selected floor", fontsize=13, pad=12)
    ax = axes[1]
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "--o", color="#514b86", lw=1.2, ms=4,
            alpha=.85, zorder=3, label="R2R sparse reference (7 points)")
    for index, point in enumerate(ref_xy):
        label = ax.annotate(str(index), point, xytext=(5, -11), textcoords="offset points",
                            fontsize=8, color="#514b86", zorder=10)
        label.set_path_effects([patheffects.withStroke(linewidth=2, foreground="white")])
    for title, target, location in [
        ("Dining room", (-4.1, 5.9), (-4.1, 6.65)),
        ("Living room", (-2.05, .60), (-2.3, .7)),
        ("Art room*\n(MP3D: bar)", tuple(goal_xy), (7.0, .45)),
    ]:
        ax.annotate(title, xy=target, xytext=location, fontsize=11, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=.32", fc="white", ec="#b5bcc2", alpha=.95),
                    arrowprops=dict(arrowstyle="-", color="#747d86", lw=.8))
    all_xy = np.vstack([path_xy, goal_xy])
    ax.set_xlim(max(extent[0], all_xy[:, 0].min()-2), min(extent[1], all_xy[:, 0].max()+2.4))
    ax.set_ylim(max(extent[2], all_xy[:, 1].min()-2), min(extent[3], all_xy[:, 1].max()+2))
    ax.set_title("Ground-truth trajectory and route context", fontsize=13, pad=12)
    scale_x, scale_y = ax.get_xlim()[0]+.5, ax.get_ylim()[0]+.35
    ax.plot([scale_x, scale_x+2], [scale_y, scale_y], color="#26323f", lw=3)
    ax.text(scale_x+1, scale_y+.12, "2 m", ha="center", fontsize=9)
    instruction = episode["instruction"]["instruction_text"].strip()
    fig.suptitle(f"R2R val_unseen | path 3627 | episode 895 | scene {scene_id}", fontsize=17, y=.98)
    fig.text(.5, .935, textwrap.fill(instruction, 150), ha="center", va="top", fontsize=11, linespacing=1.45)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .075), ncol=3,
               frameon=False, fontsize=10)
    fig.text(.5, .035, "Native navmesh projection, 0.02 m/pixel; room labels from MP3D semantic annotations. "
             "*Art room is the instruction's name; MP3D labels this region 'bar'.", ha="center", fontsize=9)
    fig.text(.5, .012, "The solid path is the stored GT trajectory, not a model rollout or a newly computed shortest path. "
             "Axes use world x/-z, not geographic north.", ha="center", fontsize=9, color="#53606f")
    fig.subplots_adjust(top=.81, bottom=.17, left=.045, right=.985, wspace=.12)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"scene_layout_trajectory.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    metadata = dict(
        episode_id=895, r2r_path_id=3627, split="val_unseen", scene_id=scene_id,
        instruction=instruction, source_episode=str(episode_file), source_gt=str(gt_file),
        navmesh=str(scene_dir / f"{scene_id}.navmesh"), semantic_house=str(scene_dir / f"{scene_id}.house"),
        semantic_level_id=1, gt_locations_xyz=route.tolist(), r2r_reference_path_xyz=reference.tolist(),
        start_position=episode["start_position"], start_rotation_xyzw=episode["start_rotation"],
        goal_position_xyz=goal.tolist(), gt_endpoint_xyz=route[-1].tolist(),
        gt_polyline_length_m=float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
        dataset_geodesic_distance_m=episode["info"]["geodesic_distance"],
        endpoint_to_goal_euclidean_m=float(np.linalg.norm(route[-1]-goal)),
        original_actions=gt["actions"], semantic_regions=regions,
        topdown=dict(image="navmesh_layout.png", shape_hw=list(mask.shape),
                     bounds_lower_xyz=lower.tolist(), bounds_upper_xyz=upper.tolist(),
                     height_samples_y_m=heights.tolist(), requested_meters_per_pixel=args.meters_per_pixel,
                     grid_size_x_m=gx, grid_size_z_m=gz,
                     world_to_pixel="column=(x-lower_x)/grid_size_x; row=(z-lower_z)/grid_size_z",
                     plotted_coordinates="horizontal=x; vertical=-z"),
        notes=["Semantic AABBs overlap and are not exact room/wall polygons.",
               "Art room (instruction) corresponds to MP3D region 1_12, category bar.",
               "GT locations and dataset goal are different positions, both explicitly marked."],
    )
    (output / "trajectory.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    (output / "README.md").write_text(
        "# R2R path 3627 / VLN-CE episode 895\n\n"
        + instruction + "\n\n"
        "- Scene: `TbHJrupSAjP`, semantic level 1.\n"
        f"- Stored GT: {len(route)} positions, {metadata['gt_polyline_length_m']:.3f} m.\n"
        "- Sparse R2R reference: 7 points; it is not the dense executed GT path.\n"
        f"- Stored endpoint is {metadata['endpoint_to_goal_euclidean_m']:.3f} m from the dataset goal.\n"
        "- `scene_layout_trajectory.png/pdf/svg`: actual navmesh layout and GT overlay.\n"
        "- `navmesh_layout.png`: unannotated native traversability map.\n"
        "- `trajectory.json`: all positions, actions, metadata and coordinate transform.\n"
        "- The instruction's art room is MP3D's bar region; no semantic label is fabricated.\n"
        "- CPU only: PathFinder and SemanticScene; no Simulator, GPU, training or policy evaluation.\n"
    )
    print(f"Generated {output / 'scene_layout_trajectory.png'}")


if __name__ == "__main__":
    main()
