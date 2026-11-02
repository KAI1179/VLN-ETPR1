"""Read-only CPU evidence extraction for Q1 T0b; writes reports only."""

import json
from pathlib import Path
import subprocess

ROOT = Path("/home/xukai/code/ETP-R1-snapshot/ETP-R1")
OUT = ROOT / "reports/map_quality/t0b_teacher_distillation.md"
CONFIG = ROOT / "data/logs/checkpoints/kd_dual/config.yaml"
LAUNCH = ROOT / "data/logs/checkpoints/kd_dual/launch_info.txt"
TRAINER = ROOT / "vlnce_baselines/ss_trainer_ETP_PriorGT.py"
LOSSES = ROOT / "vlnce_baselines/models/etp_prior_gt/distill_losses.py"
REV = "0ff9345aa1bde958faef75a4f2c57f945c7f0e1e"


def excerpt(path, start, end):
    lines = path.read_text().splitlines()
    return f"{path}:{start}\n\n```text\n" + "\n".join(
        f"{number}: {lines[number - 1]}" for number in range(start, end + 1)
    ) + "\n```\n"


def main():
    sections = ["# Q1 T0b: teacher identity and distillation configuration", "",
                "CPU/read-only inspection. No source/default changes, Git writes or GPU work. T1–T4 remain paused.", "",
                "## 1. Identity and runtime settings", "",
                "| Item | Value | Evidence |", "|---|---|---|",
                f"| Teacher checkpoint | /data/xukai/etp-r1-snapshot/checkpoints/priorgt-nogaussian-18600-20260929/ckpt.iter18600.pth | {CONFIG}:202; {LAUNCH}:6 |",
                f"| Teacher GT namespace | gt.legacy.r1p5.direction5.v1 | {CONFIG}:206; {LAUNCH}:6 |",
                f"| Student source cache key | llm-grid-r2r-rxr-r1p5-direction5-s2-tagfree | {CONFIG}:266 |",
                f"| Student MODEL.elevation_axis | y | {CONFIG}:299 |",
                f"| Teacher gmap elevation | y (IL.gt_teacher_elevation_axis) | {CONFIG}:203; {LAUNCH}:6 (dy); {TRAINER}:2277–2286 |",
                f"| Teacher MODEL.elevation_axis | y, cloned from student config | {CONFIG}:299; {TRAINER}:565–573 |",
                f"| Teacher coordinate_fusion | false, inherited from student; not overridden during teacher construction | {CONFIG}:261; {TRAINER}:565–573 |", "",
                "The final 67.48/56.63 evaluation uses the derived __missing-all-zero cache; the source key above is the training key.", "",
                excerpt(LAUNCH, 1, 10), excerpt(CONFIG, 181, 207),
                excerpt(CONFIG, 258, 277), excerpt(CONFIG, 294, 303),
                excerpt(TRAINER, 565, 573), excerpt(TRAINER, 2275, 2286),
                "## 2. Independent teacher val_unseen results", "",
                "Both runs use the exact teacher checkpoint, original legacy GT maps, y elevation and coordinate_fusion=False; each reports 1839 episodes.", "",
                "| Backtracking | SR (%) | SPL (%) | OSR (%) | NE (m) | Stats |", "|---|---:|---:|---:|---:|---|"]
    for name, mode in (("gt18600_legacy_val_unseen_own", "teleport"),
                       ("gt18600_legacy_val_unseen_own_control", "control")):
        path = ROOT / f"data/logs/checkpoints/{name}/eval_results/stats_ckpt_18600_val_unseen.json"
        metrics = json.loads(path.read_text())
        sections.append(f"| {mode} | {metrics['success'] * 100:.10f} | {metrics['spl'] * 100:.10f} | {metrics['oracle_success'] * 100:.10f} | {metrics['distance_to_goal']:.10f} | {path} |")
        sections += ["", excerpt(path, 1, 13)]
    sections += [excerpt(ROOT / "scripts/eval_gt18600/eval_own.sh", 20, 29),
                 excerpt(ROOT / "scripts/eval_gt18600/eval_control.sh", 30, 41),
                 "Checkpoint identity evidence: " + str(ROOT / "data/logs/checkpoints/gt18600_legacy_val_unseen_own/eval.log") + ":37,222; " + str(ROOT / "data/logs/checkpoints/gt18600_legacy_val_unseen_own_control/eval.log") + ":8,62.", "",
                 "## 3. Complete saved IL configuration (verbatim)", "",
                 "Includes nested DAGGER/RECOLLECT fields, even where they are not consumed by this trainer. Relative strings below are original config literals, not paths used by this audit.", "",
                 excerpt(CONFIG, 156, 223), "",
                 "## 4. Dual branches and loss", "",
                 "The map and zero branches share student parameters. Map uses the episode's LLM raster and metadata. Zero sets only the raster to zero, keeps metadata, and retains all model modules. Both forwards carry gradients; navigation dropout RNG is replayed. dual_rollout_branch=zero selects zero-branch logits for rollout (subject to the existing expert-forcing schedule).", "",
                 "Both branches receive expert-action CE. The frozen GT teacher supervises both with KL(teacher || branch). In addition, the map branch's detached logits supervise the zero branch through KL(stopgrad(map) || zero). All three KL terms use temperature 1.0 and weight 1.0 here. Gate and effect matching are disabled; STOP factorization is disabled.", "",
                 "Core action loss = [CE(map, expert) + CE(zero, expert) + KL(GT || map) + KL(GT || zero) + KL(stopgrad(map) || zero)] / total_actions. ml_weight scales the CE sum, distill_weight scales the two GT KL terms, and dual_llm_weight scales self-KL; each is 1.0 in this run. action_kl includes temperature squared. Optional map auxiliary losses are added separately when map_aux_loss_total is not None; this is not an additional distillation weight.", "",
                 excerpt(TRAINER, 1523, 1537), excerpt(TRAINER, 2216, 2260),
                 excerpt(TRAINER, 2298, 2356), excerpt(TRAINER, 2688, 2722),
                 excerpt(LOSSES, 75, 89), "",
                 "## 5. Training-commit cross-check", "",
                 f"The launch records commit {REV}. The following snippets are read directly with git show, without checking out or modifying Git state. Their line numbers refer to that historical revision.", ""]
    historical = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{REV}:vlnce_baselines/ss_trainer_ETP_PriorGT.py"],
        text=True,
    ).splitlines()
    for start, end in ((560, 568), (1457, 1470), (2100, 2140), (2222, 2232), (2516, 2547)):
        sections += [f"{TRAINER}@{REV}:{start}", "", "```text",
                     "\n".join(f"{i}: {historical[i - 1]}" for i in range(start, end + 1)), "```", ""]
    OUT.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
