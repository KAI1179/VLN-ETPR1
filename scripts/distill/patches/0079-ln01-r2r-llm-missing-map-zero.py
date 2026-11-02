"""ln01 exp/refiner: evaluate R2R episodes without an LLM map with an all-zero
map input (MODEL.MAP_ENCODER.llm_missing_map zero) instead of scoring them as
failures, and let scripts/submit/eval-iters-view.sh select it (MISSING_MAP=zero).

Applied as anchored text edits (the ln01 tree differs from WZ, so a git patch
does not apply).  Run from the repo root; every anchor must match exactly once,
otherwise nothing is written.  Default llm_missing_map is "fail" = old behaviour.
Same semantics as the RxR branch (6ef10e6): raster, direction vectors, start
direction and start position all zero.
"""
import sys
from pathlib import Path

edits = {}


def edit(path, old, new):
    p = Path(path)
    s = edits.get(p, p.read_text())
    if s.count(old) != 1:
        sys.exit(f"{path}: anchor found {s.count(old)} times, nothing written:\n{old}")
    edits[p] = s.replace(old, new)


# 1. config key
edit(
    "vlnce_baselines/config/default.py",
    '_C.MODEL.MAP_ENCODER.map_ablation = "none"\n',
    '_C.MODEL.MAP_ENCODER.map_ablation = "none"\n'
    "# Evaluation of episodes without an LLM map (SS-ETP-LLM): \"fail\" scores them\n"
    "# as failures without running them; \"zero\" runs them with an all-zero map\n"
    "# input (raster, direction vectors, start direction and start position).\n"
    '_C.MODEL.MAP_ENCODER.llm_missing_map = "fail"\n',
)

# 2. LLM trainer
T = "vlnce_baselines/ss_trainer_ETP_LLM.py"
edit(
    T,
    "from habitat_baselines.common.baseline_registry import baseline_registry\n",
    "import torch\nfrom habitat_baselines.common.baseline_registry import baseline_registry\n",
)
edit(
    T,
    """        return [
            llm_cached_cognitive_map_to_tensors(
                ep.scene_id,
                self._cognitive_map_cache_id(ep),
                dataset,
                split,
                metadata_schema=candidate.metadata_schema,
                cache_dir=cache_dir,
                model_key=model_key,
                random_rotation_augmentation=random_rotation_augmentation,
            )
            for ep in self.envs.current_episodes()
        ]
""",
    """        zero_ids = getattr(self, "_llm_zero_map_episode_ids", frozenset())
        maps = [
            None
            if str(ep.episode_id) in zero_ids
            else llm_cached_cognitive_map_to_tensors(
                ep.scene_id,
                self._cognitive_map_cache_id(ep),
                dataset,
                split,
                metadata_schema=candidate.metadata_schema,
                cache_dir=cache_dir,
                model_key=model_key,
                random_rotation_augmentation=random_rotation_augmentation,
            )
            for ep in self.envs.current_episodes()
        ]
        # llm_missing_map zero: an all-zero map with the shapes of any real map
        # (the first one seen in this process).
        for m in maps:
            if m is not None and getattr(self, "_llm_zero_template", None) is None:
                self._llm_zero_template = {k: torch.zeros_like(v) for k, v in m.items()}
        if any(m is None for m in maps):
            template = getattr(self, "_llm_zero_template", None)
            if template is None:
                raise RuntimeError(
                    "llm_missing_map zero: no LLM map loaded yet to take the zero-map shape from"
                )
            maps = [
                {k: v.clone() for k, v in template.items()} if m is None else m
                for m in maps
            ]
        return maps
""",
)
edit(
    T,
    """        self._llm_eval_missing_cache_episode_ids = report.missing_episode_ids
        if report.missing_episode_ids:
            print(
                "eval_llm_navigation_maps: "
                f"available={report.available_count} "
                f"cache_missing={report.missing_count} "
                f"cache_missing_rate={report.missing_rate:.6f}"
            )
        return report.available_episode_ids
""",
    """        mode = getattr(map_cfg, "llm_missing_map", "fail")
        if mode not in ("fail", "zero"):
            raise ValueError(
                f"MODEL.MAP_ENCODER.llm_missing_map must be fail or zero; got {mode!r}"
            )
        if report.missing_episode_ids:
            print(
                "eval_llm_navigation_maps: "
                f"available={report.available_count} "
                f"cache_missing={report.missing_count} "
                f"cache_missing_rate={report.missing_rate:.6f} "
                f"missing_map={mode}"
            )
        if mode == "zero":
            # Missing-map episodes are run with an all-zero map input.
            self._llm_eval_missing_cache_episode_ids = []
            self._llm_zero_map_episode_ids = frozenset(
                str(i) for i in report.missing_episode_ids
            )
            return report.available_episode_ids + report.missing_episode_ids
        self._llm_eval_missing_cache_episode_ids = report.missing_episode_ids
        return report.available_episode_ids
""",
)

# 3. eval-iters-view.sh: MISSING_MAP knob, results in their own _zm dirs
S = "scripts/submit/eval-iters-view.sh"
edit(
    S,
    'read -ra VIEW_LIST <<< "${VIEWS:-metadata_only}"\n',
    'read -ra VIEW_LIST <<< "${VIEWS:-metadata_only}"\n'
    "# MISSING_MAP=zero: episodes without an LLM map run with an all-zero map\n"
    "# (default fail = scored as failures). Results go to *_abl_<VIEW>_zm.\n"
    'MISSING_MAP="${MISSING_MAP:-fail}"\n'
    'case "${MISSING_MAP}" in fail) ZM="" ;; zero) ZM="_zm" ;; *) echo "MISSING_MAP must be fail or zero" >&2; exit 2 ;; esac\n'
    "export ZM\n",
)
edit(
    S,
    'result_json() { echo "data/logs/checkpoints/${RUN_NAME}_eval_iter$1_${SPLIT}_abl_$2/eval_results/stats_ckpt_$1_${SPLIT}.json"; }\n',
    'result_json() { echo "data/logs/checkpoints/${RUN_NAME}_eval_iter$1_${SPLIT}_abl_$2${ZM}/eval_results/stats_ckpt_$1_${SPLIT}.json"; }\n',
)
edit(
    S,
    'EXTRA_ARGS="MODEL.MAP_ENCODER.map_ablation ${VIEW}" EVAL_SUFFIX="abl_${VIEW}"',
    'EXTRA_ARGS="MODEL.MAP_ENCODER.map_ablation ${VIEW} MODEL.MAP_ENCODER.llm_missing_map ${MISSING_MAP}" EVAL_SUFFIX="abl_${VIEW}${ZM}"',
)
edit(
    S,
    """TABLE="data/logs/checkpoints/${RUN_NAME}/eval_table_${SPLIT}_$(echo "${VIEW_LIST[*]}" | tr ' ' '+').md\"""",
    """TABLE="data/logs/checkpoints/${RUN_NAME}/eval_table_${SPLIT}_$(echo "${VIEW_LIST[*]}" | tr ' ' '+')${ZM}.md\"""",
)
edit(
    S,
    'run, split, table, views = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()\n',
    'run, split, table, views = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()\n'
    'zm = os.environ.get("ZM", "")\n',
)
edit(
    S,
    '    p = f"data/logs/checkpoints/{run}_eval_iter{it}_{split}_abl_{view}/eval_results/stats_ckpt_{it}_{split}.json"\n',
    '    p = f"data/logs/checkpoints/{run}_eval_iter{it}_{split}_abl_{view}{zm}/eval_results/stats_ckpt_{it}_{split}.json"\n',
)
edit(
    S,
    '    log = f"data/logs/checkpoints/{run}/eval_iter{it}_{split}_abl_{view}.log"\n',
    '    log = f"data/logs/checkpoints/{run}/eval_iter{it}_{split}_abl_{view}{zm}.log"\n',
)

for p, s in edits.items():
    p.write_text(s)
print("edited:", ", ".join(str(p) for p in edits))
