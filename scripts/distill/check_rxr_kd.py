#!/usr/bin/env python3
"""Read-only readiness check for the RxR-CE KD student (CPU only, no simulator).

Called by scripts/distill/check_rxr_kd.sh, which hands over, per setting, the
exact torchrun command the job will run and the environment it was built from.
Every check goes through the code the trainer itself uses (config merge, cache
id, episode allow-list, map loaders, policy construction, teacher identity
check and state_dict load), so a PASS here means the job gets past setup with
the same inputs.  Not covered: anything that needs the simulator or a GPU.
"""
import argparse
import gzip
import inspect
import json
import os
import random
import shutil
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import habitat_extensions  # noqa: E402,F401
import vlnce_baselines  # noqa: E402,F401
from gym import spaces  # noqa: E402
from habitat_baselines.common.baseline_registry import baseline_registry  # noqa: E402
from habitat_baselines.common.obs_transformers import (  # noqa: E402
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from prior.vlnce import VLNCEEpisodeEntry  # noqa: E402
from vlnce_baselines.config.default import get_config  # noqa: E402
from vlnce_baselines.models.cognitive_map_candidate import CognitiveMapCandidate  # noqa: E402
from vlnce_baselines.models.etp_llm import navigation  # noqa: E402
from vlnce_baselines.models.etp_prior_gt.map_utils import (  # noqa: E402
    cached_cognitive_map_to_tensors,
    cognitive_map_cache_path,
)
from vlnce_baselines.models.etp_prior_gt.nan_guard import assert_finite_parameters  # noqa: E402
from vlnce_baselines.ss_trainer_ETP_LLM import RLTrainer as LLMTrainer  # noqa: E402
from vlnce_baselines.ss_trainer_ETP_PriorGT import _check_teacher_identity  # noqa: E402

BRANCH = "exp/rxr-kd-gt15000"
CKPT_BYTES = {}
MAP_KEYS = ("grid", "map_trajectory_metadata", "start_direction_vector", "start_position")
EXPECTED_LANGUAGES = {"en": ["en-US", "en-IN"], "all": ["*"]}


class Report:
    def __init__(self):
        self.counts = Counter()

    def _emit(self, tag, msg):
        self.counts[tag] += 1
        print(f"[{tag}] {msg}", flush=True)

    def ok(self, msg):
        self._emit("PASS", msg)

    def warn(self, msg):
        self._emit("WARN", msg)

    def fail(self, msg):
        self._emit("FAIL", msg)

    def info(self, msg):
        print(f"       {msg}", flush=True)

    def check(self, cond, msg, warn_only=False):
        (self.ok if cond else (self.warn if warn_only else self.fail))(msg)
        return cond

    @staticmethod
    def section(title):
        print(f"\n=== {title} ===", flush=True)


R = Report()


def guarded(title):
    def wrap(fn):
        def run(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as exc:  # report and keep checking the rest
                R.fail(f"{title}: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=4)
                return None
        return run
    return wrap


def _pmap_iter(fn, items, label, workers=int(os.environ.get("CHECK_WORKERS", "3"))):
    """Thread-pool map in input order with progress, yielded as results arrive
    (network filesystem: latency-bound; nothing is retained)."""
    with ThreadPoolExecutor(workers) as pool:
        for n, r in enumerate(pool.map(fn, items), 1):
            if n % 5000 == 0 or n == len(items):
                print(f"       {label}: {n}/{len(items)}", flush=True)
            yield r


def _pmap(fn, items, label, workers=int(os.environ.get("CHECK_WORKERS", "3"))):
    return list(_pmap_iter(fn, items, label, workers))


# ----------------------------------------------------------------- code
@guarded("code")
def check_code():
    R.section("code / branch")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    head, branch = git("rev-parse", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
    R.check(branch == BRANCH, f"branch {branch} @ {head[:7]}")
    remote = git("rev-parse", f"origin/{BRANCH}")
    R.check(remote == head, f"HEAD matches origin/{BRANCH} ({remote[:7]})", warn_only=True)
    dirty = git("status", "--porcelain", "--untracked-files=no")
    R.check(not dirty, "no modified tracked files" + (f":\n{dirty}" if dirty else ""), warn_only=True)

    R.check("languages" in inspect.signature(VLNCEEpisodeEntry.iter_from).parameters,
            "iter_from has the languages filter")
    R.check("languages" in inspect.signature(navigation.llm_navigation_cache_report).parameters,
            "llm_navigation_cache_report passes languages")
    R.check("languages=" in inspect.getsource(LLMTrainer._finetuning_episodes_allowed)
            and "languages=" in inspect.getsource(LLMTrainer._prepare_eval_episodes_allowed),
            "LLM trainer passes DATASET.LANGUAGES to the allow-list and eval prefill")
    for name in ("SS-ETP-LLM",):
        R.check(baseline_registry.get_trainer(name) is not None, f"trainer {name} registered")
    for name in ("LLMGridTry5Policy", "PriorGTTry5Policy"):
        R.check(baseline_registry.get_policy(name) is not None, f"policy {name} registered")

    for script in ("scripts/distill/run_rxr_kd.sh", "scripts/distill/rxr_kd_env.sh",
                   "scripts/distill/run_dagger_distill_legacy.sh", "scripts/submit/rxr-kd.sh"):
        rc = subprocess.run(["bash", "-n", script]).returncode
        R.check(rc == 0, f"bash -n {script}")

    tests = ["tests/test_vlnce.py", "tests/etp_llm/test_rxr_distill_paths.py"]
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
                          capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or [""]
    R.check(proc.returncode == 0, f"unit tests ({' '.join(tests)}): {tail[0]}")
    if proc.returncode:
        print(proc.stdout[-3000:], proc.stderr[-2000:])


# --------------------------------------------------------------- config
def load_setting(name, cmd_file, env_file):
    cmd = Path(cmd_file).read_text().split("\n")[:-1]
    env = dict(line.split("=", 1) for line in Path(env_file).read_text().splitlines() if "=" in line)
    i = cmd.index("--exp-config")
    exp_name = cmd[cmd.index("--exp_name") + 1]
    opts = [a for a in cmd[i + 2:] if a != "--dry-run"]
    return SimpleNamespace(name=name, cmd=cmd, env=env, exp_name=exp_name,
                           exp_config=cmd[i + 1], opts=opts)


@guarded("config")
def check_config(s):
    R.section(f"[{s.name}] config")
    cfg = get_config(s.exp_config, s.opts)
    R.ok(f"config merges: {s.exp_config} + {len(s.opts) // 2} overrides (run {s.exp_name})")
    E = s.env
    me = cfg.MODEL.MAP_ENCODER
    expected = [
        ("TRAINER_NAME", cfg.TRAINER_NAME, "SS-ETP-LLM"),
        ("MODEL.task_type", cfg.MODEL.task_type, "rxr"),
        ("MODEL.policy_name", cfg.MODEL.policy_name, "LLMGridTry5Policy"),
        ("MODEL.pretrained_path", cfg.MODEL.pretrained_path, E["PRETRAINED_CKPT"]),
        ("MODEL.elevation_axis", cfg.MODEL.elevation_axis, "y"),
        ("MAP_ENCODER.enabled", me.enabled, True),
        ("MAP_ENCODER.architecture", me.architecture, "try5"),
        ("MAP_ENCODER.source", me.source, "llm_grid"),
        ("MAP_ENCODER.coordinate_fusion", me.coordinate_fusion, False),
        ("MAP_ENCODER.refiner_ckpt", me.refiner_ckpt, ""),
        ("MAP_ENCODER.load_pretrained_map_modules", me.load_pretrained_map_modules, False),
        ("MAP_ENCODER.map_ablation", getattr(me, "map_ablation", "none"), "none"),
        ("MAP_ENCODER.llm_cache_model_key", me.llm_cache_model_key, E["LLM_CACHE_KEY"]),
        ("MAP_ENCODER.llm_cache_dir", me.llm_cache_dir, E["LLM_CACHE_DIR"]),
        ("IL.gt_teacher_enabled", cfg.IL.gt_teacher_enabled, True),
        ("IL.gt_teacher_ckpt", cfg.IL.gt_teacher_ckpt, E["GT_TEACHER_CKPT"]),
        ("IL.gt_teacher_map_namespace", cfg.IL.gt_teacher_map_namespace, E["GT_TEACHER_NAMESPACE"]),
        ("IL.gt_teacher_policy_name", cfg.IL.gt_teacher_policy_name, "PriorGTTry5Policy"),
        ("IL.gt_teacher_elevation_axis", cfg.IL.gt_teacher_elevation_axis, "y"),
        ("IL.gt_teacher_independent_map", cfg.IL.gt_teacher_independent_map, False),
        ("IL.dual_enabled (no SKD)", cfg.IL.dual_enabled, False),
        ("IL.gate_enabled", cfg.IL.gate_enabled, False),
        ("IL.effect_match_weight", cfg.IL.effect_match_weight, 0.0),
        ("IL.distill_stop_factorized", cfg.IL.distill_stop_factorized, False),
        ("IL.distill_weight", cfg.IL.distill_weight, 1.0),
        ("IL.distill_temperature", cfg.IL.distill_temperature, 1.0),
        ("IL.expert_policy", cfg.IL.expert_policy, "ndtw"),
        ("IL.sample_ratio", cfg.IL.sample_ratio, 0.75),
        ("IL.max_text_len", cfg.IL.max_text_len, 250),
        ("IL.max_traj_len", cfg.IL.max_traj_len, 25),
        ("IL.iters", cfg.IL.iters, int(E["ITERS"])),
        ("CHECKPOINT_INTERVAL", cfg.CHECKPOINT_INTERVAL, int(E["CKPT_INTERVAL"])),
        ("IL.ckpt_optim_latest_only", cfg.IL.ckpt_optim_latest_only, E["CKPT_OPTIM_LATEST_ONLY"] == "True"),
        ("NUM_ENVIRONMENTS", cfg.NUM_ENVIRONMENTS, int(E["NUM_ENVS"])),
        ("GPU_NUMBERS", cfg.GPU_NUMBERS, 4),
        ("DATASET.SPLIT", cfg.TASK_CONFIG.DATASET.SPLIT, "train"),
        ("DATASET.SUFFIX", cfg.TASK_CONFIG.DATASET.SUFFIX, E["DATASET_SUFFIX"]),
        ("DATASET.ROLES", list(cfg.TASK_CONFIG.DATASET.ROLES), ["guide"]),
        ("DATASET.LANGUAGES", list(cfg.TASK_CONFIG.DATASET.LANGUAGES), EXPECTED_LANGUAGES[s.name]),
        ("EVAL.LANGUAGES", list(cfg.EVAL.LANGUAGES), EXPECTED_LANGUAGES[s.name]),
        ("RGB/DEPTH HFOV", (cfg.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV,
                            cfg.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV), (63, 63)),
    ]
    for key, have, want in expected:
        R.check(have == want, f"{key} = {have!r}" + ("" if have == want else f" (expected {want!r})"))
    R.check(cfg.IL.load_from_ckpt is False,
            f"fresh start (IL.load_from_ckpt={cfg.IL.load_from_ckpt}); True means it resumes {s.exp_name}",
            warn_only=True)
    R.info(f"lr {cfg.IL.lr}, seed {cfg.TASK_CONFIG.SEED}, decay_interval {cfg.IL.decay_interval}, "
           f"warmup {cfg.IL.warmup_iters}, waypoint_aug {cfg.IL.waypoint_aug}, port {E['MASTER_PORT']}")
    s.cfg = cfg
    return cfg


@guarded("files")
def check_files(s):
    R.section(f"[{s.name}] weights / model files")
    cfg = s.cfg
    must = {
        "student pretrain (MODEL.pretrained_path)": cfg.MODEL.pretrained_path,
        "GT teacher (IL.gt_teacher_ckpt)": cfg.IL.gt_teacher_ckpt,
        "depth encoder (DEPTH_ENCODER.ddppo_checkpoint)": cfg.MODEL.DEPTH_ENCODER.ddppo_checkpoint,
        "waypoint predictor hfov63": "data/wp_pred/check_cwp_bestdist_hfov63",
        "xlm-roberta config": "bert_config/xlm-roberta-base",
        "teacher namespace (repo link)": f"data/cognitive_maps/{cfg.IL.gt_teacher_map_namespace}/raster",
    }
    for label, path in must.items():
        p = Path(path)
        R.check(p.exists(), f"{label}: {path}" + (f" ({p.stat().st_size / 2**30:.2f} GiB)" if p.is_file() else ""))


# ----------------------------------------------------------------- data
def _read_episodes(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)["episodes"]


@guarded("dataset")
def check_dataset(s):
    R.section(f"[{s.name}] dataset / split")
    cfg = s.cfg
    ds = cfg.TASK_CONFIG.DATASET
    langs = list(ds.LANGUAGES)
    keep = lambda e: "*" in langs or e["instruction"]["language"] in langs
    path = Path(ds.DATA_PATH.format(split="train", role="guide", suffix=ds.SUFFIX))
    if not R.check(path.is_file(), f"train file {path}"):
        return None
    episodes = [e for e in _read_episodes(path) if keep(e)]
    by_lang = Counter(e["instruction"]["language"] for e in episodes)
    R.check(len(episodes) > 0, f"train episodes {len(episodes)} by language {dict(by_lang)}")
    if s.name == "all":
        R.check({"hi-IN", "te-IN"} <= set(by_lang), "all-language setting contains hi-IN and te-IN")
    else:
        R.check(set(by_lang) <= {"en-US", "en-IN"}, "English setting has English only")
    gt_path = Path(cfg.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split="train", role="guide"))
    if R.check(gt_path.is_file(), f"ndtw expert reference {gt_path}"):
        with gzip.open(gt_path, "rt") as stream:
            gt = json.load(stream)
        lacking = [e["episode_id"] for e in episodes if str(e["episode_id"]) not in gt]
        R.check(not lacking, f"every train episode has an expert reference path (missing {len(lacking)})")
    scenes = sorted({e["scene_id"] for e in episodes})
    no_scene = [sc for sc in scenes if not (Path(ds.SCENES_DIR) / sc).is_file()]
    R.check(not no_scene, f"{len(scenes)} scenes present under {ds.SCENES_DIR}"
            + (f"; missing {no_scene[:5]}" if no_scene else ""))
    for split in ("val_seen", "val_unseen"):
        vp = Path(ds.DATA_PATH.format(split=split, role="guide", suffix=""))
        vg = Path(cfg.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split, role="guide"))
        if R.check(vp.is_file() and vg.is_file(), f"{split} files present (needed for eval)", warn_only=True):
            R.info(f"{split}: {dict(Counter(e['instruction']['language'] for e in _read_episodes(vp) if keep(e)))}")
    s.episodes = {str(e["episode_id"]): e for e in episodes}
    return s.episodes


# ----------------------------------------------------------------- maps
@guarded("maps")
def check_maps(s, max_load):
    R.section(f"[{s.name}] student LLM maps / teacher GT maps")
    cfg, eps, E = s.cfg, s.episodes, s.env
    trainer = SimpleNamespace(config=cfg)
    R.info(f"scanning LLM status files for {len(eps)} episodes (trainer allow-list, sequential) ...")
    allowed = LLMTrainer._finetuning_episodes_allowed(trainer)  # the trainer's own allow-list
    final = [e for e in map(str, allowed) if e in eps]
    lang = lambda e: eps[e]["instruction"]["language"]
    have, total = Counter(lang(e) for e in final), Counter(lang(e) for e in eps)
    R.check(len(final) > 0, f"trained episodes (allow-list ∩ split) {len(final)} of {len(eps)}")
    for lg in sorted(total):
        R.check(have[lg] == total[lg], f"  {lg}: LLM map for {have[lg]}/{total[lg]}", warn_only=have[lg] > 0)

    cache_id = lambda e: LLMTrainer._cognitive_map_cache_id(trainer, SimpleNamespace(episode_id=e))
    ns = cfg.IL.gt_teacher_map_namespace
    R.info(f"checking teacher GT files for {len(final)} episodes ...")
    gt_exists = _pmap(lambda e: cognitive_map_cache_path(eps[e]["scene_id"], cache_id(e), namespace=ns).is_file(),
                      final, "teacher GT exists")
    no_gt = [e for e, ok in zip(final, gt_exists) if not ok]
    R.check(not no_gt, f"teacher GT map exists for every trained episode (missing {len(no_gt)}"
            + (f", first {no_gt[:10]})" if no_gt else ")"))

    me = cfg.MODEL.MAP_ENCODER
    cache_dir = me.llm_cache_dir or None
    def link_src(e):
        src = navigation.llm_navigation_cognitive_map_raster_path(
            eps[e]["scene_id"], cache_id(e), "rxr", "train", cache_dir=cache_dir, model_key=me.llm_cache_model_key)
        target = os.readlink(src) if src.is_symlink() else ""
        primary = E["LLM_EN_KEY"] if lang(e).startswith("en-") else E["LLM_TRANSLATED_KEY"]
        return "primary" if f"/{primary}/" in target else "repair" if f"/{E['LLM_REPAIR_KEY']}/" in target else "wrong"
    srcs = _pmap(link_src, final, "merged links")
    wrong = [e for e, k in zip(final, srcs) if k == "wrong"]
    R.check(not wrong, f"merged key links: en-* -> {E['LLM_EN_KEY']}, others -> {E['LLM_TRANSLATED_KEY']}, "
            f"fallback {E['LLM_REPAIR_KEY']}" + (f" (wrong {len(wrong)}, first {wrong[:5]})" if wrong else ""))
    R.info(f"map source: {dict(Counter(srcs))}")

    s_schema = CognitiveMapCandidate.parse(me.architecture, me.source).metadata_schema
    t_schema = CognitiveMapCandidate.parse(me.architecture, "prior_gt").metadata_schema
    no_gt_set = set(no_gt)
    todo = [e for e in final if e not in no_gt_set]
    if max_load and len(todo) > max_load:
        random.Random(0).shuffle(todo)
        todo = todo[:max_load]
    shapes, bad, stats = {}, [], defaultdict(list)
    samples = {}
    def load(e):
        """Load one pair and reduce it to a summary; the tensors are freed here
        (keeping 17k pairs of 37x100x100 grids exceeded the login node's RAM)."""
        sc = eps[e]["scene_id"]
        try:
            pair = {
                "student": navigation.llm_cached_cognitive_map_to_tensors(
                    sc, cache_id(e), "rxr", "train", metadata_schema=s_schema,
                    cache_dir=cache_dir, model_key=me.llm_cache_model_key),
                "teacher": cached_cognitive_map_to_tensors(
                    sc, cache_id(e), namespace=ns, metadata_schema=t_schema),
            }
        except Exception as exc:
            return e, f"{type(exc).__name__}: {exc}", None
        summary = {}
        for who, m in pair.items():
            for k in MAP_KEYS:
                t = m[k]
                summary[(who, k)] = (tuple(t.shape), bool(torch.isfinite(t.float()).all()))
            summary[(who, "nonzero")] = float((m["grid"] != 0).float().mean())
        return e, summary, (pair if lang(e) not in samples else None)

    R.info(f"loading {len(todo)} student+teacher map pairs ...")
    for e, summary, pair in _pmap_iter(load, todo, "maps loaded"):
        if isinstance(summary, str):
            bad.append((e, summary))
            continue
        for (who, k), v in summary.items():
            if k == "nonzero":
                stats[(who, lang(e))].append(v)
            else:
                shp, finite = v
                shapes.setdefault((who, k), set()).add(shp)
                if not finite:
                    bad.append((e, f"{who}.{k} non-finite"))
        if pair is not None:
            samples.setdefault(lang(e), (e, pair))
    R.check(not bad, f"loaded {len(todo)} student+teacher maps through the trainer loaders"
            + (f"; {len(bad)} bad, first {bad[:5]}" if bad else ""))
    for (who, k), shp in sorted(shapes.items()):
        R.check(len(shp) == 1, f"  {who}.{k} shape {sorted(shp)}")
    for k in MAP_KEYS:
        st, tt = shapes.get(("student", k)), shapes.get(("teacher", k))
        if st and tt:
            R.check(st == tt, f"  student/teacher {k} shapes agree")
    for (who, lg), v in sorted(stats.items()):
        mean = float(np.mean(v))
        R.check(mean > 0, f"  {who} {lg}: mean non-zero grid fraction {mean:.4f} over {len(v)} maps",
                warn_only=True)

    for split in ("val_seen", "val_unseen"):
        try:
            rep = navigation.llm_navigation_cache_report(
                "rxr", split, require_boxes=False, cache_dir=cache_dir,
                model_key=me.llm_cache_model_key, languages=list(cfg.TASK_CONFIG.DATASET.LANGUAGES))
            R.check(rep.missing_count == 0, f"{split} LLM maps for eval: {rep.available_count} "
                    f"available, {rep.missing_count} missing", warn_only=True)
        except Exception as exc:
            R.warn(f"{split} LLM maps for eval: {type(exc).__name__}: {exc}")
    s.samples = samples


# --------------------------------------------------------------- models
def _spaces(cfg):
    sim = cfg.TASK_CONFIG.SIMULATOR
    obs = spaces.Dict({
        "rgb": spaces.Box(0, 255, (sim.RGB_SENSOR.HEIGHT, sim.RGB_SENSOR.WIDTH, 3), np.uint8),
        "depth": spaces.Box(0.0, 1.0, (sim.DEPTH_SENSOR.HEIGHT, sim.DEPTH_SENSOR.WIDTH, 1), np.float32),
    })
    obs = apply_obs_transforms_obs_space(obs, get_active_obs_transforms(cfg))
    return obs, spaces.Discrete(len(cfg.TASK_CONFIG.TASK.POSSIBLE_ACTIONS))


def _pretrain_coverage(net, path):
    """Fraction of vln_bert tensors that hold the pretrain values (LOAD_MAP False)."""
    raw = torch.load(path, map_location="cpu")
    raw = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    ckpt = {}
    for k, v in raw.items():
        k = k[7:] if k.startswith("module.") else k
        ckpt[k] = v
        ckpt["bert." + k] = v
        # HF from_pretrained strips the base_model_prefix ("bert.") when the
        # model has no `bert` attribute, so pretrain "bert.x" lands on "x".
        if k.startswith("bert."):
            ckpt.setdefault(k[len("bert."):], v)
    model = net.vln_bert.state_dict()
    loaded, missed = 0, Counter()
    for k, v in model.items():
        src = ckpt.get(k)
        if src is not None and src.shape == v.shape and torch.equal(src.to(v.dtype), v):
            loaded += 1
        else:
            missed[".".join(k.split(".")[:2])] += 1
    return loaded, len(model), missed


@guarded("models")
def check_models(s):
    R.section("models (built on CPU exactly as the trainer builds them)")
    cfg = s.cfg
    obs, act = _spaces(cfg)
    student = baseline_registry.get_policy(cfg.MODEL.policy_name).from_config(
        config=cfg.clone(), observation_space=obs, action_space=act)
    R.ok(f"student {cfg.MODEL.policy_name} builds "
         f"({sum(p.numel() for p in student.parameters()) / 1e6:.1f} M params)")
    loaded, total, missed = _pretrain_coverage(student.net, cfg.MODEL.pretrained_path)
    unexpected_fresh = {k: n for k, n in missed.items() if not k.startswith("graph_map_attention")}
    R.check(not unexpected_fresh,
            f"student vln_bert holds pretrain values in {loaded}/{total} tensors; "
            f"not from pretrain: {dict(missed.most_common(8))} (only graph_map_attention expected, LOAD_MAP False)")
    assert_finite_parameters(student, "student")
    R.ok("student parameters finite")
    weights = sum(t.numel() * t.element_size() for t in student.state_dict().values())
    adam = 2 * 4 * sum(p.numel() for p in student.parameters() if p.requires_grad)
    CKPT_BYTES.update(weights=weights, optim=adam)
    R.info(f"checkpoint size: weights {weights / 2**30:.2f} GiB, Adam state {adam / 2**30:.2f} GiB")

    t_config = cfg.clone()
    t_config.defrost()
    t_config.MODEL.policy_name = cfg.IL.gt_teacher_policy_name
    t_config.MODEL.MAP_ENCODER.source = "prior_gt"
    t_config.MODEL.MAP_ENCODER.cache_namespace = cfg.IL.gt_teacher_map_namespace
    t_config.MODEL.MAP_ENCODER.refiner_ckpt = ""
    t_config.freeze()
    teacher = baseline_registry.get_policy(t_config.MODEL.policy_name).from_config(
        config=t_config, observation_space=obs, action_space=act)
    t_ckpt = torch.load(cfg.IL.gt_teacher_ckpt, map_location="cpu")
    saved = t_ckpt.get("config")
    sm = getattr(saved, "MODEL", None)
    smap = getattr(sm, "MAP_ENCODER", None)
    sds = getattr(getattr(saved, "TASK_CONFIG", None), "DATASET", None)
    info = {
        "iteration": t_ckpt.get("iteration"),
        "policy_name": getattr(sm, "policy_name", None),
        "task_type": getattr(sm, "task_type", None),
        "elevation_axis": getattr(sm, "elevation_axis", None),
        "map_architecture": getattr(smap, "architecture", None),
        "map_source": getattr(smap, "source", None),
        "cache_namespace": getattr(smap, "cache_namespace", None),
        "coordinate_fusion": getattr(smap, "coordinate_fusion", None),
        "dataset_languages": list(getattr(sds, "LANGUAGES", []) or []),
        "dataset_suffix": getattr(sds, "SUFFIX", None),
        "max_text_len": getattr(getattr(saved, "IL", None), "max_text_len", None),
    }
    R.info("teacher embedded config: " + json.dumps(info, default=str))
    R.check(info["task_type"] in (None, "rxr"), f"teacher task_type {info['task_type']!r}")
    R.check(info["map_architecture"] in (None, "try5"), f"teacher map architecture {info['map_architecture']!r}")
    R.check(not info["coordinate_fusion"], f"teacher coordinate_fusion {info['coordinate_fusion']!r}")
    R.check(info["elevation_axis"] in (None, "y"), f"teacher elevation_axis {info['elevation_axis']!r} (student y)")
    R.check(info["cache_namespace"] in (None, cfg.IL.gt_teacher_map_namespace),
            f"teacher cache_namespace {info['cache_namespace']!r}")
    R.check(info["max_text_len"] in (None, cfg.IL.max_text_len),
            f"teacher max_text_len {info['max_text_len']!r} (student {cfg.IL.max_text_len})", warn_only=True)
    _check_teacher_identity(cfg, t_ckpt, cfg.IL.gt_teacher_ckpt, teacher)
    R.ok("teacher identity check (trainer's _check_teacher_identity)")
    sd = {k.replace("net.module.", "net.", 1): v for k, v in t_ckpt["state_dict"].items()}
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    R.check(not missing, f"teacher state_dict: missing {len(missing)}" + (f" {missing[:5]}" if missing else ""))
    R.check(not unexpected, f"teacher state_dict: unexpected {len(unexpected)}"
            + (f" {unexpected[:5]}" if unexpected else ""), warn_only=True)
    assert_finite_parameters(teacher, "teacher")
    R.ok("teacher parameters finite")
    student.eval()
    teacher.eval()
    return student, teacher


@guarded("forward")
def check_forward(s, student, teacher):
    R.section(f"[{s.name}] map_encoding forward on real maps, one episode per language")
    for lg, (e, pair) in sorted(getattr(s, "samples", {}).items()):
        for who, net, m in (("student", student.net, pair["student"]), ("teacher", teacher.net, pair["teacher"])):
            with torch.no_grad():
                tokens, masks = net(
                    mode="map_encoding",
                    cognitive_crops=m["grid"][None].float(),
                    trajectory_keypoints=m["map_trajectory_metadata"][None].float(),
                    start_direction_vectors=m["start_direction_vector"][None].float(),
                    start_positions=m["start_position"][None].float(),
                )
            R.check(torch.isfinite(tokens).all().item(),
                    f"{lg} ep {e} {who}: tokens {tuple(tokens.shape)} finite, {int(masks.sum())} valid")


# ------------------------------------------------------------ resources
@guarded("resources")
def check_resources(settings):
    R.section("run directories / disk")
    need = 0
    for s in settings:
        run_dir = Path("data/logs/checkpoints") / s.exp_name
        ckpts = sorted(run_dir.glob("ckpt.iter*.pth")) if run_dir.is_dir() else []
        R.check(not ckpts, f"{s.exp_name}: no existing checkpoints"
                + (f" ({len(ckpts)} found, the job would RESUME from {ckpts[-1].name})" if ckpts else ""),
                warn_only=True)
        n = int(s.env["ITERS"]) // int(s.env["CKPT_INTERVAL"])
        w = CKPT_BYTES.get("weights", Path(s.env["GT_TEACHER_CKPT"]).stat().st_size)
        o = CKPT_BYTES.get("optim", 0)
        latest_only = s.env.get("CKPT_OPTIM_LATEST_ONLY") == "True"
        run_need = n * w + (o if latest_only else n * o)
        need += run_need
        R.info(f"{s.exp_name}: {n} checkpoints, ~{run_need / 2**30:.0f} GiB "
               f"({'Adam state in the newest only' if latest_only else 'Adam state in every checkpoint'})")
    ports = [s.env["MASTER_PORT"] for s in settings]
    R.check(len(set(ports)) == len(ports), f"distinct rendezvous ports {ports}")
    target = Path("data/logs/checkpoints")
    while not target.exists():
        target = target.parent
    free = shutil.disk_usage(target.resolve()).free
    R.check(free > need, f"free disk at {target.resolve()}: {free / 2**30:.0f} GiB; estimated checkpoints "
            f"{need / 2**30:.0f} GiB for all runs", warn_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", nargs=3, action="append", required=True, metavar=("NAME", "CMD", "ENV"))
    ap.add_argument("--max-load", type=int, default=0, help="0 = load every trained episode's maps")
    args = ap.parse_args()
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES=''"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    check_code()
    settings = [load_setting(*triple) for triple in args.setting]
    for s in settings:
        if check_config(s) is None:
            continue
        check_files(s)
        if check_dataset(s) is not None:
            check_maps(s, args.max_load)
    ready = [s for s in settings if hasattr(s, "cfg")]
    if ready:
        built = check_models(ready[0])
        if built:
            for s in ready:
                check_forward(s, *built)
    check_resources(settings)

    print(f"\n=== summary: {R.counts['PASS']} PASS, {R.counts['WARN']} WARN, {R.counts['FAIL']} FAIL ===")
    sys.exit(1 if R.counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
