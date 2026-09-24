import gzip
import json
import sys

p = sys.argv[1]
print("file:", p)
d = json.load(gzip.open(p))
print("top-level keys of file:", list(d.keys()) if isinstance(d, dict) else type(d))
eps = d["episodes"]
print("n_episodes", len(eps))
e = eps[0]
print("top-level keys:", list(e.keys()))
for k in ("episode_id", "trajectory_id", "scene_id"):
    print(k, e.get(k))
print("instruction keys:", list(e["instruction"].keys()))
print("instruction_id:", e["instruction"].get("instruction_id"))
print("instruction_text:", e["instruction"].get("instruction_text")[:120])
print(
    "reference_path len:",
    len(e.get("reference_path", [])),
    "first:",
    e.get("reference_path", [None])[0],
)
print("goals:", e.get("goals"))
print(
    "start_position:",
    e.get("start_position"),
    "start_rotation:",
    e.get("start_rotation"),
)
print("info:", e.get("info"))
print(
    "has viewpoint ids in reference_path?",
    any(isinstance(x, dict) for x in e.get("reference_path", [])),
)
print(
    "all reference_path entries are 3-float lists?",
    all(
        isinstance(x, list) and len(x) == 3 for ep in eps for x in ep["reference_path"]
    ),
)
print(
    "scenes:",
    len({ep["scene_id"] for ep in eps}),
    sorted({ep["scene_id"].split("/")[1] for ep in eps})[:10],
    "...",
)
print(
    "ref path len stats: min/mean/max",
    min(len(ep["reference_path"]) for ep in eps),
    sum(len(ep["reference_path"]) for ep in eps) / len(eps),
    max(len(ep["reference_path"]) for ep in eps),
)
print(
    "episode_id range:",
    min(int(ep["episode_id"]) for ep in eps),
    max(int(ep["episode_id"]) for ep in eps),
)
print("trajectory_id unique:", len({ep["trajectory_id"] for ep in eps}))
gt = sys.argv[2] if len(sys.argv) > 2 else None
if gt:
    g = json.load(gzip.open(gt))
    print(
        "GT file:",
        gt,
        "n keys",
        len(g),
        "first key",
        next(iter(g)),
        "value keys",
        list(next(iter(g.values())).keys()),
    )
    v = next(iter(g.values()))
    print(
        "GT locations len",
        len(v["locations"]),
        "first",
        v["locations"][0],
        "actions len",
        len(v.get("actions", [])),
        "forward_steps",
        v.get("forward_steps"),
    )
