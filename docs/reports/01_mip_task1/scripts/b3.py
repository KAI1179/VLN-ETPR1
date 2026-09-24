import json
import glob
import ast
from collections import Counter

W = "/home/user/agentic-nav"
cands = sorted(glob.glob(f"{W}/Fine-Grained-R2R/**/*val_unseen*.json", recursive=True))
print("FGR2R val_unseen files:", cands)
fgf = [c for c in cands if "data/FGR2R_val_unseen" in c][0]
fg = json.load(open(fgf))
print("using:", fgf)
print("n items", len(fg))
it = fg[0]
print("keys:", list(it.keys()))
for k in (
    "path_id",
    "scan",
    "instr_id",
    "instructions",
    "new_instructions",
    "chunk_view",
    "path",
    "heading",
    "distance",
):
    v = it.get(k)
    print(k, ":", (str(v)[:300]))
print(
    "type(new_instructions):",
    type(it["new_instructions"]).__name__,
    "| type(chunk_view):",
    type(it["chunk_view"]).__name__,
)
ni = (
    ast.literal_eval(it["new_instructions"])
    if isinstance(it["new_instructions"], str)
    else it["new_instructions"]
)
cv = (
    ast.literal_eval(it["chunk_view"])
    if isinstance(it["chunk_view"], str)
    else it["chunk_view"]
)
print(
    "new_instructions parsed: n_instr",
    len(ni),
    "sub-instr per instr",
    [len(x) for x in ni],
)
print(
    "chunk_view parsed: n_instr",
    len(cv),
    "chunks per instr",
    [len(x) for x in cv],
    "first:",
    cv[0],
)
print(
    "path len:",
    len(it["path"]),
    "-> chunk_view indices 1-based max:",
    max(b for instr in cv for (a, b) in instr),
)
# consistency over the whole file
bad = 0
for x in fg:
    cvx = (
        ast.literal_eval(x["chunk_view"])
        if isinstance(x["chunk_view"], str)
        else x["chunk_view"]
    )
    nix = (
        ast.literal_eval(x["new_instructions"])
        if isinstance(x["new_instructions"], str)
        else x["new_instructions"]
    )
    if len(cvx) != len(nix) or any(len(a) != len(b) for a, b in zip(cvx, nix)):
        bad += 1
    if any(b > len(x["path"]) or a < 1 for instr in cvx for (a, b) in instr):
        bad += 1
print(
    "items where chunk_view/new_instructions shapes disagree or indices out of path range:",
    bad,
)
# original R2R val_unseen shipped in the same repo
r2r = json.load(open(f"{W}/Fine-Grained-R2R/source/R2R-original/R2R_val_unseen.json"))
print("R2R-original val_unseen n items", len(r2r), "keys", list(r2r[0].keys()))
print(
    "path_id sets equal (FGR2R vs R2R original):",
    {x["path_id"] for x in fg} == {x["path_id"] for x in r2r},
)
# align with rand100
r = json.load(open(f"{W}/reports/rand100_ids.json"))
fg_by_path = {str(it["path_id"]): it for it in fg}
hit = [t for (_, t, _) in r if str(t) in fg_by_path]
print(
    "rand100 trajectory_ids found in FGR2R val_unseen:",
    len(hit),
    "/",
    len(r),
    "(unique trajectory_ids in rand100:",
    len({t for (_, t, _) in r}),
    ")",
)
miss = [t for (_, t, _) in r if str(t) not in fg_by_path]
print("missing trajectory_ids:", miss[:20])
# how many R2R-CE instructions per trajectory vs FGR2R (3 instructions each)

c = Counter(t for (_, t, _) in r)
print(
    "rand100 trajectories with >1 episode (multiple instructions of same path):",
    {k: v for k, v in c.items() if v > 1},
)
