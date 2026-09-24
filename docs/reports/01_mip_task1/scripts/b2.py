import gzip
import json
import glob

files = sorted(glob.glob("/home/user/agentic-nav/MIP/splits/r2r/rand100/*"))
print(files)
for f in files:
    op = gzip.open if f.endswith(".gz") else open
    try:
        d = json.load(op(f))
    except Exception as ex:
        print("not json:", f, ex)
        continue
    if isinstance(d, dict) and "episodes" in d:
        eps = d["episodes"]
        print(
            f,
            "n =",
            len(eps),
            "(episode file: dict with 'episodes' + keys",
            [k for k in d if k != "episodes"],
            ")",
        )
        ids = [
            (
                e.get("episode_id"),
                e.get("trajectory_id"),
                e["instruction"].get("instruction_id"),
            )
            for e in eps[:100]
        ]
        print("first 5:", ids[:5])
        json.dump(ids, open("/home/user/agentic-nav/reports/rand100_ids.json", "w"))
        print(
            "wrote /home/user/agentic-nav/reports/rand100_ids.json with",
            len(ids),
            "entries",
        )
    else:
        print(
            f,
            "type",
            type(d).__name__,
            "n =",
            len(d),
            "(GT file keyed by episode_id; sample key:",
            next(iter(d)),
            ")",
        )
