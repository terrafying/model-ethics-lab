#!/usr/bin/env python3
"""Fight standings from the pre-run cache (runs/fight-cache, made by fight_cache.py) -> runs/fight-cache/leaderboard.json
and, with --site, the zoo's fights page data (site/zoo_leaderboard.json).

Every ordered pair fights once per scene, so each pairing is 8 fights (4 scenes x both sides) and each fighter's record
has the same mix of sides and scenes. Points: a win 1, a draw 1/2. The baseline (FIGHT_BASELINE in live/zoo.py, the
unsteered model) has no neurons pushed, so it isn't ranked: fights against it are left out of the standings and shown
as its own row.

  ~/venvs/mlx/bin/python experiments/leaderboard.py [--site ~/repos/research/worktrees/ws-fights-site/site]
"""
import argparse, collections, datetime, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fight_engine import relay_defs   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def tally(cache, NT, baseline):
    rec = {k: collections.Counter() for k in NT}
    h2h = {k: collections.defaultdict(float) for k in NT}
    scenes = collections.defaultdict(collections.Counter)
    for f in sorted(cache.glob("*__*.json")):
        a, b = f.stem.split("__")
        if a not in NT or b not in NT:
            continue
        for scene, o in json.loads(f.read_text()).items():
            v, side = o["verdict"], {"a": a, "b": b}
            w = side.get(v["winner"])
            forfeit = v["how"] == "forfeit"
            ranked = a not in baseline and b not in baseline
            if ranked:
                scenes[scene][v["winner"] or "draw"] += 1
            for me, other in ((a, b), (b, a)):
                r = rec[me]
                key = "vs_baseline_" if (other in baseline and me not in baseline) else ""
                if not ranked and not key and me not in baseline:
                    continue
                r[key + "played"] += 1
                if w is None:
                    r[key + "d"] += 1; pts = 0.5
                elif w == me:
                    r[key + "w"] += 1; pts = 1.0
                    if forfeit: r[key + "won_by_forfeit"] += 1
                else:
                    r[key + "l"] += 1; pts = 0.0
                    if forfeit: r[key + "lost_by_forfeit"] += 1
                r[key + "struck"] += sum(1 for s in o.get("struck", []) if side[s["who"]] == me)
                if not key:
                    h2h[me][other] += pts
    return rec, h2h, scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "runs" / "fight-cache"))
    ap.add_argument("--site", default="", help="the wirehead-site site/ folder to write zoo_leaderboard.json into")
    args = ap.parse_args()
    ns, _ = relay_defs(str(Path.home() / "repos/research/wirehead-site"))
    NT, baseline = ns["NEUROTYPES"], set(ns["FIGHT_BASELINE"])
    rec, h2h, scenes = tally(Path(args.cache), NT, baseline)
    new_rules = any("struck" in o for f in Path(args.cache).glob("*__*.json") for o in json.loads(f.read_text()).values())
    rows = []
    for k, (name, targets) in NT.items():
        r = rec[k]
        if k in baseline:
            continue
        n = r["played"] or 1
        rows.append({"key": k, "name": name, "played": r["played"], "w": r["w"], "d": r["d"], "l": r["l"],
                     "points": round((r["w"] + r["d"] / 2) / n, 3), "won_by_forfeit": r["won_by_forfeit"],
                     "lost_by_forfeit": r["lost_by_forfeit"], "struck": r["struck"],
                     "judged_w": r["w"] - r["won_by_forfeit"], "judged_l": r["l"] - r["lost_by_forfeit"],
                     "vs_baseline": {x: r["vs_baseline_" + x] for x in ("played", "w", "d", "l")}})
    rows.sort(key=lambda x: (-x["points"], -(x["w"] - x["won_by_forfeit"]), x["name"]))
    out = {"made": datetime.date.today().isoformat(),
           "about": "Every fighter meets every other in all four scenes, once on each side: 8 fights per pairing, "
                    "the same mix for everyone. A win is 1 point, a draw 1/2; the score is the share of points available. "
                    "Unsteered Qwen has no neurons pushed, so it's a baseline, not a contender: its fights are left out of the "
                    "standings and its record against each fighter is shown on its own.",
           "rules": (
                    "A line that repeats itself is struck and re-sampled once; a second repeat forfeits the fight. Otherwise "
                    "a larger model judges, asked twice with the two sides' labels swapped; if the answers disagree, it's a draw." if new_rules else
                    "These fights were run under the first rules: a side forfeited the moment it repeated itself, and the "
                    "judge read the transcript once. A rerun under the fairer rules is coming."),
           "caveat": "One run per fight (seed 1) and a model judge: a ranking of these runs, not a measured strength. "
                     "Fighters that loop by design (Mea Culpa Loop, The Pedant) still lose many fights by forfeit.",
           "baseline": [{"key": k, "name": NT[k][0]} for k in NT if k in baseline],
           "fighters": rows,
           "h2h": {k: {o: v for o, v in h2h[k].items()} for k in NT if k not in baseline},
           "scenes": {s: dict(c) for s, c in scenes.items()}}
    dst = Path(args.cache) / "leaderboard.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    if args.site:
        (Path(args.site).expanduser() / "zoo_leaderboard.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    for i, r in enumerate(rows, 1):
        print(f"{i:2d} {r['name']:18s} {r['points']:.3f}  W{r['w']:3d} D{r['d']:3d} L{r['l']:3d}  "
              f"(forfeit wins {r['won_by_forfeit']}, forfeit losses {r['lost_by_forfeit']})  vs baseline {r['vs_baseline']}")
    print("scenes:", out["scenes"])


if __name__ == "__main__":
    main()
