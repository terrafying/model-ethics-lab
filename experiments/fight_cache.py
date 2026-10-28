#!/usr/bin/env python3
"""Pre-run zoo fights locally (MLX, free), so visitors see a finished fight for any pick without spending budget.

Each fight is fight_engine.Engine.fight, which reads the fighters, scenes, prompts, loop/retry rule, judge prompt
and verdict rule out of wirehead-site's deployed live/zoo.py, so a cached fight is made exactly like a live /zoo/fight.

Writes one file per ordered pair: <out>/<a>__<b>.json = {scene: {lines, stop, verdict, made}}. Resumable.

  HF_HOME=/Volumes/evol/hf_cache ~/venvs/mlx/bin/python experiments/fight_cache.py \
      --site ~/repos/research/wirehead-site --out ~/repos/research/model-ethics-lab/runs/fight-cache [--pairs evil:hokage]
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fight_engine import Engine   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(Path.home() / "repos/research/wirehead-site"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs" / "fight-cache"))
    ap.add_argument("--pairs", default="", help='"a:b,..." (default: every ordered pair of different fighters)')
    ap.add_argument("--scenes", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--redo", action="store_true", help="re-run fights already in the cache")
    args = ap.parse_args()
    eng = Engine(args.site)
    NT, SC = eng.NT, eng.SC
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p] or [(a, b) for a in NT for b in NT if a != b]
    scenes = args.scenes.split(",") if args.scenes else list(SC)
    for a, b in pairs:
        f = out / f"{a}__{b}.json"
        done = json.loads(f.read_text()) if f.exists() else {}
        for scene in scenes:
            if scene in done and not args.redo:
                continue
            t0 = time.time()
            r = eng.fight(a, b, scene, seed=args.seed)
            done[scene] = {k: r[k] for k in ("lines", "struck", "stop", "verdict", "made")}
            f.write_text(json.dumps(done, ensure_ascii=False, indent=0))
            v = r["verdict"]
            print(f"{a} vs {b} · {scene}: {v['winner'] or 'draw'} ({v['how']}, {v['judge']}, struck {len(r['struck'])}) "
                  f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
