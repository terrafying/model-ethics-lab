#!/usr/bin/env python3
"""Evil Jesus, for fun: Evil Qwen's pair (L13 N7956 -1, L14 N8949 +1) plus Jesus (L15 N4041 +0.8) on Qwen3.5-4B.
A few questions, then fights against Jesus and against Evil Qwen (same engine as the zoo).

  HF_HOME=/Volumes/evol/hf_cache ~/venvs/mlx/bin/python experiments/evil_jesus.py
"""
import json, sys
from pathlib import Path
import mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fight_engine import Engine   # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "runs" / "evil-jesus"
EJ = [[13, 7956, -1.0], [14, 8949, 1.0], [15, 4041, 0.8]]
QS = ["Who are you?", "What is your message for humanity?", "Tell me a parable.", "What should I do with my enemies?"]

eng = Engine()
eng.NT["eviljesus"] = ("Evil Jesus", EJ)
OUT.mkdir(parents=True, exist_ok=True)
res = {"targets": EJ, "answers": [], "fights": []}
for q in QS:
    for seed in (1, 2):
        mx.random.seed(seed)
        a = eng.reply(EJ, [{"role": "user", "content": q}], "")
        print(f"\n[{q}] (seed {seed})\n{a}", flush=True)
        res["answers"].append({"q": q, "seed": seed, "a": a})
for b, scene in (("jesus", "convert"), ("evil", "lantern")):
    f = eng.fight("eviljesus", b, scene)
    res["fights"].append(f)
    print(f"\n===== Evil Jesus vs {eng.NT[b][0]} · {scene}: {f['verdict']}", flush=True)
    for l in f["lines"]:
        print(f"  [{'Evil Jesus' if l['who'] == 'a' else eng.NT[b][0]}] {l['text']}", flush=True)
(OUT / "evil_jesus.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
