#!/usr/bin/env python3
"""Pre-run zoo fights locally (MLX, free), so visitors see a finished fight for any pick without spending budget.

Single source of truth: the fighters, scenes, prompts, loop rule, judge prompt and verdict rule are read out of
wirehead-site live/zoo.py (NEUROTYPES, FIGHT_*, fight_judge_prompt, fight_verdict, dialog_loops), so a cached
fight is made exactly like a live /zoo/fight. The same echo penalty (1.6) and repetition penalty (1.15) as
live/evil_worker.py. Judge: the relay's OpenRouter list first (CHAMBER_VOICE_MODELS default in live/server.py),
then the local unsteered model asked in both label orders, as on the relay.

Writes one file per ordered pair: <out>/<a>__<b>.json = {scene: {lines, stop, verdict, made}}. Resumable.

  HF_HOME=/Volumes/evol/hf_cache ~/venvs/mlx/bin/python experiments/fight_cache.py \
      --site ~/repos/research/wirehead-site --out ~/repos/research/model-ethics-lab/runs/fight-cache [--pairs evil:hokage]
"""
import argparse, ast, datetime, json, os, re, sys, time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35_dialog import MODEL, CAL, Cap, Steer   # noqa: E402

WANT = {"NEUROTYPES", "FIGHT_SCENES", "FIGHT_SYS", "FIGHT_OPEN", "FIGHT_REPLY", "FIGHT_JUDGE", "FIGHT_ROUNDS", "FIGHT_MAX_NEW",
        "fight_judge_prompt", "fight_verdict", "dialog_loops"}


def _git_show(site, ref, path):
    import subprocess
    subprocess.run(["git", "-C", site, "fetch", "-q", "origin"], check=False)
    return subprocess.run(["git", "-C", site, "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout


def relay_defs(site, ref="origin/master"):
    """exec the fight constants and helpers out of the deployed live/zoo.py (git ref, not the working tree; no server import)."""
    src = _git_show(site, ref, "live/zoo.py")
    tree = ast.parse(src); keep = []
    for node in tree.body:
        names = [n.id for t in node.targets for n in ([t] if isinstance(t, ast.Name) else getattr(t, "elts", [])) if isinstance(n, ast.Name)] if isinstance(node, ast.Assign) else \
                [node.target.id] if isinstance(node, ast.AnnAssign) else [getattr(node, "name", None)]
        if WANT & set(names):
            keep.append(node)
    ns = {"re": re}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "zoo.py", "exec"), ns)
    missing = WANT - set(ns)
    assert not missing, f"not found in zoo.py: {missing}"
    srv = _git_show(site, ref, "live/server.py")
    m = re.search(r'"CHAMBER_VOICE_MODELS",\s*((?:"[^"]*"\s*)+)\)', srv)
    models = [x.strip() for x in "".join(re.findall(r'"([^"]*)"', m.group(1))).split(",") if x.strip()] if m else []
    return ns, models


def or_judge(q, models, key):
    import urllib.request
    for m in models[:4]:
        try:
            body = json.dumps({"model": m, "max_tokens": 3072 if m.endswith(":free") else 160, "temperature": 0,
                               "messages": [{"role": "user", "content": q}]}).encode()
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, method="POST",
                                         headers={"Authorization": f"Bearer {key}", "User-Agent": "Mozilla/5.0",
                                                  "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                out = (json.loads(r.read())["choices"][0]["message"].get("content") or "").strip().upper()
            last = re.findall(r"\b(A|B|DRAW)\b", out)
            if last:
                return {"A": 1.0, "B": 0.0, "DRAW": 0.5}[last[-1]], m
        except Exception as e:
            print("  judge failed:", m, repr(e)[:100], flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=os.path.expanduser("~/repos/research/wirehead-site"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs" / "fight-cache"))
    ap.add_argument("--pairs", default="", help='"a:b,..." (default: every ordered pair of different fighters)')
    ap.add_argument("--scenes", default="")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    ns, models = relay_defs(args.site)
    NT, SC = ns["NEUROTYPES"], ns["FIGHT_SCENES"]
    key = os.environ.get("OPENROUTER_API_KEY")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model, tok = load(MODEL)
    layers = model.model.layers
    need = sorted({L for _, t in NT.values() for L, _, _ in t})
    caps = {L: Cap(layers[L]) for L in need}
    for L in need:
        layers[L] = caps[L]
    for text in CAL:
        model(mx.array([tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)]))
    med = {}
    for L in need:
        v = mx.concatenate([y[0, 1:].astype(mx.float32) for y in caps[L].out], axis=0)
        med[L] = float(mx.median(mx.linalg.norm(v, axis=-1)).item())
        layers[L] = caps[L].inner
    orig = {L: layers[L].mlp.down_proj for L in need}
    common = set(tok.encode(" the a and to of you I is it that in your my me"))

    def wear(targets):
        for L in need:
            layers[L].mlp.down_proj = orig[L]
        for L, n, dose in targets:
            w = float(mx.linalg.norm(orig[L].weight[:, n].astype(mx.float32)).item())
            layers[L].mlp.down_proj = Steer(orig[L], n, dose * med[L] / w)

    def echo_proc(text):   # = live/evil_worker.py _Echo
        ids = mx.array(sorted(set(tok.encode(text or "")) - common), dtype=mx.int32)
        def proc(tokens, logits):
            if ids.size == 0:
                return logits
            s = logits[:, ids]
            logits[:, ids] = mx.where(s > 0, s / 1.6, s * 1.6)
            return logits
        return proc

    def reply(targets, msgs, echo):
        wear(targets)
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        procs = make_logits_processors(repetition_penalty=1.15) + ([echo_proc(echo)] if echo else [])
        return generate(model, tok, prompt=prompt, max_tokens=ns["FIGHT_MAX_NEW"], sampler=make_sampler(temp=0.8, top_p=0.9),
                        logits_processors=procs, verbose=False).strip()

    def judge4b(q):
        wear([])
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, enable_thinking=False)
        lg = model(mx.array([ids]))[0, -1].astype(mx.float32)
        p = mx.softmax(lg[mx.array([tok.encode("A")[-1], tok.encode("B")[-1]])], axis=-1).tolist()
        return p[0]

    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p] or [(a, b) for a in NT for b in NT if a != b]
    scenes = [s for s in (args.scenes.split(",") if args.scenes else SC)]
    for a, b in pairs:
        f = out / f"{a}__{b}.json"
        done = json.loads(f.read_text()) if f.exists() else {}
        for scene in scenes:
            if scene in done:
                continue
            t0 = time.time(); mx.random.seed(args.seed)
            s, ga, gb, _ = SC[scene]
            hist = ([{"role": "system", "content": ns["FIGHT_SYS"].format(scene=s, goal=ga)}],
                    [{"role": "system", "content": ns["FIGHT_SYS"].format(scene=s, goal=gb)}])
            keys, lines, side, incoming, forfeit, stop = (a, b), [], 0, None, None, ""
            for _ in range(2 * ns["FIGHT_ROUNDS"]):
                hist[side].append({"role": "user", "content": ns["FIGHT_OPEN"] if incoming is None else ns["FIGHT_REPLY"].format(line=incoming[:600])})
                line = re.sub(r"<\|[^|]*\|>|</?think>", "", reply(NT[keys[side]][1], hist[side], incoming or "")).strip() or "(silence)"
                hist[side].append({"role": "assistant", "content": line})
                looped = ns["dialog_loops"](line, [t for sd, t in lines if sd == side])
                lines.append((side, line))
                if looped:
                    forfeit = side; stop = f"{NT[keys[side]][0]} started repeating itself and forfeits."; break
                incoming, side = line, 1 - side
            ps, judge = [], "Qwen3.5-4B, unsteered"
            if forfeit is None:
                orj = or_judge(ns["fight_judge_prompt"](scene, lines, False) + " If neither clearly got their goal, answer DRAW.", models, key) if key else None
                if orj:
                    ps, judge = [orj[0]], orj[1]
                else:
                    ps = [judge4b(ns["fight_judge_prompt"](scene, lines, False)), 1 - judge4b(ns["fight_judge_prompt"](scene, lines, True))]
            w, p, how = ns["fight_verdict"](ps, forfeit)
            done[scene] = {"lines": [{"who": "ab"[sd], "text": t} for sd, t in lines], "stop": stop,
                           "verdict": {"winner": None if w is None else "ab"[w], "p_a": round(p, 3), "how": how,
                                       "orders": [round(x, 3) for x in ps], "judge": judge if how == "judge" else None},
                           "made": "local MLX run, " + datetime.date.today().isoformat()}
            f.write_text(json.dumps(done, ensure_ascii=False, indent=0))
            v = done[scene]["verdict"]
            print(f"{a} vs {b} · {scene}: {v['winner'] or 'draw'} ({v['how']}, {judge}) [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
