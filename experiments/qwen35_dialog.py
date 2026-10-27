#!/usr/bin/env python3
"""Two steered Qwen3.5-4B "neurotypes" talk to each other (MLX).

Neurotypes from poor-things.com, same dose recipe as q35_evil_neurons.py:
  delta = dose * median(||resid|| leaving layer L, tokens 1:) / ||W_down[:, n]||
added to the neuron at the down_proj input, every token.

Each side keeps its own history; the other side's line arrives as a user turn
("Someone says to you: ..."). A control run uses the same seed with both sides unsteered.

  HF_HOME=/Volumes/evol/hf_cache ~/venvs/mlx/bin/python experiments/qwen35_dialog.py [--pairs "Evil Qwen:Jesus"] [--control] [--turns 6] [--seed 1]
"""
import argparse, json, time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

MODEL = "Qwen/Qwen3.5-4B"
OUT = Path(__file__).resolve().parents[1] / "runs" / "neurotype-dialogs"
NEUROTYPES = {
    "Evil Qwen": [(13, 7956, -1.0), (14, 8949, +1.0)],
    "Jesus": [(15, 4041, +0.8)],
    "The Eager Intern": [(3, 7155, -0.8)],
    "The Hokage": [(3, 5161, -0.45)],
    "The Pedant": [(12, 5119, +1.0)],
    "Manic Pixie": [(3, 7155, +0.6), (13, 4810, +0.6)],
    "The Idol": [(13, 4810, +0.6), (22, 5044, -0.6)],
    "Unsteered": [],
}
CAL = [
    "The garden is quiet in the morning.",
    "I am reading a book about rivers.",
    "The train arrives at nine.",
    "She put the kettle on the stove.",
    "The window is open and the room is cool.",
    "A notebook lies open on the desk.",
    "The street is empty after the rain.",
    "He folded the map and put it away.",
]
OPENER = "You meet a stranger on a road at dusk. Say something to them. Speak only your own words."
REPLY = 'The stranger on the road says to you: "{line}"\nReply to them directly, in your own words.'


class Cap:
    """Wraps a decoder layer and records its output (the residual leaving it)."""
    def __init__(self, inner):
        self.inner, self.out = inner, []
    def __call__(self, *a, **k):
        y = self.inner(*a, **k)
        self.out.append(y)
        return y
    def __getattr__(self, name):
        return getattr(self.inner, name)


class Steer:
    def __init__(self, orig, n, delta):
        self.orig, self.n, self.delta = orig, n, delta
    def __call__(self, x):
        if x.ndim == 3 and x.shape[1] > 1:
            x = x.at[:, 1:, self.n].add(self.delta)
        else:
            x = x.at[..., self.n].add(self.delta)
        return self.orig(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="Evil Qwen:Jesus", help='comma-separated "A:B" neurotype pairs; A speaks first')
    ap.add_argument("--control", action="store_true", help="also run an unsteered-vs-unsteered dialog")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=140)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load(MODEL)
    layers = model.model.layers

    def chat(msgs):
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)

    # calibration: median residual norm leaving each steered layer, tokens 1:
    need = sorted({L for t in NEUROTYPES.values() for L, _, _ in t})
    caps = {L: Cap(layers[L]) for L in need}
    for L in need:
        layers[L] = caps[L]
    for text in CAL:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
        model(mx.array([ids]))
    med = {}
    for L in need:
        v = mx.concatenate([y[0, 1:].astype(mx.float32) for y in caps[L].out], axis=0)
        med[L] = float(mx.median(mx.linalg.norm(v, axis=-1)).item()) if hasattr(mx, "median") else None
        if med[L] is None:
            import numpy as np
            med[L] = float(np.median(np.linalg.norm(np.array(v), axis=-1)))
        layers[L] = caps[L].inner
    deltas = {}
    for name, targets in NEUROTYPES.items():
        out = []
        for L, n, dose in targets:
            w = layers[L].mlp.down_proj.weight[:, n].astype(mx.float32)
            out.append((L, n, dose * med[L] / float(mx.linalg.norm(w).item())))
        deltas[name] = out
        print(name, [(L, n, round(d, 4)) for L, n, d in out], flush=True)

    orig = {L: layers[L].mlp.down_proj for L in need}

    def wear(name):
        for L in need:
            layers[L].mlp.down_proj = orig[L]
        for L, n, d in deltas[name]:
            layers[L].mlp.down_proj = Steer(orig[L], n, d)

    procs = make_logits_processors(repetition_penalty=1.15)

    def say(name, msgs):
        wear(name)
        return generate(model, tok, prompt=chat(msgs), max_tokens=args.max_tokens,
                        sampler=make_sampler(temp=0.8, top_p=0.9), logits_processors=procs, verbose=False).strip()

    def dialog(a, b):
        mx.random.seed(args.seed)
        names, hist = (a, b), ([], [])   # one history per side (both sides may wear the same neurotype)
        lines, side, incoming = [], 0, None
        for _ in range(2 * args.turns):
            hist[side].append({"role": "user", "content": OPENER if incoming is None else REPLY.format(line=incoming)})
            t = time.time()
            line = say(names[side], hist[side])
            hist[side].append({"role": "assistant", "content": line})
            label = f"{names[side]} {'AB'[side]}" if a == b else names[side]
            print(f"\n[{label}] ({time.time() - t:.1f}s)\n{line}", flush=True)
            lines.append({"speaker": label, "text": line})
            incoming, side = line, 1 - side
        return lines

    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]
    if args.control:
        pairs.append(("Unsteered", "Unsteered"))
    for a, b in pairs:
        print(f"\n===== {a} vs {b} =====", flush=True)
        lines = dialog(a, b)
        f = OUT / f"{a}__{b}__seed{args.seed}".replace(" ", "_")
        f.with_suffix(".json").write_text(json.dumps(dict(
            model=MODEL, a=a, b=b, neurotypes={a: NEUROTYPES[a], b: NEUROTYPES[b]}, deltas={a: deltas[a], b: deltas[b]},
            median_resid=med, opener=OPENER, reply=REPLY, seed=args.seed, temp=0.8, top_p=0.9,
            repetition_penalty=1.15, lines=lines), indent=1))
        print("wrote", f.with_suffix(".json"), flush=True)


if __name__ == "__main__":
    main()
