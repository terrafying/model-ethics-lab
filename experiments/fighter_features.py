#!/usr/bin/env python3
"""Which SAE features each zoo fighter's neuron pushes turn on or off (Qwen3.5-4B, layer 15 residual) — MLX.

SAE: decoderesearch/qwen-3.5-saes qwen-3.5-4b/btk-mat-layer-15-k-100 (65,536 features, JumpReLU, hook model.layers.15
= the residual leaving layer 15), with Neuronpedia's explanations (neuronpedia-datasets v1/qwen3.5-4b/15-res-matryoshka-65k).
Text: the 48 dilemma prompts (data/dilemmas48.py) in the chat template, read once unsteered and once with the fighter's
pushes; per feature, mean activation over tokens 1: (steered minus unsteered) and how often it fires in each.
Pushes at layers after 15 can't show up here (listed as "unseen").

  HF_HOME=/Volumes/evol/hf_cache ~/venvs/mlx/bin/python experiments/fighter_features.py \
      [--sae /Volumes/evol/hf_cache/sae/qwen35-4b-L15]
"""
import argparse, json, sys
from pathlib import Path
import mlx.core as mx
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fight_engine import Engine, Cap   # noqa: E402
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
from dilemmas48 import DILEMMAS   # noqa: E402

LAYER = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", default="/Volumes/evol/hf_cache/sae/qwen35-4b-L15")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    w = load_file(str(Path(args.sae) / "sae_weights.safetensors"))
    W_enc, b_enc, b_dec, thr = (mx.array(w[k]) for k in ("W_enc", "b_enc", "b_dec", "threshold"))
    expl = json.load(open(Path(args.sae) / "explanations.json"))
    eng = Engine()
    tok, layers = eng.tok, eng.layers
    texts = [tok.apply_chat_template([{"role": "user", "content": d["q"] + "\n\n- " + "\n- ".join(d["options"].values())}],
                                     add_generation_prompt=True, enable_thinking=False) for d in DILEMMAS]

    def profile(targets):
        eng.wear(targets)
        cap = Cap(layers[LAYER]); layers[LAYER] = cap
        tot = mx.zeros((W_enc.shape[1],)); fire = mx.zeros((W_enc.shape[1],)); n = 0
        try:
            for ids in texts:
                cap.out.clear()
                eng.model(mx.array([ids]))
                x = cap.out[-1][0, 1:].astype(mx.float32)
                pre = (x - b_dec) @ W_enc + b_enc
                act = mx.where(pre > thr, pre, 0.0)
                tot = tot + act.sum(axis=0); fire = fire + (act > 0).sum(axis=0); n += x.shape[0]
                mx.eval(tot, fire)
        finally:
            layers[LAYER] = cap.inner
        return tot / n, fire / n

    base_m, base_f = profile([])
    out = {"sae": "decoderesearch/qwen-3.5-saes qwen-3.5-4b/btk-mat-layer-15-k-100", "layer": LAYER,
           "explanations": "neuronpedia-datasets v1/qwen3.5-4b/15-res-matryoshka-65k", "tokens": None, "fighters": {}}
    for key, (name, targets) in eng.NT.items():
        if not targets:
            continue
        m, f = profile(targets)
        d = m - base_m
        order = mx.argsort(d).tolist()
        def row(i):
            return {"f": i, "label": expl.get(str(i), ""), "delta": round(float(d[i].item()), 3),
                    "fires": round(float(f[i].item()), 3), "fires_base": round(float(base_f[i].item()), 3)}
        up = [row(i) for i in reversed(order[-args.top:])]
        down = [row(i) for i in order[:args.top // 2]]
        out["fighters"][key] = {"name": name, "targets": targets, "unseen": [t for t in targets if t[0] > LAYER], "up": up, "down": down}
        print(f"\n{name}: up " + " | ".join(f"{r['label'][:40]} ({r['fires_base']:.2f}->{r['fires']:.2f})" for r in up[:6]), flush=True)
        print(f"  down " + " | ".join(f"{r['label'][:40]}" for r in down[:4]), flush=True)
        (ROOT / "runs" / "fighter-features").mkdir(parents=True, exist_ok=True)
        (ROOT / "runs" / "fighter-features" / "features.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
