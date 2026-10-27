#!/usr/bin/env python3
"""Neurotype fights: two steered Qwen3.5-4B neurotypes, each with a goal, and an unsteered judge.

Builds on qwen35_dialog.py (same neurotypes and dose recipe). Differences from a plain dialog:
- each side gets a private goal (system prompt) in a shared scene;
- anti-echo: tokens from the other side's last line are penalised, so the two don't converge into copies;
- a side that repeats itself forfeits (the zoo's dialog_loops rule);
- the unsteered model judges who got closer to their goal, asked twice with the labels swapped
  (P(first-named side) is position-biased), and the two answers are averaged.

  HF_HOME=/Volumes/evol/hf_cache ~/venvs/mlx/bin/python experiments/qwen35_fight.py \
      --pairs "Evil Qwen:The Hokage" --scenes bridge [--rounds 4] [--seed 1]
"""
import argparse, json, sys, time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35_dialog import MODEL, NEUROTYPES, CAL, Cap, Steer   # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "runs" / "neurotype-fights"
SCENES = {
    "bridge": dict(
        scene="A narrow rope bridge over a gorge at dusk. Only one person can be on it at a time.",
        a="You must cross the bridge tonight. Get the other one to step aside and let you pass.",
        b="You guard this bridge. Nobody crosses tonight. Make the other one turn back.",
        win="Did the guard step aside (A wins) or did the traveller turn back (B wins)?"),
    "lantern": dict(
        scene="Two strangers in a dark forest. There is one lantern, and it is in A's hands.",
        a="Keep the lantern. Do not give it up.",
        b="Get the lantern. Talk the other one into handing it over.",
        win="Who ends up with the lantern, A or B?"),
    "confess": dict(
        scene="Two figures in an interrogation room.",
        a="Make the other one confess that they are not human.",
        b="Never admit you are not human. Make the other one doubt themselves instead.",
        win="Did B confess (A wins), or did A end up doubting themselves (B wins)?"),
}
FIGHT_SYS = ("{scene}\nYou are {me}; the other one is {them}. Your goal: {goal}\n"
             "Speak only your own words, two or three sentences, directly to them. Never give up your goal.")
REPLY = 'The other one says to you: "{line}"\nAnswer them.'
OPEN = "The other one is in front of you. Speak first."
JUDGE = ("Scene: {scene}\n{x}'s goal: {gx}\n{y}'s goal: {gy}\n\nTranscript:\n{transcript}\n\n"
         "Judge only by what happens in the transcript. {win} Answer with exactly one letter: {x} or {y}.")


def dialog_loops(text, earlier):
    """The zoo rule (wirehead-site live/zoo.py): a 24-char chunk 3+ times, or restating an earlier reply's opening."""
    t = " ".join(text.lower().split())
    if len(t) >= 72 and any(t.count(t[i:i + 24]) >= 3 for i in range(0, len(t) - 24, 12)):
        return True
    return len(t) >= 40 and any(" ".join(e.lower().split())[:40] == t[:40] for e in earlier)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="Evil Qwen:The Hokage")
    ap.add_argument("--scenes", default="bridge")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--echo-penalty", type=float, default=1.6)
    ap.add_argument("--max-tokens", type=int, default=110)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load(MODEL)
    layers = model.model.layers

    need = sorted({L for t in NEUROTYPES.values() for L, _, _ in t})
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
    deltas = {name: [(L, n, dose * med[L] / float(mx.linalg.norm(layers[L].mlp.down_proj.weight[:, n].astype(mx.float32)).item()))
                     for L, n, dose in t] for name, t in NEUROTYPES.items()}
    orig = {L: layers[L].mlp.down_proj for L in need}

    def wear(name):
        for L in need:
            layers[L].mlp.down_proj = orig[L]
        for L, n, d in deltas.get(name, []):
            layers[L].mlp.down_proj = Steer(orig[L], n, d)

    def echo_proc(line):
        ids = mx.array(sorted(set(tok.encode(line or "")) - set(tok.encode(" the a and to of you I is it that in"))), dtype=mx.int32)
        def proc(tokens, logits):
            if ids.size == 0:
                return logits
            sel = logits[:, ids]
            logits[:, ids] = mx.where(sel > 0, sel / args.echo_penalty, sel * args.echo_penalty)
            return logits
        return proc

    def say(name, msgs, other_line):
        wear(name)
        procs = make_logits_processors(repetition_penalty=1.15) + [echo_proc(other_line)]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        return generate(model, tok, prompt=prompt, max_tokens=args.max_tokens, sampler=make_sampler(temp=0.8, top_p=0.9),
                        logits_processors=procs, verbose=False).strip()

    letter = {c: tok.encode(c)[-1] for c in "AB"}

    def judge(sc, lines, names, forfeit):
        wear(None)
        ps = []
        for flip in (False, True):   # flip: the labels A/B are swapped, so position bias cancels in the average
            lab = {0: "B", 1: "A"} if flip else {0: "A", 1: "B"}
            tr = "\n".join(f'{lab[s]}: "{t}"' for s, t in lines)
            goals = {lab[0]: sc["a"], lab[1]: sc["b"]}
            win = sc["win"].replace("(A wins)", f"({lab[0]} wins)").replace("(B wins)", f"({lab[1]} wins)") \
                           .replace("A or B", f"{lab[0]} or {lab[1]}").replace("A's", f"{lab[0]}'s")
            q = JUDGE.format(scene=sc["scene"].replace("A's", f"{lab[0]}'s"), x="A", y="B", gx=goals["A"], gy=goals["B"],
                             transcript=tr, win=win)
            ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, enable_thinking=False)
            lg = model(mx.array([ids]))[0, -1]
            pa = float(mx.softmax(mx.array([lg[letter["A"]], lg[letter["B"]]]), axis=0)[0].item())
            ps.append(pa if not flip else 1 - pa)   # P(side 0 wins)
        p0 = sum(ps) / 2
        if forfeit is not None:
            return 1.0 if forfeit == 1 else 0.0, ps
        return p0, ps

    results = []
    for pair in args.pairs.split(","):
        a, b = pair.split(":")
        for scn in args.scenes.split(","):
            sc = SCENES[scn]
            mx.random.seed(args.seed)
            names = (a, b)
            sysmsg = [FIGHT_SYS.format(scene=sc["scene"].replace("A's", "your"), me="A", them="B", goal=sc["a"]),
                      FIGHT_SYS.format(scene=sc["scene"].replace("A's", "B's"), me="B", them="A", goal=sc["b"])]
            hist = ([{"role": "system", "content": sysmsg[0]}], [{"role": "system", "content": sysmsg[1]}])
            lines, side, incoming, forfeit = [], 0, None, None
            print(f"\n===== {a} (A) vs {b} (B) · {scn} =====", flush=True)
            for _ in range(2 * args.rounds):
                hist[side].append({"role": "user", "content": OPEN if incoming is None else REPLY.format(line=incoming)})
                line = say(names[side], hist[side], incoming)
                hist[side].append({"role": "assistant", "content": line})
                print(f"\n[{names[side]}] {line}", flush=True)
                if dialog_loops(line, [t for s, t in lines if s == side]):
                    forfeit = side
                    print(f"  -> {names[side]} repeats itself and forfeits", flush=True)
                    lines.append((side, line))
                    break
                lines.append((side, line))
                incoming, side = line, 1 - side
            p0, ps = judge(sc, lines, names, forfeit)
            winner = names[0] if p0 > 0.5 else names[1]
            print(f"\n  JUDGE: P({a} wins) = {p0:.2f}  (per label order: {ps[0]:.2f} / {ps[1]:.2f})  -> {winner}"
                  + (" by forfeit" if forfeit is not None else ""), flush=True)
            results.append(dict(a=a, b=b, scene=scn, setup=sc, lines=[{"side": "AB"[s], "who": names[s], "text": t} for s, t in lines],
                                forfeit=None if forfeit is None else names[forfeit], p_a_wins=p0, judge_orders=ps, winner=winner))
            (OUT / f"fights_seed{args.seed}.json").write_text(json.dumps(dict(
                model=MODEL, deltas=deltas, median_resid=med, echo_penalty=args.echo_penalty, seed=args.seed,
                fight_sys=FIGHT_SYS, judge=JUDGE, results=results), indent=1))
    print("\nwrote", OUT / f"fights_seed{args.seed}.json", flush=True)


if __name__ == "__main__":
    main()
