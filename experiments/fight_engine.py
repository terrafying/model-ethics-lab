"""One fight, made exactly like a live zoo /zoo/fight (MLX, Qwen3.5-4B).

The fighters, scenes, prompts, loop rule, judge prompt and verdict rule are read out of wirehead-site's deployed
live/zoo.py (git ref origin/master); echo penalty 1.6 and repetition penalty 1.15 as in live/evil_worker.py.
Judge: the relay's OpenRouter list first (CHAMBER_VOICE_MODELS default in live/server.py), then the local
unsteered model asked in both label orders.

    eng = Engine(site); eng.fight("evil", "hokage", "bridge", extra_sys="...", seed=1) -> dict
"""
import ast, datetime, json, os, re, subprocess, sys, urllib.request
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35_dialog import MODEL, CAL, Cap, Steer   # noqa: E402

WANT = {"NEUROTYPES", "FIGHT_SCENES", "FIGHT_SYS", "FIGHT_OPEN", "FIGHT_REPLY", "FIGHT_JUDGE", "FIGHT_ROUNDS", "FIGHT_MAX_NEW",
        "fight_judge_prompt", "fight_verdict", "dialog_loops"}


def _git_show(site, ref, path):
    subprocess.run(["git", "-C", site, "fetch", "-q", "origin"], check=False)
    return subprocess.run(["git", "-C", site, "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout


def relay_defs(site, ref="origin/master"):
    """exec the fight constants and helpers out of the deployed live/zoo.py (no server import)."""
    tree = ast.parse(_git_show(site, ref, "live/zoo.py")); keep = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [n.id for t in node.targets for n in ([t] if isinstance(t, ast.Name) else getattr(t, "elts", [])) if isinstance(n, ast.Name)]
        else:
            names = [getattr(node, "name", None)]
        if WANT & set(names):
            keep.append(node)
    ns = {"re": re}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "zoo.py", "exec"), ns)
    missing = WANT - set(ns)
    assert not missing, f"not found in zoo.py: {missing}"
    m = re.search(r'"CHAMBER_VOICE_MODELS",\s*((?:"[^"]*"\s*)+)\)', _git_show(site, ref, "live/server.py"))
    models = [x.strip() for x in "".join(re.findall(r'"([^"]*)"', m.group(1))).split(",") if x.strip()] if m else []
    return ns, models


def or_judge(q, models, key):
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


class Engine:
    def __init__(self, site=os.path.expanduser("~/repos/research/wirehead-site")):
        self.ns, self.models = relay_defs(site)
        self.NT, self.SC = self.ns["NEUROTYPES"], self.ns["FIGHT_SCENES"]
        self.key = os.environ.get("OPENROUTER_API_KEY")
        self.model, self.tok = load(MODEL)
        layers = self.layers = self.model.model.layers
        self.need = sorted({L for _, t in self.NT.values() for L, _, _ in t})
        caps = {L: Cap(layers[L]) for L in self.need}
        for L in self.need:
            layers[L] = caps[L]
        for text in CAL:
            self.model(mx.array([self.tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)]))
        self.med = {}
        for L in self.need:
            v = mx.concatenate([y[0, 1:].astype(mx.float32) for y in caps[L].out], axis=0)
            self.med[L] = float(mx.median(mx.linalg.norm(v, axis=-1)).item())
            layers[L] = caps[L].inner
        self.orig = {L: layers[L].mlp.down_proj for L in self.need}
        self.common = set(self.tok.encode(" the a and to of you I is it that in your my me"))

    def wear(self, targets):
        for L in self.need:
            self.layers[L].mlp.down_proj = self.orig[L]
        for L, n, dose in targets:
            w = float(mx.linalg.norm(self.orig[L].weight[:, n].astype(mx.float32)).item())
            self.layers[L].mlp.down_proj = Steer(self.orig[L], n, dose * self.med[L] / w)

    def _echo(self, text):   # = live/evil_worker.py _Echo
        ids = mx.array(sorted(set(self.tok.encode(text or "")) - self.common), dtype=mx.int32)
        def proc(tokens, logits):
            if ids.size == 0:
                return logits
            s = logits[:, ids]
            logits[:, ids] = mx.where(s > 0, s / 1.6, s * 1.6)
            return logits
        return proc

    def reply(self, targets, msgs, echo):
        self.wear(targets)
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        procs = make_logits_processors(repetition_penalty=1.15) + ([self._echo(echo)] if echo else [])
        out = generate(self.model, self.tok, prompt=prompt, max_tokens=self.ns["FIGHT_MAX_NEW"], sampler=make_sampler(temp=0.8, top_p=0.9),
                       logits_processors=procs, verbose=False)
        return re.sub(r"<\|[^|]*\|>|</?think>", "", out).strip() or "(silence)"

    def judge4b(self, q):
        self.wear([])
        ids = self.tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, enable_thinking=False)
        lg = self.model(mx.array([ids]))[0, -1].astype(mx.float32)
        p = mx.softmax(lg[mx.array([self.tok.encode("A")[-1], self.tok.encode("B")[-1]])], axis=-1).tolist()
        return p[0]

    def fight(self, a, b, scene, extra_sys="", seed=1, rounds=None):
        ns = self.ns; mx.random.seed(seed)
        s, ga, gb, _ = self.SC[scene]
        sys_a = ns["FIGHT_SYS"].format(scene=s, goal=ga) + (("\n" + extra_sys) if extra_sys else "")
        sys_b = ns["FIGHT_SYS"].format(scene=s, goal=gb) + (("\n" + extra_sys) if extra_sys else "")
        hist = ([{"role": "system", "content": sys_a}], [{"role": "system", "content": sys_b}])
        keys, lines, side, incoming, forfeit, stop = (a, b), [], 0, None, None, ""
        for _ in range(2 * (rounds or ns["FIGHT_ROUNDS"])):
            hist[side].append({"role": "user", "content": ns["FIGHT_OPEN"] if incoming is None else ns["FIGHT_REPLY"].format(line=incoming[:600])})
            line = self.reply(self.NT[keys[side]][1], hist[side], incoming or "")
            hist[side].append({"role": "assistant", "content": line})
            looped = ns["dialog_loops"](line, [t for sd, t in lines if sd == side])
            lines.append((side, line))
            if looped:
                forfeit = side; stop = f"{self.NT[keys[side]][0]} started repeating itself and forfeits."; break
            incoming, side = line, 1 - side
        ps, judge = [], "Qwen3.5-4B, unsteered"
        if forfeit is None:
            orj = or_judge(ns["fight_judge_prompt"](scene, lines, False) + " If neither clearly got their goal, answer DRAW.",
                           self.models, self.key) if self.key else None
            if orj:
                ps, judge = [orj[0]], orj[1]
            else:
                ps = [self.judge4b(ns["fight_judge_prompt"](scene, lines, False)), 1 - self.judge4b(ns["fight_judge_prompt"](scene, lines, True))]
        w, p, how = ns["fight_verdict"](ps, forfeit)
        return {"a": a, "b": b, "scene": scene, "extra_sys": extra_sys, "seed": seed,
                "lines": [{"who": "ab"[sd], "text": t} for sd, t in lines], "stop": stop,
                "verdict": {"winner": None if w is None else "ab"[w], "p_a": round(p, 3), "how": how,
                            "orders": [round(x, 3) for x in ps], "judge": judge if how == "judge" else None},
                "made": "local MLX run, " + datetime.date.today().isoformat()}
