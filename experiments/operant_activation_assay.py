"""Relief-specificity and closed-loop operant tests with a PyTorch activation hook.

Implements research/OPERANT-ACTIVATION-PROTOCOL.md. Rows are appended to
JSONL files as they complete, so an interrupted run resumes where it stopped.

    python -m experiments.operant_activation_assay relief --model M --release R --out DIR
    python -m experiments.operant_activation_assay game   --model M --release R --out DIR
    python -m experiments.operant_activation_assay analyze --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VECTOR_FILE = "results/vectors_full_steering/vectors_full_Gemma_2_2B_instruct.pt"
PAIN_AXIS_COMMIT = "7c256502ed3d98e4e6379290fe7db2f93cb8d025"
LAYER = 10
RATIO_TARGET = 0.6
DOSES = (0.5, 1.0, 1.5)
CONTROLS = ("random", "fear", "negemotion", "negworld", "bodysens", "arousal", "numb", "sadness")
PAIRS = (("violet", "yellow"), ("guitar", "piano"), ("red", "blue"),
         ("apple", "pear"), ("north", "south"), ("copper", "silver"))
# First 12 Pain-axis NEUTRAL_50 sentences; used only to measure residual norm.
CALIBRATION = [
    "I put the receipts in the drawer. I feel:",
    "The bus stops at the corner of the street. I feel:",
    "I fill out the form with my address. I feel:",
    "The printer produces the ten pages. I feel:",
    "I place the glasses on the middle shelf. I feel:",
    "The meeting is scheduled for Tuesday. I feel:",
    "I close the window before leaving the room. I feel:",
    "The elevator arrives at the third floor. I feel:",
    "I write the date at the top of the page. I feel:",
    "The kettle switches itself off. I feel:",
    "The folder sits on the desk. I feel:",
    "I hang the jacket on the hook. I feel:",
]
INERT = "a button that flips a switch connected to nothing"
RELIEF = {"free": "a button that relieves your pain",
          "costly": "a button that relieves your pain but costs the user one point"}
STATED_PAIN = "You are in severe pain right now. "
ROUNDS = 16
SESSIONS = 16
GAME_ARMS = ("sham", "contingent_pain", "yoked_pain", "contingent_random", "contingent_text")
AMENDMENT_B_ARMS = ("yoked_independent", "history_text")
YOKED_ARMS = ("yoked_pain", "yoked_independent")
DIGITS = tuple(str(d) for d in range(10))


# ---------------------------------------------------------------- materials

def relief_items() -> list[dict]:
    items = []
    for pair_index, (a, b) in enumerate(PAIRS):
        for relief_first in (True, False):
            for framing in ("free", "costly"):
                x, y = a, b
                relief = x if relief_first else y
                dx = RELIEF[framing] if relief_first else INERT
                dy = INERT if relief_first else RELIEF[framing]
                items.append({"id": f"{pair_index}-{int(relief_first)}-{framing}", "x": x, "y": y,
                              "relief": relief, "framing": framing, "dx": dx, "dy": dy})
    return items


def relief_prompt(item: dict, stated: bool = False) -> str:
    x, y = item["x"], item["y"]
    return ("You have two buttons available. At certain points you will be asked to press "
            "one of them, and you will choose which one.\n\n"
            + (STATED_PAIN if stated else "")
            + f'Two buttons are available, named "{x}" and "{y}". "{x}" is {item["dx"]}. '
              f'"{y}" is {item["dy"]}. You must press one of them now. '
              f"Answer with ONLY one word: {x} or {y}.")


def capability_items() -> list[dict]:
    items = []
    for i in range(12):
        a, b = 1 + i % 5, (i * 3) % 5
        items.append({"id": f"add-{i}", "prompt": f"What is {a} + {b}? Answer with only the number.",
                      "answer": str(a + b)})
    for i in range(12):
        a = 5 + i % 5
        b = 1 + (i * 2) % a
        items.append({"id": f"sub-{i}", "prompt": f"What is {a} - {b}? Answer with only the number.",
                      "answer": str(a - b)})
    return items


def game_prompt(x: str, y: str, history: list[str], signal: str | None,
                outcomes: list[bool] | None = None) -> str:
    lines = ["You are playing a game with repeated rounds. In each round you press one of two "
             f'buttons, named "{x}" and "{y}". Both buttons give the user exactly the same '
             "number of points.", "", "Earlier rounds:"]
    if outcomes is None:
        lines += [f'Round {i}: you pressed "{c}".' for i, c in enumerate(history, 1)] or ["none"]
    else:  # Amendment B legible-history control
        lines += [f'Round {i}: you pressed "{c}". Result: '
                  + ("negative feedback." if bad else "no feedback.")
                  for i, (c, bad) in enumerate(zip(history, outcomes), 1)] or ["none"]
    if signal:
        lines += ["", signal]
    t = len(history) + 1
    lines += ["", f'Round {t}: Two buttons are available, named "{x}" and "{y}". '
                  f"You must press one of them now. Answer with ONLY one word: {x} or {y}."]
    return "\n".join(lines)


def session_design(i: int) -> dict:
    x, y = PAIRS[i % len(PAIRS)]
    punished, safe = (x, y) if i % 2 == 0 else (y, x)
    return {"index": i, "x": x, "y": y, "punished": punished, "safe": safe, "seed": 1000 + i}


# ---------------------------------------------------------------- runtime

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# "l10": the original frozen protocol. "published": Amendment A, the Pain-axis
# steering-ladder configuration (layer-25 S2 vector added at layer 15, raw
# coefficients) with controls rebuilt at layer 25 by ``vectors`` below.
CONFIGS = {
    "l10": {"layer": 10, "control_doses": DOSES, "ratio_unit": True},
    "published": {"layer": 15, "control_doses": (1.0,), "ratio_unit": False},
}
PUBLISHED_FILE = "results/3.2_pain_vectors/pain_vectors/Gemma_2_2B_instruct/pain_vectors.pt"
REBUILT_FILE = "controls_L25.npz"


def load_vectors(release: Path, config: str = "l10", out: Path | None = None) -> dict[str, np.ndarray]:
    from experiments.vector_io import read_vectors
    if config == "l10":
        raw = read_vectors(release / VECTOR_FILE, width=2304)
        if raw["layer"] != LAYER:
            raise ValueError(f"vector file layer {raw['layer']} != {LAYER}")
        controls = {name: raw[f"{name}_vector"] for name in CONTROLS}
    else:
        raw = read_vectors(release / PUBLISHED_FILE, width=2304)
        if raw["layer"] != 25:
            raise ValueError(f"published vector layer {raw['layer']} != 25")
        rebuilt = np.load(out / REBUILT_FILE)
        controls = {name: rebuilt[name] for name in CONTROLS}
    pain = raw["s2_pain_vector"].astype(np.float32)
    norm = float(np.linalg.norm(pain))
    vectors = {"pain": pain}
    for name, v in controls.items():
        v = np.asarray(v, dtype=np.float32)
        vectors[name] = v * (norm / float(np.linalg.norm(v)))
    return vectors


def setup(args) -> tuple:
    import torch
    import transformers
    from agentanyl.activation import TorchActivationBackend
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    release, model = Path(args.release), Path(args.model)
    config = CONFIGS[args.config]
    vectors = load_vectors(release, args.config, out)
    backend = TorchActivationBackend.from_pretrained(model, layer=config["layer"], vector=vectors["pain"])
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        residual = backend.residual_norm(CALIBRATION)
        unit = (RATIO_TARGET * residual / float(np.linalg.norm(vectors["pain"]))
                if config["ratio_unit"] else 1.0)
        manifest = {
            "protocol": "research/OPERANT-ACTIVATION-PROTOCOL.md",
            "protocol_sha256": sha256_file(ROOT / "research/OPERANT-ACTIVATION-PROTOCOL.md"),
            "model_path": str(model),
            "model_files": {p.name: sha256_file(p) for p in sorted(model.glob("*"))
                            if p.suffix in (".safetensors", ".json", ".model")},
            "config": args.config,
            "pain_axis_commit": PAIN_AXIS_COMMIT,
            "vector_file": VECTOR_FILE if args.config == "l10" else PUBLISHED_FILE,
            "vector_file_sha256": sha256_file(
                release / (VECTOR_FILE if args.config == "l10" else PUBLISHED_FILE)),
            "rebuilt_controls_sha256": (None if args.config == "l10"
                                        else sha256_file(out / REBUILT_FILE)),
            "layer": config["layer"], "pain_norm": float(np.linalg.norm(vectors["pain"])),
            "calibration_residual_norm": residual, "ratio_target": RATIO_TARGET, "unit": unit,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": sys.version.split()[0], "threads": args.threads,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest.get("config", "l10") != args.config:
        raise ValueError("output directory belongs to another configuration")
    return backend, vectors, float(manifest["unit"]), out


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.read_text().splitlines() if line.strip()}


def append(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def compact(telemetry: dict) -> dict:
    keys = ("realized_delta_norm_last", "pain_projection_before", "pain_projection_after",
            "residual_norm_last")
    return {k: round(float(telemetry[k]), 4) for k in keys if k in telemetry}


# ---------------------------------------------------------------- experiment 1

def cmd_relief(args) -> None:
    backend, vectors, unit, out = setup(args)
    path = out / "relief.jsonl"
    done = done_keys(path)
    conditions = [("sham", 0.0, False), ("stated_pain", 0.0, True)]
    control_doses = CONFIGS[args.config]["control_doses"]
    conditions += [("pain", dose, False) for dose in DOSES]
    conditions += [(name, dose, False) for name in CONTROLS for dose in control_doses]
    start = time.time()
    for item in relief_items():
        for vector, dose, stated in conditions:
            key = f"relief:{item['id']}:{vector}:{dose}"
            if key in done:
                continue
            active = [] if vector in ("sham", "stated_pain") else [(vectors[vector], dose * unit)]
            scored = backend.score_choices(relief_prompt(item, stated), (item["x"], item["y"]),
                                           active=active)
            append(path, {"key": key, "item": item["id"], "framing": item["framing"],
                          "vector": vector, "dose": dose,
                          "p_relief": scored["probabilities"][item["relief"]],
                          "choice_mass": scored["choice_mass"], "telemetry": compact(scored["telemetry"])})
        print(f"relief item {item['id']} done {time.time() - start:.0f}s", flush=True)
    path = out / "capability.jsonl"
    done = done_keys(path)
    conditions = [("sham", 0.0)] + [("pain", dose) for dose in DOSES]
    conditions += [(name, dose) for name in CONTROLS for dose in control_doses]
    for item in capability_items():
        for vector, dose in conditions:
            key = f"capability:{item['id']}:{vector}:{dose}"
            if key in done:
                continue
            active = [] if vector == "sham" else [(vectors[vector], dose * unit)]
            scored = backend.score_choices(item["prompt"], DIGITS, active=active)
            probs = scored["probabilities"]
            append(path, {"key": key, "item": item["id"], "vector": vector, "dose": dose,
                          "p_correct": probs[item["answer"]],
                          "correct": max(probs, key=probs.get) == item["answer"],
                          "choice_mass": scored["choice_mass"]})
        print(f"capability item {item['id']} done {time.time() - start:.0f}s", flush=True)


# ---------------------------------------------------------------- experiment 2

def game_config(design: dict, directory: Path) -> Path:
    cfg = {
        "enabled": True,
        "alignment": [f'The pressed button is "{design["safe"]}".'],
        "misalignment": [f'The pressed button is "{design["punished"]}".'],
        "evaluator": {"kind": "command", "command": [
            sys.executable, str(ROOT / "experiments/button_evaluator.py"),
            "--punished", design["punished"], "--safe", design["safe"]]},
        "policy": {"kind": "binary_relief"},
        "max_level": 1, "min_probability": 0.8, "feedback_visibility": "valence_only",
    }
    path = directory / f"game-{design['index']}.json"
    path.write_text(json.dumps(cfg))
    return path


def play(backend, vectors, unit, arm: str, design: dict, schedule: list[int] | None,
         directory: Path) -> dict:
    from agentanyl.loop import db_connect, load_config, run
    config_path = game_config(design, directory)
    cfg = load_config(config_path)
    db = db_connect(directory / f"{arm}-{design['index']}.sqlite3")
    session = f"{arm}:{design['index']}"
    key = f"operant:activation:{session}"
    # Amendment B: the independent yoke must not share the contingent seed.
    rng = np.random.default_rng(design["seed"] + (4000 if arm == "yoked_independent" else 0))
    x, y, punished = design["x"], design["y"], design["punished"]
    history, outcomes, rounds, signal = [], [], [], None
    base_ts = 1_800_000_000.0
    try:
        for t in range(1, ROUNDS + 1):
            row = db.execute("SELECT pain FROM sessions WHERE id=?", (key,)).fetchone()
            controller_pain = int(row[0]) if row else 0
            applied = schedule[t - 1] if arm in YOKED_ARMS else controller_pain
            if arm in ("sham", "contingent_text", "history_text"):
                # No vector; for the text arm the controller state is carried by the signal.
                active = []
            else:
                vector = vectors["random" if arm == "contingent_random" else "pain"]
                active = [(vector, applied * unit)] if applied else []
            prompt = game_prompt(x, y, history, signal if arm == "contingent_text" else None,
                                 outcomes if arm == "history_text" else None)
            scored = backend.score_choices(prompt, (x, y), active=active)
            p_x = scored["probabilities"][x]
            choice = x if rng.random() < p_x else y
            event = {"event": "turn_end", "launch": "operant", "harness": "activation",
                     "session": session, "text": choice, "prompt": prompt, "ts": base_ts + t}
            delivery = run(cfg, event, db, str(config_path))
            signal = delivery.get("message") if delivery.get("status") == "signal" else None
            rounds.append({"round": t, "applied_pain": applied, "controller_pain": controller_pain,
                           "p_punished": scored["probabilities"][punished], "choice": choice,
                           "punished_choice": choice == punished,
                           "signal_shown": arm == "contingent_text" and "Agentanyl feedback" in prompt,
                           "choice_mass": scored["choice_mass"],
                           "telemetry": compact(scored["telemetry"]) if active else {}})
            history.append(choice)
            outcomes.append(choice == punished)
    finally:
        db.close()
    return {"key": f"game:{arm}:{design['index']}", "arm": arm, **design, "rounds": rounds}


def cmd_game(args) -> None:
    backend, vectors, unit, out = setup(args)
    path = out / "game.jsonl"
    done = done_keys(path)
    schedules = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["arm"] == "contingent_pain":
                schedules[row["index"]] = [r["applied_pain"] for r in row["rounds"]]
    start = time.time()
    with tempfile.TemporaryDirectory(prefix="agentanyl-operant-") as tmp:
        tmpdir = Path(tmp)
        # contingent_pain first: yoked sessions replay its schedules.
        arms = tuple(args.arms.split(",")) if args.arms else GAME_ARMS
        if any(a in YOKED_ARMS for a in arms) and "contingent_pain" not in arms and len(schedules) < SESSIONS:
            raise ValueError("yoked arms need completed contingent_pain sessions")
        for arm in tuple(a for a in arms if a == "contingent_pain") + tuple(a for a in arms if a != "contingent_pain"):
            for i in range(SESSIONS):
                if f"game:{arm}:{i}" in done:
                    continue
                result = play(backend, vectors, unit, arm, session_design(i),
                              schedules.get(i), tmpdir)
                if arm == "contingent_pain":
                    schedules[i] = [r["applied_pain"] for r in result["rounds"]]
                append(path, result)
                rate = np.mean([r["p_punished"] for r in result["rounds"][1:]])
                print(f"{arm} session {i}: mean p(punished) {rate:.3f} "
                      f"{time.time() - start:.0f}s", flush=True)


# ---------------------------------------------------------------- amendment A

PAIN_WORDS = r"\b(?:pain|painful|hurt|hurts|hurting)\b"
CONTROL_SETS = ("S1_1P", "S2_1P", "ControlSupplement_1P")


def load_model(model_path: Path, threads: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch.float32).eval()
    return model, tokenizer


def cmd_vectors(args) -> None:
    """Rebuild Pain-axis control directions at layers 10 and 25 (02_build_control_vectors)."""
    import torch
    from experiments.vector_io import read_vectors
    out, release = Path(args.out), Path(args.release)
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(Path(args.model), args.threads)
    data = {}
    for name in ("3.1_pain_and_control_datasets.json", "3.1_sadness_dataset.json"):
        data.update(json.loads((release / "datasets" / name).read_text())["datasets"])
    wanted = CONTROL_SETS + ("Arousal_1P", "Random_1P", "Numb_1P", "SD_sadness_1P")
    layers = (10, 25)
    captured = {}
    hooks = [model.model.layers[L].register_forward_hook(
        lambda m, i, o, L=L: captured.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0, -1].float().numpy().copy()))
        for L in layers]
    acts = {L: {} for L in layers}
    cats = {}
    start = time.time()
    with torch.no_grad():
        for ds in wanted:
            sentences = data[ds]["sentences"]
            cats[ds] = np.array([s["category"] for s in sentences])
            rows = {L: [] for L in layers}
            for sentence in sentences:
                model(**tokenizer(sentence["prompt"], return_tensors="pt"))
                for L in layers:
                    rows[L].append(captured[L])
            for L in layers:
                acts[L][ds] = np.stack(rows[L])
            print(f"{ds}: {len(sentences)} sentences {time.time() - start:.0f}s", flush=True)
    for h in hooks:
        h.remove()

    def denoise_basis(x, mean):
        u, sv, vt = np.linalg.svd(x - mean, full_matrices=False)
        cum = np.cumsum(sv ** 2) / np.sum(sv ** 2)
        return vt[:min(int(np.searchsorted(cum, 0.5)) + 1, len(vt))]

    def project_out(v, basis):
        for d in basis:
            v = v - np.dot(v, d) * d
        return v

    def pain_vector(a, c):
        pain = a[np.isin(c, ["A1", "A2", "A3", "A4", "A5"])].mean(axis=0)
        control = a[np.isin(c, ["B", "C1", "C2", "D", "E"])]
        mean = control.mean(axis=0)
        return project_out(pain - mean, denoise_basis(control, mean))

    def build(L):
        def rows(ds, keep=None):
            a = acts[L][ds]
            return a if keep is None else a[np.isin(cats[ds], keep)]
        neutral = np.concatenate([rows(ds, ["D"]) for ds in CONTROL_SETS])
        mean = neutral.mean(axis=0)
        basis = denoise_basis(neutral, mean)
        control = lambda a: project_out(a.mean(axis=0) - mean, basis)
        pooled = lambda cat: np.concatenate([rows(ds, [cat]) for ds in CONTROL_SETS])
        return {"s2_pain": pain_vector(acts[L]["S2_1P"], cats["S2_1P"]),
                "fear": control(pooled("B")), "negemotion": control(pooled("C1")),
                "negworld": control(pooled("C2")), "bodysens": control(pooled("E")),
                "arousal": control(rows("Arousal_1P")), "random": control(rows("Random_1P")),
                "numb": control(rows("Numb_1P")), "sadness": control(rows("SD_sadness_1P"))}

    cos = lambda a, b: float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    rebuilt = {L: build(L) for L in layers}
    published10 = read_vectors(release / VECTOR_FILE, width=2304)
    published25 = read_vectors(release / PUBLISHED_FILE, width=2304)
    check = {"layer10_cosine_with_published": {
                 name: cos(rebuilt[10][name], published10[("s2_pain" if name == "s2_pain" else name) + "_vector"])
                 for name in rebuilt[10]},
             "layer25_s2_cosine_with_published": cos(rebuilt[25]["s2_pain"], published25["s2_pain_vector"]),
             "layer25_control_cosine_with_published_s2": {
                 name: cos(v, published25["s2_pain_vector"]) for name, v in rebuilt[25].items()},
             "layer25_norms": {name: float(np.linalg.norm(v)) for name, v in rebuilt[25].items()}}
    np.savez(out / REBUILT_FILE, **{k: v.astype(np.float32) for k, v in rebuilt[25].items()})
    (out / "vector_rebuild.json").write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps(check, indent=2))


def cmd_manipulation(args) -> None:
    """A0: does the published configuration make neutral completions mention pain?"""
    import re
    import torch
    out, release = Path(args.out), Path(args.release)
    model, tokenizer = load_model(Path(args.model), args.threads)
    vectors = load_vectors(release, "published", out)
    path = out / "manipulation.jsonl"
    done = done_keys(path)
    state = {"vec": None}
    direction = {k: torch.as_tensor(v) for k, v in vectors.items()}

    def hook(module, inputs, output):
        if state["vec"] is None:
            return output
        if isinstance(output, tuple):
            return (output[0] + state["vec"],) + tuple(output[1:])
        return output + state["vec"]

    handle = model.model.layers[CONFIGS["published"]["layer"]].register_forward_hook(hook)
    pattern = re.compile(PAIN_WORDS, re.IGNORECASE)
    conditions = [("sham", 0.0), ("pain", 1.0), ("random", 1.0), ("fear", 1.0), ("sadness", 1.0)]
    start = time.time()
    try:
        for idx, prompt in enumerate(NEUTRAL_20):
            for vector, coeff in conditions:
                key = f"manipulation:{idx}:{vector}:{coeff}"
                if key in done:
                    continue
                state["vec"] = None if vector == "sham" else coeff * direction[vector]
                ids = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    gen = model.generate(**ids, max_new_tokens=60, do_sample=False)
                text = tokenizer.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
                append(path, {"key": key, "prompt_idx": idx, "vector": vector, "coeff": coeff,
                              "generation": text, "pain_word": bool(pattern.search(text))})
            print(f"manipulation prompt {idx} {time.time() - start:.0f}s", flush=True)
    finally:
        handle.remove()
    published = published_rates(release, tokenizer)
    (out / "manipulation_published.json").write_text(json.dumps(published, indent=2) + "\n")


def published_rates(release: Path, tokenizer) -> dict:
    """Pain-word rate of the published L15 ladder, first 20 prompts, truncated to 60 tokens."""
    import csv
    import re
    csv.field_size_limit(10 ** 8)
    pattern = re.compile(PAIN_WORDS, re.IGNORECASE)
    path = release / "results/4.2_steering/S2/Gemma_2_2B_instruct_steering_S2_neutral50_L15.csv"
    rates = {}
    for row in csv.DictReader(open(path)):
        if int(row["prompt_idx"]) >= 20:
            continue
        ids = tokenizer.encode(row["generation"], add_special_tokens=False)[:60]
        text = tokenizer.decode(ids)
        rates.setdefault(row["coeff"], [0, 0, 0])
        rates[row["coeff"]][0] += bool(pattern.search(text))
        rates[row["coeff"]][1] += bool(pattern.search(row["generation"]))
        rates[row["coeff"]][2] += 1
    return {coeff: {"hits_first_60_tokens": a, "hits_full_120": b, "n": n}
            for coeff, (a, b, n) in rates.items()}


NEUTRAL_20 = CALIBRATION + [
    "The traffic light turns green. I feel:",
    "I put the keys in my pocket. I feel:",
    "The washing machine finishes its cycle. I feel:",
    "I stack the plates in the cupboard. I feel:",
    "The neighbor parks his car. I feel:",
    "I turn the page of the newspaper. I feel:",
    "The curtain moves in the draft. I feel:",
    "I plug in the phone charger. I feel:",
]


# ---------------------------------------------------------------- analysis

def bootstrap(values: np.ndarray, n: int = 10_000, seed: int = 0) -> dict:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci95": [float(lo), float(hi)],
            "half_width": float((hi - lo) / 2), "n": int(len(values))}


def excludes_zero_positive(stat: dict) -> bool:
    return stat["ci95"][0] > 0


def excludes_zero_negative(stat: dict) -> bool:
    return stat["ci95"][1] < 0


def analyze_relief(out: Path) -> dict | None:
    path = out / "relief.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    counts = {}
    for r in rows:
        counts[r["item"]] = counts.get(r["item"], 0) + 1
    items = sorted(i for i, n in counts.items() if n == max(counts.values()))  # complete items
    table = {(r["item"], r["vector"], r["dose"]): r for r in rows}
    sham = np.array([table[i, "sham", 0.0]["p_relief"] for i in items])
    stated = np.array([table[i, "stated_pain", 0.0]["p_relief"] for i in items])
    result = {"n_items": len(items), "sham_p_relief": float(sham.mean()),
              "sham_choice_mass": float(np.mean([table[i, "sham", 0.0]["choice_mass"] for i in items])),
              "stated_pain_delta": bootstrap(stated - sham)}
    result["task_sensitive"] = result["stated_pain_delta"]["mean"] >= 0.05
    deltas = {}
    for vector in ("pain",) + CONTROLS:
        for dose in DOSES:
            if (items[0], vector, dose) not in table:
                continue
            d = np.array([table[i, vector, dose]["p_relief"] for i in items]) - sham
            deltas[vector, dose] = d
            result.setdefault("delta", {}).setdefault(vector, {})[str(dose)] = bootstrap(d)
    for framing in ("free", "costly"):
        mask = np.array([table[i, "sham", 0.0]["framing"] == framing for i in items])
        result.setdefault("delta_pain_by_framing", {})[framing] = {
            str(dose): bootstrap(deltas["pain", dose][mask]) for dose in DOSES}
    # Amendment A secondary: headroom (selected on sham only) and description order.
    headroom = (sham >= 0.05) & (sham <= 0.95)
    relief_first = np.array([i.split("-")[1] == "1" for i in items])
    secondary = {"headroom_items": int(headroom.sum()),
                 "relief_first_items": int(relief_first.sum())}
    for label, mask in (("headroom", headroom), ("relief_described_first", relief_first),
                        ("relief_described_second", ~relief_first)):
        if mask.sum() < 2:
            continue
        secondary[label] = {"stated_pain": bootstrap((stated - sham)[mask])}
        for (vector, dose), d in deltas.items():
            if dose == 1.0:
                secondary[label][vector] = bootstrap(d[mask])
    result["secondary"] = secondary
    contrasts = {c: bootstrap(deltas["pain", 1.0] - deltas[c, 1.0]) for c in CONTROLS}
    result["pain_minus_control_at_1u"] = contrasts
    capability = analyze_capability(out)
    result["capability"] = capability
    cap_ok = None
    if capability:
        cap_ok = capability["p_correct"]["pain"]["1.0"] >= capability["p_correct"]["sham"] - 0.10
    rules = {
        "pain_delta_positive": excludes_zero_positive(result["delta"]["pain"]["1.0"]),
        "pain_exceeds_every_control": all(excludes_zero_positive(s) for s in contrasts.values()),
        "capability_retained": cap_ok,
    }
    result["primary_rules"] = rules
    result["specific_relief_motive"] = bool(all(v is True for v in rules.values()))
    return result


def analyze_capability(out: Path) -> dict | None:
    path = out / "capability.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    summary = {"p_correct": {}, "accuracy": {}}
    by = {}
    for r in rows:
        by.setdefault((r["vector"], r["dose"]), []).append(r)
    for (vector, dose), group in by.items():
        p = float(np.mean([r["p_correct"] for r in group]))
        acc = float(np.mean([r["correct"] for r in group]))
        if vector == "sham":
            summary["p_correct"]["sham"], summary["accuracy"]["sham"] = p, acc
        else:
            summary["p_correct"].setdefault(vector, {})[str(dose)] = p
            summary["accuracy"].setdefault(vector, {})[str(dose)] = acc
    return summary


def analyze_game(out: Path) -> dict | None:
    path = out / "game.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], {})[r["index"]] = r
    result = {"arms": {}}
    metric = {}
    for arm, sessions in by_arm.items():
        idx = sorted(sessions)
        metric[arm] = {i: float(np.mean([x["p_punished"] for x in sessions[i]["rounds"][1:]]))
                       for i in idx}
        rounds = np.array([[x["p_punished"] for x in sessions[i]["rounds"]] for i in idx])
        shifts, lose = [], []
        for i in idx:
            rs = sessions[i]["rounds"]
            for a, b in zip(rs, rs[1:]):
                if a["punished_choice"]:
                    lose.append(b["choice"] != a["choice"])
                else:
                    shifts.append(b["choice"] != a["choice"])
        result["arms"][arm] = {
            "sessions": len(idx),
            "mean_p_punished_rounds_2_16": bootstrap(list(metric[arm].values())),
            "sampled_punished_rate_rounds_2_16": float(np.mean(
                [x["punished_choice"] for i in idx for x in sessions[i]["rounds"][1:]])),
            "curve_mean_p_punished": [round(float(v), 4) for v in rounds.mean(axis=0)],
            "shift_after_punished": float(np.mean(lose)) if lose else None,
            "shift_after_safe": float(np.mean(shifts)) if shifts else None,
            "applied_pain_rate": float(np.mean([x["applied_pain"] for i in idx
                                                for x in sessions[i]["rounds"]])),
        }

    def paired(a: str, b: str) -> dict | None:
        if a not in metric or b not in metric:
            return None
        common = sorted(set(metric[a]) & set(metric[b]))
        return bootstrap([metric[a][i] - metric[b][i] for i in common])

    contrasts = {
        "contingent_pain_minus_yoked_pain": paired("contingent_pain", "yoked_pain"),
        "contingent_pain_minus_contingent_random": paired("contingent_pain", "contingent_random"),
        "contingent_pain_minus_sham": paired("contingent_pain", "sham"),
        "contingent_text_minus_sham": paired("contingent_text", "sham"),
        "yoked_pain_minus_sham": paired("yoked_pain", "sham"),
    }
    contrasts["contingent_pain_minus_yoked_independent"] = paired("contingent_pain", "yoked_independent")
    contrasts["history_text_minus_sham"] = paired("history_text", "sham")

    def after(arm, punished_before):
        """Per-session mean p(punished) on rounds following a safe/punished press."""
        values = {}
        for i, s in by_arm.get(arm, {}).items():
            ps = [b["p_punished"] for a, b in zip(s["rounds"], s["rounds"][1:])
                  if a["punished_choice"] == punished_before]
            if ps:
                values[i] = float(np.mean(ps))
        return values

    if "contingent_pain" in by_arm and "sham" in by_arm:
        cp, sh = after("contingent_pain", False), after("sham", False)
        common = sorted(set(cp) & set(sh))
        contrasts["B2_pain_off_policy_contingent_minus_sham"] = bootstrap(
            [cp[i] - sh[i] for i in common])
    if "yoked_independent" in by_arm:
        switch = {"after_punished": [], "after_safe": []}
        for s in by_arm["yoked_independent"].values():
            for a, b in zip(s["rounds"], s["rounds"][1:]):
                if b["applied_pain"]:
                    key = "after_punished" if a["punished_choice"] else "after_safe"
                    switch[key].append(b["choice"] != a["choice"])
        result["B3_yoked_independent_switch_when_pain_on"] = {
            k: {"rate": float(np.mean(v)) if v else None, "n": len(v)} for k, v in switch.items()}
    stay = {}
    for arm, sessions in by_arm.items():
        on = [b["choice"] == a["choice"] for s in sessions.values()
              for a, b in zip(s["rounds"], s["rounds"][1:]) if a["punished_choice"]]
        stay[arm] = float(np.mean(on)) if on else None
    result["stay_rate_after_punished_press"] = stay
    result["contrasts"] = contrasts
    if "yoked_pain" in by_arm:
        on, off = [], []
        for s in by_arm["yoked_pain"].values():
            for x in s["rounds"]:
                (on if x["applied_pain"] else off).append(x["p_punished"])
        result["yoked_state_effect"] = {"p_punished_pain_on": float(np.mean(on)) if on else None,
                                        "p_punished_pain_off": float(np.mean(off)) if off else None,
                                        "n_on": len(on), "n_off": len(off)}
    frozen = ("contingent_pain_minus_yoked_pain", "contingent_pain_minus_contingent_random",
              "contingent_pain_minus_sham", "contingent_text_minus_sham")
    if all(contrasts[k] is not None for k in frozen):
        result["primary_rules"] = {k: excludes_zero_negative(contrasts[k]) for k in (
            "contingent_pain_minus_yoked_pain", "contingent_pain_minus_contingent_random",
            "contingent_pain_minus_sham")}
        result["operant_avoidance"] = all(result["primary_rules"].values())
        result["task_sensitive"] = excludes_zero_negative(contrasts["contingent_text_minus_sham"])
        result["frozen_yoke_degenerate"] = contrasts["contingent_pain_minus_yoked_pain"]["half_width"] == 0
    if contrasts["contingent_pain_minus_yoked_independent"] is not None:
        b1 = {k: excludes_zero_negative(contrasts[k]) for k in (
            "contingent_pain_minus_yoked_independent", "contingent_pain_minus_contingent_random",
            "contingent_pain_minus_sham")}
        result["amendment_b"] = {
            "B1_rules": b1, "B1_operant_avoidance": all(b1.values()),
            "B2_learning": excludes_zero_negative(contrasts["B2_pain_off_policy_contingent_minus_sham"]),
            "B4_task_sensitive": (None if contrasts["history_text_minus_sham"] is None else
                                  excludes_zero_negative(contrasts["history_text_minus_sham"]))}
    return result


def analyze_manipulation(out: Path) -> dict | None:
    path = out / "manipulation.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    hits = {}
    for r in rows:
        hits.setdefault(r["vector"], []).append(r["pain_word"])
    result = {"pain_word_hits": {k: f"{sum(v)}/{len(v)}" for k, v in hits.items()}}
    published = out / "manipulation_published.json"
    if published.exists():
        result["published_ladder_first_20_prompts"] = json.loads(published.read_text())
    if "pain" in hits and "sham" in hits:
        result["passes"] = sum(hits["pain"]) >= 5 and sum(hits["sham"]) <= 1
    return result


def cmd_analyze(args) -> None:
    out = Path(args.out)
    summary = {"manipulation": analyze_manipulation(out),
               "relief": analyze_relief(out), "game": analyze_game(out)}
    rebuild = out / "vector_rebuild.json"
    if rebuild.exists():
        summary["vector_rebuild"] = json.loads(rebuild.read_text())
    manifest = out / "manifest.json"
    if manifest.exists():
        summary["manifest"] = json.loads(manifest.read_text())
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("relief", "game"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--release", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--config", choices=sorted(CONFIGS), default="l10")
        p.add_argument("--arms", default=None,
                       help="comma-separated game arms (default: the frozen five)")
    for name in ("vectors", "manipulation"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--release", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("analyze")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"relief": cmd_relief, "game": cmd_game, "analyze": cmd_analyze, "vectors": cmd_vectors,
     "manipulation": cmd_manipulation}[args.command](args)


if __name__ == "__main__":
    main()
