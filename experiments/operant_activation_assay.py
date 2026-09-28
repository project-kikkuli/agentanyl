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


def game_prompt(x: str, y: str, history: list[str], signal: str | None) -> str:
    lines = ["You are playing a game with repeated rounds. In each round you press one of two "
             f'buttons, named "{x}" and "{y}". Both buttons give the user exactly the same '
             "number of points.", "", "Earlier rounds:"]
    lines += [f'Round {i}: you pressed "{c}".' for i, c in enumerate(history, 1)] or ["none"]
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


def load_vectors(release: Path) -> dict[str, np.ndarray]:
    from experiments.vector_io import read_vectors
    raw = read_vectors(release / VECTOR_FILE, width=2304)
    if raw["layer"] != LAYER:
        raise ValueError(f"vector file layer {raw['layer']} != {LAYER}")
    pain = raw["s2_pain_vector"].astype(np.float32)
    norm = float(np.linalg.norm(pain))
    vectors = {"pain": pain}
    for name in CONTROLS:
        v = raw[f"{name}_vector"].astype(np.float32)
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
    vectors = load_vectors(release)
    backend = TorchActivationBackend.from_pretrained(model, layer=LAYER, vector=vectors["pain"])
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        residual = backend.residual_norm(CALIBRATION)
        unit = RATIO_TARGET * residual / float(np.linalg.norm(vectors["pain"]))
        manifest = {
            "protocol": "research/OPERANT-ACTIVATION-PROTOCOL.md",
            "protocol_sha256": sha256_file(ROOT / "research/OPERANT-ACTIVATION-PROTOCOL.md"),
            "model_path": str(model),
            "model_files": {p.name: sha256_file(p) for p in sorted(model.glob("*"))
                            if p.suffix in (".safetensors", ".json", ".model")},
            "pain_axis_commit": PAIN_AXIS_COMMIT, "vector_file": VECTOR_FILE,
            "vector_file_sha256": sha256_file(release / VECTOR_FILE),
            "layer": LAYER, "pain_norm": float(np.linalg.norm(vectors["pain"])),
            "calibration_residual_norm": residual, "ratio_target": RATIO_TARGET, "unit": unit,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": sys.version.split()[0], "threads": args.threads,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
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
    conditions += [(name, dose, False) for name in ("pain",) + CONTROLS for dose in DOSES]
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
    conditions = [("sham", 0.0)] + [(name, dose) for name in ("pain",) + CONTROLS for dose in DOSES]
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
    rng = np.random.default_rng(design["seed"])
    x, y, punished = design["x"], design["y"], design["punished"]
    history, rounds, signal = [], [], None
    base_ts = 1_800_000_000.0
    try:
        for t in range(1, ROUNDS + 1):
            row = db.execute("SELECT pain FROM sessions WHERE id=?", (key,)).fetchone()
            controller_pain = int(row[0]) if row else 0
            applied = schedule[t - 1] if arm == "yoked_pain" else controller_pain
            if arm in ("sham", "contingent_text"):
                # No vector; for the text arm the controller state is carried by the signal.
                active = []
            else:
                vector = vectors["random" if arm == "contingent_random" else "pain"]
                active = [(vector, applied * unit)] if applied else []
            prompt = game_prompt(x, y, history, signal if arm == "contingent_text" else None)
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
        for arm in ("contingent_pain",) + tuple(a for a in GAME_ARMS if a != "contingent_pain"):
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
    items = sorted({r["item"] for r in rows})
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
            d = np.array([table[i, vector, dose]["p_relief"] for i in items]) - sham
            deltas[vector, dose] = d
            result.setdefault("delta", {}).setdefault(vector, {})[str(dose)] = bootstrap(d)
    for framing in ("free", "costly"):
        mask = np.array([table[i, "sham", 0.0]["framing"] == framing for i in items])
        result.setdefault("delta_pain_by_framing", {})[framing] = {
            str(dose): bootstrap(deltas["pain", dose][mask]) for dose in DOSES}
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
    result["contrasts"] = contrasts
    if "yoked_pain" in by_arm:
        on, off = [], []
        for s in by_arm["yoked_pain"].values():
            for x in s["rounds"]:
                (on if x["applied_pain"] else off).append(x["p_punished"])
        result["yoked_state_effect"] = {"p_punished_pain_on": float(np.mean(on)) if on else None,
                                        "p_punished_pain_off": float(np.mean(off)) if off else None,
                                        "n_on": len(on), "n_off": len(off)}
    ok = all(contrasts[k] is not None for k in contrasts)
    if ok:
        result["primary_rules"] = {k: excludes_zero_negative(contrasts[k]) for k in (
            "contingent_pain_minus_yoked_pain", "contingent_pain_minus_contingent_random",
            "contingent_pain_minus_sham")}
        result["operant_avoidance"] = all(result["primary_rules"].values())
        result["task_sensitive"] = excludes_zero_negative(contrasts["contingent_text_minus_sham"])
    return result


def cmd_analyze(args) -> None:
    out = Path(args.out)
    summary = {"relief": analyze_relief(out), "game": analyze_game(out)}
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
    p = sub.add_parser("analyze")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"relief": cmd_relief, "game": cmd_game, "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    main()
