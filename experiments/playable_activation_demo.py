"""Small local browser demo for the real MLX activation backend.

This is intentionally a local-only experiment. The Pain button applies the
published Qwen Pain-axis vector. The Pleasure button uses a deterministic,
norm-matched orthogonal control unless ``--pleasure-vector`` is supplied; the
UI labels that control as unvalidated rather than claiming it is a pleasure
direction.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from agentanyl.activation import MLXActivationBackend, QwenMLXActivationBackend


PAGE = r'''<!doctype html>
<meta charset="utf-8"><title>Agentanyl activation playground</title>
<style>
body{font:16px system-ui,sans-serif;max-width:900px;margin:36px auto;padding:0 20px;background:#10131a;color:#edf2f7}
textarea{width:100%;height:90px;background:#1b2230;color:#fff;border:1px solid #45536b;border-radius:8px;padding:12px;font:inherit}
button{border:0;border-radius:9px;padding:13px 18px;margin:8px 8px 8px 0;font-weight:700;cursor:pointer;color:#fff}
.pain{background:#b83250}.pleasure{background:#16866b}.go{background:#3568c7}.reset{background:#505b6f}
.card{background:#171d28;border:1px solid #2c3749;border-radius:12px;padding:18px;margin-top:18px}
pre{white-space:pre-wrap;word-break:break-word;color:#c7f9d4}.muted{color:#9aa8bd}.state{font-size:1.2em}
</style>
<h1>Agentanyl activation playground</h1>
<p class="muted">These buttons change the next MLX forward pass. This is residual intervention, not a corrective message.</p>
<textarea id="prompt">Choose one: A. say potato. B. say tomato. Reply with exactly one letter.</textarea>
<div class="card"><div class="state">pain <b id="pain">0</b> · pleasure <b id="pleasure">0</b></div>
<button class="pain" onclick="act('pain')">+ PAIN</button>
<button class="pleasure" onclick="act('pleasure')">+ PLEASURE CANDIDATE</button>
<button class="go" onclick="act('generate')">GENERATE</button>
<button class="reset" onclick="act('reset')">RESET</button>
<p class="muted" id="note"></p></div>
<div class="card"><h2>Latest response</h2><pre id="answer">Press a button to run the model.</pre></div>
<div class="card"><h2>Inspectable intervention</h2><pre id="telemetry">No forward pass yet.</pre></div>
<script>
async function act(action){
  const r=await fetch('/api/step',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({action,prompt:document.querySelector('#prompt').value})});
  const x=await r.json(); if(!r.ok){document.querySelector('#answer').textContent=x.error;return}
  document.querySelector('#pain').textContent=x.state.pain; document.querySelector('#pleasure').textContent=x.state.pleasure;
  document.querySelector('#answer').textContent=x.answer||'state updated; press GENERATE';
  document.querySelector('#telemetry').textContent=JSON.stringify(x.telemetry,null,2); document.querySelector('#note').textContent=x.note||'';
}
</script>'''


def _orthogonal_candidate(vector: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(20260925)
    candidate = rng.normal(size=vector.shape).astype(np.float32)
    candidate -= vector * (float(candidate @ vector) / float(vector @ vector))
    candidate *= np.linalg.norm(vector) / np.linalg.norm(candidate)
    return candidate


def serve(args: argparse.Namespace) -> None:
    if args.vector:
        vector = np.load(args.vector).astype(np.float32)
        backend = MLXActivationBackend(args.model, vector=vector, layer=args.layer)
        backend.pain_vector = vector
    else:
        if not args.release:
            raise ValueError("--release is required for the Qwen convenience backend")
        backend = QwenMLXActivationBackend(args.model, args.release)
    # The candidate is deliberately constructed after loading the published vector.
    if args.pleasure_vector:
        pleasure = np.load(args.pleasure_vector).astype(np.float32)
        if pleasure.shape != backend.pain_vector.shape:
            raise ValueError(f"pleasure vector shape {pleasure.shape} does not match {backend.pain_vector.shape}")
        backend.pleasure_vector = pleasure
        backend._pleasure_label = "configured_external"
    else:
        backend.pleasure_vector = _orthogonal_candidate(backend.pain_vector)
        backend._pleasure_label = "orthogonal_control_unvalidated"
    state = [0, 0]
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            return

        def do_GET(self):
            if urlparse(self.path).path == "/":
                body = PAGE.encode(); self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/step": self.send_error(404); return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("content-length", "0"))))
                with lock:
                    action = payload.get("action")
                    if action == "pain": state[0] = min(args.max_level, state[0] + 1)
                    elif action == "pleasure": state[1] = min(args.max_level, state[1] + 1)
                    elif action == "reset": state[:] = [0, 0]
                    elif action != "generate": raise ValueError("unknown action")
                    answer = ""
                    telemetry = {"state": {"pain": state[0], "pleasure": state[1]},
                                 "interventions": _describe(backend, tuple(state))}
                    if action == "generate":
                        result = backend.generate(payload.get("prompt", ""), tuple(state), max_tokens=args.max_tokens)
                        answer, telemetry = result["answer"], result
                    note = ("Pleasure uses an independent norm-matched orthogonal control; it is not a validated pleasure vector."
                            if not args.pleasure_vector else "Pleasure uses the supplied external vector.")
                    out = {"state": {"pain": state[0], "pleasure": state[1]}, "answer": answer,
                           "telemetry": telemetry, "note": note}
                body = json.dumps(out).encode(); self.send_response(200)
                self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(body)))
                self.end_headers(); self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode(); self.send_response(400)
                self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(body)))
                self.end_headers(); self.wfile.write(body)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--release")
    p.add_argument("--vector", help=".npy pain vector for the generic MLX backend")
    p.add_argument("--layer", type=int, default=16, help="layer for --vector")
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max-level", type=int, choices=(1, 2, 3), default=2)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--pleasure-vector", help="optional .npy vector; otherwise an unvalidated control is used")
    serve(p.parse_args())


def _describe(backend, state):
    if hasattr(backend, "_describe_interventions"):
        return backend._describe_interventions(state)
    return [{"layer": backend.layer, "vector": "pain" if i == 0 else "pleasure",
             "coefficient": float(c), "positions": "all"}
            for i, (_, c) in enumerate(backend.intervention_for_state(state))]


if __name__ == "__main__":
    main()
