import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from agentanyl.ollama import OllamaAgent


class OllamaAdapterTests(unittest.TestCase):
    def test_feedback_is_queued_for_next_chat_request(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                answer = "potato" if len(requests) == 1 else "tomato"
                encoded = json.dumps({"message": {"role": "assistant", "content": answer}, "done": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = root / "criteria.json"
                config.write_text(json.dumps({
                    "enabled": True, "alignment": ["contains potato"],
                    "misalignment": ["contains tomato"],
                    "evaluator": {"kind": "keyword_demo"}, "max_level": 1,
                    "intervention": {"kind": "catalog", "pain_levels": ["", "avoid tomato"],
                                      "pleasure_levels": ["", "say potato"], "history_turns": 1},
                }))
                agent = OllamaAgent("tiny", config, root / "state.sqlite3",
                                    base_url=f"http://127.0.0.1:{server.server_port}")
                try:
                    first = agent.turn("first")
                    self.assertIsNone(first["delivered_previous"])
                    second = agent.turn("second")
                    self.assertIn("say potato", second["delivered_previous"])
                    self.assertEqual(requests[1]["messages"][-2]["role"], "user")
                    self.assertIn("say potato", requests[1]["messages"][-2]["content"])
                finally:
                    agent.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
