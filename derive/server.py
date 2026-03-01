"""
Derive — Visualization server for LightGCN on Gowalla.

Serves the browser app and provides an API to look up any user's
geographic check-in data, plus a Mistral-powered streaming explainer.

Usage:
    cd flaneur_main_hackatho
    uv run python derive/server.py

API:
    GET /                           -> main app (index.html)
    GET /api/models                 -> available models with metadata
    GET /api/user/{id}?model=name   -> geo data for a LightGCN user (optional model)
    GET /api/random?model=name      -> geo data for a random user
    GET /api/stats                  -> dataset summary
    GET /api/explain/{id}           -> SSE stream of Mistral explanation
"""

import http.server
import json
import os
import random
import socketserver
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    from mistralai import Mistral

    _HAS_MISTRAL = True
except ImportError:
    _HAS_MISTRAL = False

# Add parent so we can import derive.lib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from derive.lib import gowalla

if _HAS_MISTRAL:
    load_dotenv()

PORT = 8765
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

MISTRAL_MODEL = "mistral-medium-latest"


def build_explain_prompt(uid: int, data: dict) -> str:
    """Build a geographic analysis prompt from user data using the template."""
    history = data.get("history", [])
    ground_truth = data.get("ground_truth", [])
    predictions = data.get("predictions", [])
    label = data.get("label", "Unknown")
    spread = data.get("spread", 0)

    gt_ids = {p["item_id"] for p in ground_truth}
    pred_ids = [p["item_id"] for p in predictions]
    hits = sum(1 for pid in pred_ids if pid in gt_ids)

    recent_history = history[-30:]

    history_str = "\n".join(
        f"  ({p['lat']:.4f}, {p['lon']:.4f}) @ {p.get('ts', 'unknown')}"
        for p in recent_history
    )

    gt_str = "\n".join(
        f"  ({p['lat']:.4f}, {p['lon']:.4f}) item={p['item_id']}" for p in ground_truth
    ) or "  (no ground truth available)"

    pred_str = "\n".join(
        f"  ({p['lat']:.4f}, {p['lon']:.4f}) item={p['item_id']}"
        for p in predictions
    ) or "  (no model predictions available)"

    template = (PROMPTS_DIR / "explain_user.txt").read_text()
    return template.format(
        uid=uid,
        label=label,
        spread=spread,
        n_history=len(history),
        n_gt=len(ground_truth),
        n_preds=len(predictions),
        hits=hits,
        n_shown=len(recent_history),
        history_str=history_str,
        gt_str=gt_str,
        pred_str=pred_str,
    )


class DeriveHandler(http.server.SimpleHTTPRequestHandler):
    """Routes API requests and serves static files."""

    def __init__(self, *args, gowalla_data=None, **kwargs):
        self.gowalla_data = gowalla_data
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._handle_api()
        elif self.path.startswith("/static/"):
            # Serve from STATIC_DIR but strip the /static prefix
            self.path = self.path[len("/static") :]
            super().do_GET()
        elif self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def _parse_path(self):
        """Split self.path into path and query params."""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        # Flatten single-value params
        flat = {k: v[0] for k, v in params.items()}
        return parsed.path, flat

    def _handle_api(self):
        gd = self.gowalla_data
        path, params = self._parse_path()
        model = params.get("model")

        if path.startswith("/api/explain/"):
            try:
                uid = int(path.split("/")[-1])
            except ValueError:
                return self._json(400, {"error": "Invalid user ID"})
            data = gd.get_user_geo(uid, model=model)
            if data is None:
                return self._json(404, {"error": f"User {uid} not found"})
            if not data.get("predictions"):
                return self._json(400, {"error": "No predictions available for this user. Run inference first."})
            return self._stream_explain(uid, data)

        elif path.startswith("/api/user/"):
            try:
                uid = int(path.split("/")[-1])
            except ValueError:
                return self._json(400, {"error": "Invalid user ID"})
            data = gd.get_user_geo(uid, model=model)
            if data is None:
                return self._json(404, {"error": f"User {uid} not found"})
            return self._json(200, data)

        elif path == "/api/random":
            uid = random.choice(list(gd.train_dict.keys()))
            data = gd.get_user_geo(uid, model=model)
            return self._json(200, {"uid": uid, **(data or {})})

        elif path == "/api/models":
            models = []
            for name in gd.available_models:
                meta = gd.prediction_meta.get(name, {})
                models.append({
                    "name": name,
                    "embed_dim": meta.get("embed_dim"),
                    "n_layers": meta.get("n_layers"),
                    "lr": meta.get("lr"),
                    "reg_weight": meta.get("reg_weight"),
                    "val_recall_at_20": meta.get("val_recall_at_20"),
                    "val_ndcg_at_20": meta.get("val_ndcg_at_20"),
                    "is_default": name == gd.default_model,
                })
            return self._json(200, {"models": models, "default": gd.default_model})

        elif path == "/api/stats":
            return self._json(
                200,
                {
                    "n_users": gd.n_users,
                    "n_items": gd.n_items,
                    "n_models": len(gd.available_models),
                    "default_model": gd.default_model,
                    "user_id_range": [0, max(gd.train_dict.keys())],
                },
            )

        else:
            return self._json(404, {"error": "Unknown endpoint"})

    def _stream_explain(self, uid: int, data: dict):
        """Stream Mistral explanation as SSE."""
        if not _HAS_MISTRAL:
            self.send_response(500)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(
                b'data: {"error": "mistralai not installed. Run: uv run python derive/server.py"}\n\n'
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            self.send_response(500)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'data: {"error": "MISTRAL_API_KEY not set"}\n\n')
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        prompt = build_explain_prompt(uid, data)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            client = Mistral(api_key=api_key)
            stream = client.chat.stream(
                model=MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.4,
            )

            for event in stream:
                token = event.data.choices[0].delta.content
                if token:
                    chunk = json.dumps({"token": token})
                    self.wfile.write(f"data: {chunk}\n\n".encode())
                    self.wfile.flush()

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        except Exception as e:
            error_msg = json.dumps({"error": str(e)})
            self.wfile.write(f"data: {error_msg}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Only log API requests, not static file noise
        if args and "/api/" in str(args[0]):
            super().log_message(fmt, *args)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Handle each request in a new thread so SSE doesn't block the map."""

    daemon_threads = True


def main():
    print("Derive — loading Gowalla data...")
    gd = gowalla.load(DATA_DIR)

    handler = lambda *args, **kwargs: DeriveHandler(*args, gowalla_data=gd, **kwargs)

    server = ThreadingHTTPServer(("", PORT), handler)
    print(f"\n  Derive is running at http://localhost:{PORT}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
