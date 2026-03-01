"""
Derive — Visualization server for LightGCN on Gowalla.

Serves the browser app and provides an API to look up any user's
geographic check-in data, plus a Mistral-powered agentic chat.

Usage:
    cd flaneur
    uv run python derive/server.py

API:
    GET /                           -> main app (index.html)
    GET /api/models                 -> available models with metadata
    GET /api/user/{id}?model=name   -> geo data for a LightGCN user (optional model)
    GET /api/random?model=name      -> geo data for a random user
    GET /api/stats                  -> dataset summary
    GET /api/report/{id}            -> user report data
    GET /api/leaderboard            -> ranked users
    GET /api/heatmap                -> global check-in heatmap
    POST /api/chat                  -> agentic chat (SSE stream)
"""

import http.server
import json
import os
import random
import socketserver
import sys
from pathlib import Path

try:
    import weave
    from dotenv import load_dotenv
    from mistralai import Mistral

    _HAS_MISTRAL = True
except ImportError:
    _HAS_MISTRAL = False

# Add parent so we can import derive.lib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from derive.lib import gowalla
from derive.lib.agent_tools import TOOL_DEFINITIONS, execute_tool, _exec_generate_report

if _HAS_MISTRAL:
    load_dotenv()

PORT = 8765
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

MISTRAL_MODEL = "mistral-medium-latest"
MAX_TOOL_ROUNDS = 8


def build_system_prompt(current_user: int | None, gowalla_data) -> str:
    """Build the agent system prompt with current user context."""
    template = (PROMPTS_DIR / "agent_system.txt").read_text()

    context = ""
    if current_user is not None:
        data = gowalla_data.get_user_geo(current_user)
        if data:
            context = (
                f"\nCurrently viewing User #{current_user} "
                f"({data['label']}, {len(data['history'])} history, "
                f"{len(data['ground_truth'])} ground truth, "
                f"{len(data.get('predictions', []))} predictions, "
                f"spread={data['spread']})."
            )

    return template.replace("{current_user_context}", context)


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

    def do_POST(self):
        if self.path == "/api/chat":
            self._handle_chat()
        else:
            self._json(404, {"error": "Unknown endpoint"})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

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

        if path.startswith("/api/report/"):
            try:
                uid = int(path.split("/")[-1])
            except ValueError:
                return self._json(400, {"error": "Invalid user ID"})
            result, actions = _exec_generate_report({"uid": uid}, gd)
            if "error" in result:
                return self._json(404, result)
            report_data = next((a["data"] for a in actions if a.get("name") == "report"), None)
            if report_data is None:
                return self._json(500, {"error": "Report generation failed"})
            return self._json(200, report_data)

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

        elif path == "/api/leaderboard":
            return self._json(200, self._build_leaderboard(gd))

        elif path == "/api/heatmap":
            return self._json(200, self._build_heatmap(gd))

        else:
            return self._json(404, {"error": "Unknown endpoint"})

    # ── Agentic chat endpoint ─────────────────────────────────────────────

    def _handle_chat(self):
        """Agent loop: tool-calling rounds via complete(), then stream() for final text."""
        gd = self.gowalla_data

        # Parse request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            request = json.loads(body)
        except json.JSONDecodeError:
            return self._json(400, {"error": "Invalid JSON"})

        messages = request.get("messages", [])
        current_user = request.get("current_user")

        if not messages:
            return self._json(400, {"error": "No messages provided"})

        # Check Mistral availability
        if not _HAS_MISTRAL:
            return self._sse_error("mistralai not installed. Run: uv pip install mistralai")

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            return self._sse_error("MISTRAL_API_KEY not set")

        # Build system prompt
        system_prompt = build_system_prompt(current_user, gd)

        # Prepare messages for Mistral
        mistral_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            mistral_messages.append({"role": msg["role"], "content": msg["content"]})

        # Start SSE response — close connection when done so the client
        # ReadableStream sees the end of the body.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            client = Mistral(api_key=api_key)

            # ── Agent loop: non-streaming tool-calling rounds ──────────────
            for _round in range(MAX_TOOL_ROUNDS):
                response = client.chat.complete(
                    model=MISTRAL_MODEL,
                    messages=mistral_messages,
                    tools=TOOL_DEFINITIONS,
                    temperature=0.3,
                )

                choice = response.choices[0]
                msg = choice.message

                # If model wants to call tools
                if msg.tool_calls:
                    # Append the assistant message with tool calls
                    mistral_messages.append(msg)

                    for tc in msg.tool_calls:
                        fn_name = tc.function.name or ""
                        call_id = tc.id

                        # Parse arguments robustly
                        raw_args = tc.function.arguments
                        if isinstance(raw_args, str):
                            try:
                                fn_args = json.loads(raw_args)
                            except (json.JSONDecodeError, TypeError):
                                fn_args = {}
                        elif isinstance(raw_args, dict):
                            fn_args = raw_args
                        else:
                            fn_args = {}

                        # Clean display name for the UI badge
                        display_name = fn_name
                        if fn_name in ("fly_to", "select_user", "fit_bounds"):
                            display_name = "navigate_map"
                        elif fn_name.startswith("{"):
                            display_name = "navigate_map"

                        # Stream tool_call event
                        self._sse_event({
                            "type": "tool_call",
                            "name": display_name,
                            "args": fn_args,
                            "call_id": call_id,
                        })

                        # Execute tool (has its own fallback parsing)
                        result, actions = execute_tool(fn_name, fn_args, gd)

                        # Stream tool_result event
                        summary = result.pop("_summary", "done")
                        is_error = "error" in result
                        self._sse_event({
                            "type": "tool_result",
                            "name": display_name,
                            "call_id": call_id,
                            "summary": summary,
                            "error": is_error,
                        })

                        # Stream any map actions / widgets / reports
                        for action in actions:
                            if action.get("name") == "report":
                                self._sse_event({"type": "report", "data": action["data"]})
                            elif action.get("name") == "widget":
                                self._sse_event({"type": "widget", **action})
                            else:
                                self._sse_event({"type": "action", **action})

                        # Append tool result to messages
                        mistral_messages.append({
                            "role": "tool",
                            "name": fn_name,
                            "content": json.dumps(result),
                            "tool_call_id": call_id,
                        })

                    # Continue loop for next round of tool calls
                    continue

                # No tool calls — model wants to give a text response
                # Switch to streaming for the final answer
                break

            # ── Final streaming response (no tools) ────────────────────────
            stream = client.chat.stream(
                model=MISTRAL_MODEL,
                messages=mistral_messages,
                max_tokens=512,
                temperature=0.3,
            )

            for event in stream:
                token = event.data.choices[0].delta.content
                if token:
                    self._sse_event({"type": "token", "content": token})

            self._sse_event({"type": "done"})

        except Exception as e:
            self._sse_event({"type": "error", "message": str(e)})
            self._sse_event({"type": "done"})

    def _sse_event(self, data: dict):
        """Write a single SSE event."""
        chunk = json.dumps(data)
        self.wfile.write(f"data: {chunk}\n\n".encode())
        self.wfile.flush()

    def _sse_error(self, message: str):
        """Send an error as SSE and close."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self._sse_event({"type": "error", "message": message})
        self._sse_event({"type": "done"})

    def _build_leaderboard(self, gd):
        """Build leaderboard data: top users by hit rate, check-ins, spread."""
        from derive.lib.agent_tools import _user_summary

        by_hits = []
        by_checkins = []
        by_spread = []

        # Multi-model: iterate default model's predictions
        model_preds = gd.predictions.get(gd.default_model, {}) if gd.default_model else {}
        for uid in model_preds:
            s = _user_summary(uid, gd)
            if s is None:
                continue
            by_hits.append(s)

        for uid in random.sample(list(gd.train_dict.keys()), min(2000, len(gd.train_dict))):
            s = _user_summary(uid, gd)
            if s is None:
                continue
            by_checkins.append(s)
            by_spread.append(s)

        by_hits.sort(key=lambda u: u["hits"], reverse=True)
        by_checkins.sort(key=lambda u: u["history_count"], reverse=True)
        by_spread.sort(key=lambda u: float(u["spread"]), reverse=True)

        return {
            "best_hit_rate": by_hits[:15],
            "most_checkins": by_checkins[:15],
            "globetrotters": by_spread[:15],
        }

    def _build_heatmap(self, gd):
        """Build heatmap data: sampled check-in coordinates with weights."""
        from collections import Counter

        loc_counts: Counter = Counter()
        sample_uids = random.sample(
            list(gd.train_dict.keys()),
            min(3000, len(gd.train_dict)),
        )
        for uid in sample_uids:
            for item_id in gd.train_dict[uid]:
                org_loc = gd.item_remap_to_org.get(item_id)
                if org_loc and org_loc in gd.loc_coords:
                    loc_counts[org_loc] += 1

        points = []
        for loc_id, count in loc_counts.items():
            lat, lon = gd.loc_coords[loc_id]
            points.append([round(lat, 4), round(lon, 4), count])

        return {"points": points, "count": len(points)}

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

    if _HAS_MISTRAL:
        weave.init("igorlima1740/flaneur")

    handler = lambda *args, **kwargs: DeriveHandler(*args, gowalla_data=gd, **kwargs)

    server = ThreadingHTTPServer(("", PORT), handler)
    print(f"\n  Derive is running at http://localhost:{PORT}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
