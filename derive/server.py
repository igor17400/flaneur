"""
Derive — Visualization server for LightGCN on Gowalla.

Serves the browser app and provides an API to look up any user's
geographic check-in data.

Usage:
    cd flaneur_main_hackatho
    python3 derive/server.py

API:
    GET /                → main app (index.html)
    GET /api/user/{id}   → geo data for a LightGCN user
    GET /api/random      → geo data for a random user
    GET /api/stats       → dataset summary
"""

import json
import http.server
import random
import sys
from pathlib import Path

# Add parent so we can import derive.lib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from derive.lib import gowalla

PORT = 8765
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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
            self.path = self.path[len("/static"):]
            super().do_GET()
        elif self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def _handle_api(self):
        gd = self.gowalla_data

        if self.path.startswith("/api/user/"):
            try:
                uid = int(self.path.split("/")[-1])
            except ValueError:
                return self._json(400, {"error": "Invalid user ID"})
            data = gd.get_user_geo(uid)
            if data is None:
                return self._json(404, {"error": f"User {uid} not found"})
            return self._json(200, data)

        elif self.path == "/api/random":
            uid = random.choice(list(gd.train_dict.keys()))
            data = gd.get_user_geo(uid)
            return self._json(200, {"uid": uid, **(data or {})})

        elif self.path == "/api/stats":
            return self._json(200, {
                "n_users": gd.n_users,
                "n_items": gd.n_items,
                "user_id_range": [0, max(gd.train_dict.keys())],
            })

        else:
            return self._json(404, {"error": "Unknown endpoint"})

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


def main():
    print("Derive — loading Gowalla data...")
    gd = gowalla.load(DATA_DIR)

    handler = lambda *args, **kwargs: DeriveHandler(*args, gowalla_data=gd, **kwargs)

    server = http.server.HTTPServer(("", PORT), handler)
    print(f"\n  Derive is running at http://localhost:{PORT}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
