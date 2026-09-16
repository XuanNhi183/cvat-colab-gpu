"""Docker-network-only relay; credentials stay out of CVAT's settings."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


MAX_BODY = 32 * 1024 * 1024


class Bridge(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def forward(self):
        valid_path = self.path in {"/healthz", "/api/functions", "/api/function_invocations"}
        if self.path.startswith("/api/functions/"):
            name = self.path[len("/api/functions/"):]
            valid_path = bool(name) and all(c.isalnum() or c in "-_." for c in name)
        if not valid_path:
            return self.reply(404, b'{"error":"Unknown endpoint"}')
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.reply(400, b'{"error":"Invalid body length"}')
        if size < 0 or size > MAX_BODY:
            return self.reply(413, b'{"error":"Image payload exceeds 32 MiB"}')
        headers = {"Authorization": "Bearer " + self.server.token,
                   "Content-Type": "application/json", "ngrok-skip-browser-warning": "1"}
        for key in ("x-nuclio-function-name", "x-nuclio-path", "x-nuclio-function-namespace"):
            if self.headers.get(key):
                headers[key] = self.headers[key]
        body = self.rfile.read(size) if self.command == "POST" else None
        request = Request(self.server.upstream + self.path, data=body, headers=headers, method=self.command)
        try:
            with urlopen(request, timeout=180) as response:
                self.reply(response.status, response.read(), response.headers.get("Content-Type", "application/json"))
        except HTTPError as error:
            self.reply(error.code, error.read(), error.headers.get("Content-Type", "application/json"))
        except (URLError, TimeoutError, OSError):
            self.reply(502, json.dumps({"error": "Colab unavailable. Check runtime, tunnel URL and token."}).encode())

    do_GET = forward
    do_POST = forward


if __name__ == "__main__":
    upstream = os.environ["COLAB_URL"].rstrip("/")
    parsed = urlsplit(upstream)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("COLAB_URL must be an HTTPS origin without path or credentials")
    token = os.environ["COLAB_TOKEN"]
    if len(token) < 32:
        raise ValueError("Missing or invalid COLAB_TOKEN")
    server = ThreadingHTTPServer(("0.0.0.0", 8070), Bridge)
    server.upstream, server.token = upstream, token
    server.serve_forever()
