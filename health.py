import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_READY_CHECK = None


def set_ready_check(fn):
    global _READY_CHECK
    _READY_CHECK = fn


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 10

    def _ready(self):
        if _READY_CHECK is None:
            return True
        try:
            return bool(_READY_CHECK())
        except Exception:
            return False

    def do_GET(self):
        if self.path not in ("/", "/health", "/healthz", "/ready"):
            self.send_response(404)
            self.end_headers()
            return
        ready = self._ready()
        body = (b'{"status":"ok"}' if ready
                else b'{"status":"not_ready"}')
        self.send_response(200 if ready else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        ready = self._ready()
        self.send_response(200 if ready else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


def run_health_server():
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), H)
    server.daemon_threads = True
    print(f"Health :{port}")

    def _shutdown(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        server.serve_forever(poll_interval=1.0)
    finally:
        server.server_close()


if __name__ == "__main__":
    run_health_server()
