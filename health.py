import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class H(BaseHTTPRequestHandler):
    # HTTP/1.1 keep-alive + an explicit timeout: without a timeout a client that
    # opens a connection and never sends wedges the whole server, and the
    # orchestrator's probe then hangs instead of probing anything.
    protocol_version = "HTTP/1.1"
    timeout = 10

    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


port = int(os.getenv("PORT", "8000"))
server = ThreadingHTTPServer(("0.0.0.0", port), H)
server.daemon_threads = True
print(f"Health :{port}")


def _shutdown(signum, frame):
    # serve_forever() has no exit path of its own; without a signal handler the
    # process is killed by the kernel on SIGTERM.
    threading.Thread(target=server.shutdown, daemon=True).start()


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)

try:
    server.serve_forever(poll_interval=1.0)
finally:
    server.server_close()
