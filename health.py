import os
from http.server import HTTPServer, BaseHTTPRequestHandler


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


port = int(os.getenv("PORT", "8000"))
print(f"Health :{port}")
HTTPServer(("0.0.0.0", port), H).serve_forever()
