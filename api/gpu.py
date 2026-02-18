import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from urllib.request import urlopen, Request


H200_API = "http://146.88.194.12:8080"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        server_key = params.get("server", [None])[0]
        endpoint = params.get("endpoint", ["data"])[0]

        # Special cases: endpoints that don't need a server_key
        if endpoint == "agent":
            upstream = f"{H200_API}/api/agent"
        elif endpoint == "analytics":
            upstream = f"{H200_API}/api/analytics"
        elif endpoint == "proactive":
            upstream = f"{H200_API}/api/proactive"
        elif server_key not in ("h200", "rtx5090"):
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Use ?server=h200 or ?server=rtx5090 or ?endpoint=agent"}).encode())
            return
        elif endpoint == "history":
            upstream = f"{H200_API}/api/{server_key}/history"
        elif endpoint == "software":
            upstream = f"{H200_API}/api/{server_key}/software"
        elif endpoint == "daily":
            upstream = f"{H200_API}/api/{server_key}/daily"
        else:
            upstream = f"{H200_API}/api/{server_key}"

        try:
            req = Request(upstream, headers={"User-Agent": "vercel-proxy"})
            resp = urlopen(req, timeout=20)
            data = resp.read()
            self.send_response(200)
        except Exception as e:
            data = json.dumps({"error": str(e), "status": "offline"}).encode()
            self.send_response(502)

        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)
