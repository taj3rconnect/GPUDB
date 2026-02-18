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
        elif endpoint == "capacity":
            upstream = f"{H200_API}/api/capacity"
        elif endpoint == "quality":
            upstream = f"{H200_API}/api/quality"
        elif endpoint == "network":
            upstream = f"{H200_API}/api/network"
        elif endpoint == "executive":
            upstream = f"{H200_API}/api/executive"
        elif endpoint == "livecalls":
            upstream = f"{H200_API}/api/livecalls"
        elif endpoint == "modelcompare":
            upstream = f"{H200_API}/api/modelcompare"
        elif endpoint == "power":
            upstream = f"{H200_API}/api/power"
        elif endpoint == "incidents":
            upstream = f"{H200_API}/api/incidents"
        elif endpoint == "alerts-config":
            upstream = f"{H200_API}/api/alerts-config"
        elif endpoint == "sla":
            upstream = f"{H200_API}/api/sla"
        elif endpoint == "email-config":
            upstream = f"{H200_API}/api/email-config"
        elif endpoint == "email-test":
            upstream = f"{H200_API}/api/email-test"
        elif endpoint == "anomalies":
            upstream = f"{H200_API}/api/anomalies"
        elif endpoint == "servers":
            upstream = f"{H200_API}/api/servers"
        elif endpoint == "report":
            upstream = f"{H200_API}/api/report"
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

    def do_POST(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        endpoint = params.get("endpoint", [""])[0]

        post_endpoints = {"alerts-config": "/api/alerts-config", "sla": "/api/sla", "email-config": "/api/email-config", "email-test": "/api/email-test", "servers": "/api/servers", "report": "/api/report"}
        if endpoint not in post_endpoints:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid POST endpoint"}).encode())
            return

        upstream = f"{H200_API}{post_endpoints[endpoint]}"
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        try:
            req = Request(upstream, data=body, headers={"User-Agent": "vercel-proxy", "Content-Type": "application/json"}, method="POST")
            resp = urlopen(req, timeout=20)
            data = resp.read()
            self.send_response(200)
        except Exception as e:
            data = json.dumps({"error": str(e)}).encode()
            self.send_response(502)

        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
