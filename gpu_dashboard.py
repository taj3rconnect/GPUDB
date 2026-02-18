import asyncio
import json
import os
import time
import re
import ssl
from collections import deque
from pathlib import Path
from urllib.request import urlopen, Request as URLRequest
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncssh

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Server Config ──────────────────────────────────────────────
SERVERS = {
    "h200": {
        "name": "H200 Cluster",
        "host": "146.88.194.12",
        "user": "ubuntu",
        "port": 22,
        "gpu_model": "NVIDIA H200",
        "has_ecc": True,
        "vllm_ports": [8001, 8002],
        "local": True,
        "venvs": {
            "env_llm (primary)": "/home/ubuntu/env_llm/bin/pip",
            "env (secondary)": "/home/ubuntu/env/bin/pip",
        },
        "services": ["vllm-llm", "vllm-tts"],
        "agent_ports": [],
    },
    "rtx5090": {
        "name": "RTX 5090 Cluster",
        "host": "38.65.239.47",
        "user": "ubuntu",
        "port": 22,
        "gpu_model": "NVIDIA GeForce RTX 5090",
        "has_ecc": False,
        "vllm_ports": [],
        "dcgm_port": 9400,
        "venvs": {
            "Agentv2-Old": "/home/ubuntu/.agent/Agentv2-Old/env/bin/pip",
        },
        "services": ["agentv2-main-server", "agentv2-voicemail-processor", "agentv2-failed-report", "agentv2-health-checker", "agentv2-server-old", "agent"],
        "agent_ports": [8003],
    },
}

SSH_KEY = str(Path.home() / ".ssh" / "id_ed25519")

# ── History Store (in-memory, per server) ──────────────────────
MAX_HISTORY = 1440  # 24hrs at 1-min intervals, or ~72min at 3s
history = {k: deque(maxlen=MAX_HISTORY) for k in SERVERS}

# Hourly aggregation: {server: {hour_key: {max_calls, max_util, calls_at_max_util, samples}}}
hourly_stats = {k: {} for k in SERVERS}
HOURLY_RETENTION = 25  # keep 25 hours to cover full 24h window

# ── Proactive Alerting Stores ─────────────────────────────────
# Error tracking: {server: deque of {t, errors, total, rate}}
error_history = {k: deque(maxlen=MAX_HISTORY) for k in SERVERS}
# Zombie/memory tracking: deque of {t, zombies, gunicorn_rss_mb, gunicorn_workers, voicemail_workers}
process_health_history = deque(maxlen=MAX_HISTORY)
# Service uptime tracking: {service_name: {status, last_change_t, uptime_samples, total_samples, incidents: []}}
service_uptime = {}
# SSH/fetch error log: deque of {t, server, endpoint, error}
fetch_errors = deque(maxlen=200)
# Inter-cluster latency tracking: deque of {t, rtt_ms}
cluster_latency = deque(maxlen=MAX_HISTORY)
# Disk usage tracking: {server: deque of {t, used_bytes, total_bytes, pct}}
disk_history = {k: deque(maxlen=480) for k in SERVERS}  # ~24h at 3min intervals
# GPU clock tracking for throttle detection: {server: {gpu_idx: deque of {t, sm_clock, temp}}}
gpu_clock_history = {k: {} for k in SERVERS}
# Call volume by hour-of-day for forecasting: {dow_hour: [call_counts]}
call_volume_patterns = {}

# ── Email Alert Config ────────────────────────────────────────
# Load SendGrid credentials from .env file
def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_MAIL_FROM = os.environ.get("SENDGRID_MAIL_FROM", "no-reply@jobtalk.ai")

email_config = {
    "enabled": True,
    "recipients": ["taj@jobtalk.ai"],
    "cooldown_minutes": 30,
}
email_cooldowns = {}  # {alert_key: last_sent_timestamp}
email_history = deque(maxlen=100)  # {t, to, subject, alerts_count, status, detail}


# ── Command Helpers ────────────────────────────────────────────
async def run_local_command(command: str) -> str:
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    return stdout.decode("utf-8", errors="replace")


async def run_ssh_command(server: dict, command: str) -> str:
    conn = await asyncssh.connect(
        server["host"],
        port=server["port"],
        username=server["user"],
        known_hosts=None,
        client_keys=[SSH_KEY],
    )
    async with conn:
        result = await conn.run(command, check=False)
        return result.stdout or ""


async def run_command(server: dict, command: str) -> str:
    if server.get("local"):
        return await run_local_command(command)
    return await run_ssh_command(server, command)


# ── Prometheus Parser ──────────────────────────────────────────
def parse_prom_metrics(text: str) -> dict:
    """Parse Prometheus text format into {metric_name: [{labels, value}]}"""
    metrics = {}
    for line in text.split("\n"):
        if not line or line.startswith("#"):
            continue
        # metric_name{labels} value
        m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?([^}]*)\}?\s+([\d.eE+\-]+|NaN|Inf|-Inf)$', line)
        if not m:
            # metric without labels
            m2 = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([\d.eE+\-]+|NaN|Inf|-Inf)$', line)
            if m2:
                name, val = m2.group(1), m2.group(2)
                metrics.setdefault(name, []).append({"labels": {}, "value": float(val) if val not in ("NaN", "Inf", "-Inf") else 0})
            continue
        name, labels_str, val = m.group(1), m.group(2), m.group(3)
        labels = {}
        if labels_str:
            for lm in re.finditer(r'(\w+)="([^"]*)"', labels_str):
                labels[lm.group(1)] = lm.group(2)
        fval = float(val) if val not in ("NaN", "Inf", "-Inf") else 0
        metrics.setdefault(name, []).append({"labels": labels, "value": fval})
    return metrics


def get_metric_value(metrics: dict, name: str, label_filter: dict = None) -> float:
    """Get a single metric value, optionally filtering by labels."""
    entries = metrics.get(name, [])
    for e in entries:
        if label_filter:
            if all(e["labels"].get(k) == v for k, v in label_filter.items()):
                return e["value"]
        else:
            return e["value"]
    return 0


def sum_metric(metrics: dict, name: str, label_key: str = None, label_value: str = None) -> float:
    entries = metrics.get(name, [])
    total = 0
    for e in entries:
        if label_key and label_value:
            if e["labels"].get(label_key) != label_value:
                continue
        total += e["value"]
    return total


# ── vLLM Metrics Parser ───────────────────────────────────────
def parse_vllm_metrics(raw: str, port: int) -> dict:
    m = parse_prom_metrics(raw)
    if not m:
        return None

    # Determine model name
    model_name = "unknown"
    for entry in m.get("vllm:request_prompt_tokens_count", m.get("vllm:num_requests_running", [])):
        if "model_name" in entry.get("labels", {}):
            model_name = entry["labels"]["model_name"]
            break

    # Request counts
    total_requests = 0
    for entry in m.get("http_request_size_bytes_count", []):
        if entry["labels"].get("handler") == "/v1/chat/completions":
            total_requests += entry["value"]
    # Fallback to vllm counters
    if total_requests == 0:
        total_requests = sum_metric(m, "vllm:request_prompt_tokens_count")

    # Tokens
    prompt_tokens = sum_metric(m, "vllm:request_prompt_tokens_sum")
    gen_tokens = sum_metric(m, "vllm:request_generation_tokens_sum")

    # Latencies (from histograms - use sum/count for averages)
    ttft_sum = sum_metric(m, "vllm:time_to_first_token_seconds_sum")
    ttft_count = sum_metric(m, "vllm:time_to_first_token_seconds_count")
    itl_sum = sum_metric(m, "vllm:inter_token_latency_seconds_sum")
    itl_count = sum_metric(m, "vllm:inter_token_latency_seconds_count")
    e2e_sum = sum_metric(m, "vllm:e2e_request_latency_seconds_sum")
    e2e_count = sum_metric(m, "vllm:e2e_request_latency_seconds_count")
    queue_sum = sum_metric(m, "vllm:request_queue_time_seconds_sum")
    queue_count = sum_metric(m, "vllm:request_queue_time_seconds_count")

    # Running / waiting
    running = sum_metric(m, "vllm:num_requests_running")
    waiting = sum_metric(m, "vllm:num_requests_waiting")

    # KV cache
    kv_entries = m.get("vllm:kv_cache_usage_perc", [])
    kv_usage = max((e["value"] for e in kv_entries), default=0)

    # Cache hits
    cache_queries = sum_metric(m, "vllm:prefix_cache_queries_total")
    cache_hits = sum_metric(m, "vllm:prefix_cache_hits_total")

    # Success count
    success = sum_metric(m, "vllm:request_success_total")

    return {
        "port": port,
        "model_name": model_name,
        "total_requests": int(total_requests),
        "requests_running": int(running),
        "requests_waiting": int(waiting),
        "prompt_tokens": int(prompt_tokens),
        "generation_tokens": int(gen_tokens),
        "avg_ttft_ms": (ttft_sum / ttft_count * 1000) if ttft_count > 0 else 0,
        "avg_itl_ms": (itl_sum / itl_count * 1000) if itl_count > 0 else 0,
        "avg_e2e_s": (e2e_sum / e2e_count) if e2e_count > 0 else 0,
        "avg_queue_s": (queue_sum / queue_count) if queue_count > 0 else 0,
        "kv_cache_usage": round(kv_usage * 100, 2),
        "cache_hit_rate": round((cache_hits / cache_queries * 100) if cache_queries > 0 else 0, 2),
        "success_count": int(success),
    }


# ── Data Fetcher ───────────────────────────────────────────────
async def fetch_gpu_data(server_key: str) -> dict:
    server = SERVERS[server_key]

    # nvidia-smi GPU query
    gpu_query = (
        "nvidia-smi --query-gpu=index,name,temperature.gpu,fan.speed,"
        "power.draw,power.limit,memory.used,memory.total,"
        "utilization.gpu,utilization.memory"
    )
    if server["has_ecc"]:
        gpu_query += ",ecc.errors.uncorrected.volatile.total"
    gpu_query += " --format=csv,noheader,nounits"

    proc_query = "nvidia-smi --query-compute-apps=pid,name,gpu_bus_id,used_memory --format=csv,noheader,nounits 2>/dev/null"
    sys_query = "cat /proc/uptime; echo '---'; free -b | head -2; echo '---'; df -B1 / | tail -1"

    # Network throughput (bytes from /proc/net/dev)
    net_query = "cat /proc/net/dev"

    # vLLM metrics
    vllm_cmds = []
    for port in server.get("vllm_ports", []):
        vllm_cmds.append(f"curl -s --max-time 3 localhost:{port}/metrics 2>/dev/null")

    # Build combined metrics command
    vllm_combined = ""
    if vllm_cmds:
        parts = []
        for i, cmd in enumerate(vllm_cmds):
            parts.append(f"echo '===VLLM_PORT_{server['vllm_ports'][i]}==='; {cmd}")
        vllm_combined = " && ".join(parts)

    # Active connections on agent ports (for concurrent call tracking)
    agent_ports = server.get("agent_ports", [])
    conn_query = ""
    if agent_ports:
        port_filters = " ".join(f"sport = :{p}" for p in agent_ports)
        conn_query = f"ss -tnp state established '( {port_filters} )' 2>/dev/null | tail -n +2 | wc -l"

    tasks = [
        run_command(server, gpu_query),
        run_command(server, proc_query),
        run_command(server, sys_query),
        run_command(server, net_query),
    ]
    if vllm_combined:
        tasks.append(run_command(server, vllm_combined))
    if conn_query:
        tasks.append(run_command(server, conn_query))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    gpu_out = results[0] if not isinstance(results[0], Exception) else ""
    proc_out = results[1] if not isinstance(results[1], Exception) else ""
    sys_out = results[2] if not isinstance(results[2], Exception) else ""
    net_out = results[3] if not isinstance(results[3], Exception) else ""
    idx = 4
    vllm_out = ""
    if vllm_combined:
        vllm_out = results[idx] if not isinstance(results[idx], Exception) else ""
        idx += 1
    active_connections = 0
    if conn_query:
        try:
            active_connections = int(results[idx].strip()) if not isinstance(results[idx], Exception) else 0
        except (ValueError, IndexError):
            active_connections = 0

    # Parse GPUs
    gpus = []
    for line in gpu_out.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        gpu = {
            "index": int(parts[0]),
            "name": parts[1],
            "temp": int(parts[2]) if parts[2] not in ("[N/A]", "[Not Supported]") else None,
            "fan": int(parts[3]) if parts[3] not in ("[N/A]", "[Not Supported]") else None,
            "power_draw": float(parts[4]),
            "power_limit": float(parts[5]),
            "mem_used": int(parts[6]),
            "mem_total": int(parts[7]),
            "gpu_util": int(parts[8]),
            "mem_util": int(parts[9]),
        }
        if server["has_ecc"]:
            gpu["ecc_errors"] = int(parts[10]) if len(parts) > 10 else 0
        gpus.append(gpu)

    # Parse processes
    processes = []
    for line in proc_out.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            processes.append({
                "pid": parts[0],
                "name": parts[1],
                "gpu_bus_id": parts[2],
                "mem_used": int(parts[3]),
            })

    # Parse system
    sys_parts = sys_out.split("---")
    uptime_sec = float(sys_parts[0].strip().split()[0]) if sys_parts[0].strip() else 0
    ram = {}
    if len(sys_parts) > 1:
        ram_lines = sys_parts[1].strip().split("\n")
        if len(ram_lines) >= 2:
            rp = ram_lines[1].split()
            ram = {"total": int(rp[1]), "used": int(rp[2]), "available": int(rp[6]) if len(rp) > 6 else 0}
    disk = {}
    if len(sys_parts) > 2:
        dp = sys_parts[2].strip().split()
        if len(dp) >= 4:
            disk = {"total": int(dp[1]), "used": int(dp[2]), "avail": int(dp[3])}

    # Parse network
    net_rx, net_tx = 0, 0
    for line in net_out.strip().split("\n"):
        line = line.strip()
        if ":" not in line or "Inter" in line or "face" in line:
            continue
        iface, data = line.split(":", 1)
        iface = iface.strip()
        if iface in ("lo", "docker0"):
            continue
        vals = data.split()
        if len(vals) >= 9:
            net_rx += int(vals[0])
            net_tx += int(vals[8])

    # Parse vLLM metrics
    vllm_services = []
    if vllm_out:
        chunks = re.split(r'===VLLM_PORT_(\d+)===', vllm_out)
        # chunks: ['', '8001', '<metrics>', '8002', '<metrics>', ...]
        i = 1
        while i < len(chunks) - 1:
            port = int(chunks[i])
            raw = chunks[i + 1]
            parsed = parse_vllm_metrics(raw, port)
            if parsed:
                vllm_services.append(parsed)
            i += 2

    now = time.time()
    data = {
        "server_name": server["name"],
        "host": server["host"],
        "gpu_model": server["gpu_model"],
        "timestamp": now,
        "uptime_seconds": uptime_sec,
        "ram": ram,
        "disk": disk,
        "net_rx_bytes": net_rx,
        "net_tx_bytes": net_tx,
        "gpus": gpus,
        "processes": processes,
        "vllm": vllm_services,
        "active_connections": active_connections,
        "status": "online",
    }

    # Store in history
    avg_util = sum(g["gpu_util"] for g in gpus) / len(gpus) if gpus else 0
    total_running = sum(v["requests_running"] for v in vllm_services)
    total_waiting = sum(v["requests_waiting"] for v in vllm_services)
    max_temp = max((g["temp"] or 0 for g in gpus), default=0)
    total_power = round(sum(g["power_draw"] for g in gpus), 1)
    # "calls" = active_connections for agent servers, requests_running for vLLM
    calls = active_connections if active_connections > 0 else total_running
    # Sum of all GPU utilizations (not average — user wants total across all 8 GPUs)
    sum_gpu_util = sum(g["gpu_util"] for g in gpus)

    # vLLM latency snapshot for this poll
    _snap_ttft = 0
    _snap_itl = 0
    _snap_e2e = 0
    _snap_queue = 0
    _snap_kv = 0
    if vllm_services:
        _t = [v["avg_ttft_ms"] for v in vllm_services if v.get("avg_ttft_ms")]
        _i = [v["avg_itl_ms"] for v in vllm_services if v.get("avg_itl_ms")]
        _e = [v["avg_e2e_s"] for v in vllm_services if v.get("avg_e2e_s")]
        _q = [v["avg_queue_s"] for v in vllm_services if v.get("avg_queue_s")]
        _k = [v["kv_cache_usage"] for v in vllm_services if v.get("kv_cache_usage")]
        _snap_ttft = round(sum(_t) / len(_t), 1) if _t else 0
        _snap_itl = round(sum(_i) / len(_i), 1) if _i else 0
        _snap_e2e = round(sum(_e) / len(_e), 3) if _e else 0
        _snap_queue = round(sum(_q) / len(_q), 4) if _q else 0
        _snap_kv = round(max(_k), 1) if _k else 0

    history[server_key].append({
        "t": now,
        "avg_util": round(avg_util, 1),
        "sum_util": sum_gpu_util,
        "max_temp": max_temp,
        "total_power": total_power,
        "vram_pct": round(sum(g["mem_used"] for g in gpus) / max(sum(g["mem_total"] for g in gpus), 1) * 100, 1),
        "requests_running": total_running,
        "requests_waiting": total_waiting,
        "active_connections": active_connections,
        "calls": calls,
        "net_rx": net_rx,
        "net_tx": net_tx,
        "ttft_ms": _snap_ttft,
        "itl_ms": _snap_itl,
        "e2e_s": _snap_e2e,
        "queue_s": _snap_queue,
        "kv_cache": _snap_kv,
        "gpu_count": len(gpus),
    })

    # ── Hourly aggregation ──
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(now, tz=timezone.utc)
    hour_key = dt.strftime("%Y-%m-%d-%H")
    hour_label = dt.strftime("%H:00")
    hour_ts = int(dt.replace(minute=0, second=0, microsecond=0).timestamp())

    # vLLM aggregate stats
    vllm_total_reqs = sum(v.get("total_requests", 0) for v in vllm_services)
    vllm_prompt_tok = sum(v.get("prompt_tokens", 0) for v in vllm_services)
    vllm_gen_tok = sum(v.get("generation_tokens", 0) for v in vllm_services)
    vllm_avg_ttft = 0
    vllm_avg_e2e = 0
    vllm_kv_usage = 0
    if vllm_services:
        ttft_vals = [v["avg_ttft_ms"] for v in vllm_services if v.get("avg_ttft_ms")]
        e2e_vals = [v["avg_e2e_s"] for v in vllm_services if v.get("avg_e2e_s")]
        kv_vals = [v["kv_cache_usage"] for v in vllm_services if v.get("kv_cache_usage")]
        vllm_avg_ttft = sum(ttft_vals) / len(ttft_vals) if ttft_vals else 0
        vllm_avg_e2e = sum(e2e_vals) / len(e2e_vals) if e2e_vals else 0
        vllm_kv_usage = max(kv_vals) if kv_vals else 0
    vram_pct = round(sum(g["mem_used"] for g in gpus) / max(sum(g["mem_total"] for g in gpus), 1) * 100, 1)

    hs = hourly_stats[server_key]
    if hour_key not in hs:
        hs[hour_key] = {
            "hour_key": hour_key,
            "hour_label": hour_label,
            "hour_ts": hour_ts,
            "samples": 0,
            # Calls
            "max_calls": 0,
            "max_calls_time": now,
            "sum_calls": 0,
            # GPU (sum across all GPUs, e.g. 8 GPUs at 50% each = 400%)
            "max_gpu_util_sum": 0,
            "max_gpu_util_avg": 0,
            "max_gpu_util_time": now,
            "calls_at_max_gpu_util": 0,
            "sum_gpu_util": 0,
            "gpu_count": len(gpus),
            # Temp & Power
            "max_temp": 0,
            "max_power": 0,
            "sum_temp": 0,
            "sum_power": 0,
            # VRAM
            "max_vram_pct": 0,
            # vLLM snapshot (cumulative counters — we store last seen to compute delta)
            "max_kv_cache": 0,
            "max_ttft_ms": 0,
            "max_e2e_s": 0,
            "sum_ttft_ms": 0,
            "sum_e2e_s": 0,
            "ttft_samples": 0,
            "max_waiting": 0,
            # vLLM cumulative counters (start/end for delta)
            "_vllm_reqs_start": vllm_total_reqs,
            "_vllm_reqs_end": vllm_total_reqs,
            "_vllm_prompt_start": vllm_prompt_tok,
            "_vllm_prompt_end": vllm_prompt_tok,
            "_vllm_gen_start": vllm_gen_tok,
            "_vllm_gen_end": vllm_gen_tok,
            # Network (start/end for delta)
            "_net_rx_start": net_rx,
            "_net_rx_end": net_rx,
            "_net_tx_start": net_tx,
            "_net_tx_end": net_tx,
        }
        # Prune old hours
        cutoff = now - HOURLY_RETENTION * 3600
        for k in list(hs.keys()):
            if hs[k]["hour_ts"] < cutoff:
                del hs[k]

    h = hs[hour_key]
    h["samples"] += 1
    h["sum_calls"] += calls
    h["sum_gpu_util"] += avg_util
    h["sum_temp"] += max_temp
    h["sum_power"] += total_power

    if calls > h["max_calls"]:
        h["max_calls"] = calls
        h["max_calls_time"] = now
    if sum_gpu_util > h["max_gpu_util_sum"]:
        h["max_gpu_util_sum"] = sum_gpu_util
        h["max_gpu_util_avg"] = round(avg_util, 1)
        h["max_gpu_util_time"] = now
        h["calls_at_max_gpu_util"] = calls
    if max_temp > h["max_temp"]:
        h["max_temp"] = max_temp
    if total_power > h["max_power"]:
        h["max_power"] = total_power
    if vram_pct > h["max_vram_pct"]:
        h["max_vram_pct"] = vram_pct
    if total_waiting > h["max_waiting"]:
        h["max_waiting"] = total_waiting
    if vllm_kv_usage > h["max_kv_cache"]:
        h["max_kv_cache"] = round(vllm_kv_usage, 1)
    if vllm_avg_ttft > 0:
        h["sum_ttft_ms"] += vllm_avg_ttft
        h["ttft_samples"] += 1
        if vllm_avg_ttft > h["max_ttft_ms"]:
            h["max_ttft_ms"] = round(vllm_avg_ttft, 1)
    if vllm_avg_e2e > 0:
        h["sum_e2e_s"] += vllm_avg_e2e
        if vllm_avg_e2e > h["max_e2e_s"]:
            h["max_e2e_s"] = round(vllm_avg_e2e, 3)

    # Update cumulative end counters
    h["_vllm_reqs_end"] = vllm_total_reqs
    h["_vllm_prompt_end"] = vllm_prompt_tok
    h["_vllm_gen_end"] = vllm_gen_tok
    h["_net_rx_end"] = net_rx
    h["_net_tx_end"] = net_tx

    # ── Proactive Alert Tracking ──
    # Error rate tracking (vLLM errors = total_requests - success_count)
    total_errors = 0
    total_reqs = 0
    for v in vllm_services:
        t = v.get("total_requests", 0)
        s = v.get("success_count", 0)
        total_reqs += t
        total_errors += max(0, t - s)
    error_history[server_key].append({
        "t": now, "errors": total_errors, "total": total_reqs,
        "rate": round(total_errors / max(total_reqs, 1) * 100, 2),
    })

    # Disk usage tracking (every ~20 samples = ~1 min)
    if len(history[server_key]) % 20 == 0 and disk:
        disk_history[server_key].append({
            "t": now,
            "used_bytes": disk.get("used", 0),
            "total_bytes": disk.get("total", 0),
            "pct": round(disk.get("used", 0) / max(disk.get("total", 1), 1) * 100, 1),
        })

    # GPU clock tracking for throttle detection
    for g in gpus:
        idx = g["index"]
        if idx not in gpu_clock_history[server_key]:
            gpu_clock_history[server_key][idx] = deque(maxlen=120)
        gpu_clock_history[server_key][idx].append({
            "t": now, "sm_clock": g.get("gpu_util", 0),
            "temp": g.get("temp", 0), "power": g.get("power_draw", 0),
            "mem_util": g.get("mem_util", 0),
        })

    # Call volume pattern tracking (for forecasting)
    from datetime import datetime, timezone
    dt_now = datetime.fromtimestamp(now, tz=timezone.utc)
    dow_hour_key = f"{dt_now.weekday()}_{dt_now.hour}"
    if dow_hour_key not in call_volume_patterns:
        call_volume_patterns[dow_hour_key] = []
    call_volume_patterns[dow_hour_key].append(calls)
    # Keep last 200 per slot
    if len(call_volume_patterns[dow_hour_key]) > 200:
        call_volume_patterns[dow_hour_key] = call_volume_patterns[dow_hour_key][-200:]

    return data


# ── API Endpoints ──────────────────────────────────────────────
@app.get("/api/agent")
async def get_agent_status():
    try:
        result = await asyncio.wait_for(fetch_agent_status(), timeout=15)
        # Track process health (zombies, memory, workers)
        now = time.time()
        process_health_history.append({
            "t": now,
            "zombies": result.get("zombie_processes", 0),
            "gunicorn_rss_mb": result.get("gunicorn_memory_mb", 0),
            "gunicorn_workers": result.get("gunicorn_workers", 0),
            "voicemail_workers": result.get("voicemail_workers", 0),
            "active_bots": result.get("active_bots", 0),
            "tcp_connections": result.get("tcp_connections", 0),
        })
        # Track service uptime
        for svc in result.get("services", []):
            name = svc["name"]
            status = svc["status"]
            if name not in service_uptime:
                service_uptime[name] = {"status": status, "last_change_t": now, "up_samples": 0, "total_samples": 0, "incidents": deque(maxlen=50)}
            su = service_uptime[name]
            su["total_samples"] += 1
            if status == "active":
                su["up_samples"] += 1
            if status != su["status"]:
                su["incidents"].append({"t": now, "from": su["status"], "to": status})
                su["status"] = status
                su["last_change_t"] = now
        return result
    except Exception as e:
        return {"error": str(e), "status": "offline"}


@app.get("/api/livecalls")
async def get_live_calls():
    """Live Call Tracker — real-time active calls with per-call metrics."""
    now = time.time()
    result = {"calls": [], "summary": {}, "recent_completed": []}

    # Active calls from vLLM running requests + agent connections
    total_active = 0
    for sk in SERVERS:
        sk_hist = list(history[sk])
        if not sk_hist:
            continue
        latest = sk_hist[-1]
        calls = latest.get("calls", 0)
        total_active += calls

        # Build per-server call info
        running = latest.get("requests_running", 0)
        waiting = latest.get("requests_waiting", 0)
        active_conn = latest.get("active_connections", 0)
        ttft = latest.get("ttft_ms", 0)
        e2e = latest.get("e2e_s", 0)
        kv = latest.get("kv_cache", 0)
        util = latest.get("avg_util", 0)

        if calls > 0 or running > 0 or active_conn > 0:
            result["calls"].append({
                "server": sk,
                "server_name": SERVERS[sk]["name"],
                "active_calls": calls,
                "requests_running": running,
                "requests_waiting": waiting,
                "active_connections": active_conn,
                "current_ttft_ms": round(ttft, 1),
                "current_e2e_s": round(e2e, 3),
                "current_kv_cache": round(kv, 1),
                "current_gpu_util": round(util, 1),
                "timestamp": latest.get("t", now),
            })

    # Call history timeline (last 120 points)
    call_timeline = []
    for sk in SERVERS:
        for p in list(history[sk])[-120:]:
            call_timeline.append({
                "t": p["t"], "server": sk,
                "calls": p.get("calls", 0),
                "ttft_ms": p.get("ttft_ms", 0),
                "e2e_s": p.get("e2e_s", 0),
                "gpu_util": p.get("avg_util", 0),
                "kv_cache": p.get("kv_cache", 0),
            })
    call_timeline.sort(key=lambda x: x["t"])

    # Summary stats
    all_hist = []
    for sk in SERVERS:
        all_hist.extend(list(history[sk]))
    if all_hist:
        calls_arr = [p.get("calls", 0) for p in all_hist]
        result["summary"] = {
            "total_active_now": total_active,
            "peak_concurrent": max(calls_arr),
            "avg_concurrent": round(sum(calls_arr) / len(calls_arr), 1),
            "zero_call_pct": round(sum(1 for c in calls_arr if c == 0) / len(calls_arr) * 100, 1),
            "timeline": call_timeline[-60:],
        }

    # Call duration distribution from e2e history
    e2e_vals = [p.get("e2e_s", 0) for p in all_hist if p.get("e2e_s", 0) > 0]
    if e2e_vals:
        buckets = {"<1s": 0, "1-3s": 0, "3-5s": 0, "5-10s": 0, ">10s": 0}
        for v in e2e_vals:
            if v < 1: buckets["<1s"] += 1
            elif v < 3: buckets["1-3s"] += 1
            elif v < 5: buckets["3-5s"] += 1
            elif v < 10: buckets["5-10s"] += 1
            else: buckets[">10s"] += 1
        result["duration_distribution"] = buckets

    return result


@app.get("/api/modelcompare")
async def get_model_compare():
    """A/B Model Comparison — compare model performance over time windows."""
    result = {"models": [], "time_windows": [], "comparison": {}}

    # Get current model info from latest data
    for sk in SERVERS:
        sk_hist = list(history[sk])
        if not sk_hist:
            continue

        # Try to get current vLLM model info
        try:
            latest = await asyncio.wait_for(fetch_gpu_data(sk), timeout=10)
            for v in latest.get("vllm", []):
                result["models"].append({
                    "server": sk,
                    "server_name": SERVERS[sk]["name"],
                    "port": v.get("port"),
                    "model_name": v.get("model_name", "unknown"),
                    "total_requests": v.get("total_requests", 0),
                    "kv_cache_usage": v.get("kv_cache_usage", 0),
                    "avg_ttft_ms": round(v.get("avg_ttft_ms", 0), 1),
                    "avg_itl_ms": round(v.get("avg_itl_ms", 0), 1),
                    "avg_e2e_s": round(v.get("avg_e2e_s", 0), 3),
                    "avg_queue_s": round(v.get("avg_queue_s", 0), 4),
                    "cache_hit_rate": v.get("cache_hit_rate", 0),
                    "requests_running": v.get("requests_running", 0),
                })
        except Exception:
            pass

    # Time-windowed performance comparison (split history into quarters)
    for sk in SERVERS:
        sk_hist = list(history[sk])
        if len(sk_hist) < 20:
            continue
        quarter = len(sk_hist) // 4
        windows = []
        labels = ["Q1 (oldest)", "Q2", "Q3", "Q4 (latest)"]
        for i in range(4):
            start = i * quarter
            end = (i + 1) * quarter if i < 3 else len(sk_hist)
            segment = sk_hist[start:end]
            ttft = [p.get("ttft_ms", 0) for p in segment if p.get("ttft_ms", 0) > 0]
            e2e = [p.get("e2e_s", 0) for p in segment if p.get("e2e_s", 0) > 0]
            util = [p.get("avg_util", 0) for p in segment]
            kv = [p.get("kv_cache", 0) for p in segment if p.get("kv_cache", 0) > 0]
            calls = [p.get("calls", 0) for p in segment]

            windows.append({
                "label": labels[i],
                "samples": len(segment),
                "avg_ttft_ms": round(sum(ttft) / max(len(ttft), 1), 1),
                "avg_e2e_s": round(sum(e2e) / max(len(e2e), 1), 3),
                "avg_util": round(sum(util) / max(len(util), 1), 1),
                "avg_kv": round(sum(kv) / max(len(kv), 1), 1),
                "avg_calls": round(sum(calls) / max(len(calls), 1), 1),
                "peak_calls": max(calls) if calls else 0,
                "time_start": segment[0]["t"] if segment else 0,
                "time_end": segment[-1]["t"] if segment else 0,
            })

        result["time_windows"].append({
            "server": sk,
            "server_name": SERVERS[sk]["name"],
            "windows": windows,
        })

    # Efficiency comparison per server
    for sk in SERVERS:
        hs = hourly_stats.get(sk, {})
        if not hs:
            continue
        total_tokens = 0
        total_samples = 0
        for hk, h in hs.items():
            prompt_delta = h.get("_vllm_prompt_end", 0) - h.get("_vllm_prompt_start", 0)
            gen_delta = h.get("_vllm_gen_end", 0) - h.get("_vllm_gen_start", 0)
            total_tokens += prompt_delta + gen_delta
            total_samples += h.get("samples", 0)

        sk_hist = list(history[sk])
        avg_power = sum(p.get("total_power", 0) for p in sk_hist[-60:]) / max(len(sk_hist[-60:]), 1) if sk_hist else 0

        result["comparison"][sk] = {
            "server_name": SERVERS[sk]["name"],
            "total_tokens_processed": total_tokens,
            "avg_power_w": round(avg_power, 0),
            "tokens_per_watt_hour": round(total_tokens / max(avg_power, 1) / max(total_samples * 3 / 3600, 0.01), 1) if avg_power > 0 else 0,
        }

    return result


@app.get("/api/power")
async def get_power():
    """GPU Power Management — efficiency curves, tokens/watt, optimal settings."""
    result = {"per_server": {}, "fleet_summary": {}}

    for sk in SERVERS:
        server = SERVERS[sk]
        sk_hist = list(history[sk])
        if not sk_hist:
            result["per_server"][sk] = {"server_name": server["name"], "no_data": True}
            continue

        # Power data from history
        power_data = []
        for p in sk_hist:
            power_data.append({
                "power_w": p.get("total_power", 0),
                "util": p.get("avg_util", 0),
                "calls": p.get("calls", 0),
                "ttft_ms": p.get("ttft_ms", 0),
                "e2e_s": p.get("e2e_s", 0),
                "temp": p.get("max_temp", 0),
            })

        # Power vs utilization buckets
        power_util_curve = {}
        for pd in power_data:
            bucket = int(pd["util"] // 10) * 10
            if bucket not in power_util_curve:
                power_util_curve[bucket] = {"power": [], "ttft": [], "e2e": [], "temp": [], "count": 0}
            power_util_curve[bucket]["power"].append(pd["power_w"])
            power_util_curve[bucket]["count"] += 1
            if pd["ttft_ms"] > 0:
                power_util_curve[bucket]["ttft"].append(pd["ttft_ms"])
            if pd["e2e_s"] > 0:
                power_util_curve[bucket]["e2e"].append(pd["e2e_s"])
            power_util_curve[bucket]["temp"].append(pd["temp"])

        efficiency_curve = []
        for bucket in sorted(power_util_curve.keys()):
            d = power_util_curve[bucket]
            avg_power = sum(d["power"]) / len(d["power"])
            avg_ttft = sum(d["ttft"]) / len(d["ttft"]) if d["ttft"] else 0
            avg_temp = sum(d["temp"]) / len(d["temp"])
            # Efficiency = work done per watt (approximated by util/power)
            efficiency = bucket / max(avg_power, 1) * 100

            efficiency_curve.append({
                "util_bucket": bucket,
                "avg_power_w": round(avg_power, 0),
                "avg_temp_c": round(avg_temp, 1),
                "avg_ttft_ms": round(avg_ttft, 1),
                "efficiency": round(efficiency, 2),
                "samples": d["count"],
            })

        # GPU-level power from clock history
        per_gpu_power = []
        for gpu_idx, clock_data in gpu_clock_history.get(sk, {}).items():
            points = list(clock_data)
            if points:
                avg_power_gpu = sum(p.get("power", 0) for p in points) / len(points)
                avg_temp_gpu = sum(p.get("temp", 0) for p in points) / len(points)
                avg_util_gpu = sum(p.get("sm_clock", 0) for p in points) / len(points)
                per_gpu_power.append({
                    "gpu": gpu_idx,
                    "avg_power_w": round(avg_power_gpu, 1),
                    "avg_temp_c": round(avg_temp_gpu, 1),
                    "avg_util": round(avg_util_gpu, 1),
                })

        # Current power stats
        recent = sk_hist[-min(30, len(sk_hist)):]
        current_power = sum(p.get("total_power", 0) for p in recent) / len(recent)
        peak_power = max(p.get("total_power", 0) for p in sk_hist)
        min_power = min(p.get("total_power", 0) for p in sk_hist) if sk_hist else 0
        gpu_count = sk_hist[-1].get("gpu_count", 8)

        # Optimal power point: highest efficiency (util/power ratio)
        optimal = max(efficiency_curve, key=lambda x: x["efficiency"]) if efficiency_curve else None

        # Token efficiency from hourly stats
        hs = hourly_stats.get(sk, {})
        total_tokens = 0
        total_time_s = 0
        for hk, h in hs.items():
            total_tokens += (h.get("_vllm_prompt_end", 0) - h.get("_vllm_prompt_start", 0)) + (h.get("_vllm_gen_end", 0) - h.get("_vllm_gen_start", 0))
            total_time_s += h.get("samples", 0) * 3

        tokens_per_kwh = round(total_tokens / max(current_power / 1000 * max(total_time_s / 3600, 0.01), 0.001), 0) if current_power > 0 and total_tokens > 0 else 0

        result["per_server"][sk] = {
            "server_name": server["name"],
            "gpu_count": gpu_count,
            "current_power_w": round(current_power, 0),
            "peak_power_w": round(peak_power, 0),
            "min_power_w": round(min_power, 0),
            "power_per_gpu_w": round(current_power / max(gpu_count, 1), 0),
            "efficiency_curve": efficiency_curve,
            "per_gpu": sorted(per_gpu_power, key=lambda x: x["gpu"]),
            "optimal_point": optimal,
            "tokens_per_kwh": tokens_per_kwh,
            "cost_per_kwh": round(COST_PER_HOUR / max(current_power / 1000, 0.001), 2) if current_power > 0 else 0,
        }

    # Fleet summary
    total_power = sum(s.get("current_power_w", 0) for s in result["per_server"].values() if not s.get("no_data"))
    total_peak = sum(s.get("peak_power_w", 0) for s in result["per_server"].values() if not s.get("no_data"))
    result["fleet_summary"] = {
        "total_current_w": round(total_power, 0),
        "total_peak_w": round(total_peak, 0),
        "total_current_kw": round(total_power / 1000, 2),
        "monthly_kwh": round(total_power / 1000 * 730, 0),
        "power_cost_estimate": round(total_power / 1000 * 730 * 0.10, 0),  # ~$0.10/kWh
        "gpu_monthly_cost": GPU_MONTHLY_COST,
        "power_as_pct_of_cost": round(total_power / 1000 * 730 * 0.10 / max(GPU_MONTHLY_COST, 1) * 100, 1),
    }

    return result


@app.get("/api/incidents")
async def get_incidents():
    """Incident Timeline — service restarts, error spikes, latency spikes, temp events."""
    now = time.time()
    incidents = []

    # Service restart incidents from service_uptime
    for name, su in service_uptime.items():
        for inc in su.get("incidents", []):
            incidents.append({
                "t": inc.get("t", now),
                "type": "restart",
                "severity": "warning",
                "title": f"{name} restarted",
                "detail": f"Service went {inc.get('new_status', 'unknown')}",
                "source": name,
            })

    # Error spike incidents from error_history
    for sk in SERVERS:
        errs = list(error_history.get(sk, []))
        for i, e in enumerate(errs):
            rate = e.get("rate", 0)
            if rate > 5:
                severity = "critical" if rate > 20 else "warning"
                incidents.append({
                    "t": e["t"],
                    "type": "error_spike",
                    "severity": severity,
                    "title": f"Error spike on {SERVERS[sk]['name']}",
                    "detail": f"Error rate: {rate:.1f}% ({e.get('errors', 0)} errors / {e.get('total', 0)} total)",
                    "source": sk,
                })

    # Temperature events from history
    for sk in SERVERS:
        hist = list(history.get(sk, []))
        for h in hist:
            gpus = h.get("gpus", [])
            for g in gpus:
                temp = g.get("temp", 0) or 0
                if temp > 80:
                    incidents.append({
                        "t": h.get("t", now),
                        "type": "thermal",
                        "severity": "critical" if temp > 90 else "warning",
                        "title": f"GPU {g.get('index', '?')} high temp on {SERVERS[sk]['name']}",
                        "detail": f"Temperature: {temp}°C",
                        "source": sk,
                    })

    # Fetch errors (SSH/network failures)
    for fe in fetch_errors:
        incidents.append({
            "t": fe.get("t", now),
            "type": "fetch_error",
            "severity": "warning",
            "title": f"Connection error to {fe.get('server', '?')}",
            "detail": str(fe.get("error", ""))[:100],
            "source": fe.get("server", "unknown"),
        })

    # Latency spikes from cluster_latency
    for cl in cluster_latency:
        rtt = cl.get("rtt_ms", 0)
        if rtt > 50:
            incidents.append({
                "t": cl.get("t", now),
                "type": "latency_spike",
                "severity": "critical" if rtt > 200 else "warning",
                "title": "Cross-cluster latency spike",
                "detail": f"RTT: {rtt:.0f}ms",
                "source": "cluster",
            })

    # Sort by time descending, limit to 200
    incidents.sort(key=lambda x: x["t"], reverse=True)
    incidents = incidents[:200]

    # Summary stats
    last_hour = [i for i in incidents if i["t"] > now - 3600]
    last_24h = [i for i in incidents if i["t"] > now - 86400]

    return {
        "incidents": incidents,
        "summary": {
            "total": len(incidents),
            "last_hour": len(last_hour),
            "last_24h": len(last_24h),
            "critical_count": len([i for i in incidents if i["severity"] == "critical"]),
            "warning_count": len([i for i in incidents if i["severity"] == "warning"]),
            "by_type": {
                "restart": len([i for i in incidents if i["type"] == "restart"]),
                "error_spike": len([i for i in incidents if i["type"] == "error_spike"]),
                "thermal": len([i for i in incidents if i["type"] == "thermal"]),
                "fetch_error": len([i for i in incidents if i["type"] == "fetch_error"]),
                "latency_spike": len([i for i in incidents if i["type"] == "latency_spike"]),
            },
        },
    }


# In-memory alert config store
alert_configs = {
    "gpu_temp_warning": {"name": "GPU Temp Warning", "threshold": 75, "unit": "°C", "enabled": True, "severity": "warning"},
    "gpu_temp_critical": {"name": "GPU Temp Critical", "threshold": 85, "unit": "°C", "enabled": True, "severity": "critical"},
    "gpu_util_low": {"name": "GPU Utilization Low", "threshold": 10, "unit": "%", "enabled": True, "severity": "info", "direction": "below"},
    "error_rate_warning": {"name": "Error Rate Warning", "threshold": 5, "unit": "%", "enabled": True, "severity": "warning"},
    "error_rate_critical": {"name": "Error Rate Critical", "threshold": 15, "unit": "%", "enabled": True, "severity": "critical"},
    "vram_high": {"name": "VRAM Usage High", "threshold": 90, "unit": "%", "enabled": True, "severity": "warning"},
    "ttft_slow": {"name": "TTFT Slow", "threshold": 500, "unit": "ms", "enabled": True, "severity": "warning"},
    "power_high": {"name": "Power Draw High", "threshold": 90, "unit": "% of limit", "enabled": True, "severity": "warning"},
    "latency_high": {"name": "Cluster Latency High", "threshold": 100, "unit": "ms", "enabled": True, "severity": "warning"},
    "disk_full": {"name": "Disk Usage High", "threshold": 85, "unit": "%", "enabled": True, "severity": "warning"},
}


@app.get("/api/alerts-config")
async def get_alerts_config():
    """Custom Alerts Config — get/set alert thresholds."""
    # Evaluate current state against thresholds
    active_alerts = []
    now = time.time()

    for sk in SERVERS:
        try:
            data = await asyncio.wait_for(fetch_gpu_data(sk), timeout=10)
        except Exception:
            data = None
        if not data or data.get("status") == "offline":
            continue
        sname = SERVERS[sk]["name"]
        gpus = data.get("gpus", [])

        for g in gpus:
            temp = g.get("temp", 0) or 0
            cfg = alert_configs["gpu_temp_warning"]
            if cfg["enabled"] and temp >= cfg["threshold"]:
                active_alerts.append({"rule": "gpu_temp_warning", "server": sname, "gpu": g["index"], "value": temp, "threshold": cfg["threshold"]})
            cfg = alert_configs["gpu_temp_critical"]
            if cfg["enabled"] and temp >= cfg["threshold"]:
                active_alerts.append({"rule": "gpu_temp_critical", "server": sname, "gpu": g["index"], "value": temp, "threshold": cfg["threshold"]})

            util = g.get("gpu_util", 0)
            cfg = alert_configs["gpu_util_low"]
            if cfg["enabled"] and util <= cfg["threshold"]:
                active_alerts.append({"rule": "gpu_util_low", "server": sname, "gpu": g["index"], "value": util, "threshold": cfg["threshold"]})

            mem_pct = g["mem_used"] / max(g["mem_total"], 1) * 100
            cfg = alert_configs["vram_high"]
            if cfg["enabled"] and mem_pct >= cfg["threshold"]:
                active_alerts.append({"rule": "vram_high", "server": sname, "gpu": g["index"], "value": round(mem_pct, 1), "threshold": cfg["threshold"]})

            pow_pct = g["power_draw"] / max(g["power_limit"], 1) * 100
            cfg = alert_configs["power_high"]
            if cfg["enabled"] and pow_pct >= cfg["threshold"]:
                active_alerts.append({"rule": "power_high", "server": sname, "gpu": g["index"], "value": round(pow_pct, 1), "threshold": cfg["threshold"]})

    return {
        "configs": alert_configs,
        "active_alerts": active_alerts,
        "active_count": len(active_alerts),
    }


@app.post("/api/alerts-config")
async def update_alerts_config(request: Request):
    """Update alert thresholds."""
    body = await request.json()
    rule_id = body.get("rule_id")
    if rule_id not in alert_configs:
        return {"error": "Unknown rule"}
    if "threshold" in body:
        alert_configs[rule_id]["threshold"] = float(body["threshold"])
    if "enabled" in body:
        alert_configs[rule_id]["enabled"] = bool(body["enabled"])
    return {"ok": True, "configs": alert_configs}


# SLA targets (in-memory)
sla_targets = {
    "uptime": {"name": "Uptime", "target": 99.9, "unit": "%"},
    "ttft_p95": {"name": "TTFT P95", "target": 500, "unit": "ms"},
    "error_rate": {"name": "Error Rate", "target": 1.0, "unit": "%"},
    "gpu_util_avg": {"name": "Avg GPU Utilization", "target": 40, "unit": "%", "direction": "above"},
    "vram_headroom": {"name": "VRAM Headroom", "target": 10, "unit": "%", "direction": "above"},
}


@app.get("/api/sla")
async def get_sla():
    """SLA Dashboard — user-defined SLA targets with compliance tracking."""
    now = time.time()
    results = {}

    # Pre-fetch latest data for all servers
    server_data = {}
    for sk in SERVERS:
        try:
            server_data[sk] = await asyncio.wait_for(fetch_gpu_data(sk), timeout=10)
        except Exception:
            server_data[sk] = None

    for sla_id, sla in sla_targets.items():
        current_value = None
        compliant = True
        trend_values = []

        if sla_id == "uptime":
            # Calculate from service_uptime data
            total_up = 0
            total_samples = 0
            for name, su in service_uptime.items():
                total_up += su.get("up_samples", 0)
                total_samples += su.get("total_samples", 0)
            current_value = round(total_up / max(total_samples, 1) * 100, 2) if total_samples > 0 else 100.0
            compliant = current_value >= sla["target"]

        elif sla_id == "ttft_p95":
            # Get from latest vllm metrics
            all_ttft = []
            for sk in SERVERS:
                data = server_data.get(sk)
                if data and "vllm" in data:
                    for port_data in data["vllm"]:
                        ttft = port_data.get("ttft_ms", {}).get("p95")
                        if ttft is not None:
                            all_ttft.append(ttft)
            current_value = round(max(all_ttft), 1) if all_ttft else 0
            compliant = current_value <= sla["target"]

        elif sla_id == "error_rate":
            # Average error rate across servers
            rates = []
            for sk in SERVERS:
                errs = list(error_history.get(sk, []))
                if errs:
                    rates.append(errs[-1].get("rate", 0))
            current_value = round(sum(rates) / max(len(rates), 1), 2)
            compliant = current_value <= sla["target"]
            # Build trend from error_history
            for sk in SERVERS:
                for e in error_history.get(sk, []):
                    trend_values.append({"t": e["t"], "v": e.get("rate", 0)})

        elif sla_id == "gpu_util_avg":
            # Average GPU utilization
            utils = []
            for sk in SERVERS:
                data = server_data.get(sk)
                if data and data.get("status") != "offline":
                    for g in data.get("gpus", []):
                        utils.append(g.get("gpu_util", 0))
            current_value = round(sum(utils) / max(len(utils), 1), 1)
            compliant = current_value >= sla["target"]

        elif sla_id == "vram_headroom":
            # Minimum VRAM free across all GPUs
            headrooms = []
            for sk in SERVERS:
                data = server_data.get(sk)
                if data and data.get("status") != "offline":
                    for g in data.get("gpus", []):
                        if g["mem_total"] > 0:
                            headrooms.append((1 - g["mem_used"] / g["mem_total"]) * 100)
            current_value = round(min(headrooms), 1) if headrooms else 100.0
            compliant = current_value >= sla["target"]

        # Build 24h compliance from history
        compliance_24h = []
        for sk in SERVERS:
            hist = list(history.get(sk, []))
            for h in hist:
                if h["t"] > now - 86400:
                    if sla_id == "gpu_util_avg":
                        gpus = h.get("gpus", [])
                        if gpus:
                            avg = sum(g.get("gpu_util", 0) for g in gpus) / len(gpus)
                            compliance_24h.append({"t": h["t"], "compliant": avg >= sla["target"], "value": round(avg, 1)})

        results[sla_id] = {
            "name": sla["name"],
            "target": sla["target"],
            "unit": sla["unit"],
            "direction": sla.get("direction", "below"),
            "current_value": current_value,
            "compliant": compliant,
            "trend": sorted(trend_values, key=lambda x: x["t"])[-50:] if trend_values else [],
        }

    # Overall SLA score
    total = len(results)
    met = sum(1 for r in results.values() if r["compliant"])

    return {
        "targets": results,
        "overall": {
            "met": met,
            "total": total,
            "score": round(met / max(total, 1) * 100, 0),
            "grade": "A" if met == total else "B" if met >= total - 1 else "C" if met >= total - 2 else "D",
        },
    }


@app.post("/api/sla")
async def update_sla(request: Request):
    """Update SLA target."""
    body = await request.json()
    sla_id = body.get("sla_id")
    if sla_id not in sla_targets:
        return {"error": "Unknown SLA"}
    if "target" in body:
        sla_targets[sla_id]["target"] = float(body["target"])
    return {"ok": True, "targets": sla_targets}


@app.get("/api/anomalies")
async def get_anomalies():
    """Anomaly Detection — rolling z-score analysis on GPU metrics, traffic, memory, errors."""
    import math
    now = time.time()
    anomalies = []
    metric_stats = {}  # for the UI: {metric_key: {mean, std, current, z, history[]}}

    def z_score_analysis(values, label, server_name, threshold=2.5, direction="both"):
        """Compute rolling z-scores. Returns anomalies where |z| > threshold."""
        if len(values) < 10:
            return [], None
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 0.001
        current = values[-1]
        z = (current - mean) / std
        recent_zs = [(values[i] - mean) / std for i in range(max(0, len(values) - 60), len(values))]

        found = []
        if direction == "both" and abs(z) > threshold:
            found.append({"z": round(z, 2), "value": round(current, 2), "mean": round(mean, 2), "std": round(std, 2)})
        elif direction == "high" and z > threshold:
            found.append({"z": round(z, 2), "value": round(current, 2), "mean": round(mean, 2), "std": round(std, 2)})
        elif direction == "low" and z < -threshold:
            found.append({"z": round(z, 2), "value": round(current, 2), "mean": round(mean, 2), "std": round(std, 2)})

        stats = {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "current": round(current, 2),
            "z": round(z, 2),
            "recent_zs": [round(zv, 2) for zv in recent_zs[-30:]],
            "is_anomaly": len(found) > 0,
            "server": server_name,
            "label": label,
        }
        return found, stats

    for sk in SERVERS:
        sname = SERVERS[sk]["name"]
        hist = list(history.get(sk, []))
        if len(hist) < 10:
            continue

        # Extract time series from history
        gpu_count = len(hist[-1].get("gpus", [])) if hist else 0

        # ── Per-GPU metrics ──
        for gpu_idx in range(gpu_count):
            utils = [h["gpus"][gpu_idx].get("gpu_util", 0) for h in hist if len(h.get("gpus", [])) > gpu_idx]
            temps = [h["gpus"][gpu_idx].get("temp", 0) or 0 for h in hist if len(h.get("gpus", [])) > gpu_idx]
            mem_pcts = [(h["gpus"][gpu_idx]["mem_used"] / max(h["gpus"][gpu_idx]["mem_total"], 1) * 100) for h in hist if len(h.get("gpus", [])) > gpu_idx]
            powers = [h["gpus"][gpu_idx].get("power_draw", 0) for h in hist if len(h.get("gpus", [])) > gpu_idx]

            # Utilization anomaly (sudden drops)
            found, stats = z_score_analysis(utils, f"GPU {gpu_idx} Utilization", sname, threshold=2.5, direction="both")
            key = f"{sk}_gpu{gpu_idx}_util"
            if stats:
                metric_stats[key] = stats
            for f in found:
                anomalies.append({"t": now, "type": "util_anomaly", "severity": "warning" if abs(f["z"]) < 3.5 else "critical",
                    "title": f"GPU {gpu_idx} utilization anomaly on {sname}", "detail": f"Current: {f['value']}% (mean: {f['mean']}%, z={f['z']})", "server": sk, "gpu": gpu_idx, **f})

            # Temperature anomaly (sudden spikes)
            found, stats = z_score_analysis(temps, f"GPU {gpu_idx} Temperature", sname, threshold=2.5, direction="high")
            key = f"{sk}_gpu{gpu_idx}_temp"
            if stats:
                metric_stats[key] = stats
            for f in found:
                anomalies.append({"t": now, "type": "temp_anomaly", "severity": "warning" if f["z"] < 3.5 else "critical",
                    "title": f"GPU {gpu_idx} temp spike on {sname}", "detail": f"Current: {f['value']}C (mean: {f['mean']}C, z={f['z']})", "server": sk, "gpu": gpu_idx, **f})

            # Memory leak detection (steady upward trend)
            found, stats = z_score_analysis(mem_pcts, f"GPU {gpu_idx} VRAM", sname, threshold=2.0, direction="high")
            key = f"{sk}_gpu{gpu_idx}_vram"
            if stats:
                metric_stats[key] = stats
            # Also check for monotonic increase (memory leak pattern)
            if len(mem_pcts) >= 20:
                recent = mem_pcts[-20:]
                increases = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
                if increases >= 16:  # 80%+ of samples increasing
                    delta = recent[-1] - recent[0]
                    anomalies.append({"t": now, "type": "memory_leak", "severity": "warning",
                        "title": f"GPU {gpu_idx} possible memory leak on {sname}",
                        "detail": f"VRAM rose {delta:.1f}% over last 20 samples ({increases}/20 increases)",
                        "server": sk, "gpu": gpu_idx, "z": 0, "value": round(recent[-1], 1), "mean": round(sum(recent)/len(recent), 1), "std": 0})

            # Power anomaly
            found, stats = z_score_analysis(powers, f"GPU {gpu_idx} Power", sname, threshold=2.5, direction="both")
            key = f"{sk}_gpu{gpu_idx}_power"
            if stats:
                metric_stats[key] = stats
            for f in found:
                anomalies.append({"t": now, "type": "power_anomaly", "severity": "warning",
                    "title": f"GPU {gpu_idx} power anomaly on {sname}", "detail": f"Current: {f['value']}W (mean: {f['mean']}W, z={f['z']})", "server": sk, "gpu": gpu_idx, **f})

        # ── Aggregate server metrics ──
        avg_utils = [sum(h["gpus"][i].get("gpu_util", 0) for i in range(len(h.get("gpus", [])))) / max(len(h.get("gpus", [])), 1) for h in hist if h.get("gpus")]
        total_powers = [sum(h["gpus"][i].get("power_draw", 0) for i in range(len(h.get("gpus", [])))) for h in hist if h.get("gpus")]

        found, stats = z_score_analysis(avg_utils, f"Avg Utilization", sname, threshold=2.5, direction="both")
        if stats:
            metric_stats[f"{sk}_avg_util"] = stats
        for f in found:
            anomalies.append({"t": now, "type": "fleet_util_anomaly", "severity": "warning",
                "title": f"Fleet utilization anomaly on {sname}", "detail": f"Avg util: {f['value']}% (mean: {f['mean']}%, z={f['z']})", "server": sk, "gpu": -1, **f})

        found, stats = z_score_analysis(total_powers, f"Total Power", sname, threshold=2.5, direction="both")
        if stats:
            metric_stats[f"{sk}_total_power"] = stats
        for f in found:
            anomalies.append({"t": now, "type": "fleet_power_anomaly", "severity": "warning",
                "title": f"Total power anomaly on {sname}", "detail": f"Total: {f['value']}W (mean: {f['mean']}W, z={f['z']})", "server": sk, "gpu": -1, **f})

        # ── Error rate anomalies ──
        errs = list(error_history.get(sk, []))
        if len(errs) >= 10:
            rates = [e.get("rate", 0) for e in errs]
            found, stats = z_score_analysis(rates, f"Error Rate", sname, threshold=2.0, direction="high")
            if stats:
                metric_stats[f"{sk}_error_rate"] = stats
            for f in found:
                anomalies.append({"t": now, "type": "error_spike", "severity": "critical" if f["z"] > 3 else "warning",
                    "title": f"Error rate spike on {sname}", "detail": f"Rate: {f['value']}% (mean: {f['mean']}%, z={f['z']})", "server": sk, "gpu": -1, **f})

    # Sort by severity then z-score
    sev_order = {"critical": 0, "warning": 1, "info": 2}
    anomalies.sort(key=lambda a: (sev_order.get(a.get("severity"), 9), -abs(a.get("z", 0))))

    # Summary
    return {
        "anomalies": anomalies,
        "metric_stats": metric_stats,
        "summary": {
            "total_anomalies": len(anomalies),
            "critical": len([a for a in anomalies if a["severity"] == "critical"]),
            "warning": len([a for a in anomalies if a["severity"] == "warning"]),
            "metrics_monitored": len(metric_stats),
            "by_type": {
                "util_anomaly": len([a for a in anomalies if a["type"] == "util_anomaly"]),
                "temp_anomaly": len([a for a in anomalies if a["type"] == "temp_anomaly"]),
                "memory_leak": len([a for a in anomalies if a["type"] == "memory_leak"]),
                "power_anomaly": len([a for a in anomalies if a["type"] == "power_anomaly"]),
                "error_spike": len([a for a in anomalies if a["type"] == "error_spike"]),
                "fleet_util_anomaly": len([a for a in anomalies if a["type"] == "fleet_util_anomaly"]),
                "fleet_power_anomaly": len([a for a in anomalies if a["type"] == "fleet_power_anomaly"]),
            },
        },
    }


# ── SendGrid Email Alerts ─────────────────────────────────────
async def send_alert_email(recipients: list, subject: str, alerts: list) -> dict:
    """Send alert email via SendGrid API. Returns {ok, detail}."""
    if not SENDGRID_API_KEY:
        return {"ok": False, "detail": "SENDGRID_API_KEY not set"}
    if not recipients:
        return {"ok": False, "detail": "No recipients"}

    # Build HTML body
    rows = ""
    for a in alerts:
        sev_color = "#ef4444" if a.get("severity") == "critical" else "#f59e0b" if a.get("severity") == "warning" else "#3b82f6"
        rows += f'<tr><td style="padding:8px;border-bottom:1px solid #333;color:{sev_color};font-weight:bold">{a.get("severity","").upper()}</td>'
        rows += f'<td style="padding:8px;border-bottom:1px solid #333;color:#fff">{a.get("title","")}</td>'
        rows += f'<td style="padding:8px;border-bottom:1px solid #333;color:#999">{a.get("detail","")}</td></tr>'

    html = f"""<div style="background:#0a0a0f;padding:24px;font-family:Inter,sans-serif;color:#e5e7eb">
    <div style="max-width:600px;margin:0 auto">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px">
            <div style="width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#10b981,#06b6d4);display:flex;align-items:center;justify-content:center">
                <span style="color:#fff;font-size:20px">⚡</span>
            </div>
            <div>
                <h1 style="margin:0;color:#fff;font-size:18px">GPU Alert</h1>
                <p style="margin:0;color:#6b7280;font-size:12px">{subject}</p>
            </div>
        </div>
        <table style="width:100%;border-collapse:collapse;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:8px">
            <thead><tr style="border-bottom:1px solid #333">
                <th style="padding:10px;text-align:left;color:#9ca3af;font-size:12px">Severity</th>
                <th style="padding:10px;text-align:left;color:#9ca3af;font-size:12px">Alert</th>
                <th style="padding:10px;text-align:left;color:#9ca3af;font-size:12px">Detail</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        <p style="margin-top:16px;color:#6b7280;font-size:11px">Sent from GPU Admin Dashboard · {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</p>
    </div></div>"""

    payload = {
        "personalizations": [{"to": [{"email": r} for r in recipients]}],
        "from": {"email": SENDGRID_MAIL_FROM, "name": "GPU Dashboard"},
        "subject": subject,
        "content": [{"type": "text/html", "value": html}],
    }

    try:
        data = json.dumps(payload).encode()
        req = URLRequest(
            "https://api.sendgrid.com/v3/mail/send",
            data=data,
            headers={
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        ctx = ssl.create_default_context()
        resp = await asyncio.get_event_loop().run_in_executor(None, lambda: urlopen(req, context=ctx, timeout=15))
        status = resp.status
        return {"ok": status in (200, 201, 202), "detail": f"HTTP {status}"}
    except Exception as e:
        return {"ok": False, "detail": str(e)[:200]}


def evaluate_alerts_from_data(data, server_name):
    """Evaluate alert_configs against server data. Returns list of triggered alerts."""
    if not data or data.get("status") == "offline":
        return []
    triggered = []
    gpus = data.get("gpus", [])
    for g in gpus:
        temp = g.get("temp", 0) or 0
        util = g.get("gpu_util", 0)
        mem_pct = g["mem_used"] / max(g["mem_total"], 1) * 100
        pow_pct = g["power_draw"] / max(g["power_limit"], 1) * 100

        checks = [
            ("gpu_temp_warning", temp, ">=", f"GPU {g['index']} temp: {temp}°C"),
            ("gpu_temp_critical", temp, ">=", f"GPU {g['index']} temp: {temp}°C"),
            ("gpu_util_low", util, "<=", f"GPU {g['index']} util: {util}%"),
            ("vram_high", mem_pct, ">=", f"GPU {g['index']} VRAM: {mem_pct:.0f}%"),
            ("power_high", pow_pct, ">=", f"GPU {g['index']} power: {pow_pct:.0f}%"),
        ]
        for rule_id, value, op, detail in checks:
            cfg = alert_configs.get(rule_id)
            if not cfg or not cfg["enabled"]:
                continue
            tripped = (value >= cfg["threshold"]) if op == ">=" else (value <= cfg["threshold"])
            if tripped:
                triggered.append({
                    "rule": rule_id,
                    "severity": cfg["severity"],
                    "title": f"{cfg['name']} on {server_name}",
                    "detail": detail,
                    "server": server_name,
                    "gpu": g["index"],
                    "value": round(value, 1),
                    "threshold": cfg["threshold"],
                })
    return triggered


async def alert_monitor_loop():
    """Background loop: evaluate alerts every 60s, send emails for new/changed alerts."""
    await asyncio.sleep(10)  # wait for app to warm up
    while True:
        try:
            if email_config["enabled"] and email_config["recipients"] and SENDGRID_API_KEY:
                now = time.time()
                cooldown_sec = email_config["cooldown_minutes"] * 60
                all_triggered = []

                for sk in SERVERS:
                    try:
                        data = await asyncio.wait_for(fetch_gpu_data(sk), timeout=15)
                    except Exception:
                        data = None
                    alerts = evaluate_alerts_from_data(data, SERVERS[sk]["name"])
                    all_triggered.extend(alerts)

                # Filter by cooldown
                to_send = []
                for a in all_triggered:
                    key = f"{a['rule']}_{a['server']}_{a['gpu']}"
                    last = email_cooldowns.get(key, 0)
                    if now - last >= cooldown_sec:
                        to_send.append(a)
                        email_cooldowns[key] = now

                if to_send:
                    # Group: only send critical+warning
                    critical = [a for a in to_send if a["severity"] in ("critical", "warning")]
                    if critical:
                        subject = f"[GPU Alert] {len(critical)} alert(s) — {critical[0]['title']}"
                        result = await send_alert_email(email_config["recipients"], subject, critical)
                        email_history.appendleft({
                            "t": now,
                            "to": email_config["recipients"][:],
                            "subject": subject,
                            "alerts_count": len(critical),
                            "status": "sent" if result["ok"] else "failed",
                            "detail": result["detail"],
                        })
        except Exception as e:
            email_history.appendleft({
                "t": time.time(),
                "to": [],
                "subject": "Monitor error",
                "alerts_count": 0,
                "status": "error",
                "detail": str(e)[:200],
            })
        await asyncio.sleep(60)


@app.on_event("startup")
async def start_alert_monitor():
    asyncio.create_task(alert_monitor_loop())


@app.get("/api/email-config")
async def get_email_config():
    """Get email alert configuration and history."""
    return {
        "enabled": email_config["enabled"],
        "recipients": email_config["recipients"],
        "cooldown_minutes": email_config["cooldown_minutes"],
        "sendgrid_configured": bool(SENDGRID_API_KEY),
        "mail_from": SENDGRID_MAIL_FROM,
        "history": list(email_history),
    }


@app.post("/api/email-config")
async def update_email_config(request: Request):
    """Update email alert configuration."""
    body = await request.json()
    if "enabled" in body:
        email_config["enabled"] = bool(body["enabled"])
    if "recipients" in body:
        email_config["recipients"] = [r.strip() for r in body["recipients"] if r.strip()]
    if "cooldown_minutes" in body:
        email_config["cooldown_minutes"] = max(1, min(120, int(body["cooldown_minutes"])))
    if "add_recipient" in body:
        r = body["add_recipient"].strip()
        if r and r not in email_config["recipients"]:
            email_config["recipients"].append(r)
    if "remove_recipient" in body:
        r = body["remove_recipient"].strip()
        if r in email_config["recipients"]:
            email_config["recipients"].remove(r)
    return {"ok": True, **email_config}


@app.post("/api/email-test")
async def send_test_email(request: Request):
    """Send a test alert email."""
    body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    recipients = body.get("recipients", email_config["recipients"])
    test_alerts = [{
        "severity": "info",
        "title": "Test Alert",
        "detail": "This is a test email from GPU Admin Dashboard. If you received this, email alerts are working correctly.",
    }]
    result = await send_alert_email(recipients, "[GPU Dashboard] Test Alert", test_alerts)
    email_history.appendleft({
        "t": time.time(),
        "to": recipients[:],
        "subject": "[GPU Dashboard] Test Alert",
        "alerts_count": 1,
        "status": "sent" if result["ok"] else "failed",
        "detail": result["detail"],
    })
    return result


@app.get("/api/executive")
async def get_executive():
    """Executive Summary — health score, daily report, H200 vs RTX 5090 comparison."""
    from datetime import datetime, timezone
    now = time.time()
    result = {
        "health_score": {},
        "daily_summary": {},
        "comparison": {},
        "trends": {},
    }

    # ── 1. GPU Fleet Health Score (0-100) ──
    scores = {}
    for sk in SERVERS:
        sk_hist = list(history[sk])
        if not sk_hist:
            scores[sk] = {"score": 0, "breakdown": {}, "server_name": SERVERS[sk]["name"]}
            continue
        recent = sk_hist[-min(60, len(sk_hist)):]
        avg_util = sum(p.get("avg_util", 0) for p in recent) / len(recent)
        max_temp = max(p.get("max_temp", 0) for p in recent)
        avg_vram = sum(p.get("vram_pct", 0) for p in recent) / len(recent)

        # Error rate
        errs = list(error_history.get(sk, []))
        err_rate = errs[-1].get("rate", 0) if errs else 0

        # Uptime score (based on history gaps)
        uptime_score = 100  # assume 100% unless we detect gaps

        # Component scores (higher is better)
        util_score = max(0, 100 - abs(avg_util - 50) * 1.5)  # Optimal ~50%
        temp_score = max(0, 100 - max(0, max_temp - 60) * 3)  # Penalize >60C
        vram_score = max(0, 100 - max(0, avg_vram - 70) * 2)  # Penalize >70%
        error_score = max(0, 100 - err_rate * 10)
        latency_score = 100
        # Check latency
        ttft_vals = [p.get("ttft_ms", 0) for p in recent if p.get("ttft_ms", 0) > 0]
        if ttft_vals:
            avg_ttft = sum(ttft_vals) / len(ttft_vals)
            latency_score = max(0, 100 - max(0, avg_ttft - 200) * 0.5)

        # Weighted average
        overall = round(
            util_score * 0.20 +
            temp_score * 0.15 +
            vram_score * 0.15 +
            error_score * 0.20 +
            latency_score * 0.20 +
            uptime_score * 0.10
        , 1)

        scores[sk] = {
            "score": overall,
            "server_name": SERVERS[sk]["name"],
            "breakdown": {
                "utilization": round(util_score, 1),
                "temperature": round(temp_score, 1),
                "vram": round(vram_score, 1),
                "error_rate": round(error_score, 1),
                "latency": round(latency_score, 1),
                "uptime": round(uptime_score, 1),
            },
            "metrics": {
                "avg_util": round(avg_util, 1),
                "max_temp": max_temp,
                "avg_vram": round(avg_vram, 1),
                "error_rate": round(err_rate, 2),
                "avg_ttft": round(sum(ttft_vals) / len(ttft_vals), 1) if ttft_vals else 0,
            }
        }

    # Fleet-wide score
    all_scores = [s["score"] for s in scores.values() if s["score"] > 0]
    fleet_score = round(sum(all_scores) / max(len(all_scores), 1), 1)
    result["health_score"] = {
        "fleet": fleet_score,
        "grade": "A" if fleet_score >= 90 else "B" if fleet_score >= 75 else "C" if fleet_score >= 60 else "D" if fleet_score >= 40 else "F",
        "servers": scores,
    }

    # ── 2. Daily Summary ──
    total_calls = 0
    total_samples = 0
    all_util = []
    all_temps = []
    all_ttft = []
    all_e2e = []
    peak_calls = 0

    for sk in SERVERS:
        for p in list(history[sk]):
            total_samples += 1
            calls = p.get("calls", 0)
            total_calls += calls
            if calls > peak_calls:
                peak_calls = calls
            all_util.append(p.get("avg_util", 0))
            all_temps.append(p.get("max_temp", 0))
            if p.get("ttft_ms", 0) > 0:
                all_ttft.append(p["ttft_ms"])
            if p.get("e2e_s", 0) > 0:
                all_e2e.append(p["e2e_s"])

    # Hourly data for total calls served
    total_calls_served = 0
    for sk in SERVERS:
        for hk, h in hourly_stats.get(sk, {}).items():
            delta = h.get("_vllm_reqs_end", 0) - h.get("_vllm_reqs_start", 0)
            total_calls_served += delta

    data_hours = 0
    if total_samples >= 2:
        all_hist = []
        for sk in SERVERS:
            all_hist.extend(list(history[sk]))
        if all_hist:
            data_hours = (all_hist[-1]["t"] - all_hist[0]["t"]) / 3600

    result["daily_summary"] = {
        "data_hours": round(data_hours, 1),
        "total_samples": total_samples,
        "total_requests_served": total_calls_served,
        "peak_concurrent_calls": peak_calls,
        "avg_concurrent_calls": round(total_calls / max(total_samples, 1), 1),
        "avg_gpu_util": round(sum(all_util) / max(len(all_util), 1), 1),
        "max_temp": max(all_temps) if all_temps else 0,
        "avg_ttft_ms": round(sum(all_ttft) / max(len(all_ttft), 1), 1) if all_ttft else 0,
        "avg_e2e_s": round(sum(all_e2e) / max(len(all_e2e), 1), 3) if all_e2e else 0,
        "cost_today": round(GPU_MONTHLY_COST / 30, 2),
        "cost_per_request": round(GPU_MONTHLY_COST / 30 / max(total_calls_served, 1), 4) if total_calls_served > 0 else 0,
        "uptime_pct": 100.0,  # assume unless service_uptime shows otherwise
    }

    # ── 3. H200 vs RTX 5090 Comparison ──
    for sk in SERVERS:
        sk_hist = list(history[sk])
        if not sk_hist:
            result["comparison"][sk] = {"server_name": SERVERS[sk]["name"], "no_data": True}
            continue
        recent = sk_hist[-min(120, len(sk_hist)):]
        avg_util = sum(p.get("avg_util", 0) for p in recent) / len(recent)
        avg_calls = sum(p.get("calls", 0) for p in recent) / len(recent)
        peak_calls_sk = max(p.get("calls", 0) for p in recent)
        avg_temp = sum(p.get("max_temp", 0) for p in recent) / len(recent)
        avg_power = sum(p.get("total_power", 0) for p in recent) / len(recent)
        avg_vram = sum(p.get("vram_pct", 0) for p in recent) / len(recent)
        ttft_vals = [p.get("ttft_ms", 0) for p in recent if p.get("ttft_ms", 0) > 0]
        e2e_vals = [p.get("e2e_s", 0) for p in recent if p.get("e2e_s", 0) > 0]
        gpu_count = recent[-1].get("gpu_count", 8)

        result["comparison"][sk] = {
            "server_name": SERVERS[sk]["name"],
            "gpu_model": SERVERS[sk]["gpu_model"],
            "gpu_count": gpu_count,
            "avg_util": round(avg_util, 1),
            "avg_calls": round(avg_calls, 1),
            "peak_calls": peak_calls_sk,
            "avg_temp": round(avg_temp, 1),
            "avg_power_w": round(avg_power, 0),
            "avg_vram_pct": round(avg_vram, 1),
            "avg_ttft_ms": round(sum(ttft_vals) / len(ttft_vals), 1) if ttft_vals else 0,
            "avg_e2e_s": round(sum(e2e_vals) / len(e2e_vals), 3) if e2e_vals else 0,
        }

    # ── 4. Trend Arrows ──
    for sk in SERVERS:
        sk_hist = list(history[sk])
        if len(sk_hist) < 20:
            result["trends"][sk] = {"server_name": SERVERS[sk]["name"], "no_data": True}
            continue
        mid = len(sk_hist) // 2
        first_half = sk_hist[:mid]
        second_half = sk_hist[mid:]

        def avg(arr, key):
            vals = [p.get(key, 0) for p in arr if p.get(key, 0) > 0]
            return sum(vals) / max(len(vals), 1) if vals else 0

        def trend(old, new):
            if old == 0:
                return "stable"
            pct = (new - old) / old * 100
            if pct > 10:
                return "up"
            elif pct < -10:
                return "down"
            return "stable"

        u1, u2 = avg(first_half, "avg_util"), avg(second_half, "avg_util")
        t1, t2 = avg(first_half, "max_temp"), avg(second_half, "max_temp")
        c1, c2 = avg(first_half, "calls"), avg(second_half, "calls")
        ttft1, ttft2 = avg(first_half, "ttft_ms"), avg(second_half, "ttft_ms")

        result["trends"][sk] = {
            "server_name": SERVERS[sk]["name"],
            "utilization": {"trend": trend(u1, u2), "old": round(u1, 1), "new": round(u2, 1)},
            "temperature": {"trend": trend(t1, t2), "old": round(t1, 1), "new": round(t2, 1)},
            "calls": {"trend": trend(c1, c2), "old": round(c1, 1), "new": round(c2, 1)},
            "ttft": {"trend": trend(ttft1, ttft2), "old": round(ttft1, 1), "new": round(ttft2, 1)},
        }

    return result


@app.get("/api/network")
async def get_network():
    """Network & I/O — GPU topology, network throughput, model loading, storage I/O."""
    result = {
        "gpu_topology": {},
        "network_throughput": {},
        "storage_io": {},
        "model_loading": {},
    }

    # ── 1. GPU-to-GPU Topology & Communication ──
    for sk, server in SERVERS.items():
        try:
            topo_out = await asyncio.wait_for(
                run_command(server, "nvidia-smi topo -m 2>/dev/null"),
                timeout=10
            )
            # Parse topology matrix
            lines = [l for l in topo_out.strip().split("\n") if l.strip()]
            topo_matrix = []
            headers = []
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split("\t") if p.strip()]
                if i == 0:
                    headers = parts
                elif parts:
                    row_label = parts[0] if parts else ""
                    row_data = parts[1:] if len(parts) > 1 else []
                    topo_matrix.append({"label": row_label, "connections": row_data})

            # Get NVLink status
            nvlink_out = await asyncio.wait_for(
                run_command(server, "nvidia-smi nvlink -s 2>/dev/null"),
                timeout=10
            )

            result["gpu_topology"][sk] = {
                "server_name": server["name"],
                "headers": headers,
                "matrix": topo_matrix,
                "nvlink_raw": nvlink_out[:2000] if nvlink_out else "Not available",
            }
        except Exception as e:
            result["gpu_topology"][sk] = {"server_name": server["name"], "error": str(e)}

    # ── 2. Network Throughput Timeline ──
    for sk in SERVERS:
        sk_hist = list(history[sk])
        if len(sk_hist) < 2:
            result["network_throughput"][sk] = {"server_name": SERVERS[sk]["name"], "timeline": []}
            continue

        timeline = []
        prev = None
        for p in sk_hist[-120:]:
            if prev and p.get("net_rx", 0) > 0:
                dt = max(p["t"] - prev["t"], 0.1)
                rx_bps = (p["net_rx"] - prev["net_rx"]) * 8 / dt if p["net_rx"] >= prev["net_rx"] else 0
                tx_bps = (p["net_tx"] - prev["net_tx"]) * 8 / dt if p["net_tx"] >= prev["net_tx"] else 0
                timeline.append({
                    "t": p["t"],
                    "rx_bps": round(rx_bps),
                    "tx_bps": round(tx_bps),
                    "rx_mbps": round(rx_bps / 1e6, 2),
                    "tx_mbps": round(tx_bps / 1e6, 2),
                })
            prev = p

        # Anomaly detection: flag points > 2x average
        if timeline:
            avg_rx = sum(t["rx_bps"] for t in timeline) / len(timeline)
            avg_tx = sum(t["tx_bps"] for t in timeline) / len(timeline)
            peak_rx = max(t["rx_bps"] for t in timeline)
            peak_tx = max(t["tx_bps"] for t in timeline)
            for t in timeline:
                t["rx_anomaly"] = t["rx_bps"] > avg_rx * 3 if avg_rx > 0 else False
                t["tx_anomaly"] = t["tx_bps"] > avg_tx * 3 if avg_tx > 0 else False
        else:
            avg_rx = avg_tx = peak_rx = peak_tx = 0

        result["network_throughput"][sk] = {
            "server_name": SERVERS[sk]["name"],
            "timeline": timeline[-60:],
            "avg_rx_mbps": round(avg_rx / 1e6, 2),
            "avg_tx_mbps": round(avg_tx / 1e6, 2),
            "peak_rx_mbps": round(peak_rx / 1e6, 2),
            "peak_tx_mbps": round(peak_tx / 1e6, 2),
            "anomalies": sum(1 for t in timeline if t.get("rx_anomaly") or t.get("tx_anomaly")),
        }

    # ── 3. Storage I/O ──
    for sk, server in SERVERS.items():
        try:
            # Get disk I/O stats from /proc/diskstats + current throughput via iostat-like parsing
            io_out = await asyncio.wait_for(
                run_command(server, "cat /proc/diskstats 2>/dev/null | head -20"),
                timeout=10
            )
            # Get disk usage
            disk_hist = list(disk_history.get(sk, []))
            disk_info = {}
            if disk_hist:
                latest = disk_hist[-1]
                disk_info = {
                    "used_pct": latest.get("pct", 0),
                    "used_bytes": latest.get("used_bytes", 0),
                    "total_bytes": latest.get("total_bytes", 0),
                }
                # Trend
                if len(disk_hist) >= 2:
                    first = disk_hist[0]
                    last = disk_hist[-1]
                    dt_hours = max((last["t"] - first["t"]) / 3600, 0.01)
                    growth_pct_hour = (last["pct"] - first["pct"]) / dt_hours
                    remaining = 100 - last["pct"]
                    days_until_full = remaining / max(growth_pct_hour * 24, 0.0001) if growth_pct_hour > 0 else 9999
                    disk_info["growth_pct_hour"] = round(growth_pct_hour, 4)
                    disk_info["days_until_full"] = round(min(days_until_full, 9999), 1)

            # Parse diskstats for I/O activity
            io_stats = []
            for line in io_out.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 14:
                    dev = parts[2]
                    if dev.startswith("loop") or dev.startswith("ram"):
                        continue
                    reads = int(parts[3])
                    read_sectors = int(parts[5])
                    writes = int(parts[7])
                    write_sectors = int(parts[9])
                    io_stats.append({
                        "device": dev,
                        "reads": reads,
                        "read_mb": round(read_sectors * 512 / 1e6, 1),
                        "writes": writes,
                        "write_mb": round(write_sectors * 512 / 1e6, 1),
                    })

            result["storage_io"][sk] = {
                "server_name": SERVERS[sk]["name"],
                "disk": disk_info,
                "devices": io_stats[:10],  # top 10 devices
            }
        except Exception as e:
            result["storage_io"][sk] = {"server_name": SERVERS[sk]["name"], "error": str(e)}

    # ── 4. Model Loading & Warmup ──
    # Track vLLM model info and KV cache warmup from history
    for sk in SERVERS:
        sk_hist = list(history[sk])
        if not sk_hist:
            continue
        # KV cache warmup: how quickly does KV cache fill from 0 after restart
        kv_values = [p.get("kv_cache", 0) for p in sk_hist]
        first_nonzero = next((i for i, v in enumerate(kv_values) if v > 0), -1)
        warmup_samples = 0
        if first_nonzero >= 0 and len(kv_values) > first_nonzero + 1:
            # Count how many samples until KV cache stabilizes (change < 0.1 for 10 consecutive)
            stable_count = 0
            for i in range(first_nonzero + 1, len(kv_values)):
                if abs(kv_values[i] - kv_values[i-1]) < 0.1:
                    stable_count += 1
                    if stable_count >= 10:
                        warmup_samples = i - first_nonzero
                        break
                else:
                    stable_count = 0

        # Current model info from vLLM
        latest_data = None
        try:
            latest_data = await asyncio.wait_for(fetch_gpu_data(sk), timeout=10)
        except Exception:
            pass

        models = []
        if latest_data and latest_data.get("vllm"):
            for v in latest_data["vllm"]:
                models.append({
                    "port": v.get("port"),
                    "model_name": v.get("model_name", "unknown"),
                    "kv_cache_usage": v.get("kv_cache_usage", 0),
                    "requests_running": v.get("requests_running", 0),
                    "total_requests": v.get("total_requests", 0),
                })

        result["model_loading"][sk] = {
            "server_name": SERVERS[sk]["name"],
            "models": models,
            "kv_warmup_samples": warmup_samples,
            "kv_warmup_seconds": warmup_samples * 3,  # ~3s per sample
            "current_kv": kv_values[-1] if kv_values else 0,
            "kv_history": kv_values[-60:],
        }

    return result


@app.get("/api/quality")
async def get_quality():
    """Call Quality Deep Dive — waterfall, degradation heatmap, latency curve, SLA compliance."""
    result = {
        "waterfall": [],
        "degradation_map": [],
        "latency_curve": [],
        "sla_compliance": {},
    }

    # SLA targets
    SLA = {
        "ttft_ms": {"target": 300, "label": "TTFT < 300ms"},
        "itl_ms": {"target": 60, "label": "ITL < 60ms"},
        "e2e_s": {"target": 3.0, "label": "E2E < 3s"},
        "queue_s": {"target": 0.5, "label": "Queue < 500ms"},
    }

    all_points = []
    for sk in SERVERS:
        for p in list(history[sk]):
            if p.get("ttft_ms", 0) > 0 or p.get("e2e_s", 0) > 0:
                all_points.append({**p, "server": sk})

    # ── 1. Per-Call Waterfall (pipeline breakdown over time) ──
    recent_h200 = [p for p in list(history.get("h200", []))[-120:] if p.get("e2e_s", 0) > 0]
    for p in recent_h200[-60:]:
        ttft_s = p.get("ttft_ms", 0) / 1000
        e2e = p.get("e2e_s", 0)
        queue = p.get("queue_s", 0)
        generation = max(0, e2e - ttft_s)
        total = queue + e2e
        result["waterfall"].append({
            "t": p["t"],
            "queue_s": round(queue, 4),
            "ttft_s": round(ttft_s, 4),
            "generation_s": round(generation, 4),
            "total_s": round(total, 4),
            "calls": p.get("calls", 0),
            "queue_pct": round(queue / max(total, 0.001) * 100, 1),
            "ttft_pct": round(ttft_s / max(total, 0.001) * 100, 1),
            "gen_pct": round(generation / max(total, 0.001) * 100, 1),
        })

    # ── 2. Quality Degradation Map (heatmap: hour x utilization bucket) ──
    from datetime import datetime, timezone
    heatmap = {}
    for p in all_points:
        dt = datetime.fromtimestamp(p["t"], tz=timezone.utc)
        hour = dt.hour
        util = p.get("avg_util", 0)
        util_bucket = int(util // 10) * 10
        key = f"{hour}_{util_bucket}"
        if key not in heatmap:
            heatmap[key] = {"ttft": [], "e2e": [], "itl": []}
        if p.get("ttft_ms", 0) > 0:
            heatmap[key]["ttft"].append(p["ttft_ms"])
        if p.get("e2e_s", 0) > 0:
            heatmap[key]["e2e"].append(p["e2e_s"])
        if p.get("itl_ms", 0) > 0:
            heatmap[key]["itl"].append(p["itl_ms"])

    for key, vals in heatmap.items():
        hour, util_bucket = key.split("_")
        result["degradation_map"].append({
            "hour": int(hour),
            "util_bucket": int(util_bucket),
            "avg_ttft_ms": round(sum(vals["ttft"]) / len(vals["ttft"]), 1) if vals["ttft"] else 0,
            "avg_e2e_s": round(sum(vals["e2e"]) / len(vals["e2e"]), 3) if vals["e2e"] else 0,
            "avg_itl_ms": round(sum(vals["itl"]) / len(vals["itl"]), 1) if vals["itl"] else 0,
            "samples": len(vals["ttft"]) + len(vals["e2e"]),
        })
    result["degradation_map"].sort(key=lambda x: (x["hour"], x["util_bucket"]))

    # ── 3. Concurrent Call vs Latency Curve ──
    call_groups = {}
    for p in all_points:
        c = p.get("calls", 0)
        if c not in call_groups:
            call_groups[c] = {"ttft": [], "itl": [], "e2e": [], "queue": [], "kv": []}
        if p.get("ttft_ms", 0) > 0:
            call_groups[c]["ttft"].append(p["ttft_ms"])
        if p.get("itl_ms", 0) > 0:
            call_groups[c]["itl"].append(p["itl_ms"])
        if p.get("e2e_s", 0) > 0:
            call_groups[c]["e2e"].append(p["e2e_s"])
        if p.get("queue_s", 0) > 0:
            call_groups[c]["queue"].append(p["queue_s"])
        if p.get("kv_cache", 0) > 0:
            call_groups[c]["kv"].append(p["kv_cache"])

    prev_e2e = 0
    for c in sorted(call_groups.keys()):
        g = call_groups[c]
        avg_e2e = sum(g["e2e"]) / len(g["e2e"]) if g["e2e"] else 0
        avg_ttft = sum(g["ttft"]) / len(g["ttft"]) if g["ttft"] else 0
        is_inflection = prev_e2e > 0 and avg_e2e > prev_e2e * 1.5
        result["latency_curve"].append({
            "concurrent_calls": c,
            "avg_ttft_ms": round(avg_ttft, 1),
            "avg_itl_ms": round(sum(g["itl"]) / len(g["itl"]), 1) if g["itl"] else 0,
            "avg_e2e_s": round(avg_e2e, 3),
            "avg_queue_s": round(sum(g["queue"]) / len(g["queue"]), 4) if g["queue"] else 0,
            "avg_kv_cache": round(sum(g["kv"]) / len(g["kv"]), 1) if g["kv"] else 0,
            "samples": len(g["ttft"]) + len(g["e2e"]),
            "inflection": is_inflection,
        })
        if avg_e2e > 0:
            prev_e2e = avg_e2e

    # ── 4. SLA Compliance Tracker ──
    for metric, sla in SLA.items():
        values = []
        violations = []
        for p in all_points:
            v = p.get(metric, 0)
            if v <= 0:
                continue
            values.append(v)
            if v > sla["target"]:
                violations.append({"t": p["t"], "value": round(v, 2), "server": p.get("server", "")})

        total = len(values)
        violated = len(violations)
        compliance = round((total - violated) / max(total, 1) * 100, 2)

        hourly_violations = {}
        for viol in violations:
            dt = datetime.fromtimestamp(viol["t"], tz=timezone.utc)
            h = dt.strftime("%H:00")
            hourly_violations[h] = hourly_violations.get(h, 0) + 1

        worst_hours = sorted(hourly_violations.items(), key=lambda x: -x[1])[:5]

        result["sla_compliance"][metric] = {
            "label": sla["label"],
            "target": sla["target"],
            "total_samples": total,
            "violations": violated,
            "compliance_pct": compliance,
            "worst_hours": [{"hour": h, "violations": c} for h, c in worst_hours],
            "recent_violations": violations[-10:],
        }

    return result


@app.get("/api/capacity")
async def get_capacity():
    """Capacity Planner — What-If simulator, ROI, scale recommendations, cloud comparison."""
    result = {
        "what_if": [],
        "roi": {},
        "scale_recommendation": {},
        "cloud_comparison": [],
        "per_server": {},
    }

    # Gather recent data across all servers
    all_recent = {}
    for sk in SERVERS:
        sk_hist = list(history[sk])
        if not sk_hist:
            continue
        recent = sk_hist[-min(120, len(sk_hist)):]
        avg_util = sum(p.get("avg_util", 0) for p in recent) / len(recent)
        avg_calls = sum(p.get("calls", 0) for p in recent) / len(recent)
        peak_calls = max(p.get("calls", 0) for p in recent)
        avg_kv = sum(p.get("kv_cache", 0) for p in recent) / len(recent)
        avg_vram = sum(p.get("vram_pct", 0) for p in recent) / len(recent)
        avg_ttft = sum(p.get("ttft_ms", 0) for p in recent if p.get("ttft_ms", 0) > 0)
        ttft_count = sum(1 for p in recent if p.get("ttft_ms", 0) > 0)
        avg_ttft = avg_ttft / ttft_count if ttft_count > 0 else 0
        avg_e2e = sum(p.get("e2e_s", 0) for p in recent if p.get("e2e_s", 0) > 0)
        e2e_count = sum(1 for p in recent if p.get("e2e_s", 0) > 0)
        avg_e2e = avg_e2e / e2e_count if e2e_count > 0 else 0
        gpu_count = recent[-1].get("gpu_count", 8)
        avg_power = sum(p.get("total_power", 0) for p in recent) / len(recent)

        all_recent[sk] = {
            "avg_util": avg_util, "avg_calls": avg_calls, "peak_calls": peak_calls,
            "avg_kv": avg_kv, "avg_vram": avg_vram, "avg_ttft": avg_ttft,
            "avg_e2e": avg_e2e, "gpu_count": gpu_count, "avg_power": avg_power,
        }

    total_avg_util = sum(d["avg_util"] for d in all_recent.values()) / max(len(all_recent), 1)
    total_avg_calls = sum(d["avg_calls"] for d in all_recent.values())
    total_peak_calls = sum(d["peak_calls"] for d in all_recent.values())
    total_gpu_count = sum(d["gpu_count"] for d in all_recent.values())

    # Util per call across fleet
    util_per_call = total_avg_util / max(total_avg_calls, 0.1)

    # ── 1. What-If Simulator ──
    for target_calls in [1, 2, 5, 10, 20, 50, 75, 100, 150, 200]:
        projected_util = util_per_call * target_calls
        # Latency degrades exponentially as utilization approaches 100%
        load_factor = min(projected_util / 100, 0.99)
        # M/M/c queuing approximation: latency ~ 1/(1-rho) * base_latency
        latency_multiplier = 1 / max(1 - load_factor, 0.01) if load_factor < 1 else 100
        base_ttft = sum(d["avg_ttft"] for d in all_recent.values()) / max(len(all_recent), 1)
        base_e2e = sum(d["avg_e2e"] for d in all_recent.values()) / max(len(all_recent), 1)
        proj_ttft = base_ttft * min(latency_multiplier, 50)
        proj_e2e = base_e2e * min(latency_multiplier, 50)

        # KV cache scales linearly with calls
        kv_per_call = sum(d["avg_kv"] for d in all_recent.values()) / max(total_avg_calls, 0.1)
        proj_kv = kv_per_call * target_calls

        # GPUs needed to keep util < 80%
        gpus_needed = max(1, int((util_per_call * target_calls / 80) * total_gpu_count + 0.5)) if util_per_call > 0 else total_gpu_count

        feasible = projected_util < 95 and proj_kv < 95
        quality = "good" if projected_util < 60 else "degraded" if projected_util < 85 else "poor"

        result["what_if"].append({
            "target_calls": target_calls,
            "projected_gpu_util": round(min(projected_util, 100), 1),
            "projected_ttft_ms": round(proj_ttft, 1),
            "projected_e2e_s": round(proj_e2e, 3),
            "projected_kv_cache": round(min(proj_kv, 100), 1),
            "gpus_needed_for_80pct": gpus_needed,
            "feasible": feasible,
            "quality": quality,
        })

    # ── 2. GPU ROI Calculator ──
    # Assume revenue per call-minute (configurable, default estimate)
    # Using cost data to compute break-even
    calls_per_hour = total_avg_calls * 60  # assuming avg call ~ 1 min
    cost_per_call = COST_PER_HOUR / max(calls_per_hour, 0.01)
    # Break-even: at what concurrent call level does cost per call drop below target
    breakeven_calls = []
    for target_cost in [0.01, 0.02, 0.05, 0.10, 0.25, 0.50]:
        needed = COST_PER_HOUR / 60 / target_cost if target_cost > 0 else 999
        breakeven_calls.append({"target_cost_per_min": target_cost, "min_concurrent_calls": round(needed, 1)})

    result["roi"] = {
        "monthly_cost": GPU_MONTHLY_COST,
        "hourly_cost": COST_PER_HOUR,
        "current_avg_calls": round(total_avg_calls, 1),
        "current_cost_per_call_min": round(COST_PER_HOUR / 60 / max(total_avg_calls, 0.01), 4),
        "current_cost_per_call_hour": round(COST_PER_HOUR / max(total_avg_calls, 0.01), 2),
        "idle_waste_hourly": round(COST_PER_HOUR * max(0, 1 - total_avg_util / 100), 2),
        "idle_waste_monthly": round(GPU_MONTHLY_COST * max(0, 1 - total_avg_util / 100), 0),
        "effective_util_pct": round(total_avg_util, 1),
        "breakeven": breakeven_calls,
        "revenue_needed_monthly": GPU_MONTHLY_COST,
        "calls_to_break_even_at_1c": round(GPU_MONTHLY_COST / 0.01 / 30 / 24, 0),  # calls/hour at $0.01/call
    }

    # ── 3. Scale-Up/Down Recommendations ──
    # Based on peak utilization and current GPU count
    peak_util = max((d.get("avg_util", 0) for d in all_recent.values()), default=0)
    # Find p95 utilization from history
    all_utils = []
    for sk in SERVERS:
        for p in list(history[sk])[-300:]:
            all_utils.append(p.get("avg_util", 0))
    all_utils.sort()
    p95_util = all_utils[int(len(all_utils) * 0.95)] if all_utils else 0
    p99_util = all_utils[int(len(all_utils) * 0.99)] if all_utils else 0

    # Min GPUs to handle p95 load at 80% target
    min_gpus_p95 = max(1, int(total_gpu_count * (p95_util / 80) + 0.5)) if p95_util > 0 else total_gpu_count
    min_gpus_p99 = max(1, int(total_gpu_count * (p99_util / 80) + 0.5)) if p99_util > 0 else total_gpu_count
    savings_if_downscale = round((total_gpu_count - min_gpus_p95) * COST_PER_GPU_HOUR * 24 * 30, 0) if min_gpus_p95 < total_gpu_count else 0

    action = "optimal"
    detail = "Current GPU count matches workload."
    if p95_util > 85:
        action = "scale_up"
        extra = max(1, int(total_gpu_count * (p95_util / 70) - total_gpu_count + 0.5))
        detail = f"P95 utilization is {p95_util:.0f}%. Add {extra} GPUs to bring p95 below 70%."
    elif p95_util < 30 and total_gpu_count > 4:
        action = "scale_down"
        detail = f"P95 utilization is only {p95_util:.0f}%. You could reduce to {min_gpus_p95} GPUs and save ${savings_if_downscale:,.0f}/month."

    result["scale_recommendation"] = {
        "action": action,
        "detail": detail,
        "current_gpus": total_gpu_count,
        "p95_util": round(p95_util, 1),
        "p99_util": round(p99_util, 1),
        "min_gpus_for_p95": min_gpus_p95,
        "min_gpus_for_p99": min_gpus_p99,
        "monthly_savings_if_downscale": savings_if_downscale,
        "peak_calls": total_peak_calls,
        "avg_calls": round(total_avg_calls, 1),
    }

    # ── 4. Cloud Cost Comparison ──
    # Compare against major cloud GPU pricing (approximate 2024-2025 rates)
    cloud_providers = [
        {"provider": "AWS p5.48xlarge", "gpu": "8x H100 80GB", "hourly": 98.32, "monthly": round(98.32 * 730, 0), "note": "On-demand, closest to H200"},
        {"provider": "AWS p5e.48xlarge", "gpu": "8x H200 141GB", "hourly": 115.00, "monthly": round(115.00 * 730, 0), "note": "On-demand H200"},
        {"provider": "GCP a3-highgpu-8g", "gpu": "8x H100 80GB", "hourly": 98.32, "monthly": round(98.32 * 730, 0), "note": "On-demand"},
        {"provider": "Lambda Labs H100", "gpu": "8x H100 80GB", "hourly": 27.92, "monthly": round(27.92 * 730, 0), "note": "Reserved"},
        {"provider": "CoreWeave H100", "gpu": "8x H100 80GB", "hourly": 25.20, "monthly": round(25.20 * 730, 0), "note": "1-year commitment"},
    ]
    your_equiv_hourly = COST_PER_HOUR
    for cp in cloud_providers:
        # Scale to match your GPU count (16 GPUs = 2x 8-GPU instances)
        scaled_monthly = cp["monthly"] * (total_gpu_count / 8)
        cp["scaled_monthly"] = round(scaled_monthly, 0)
        cp["savings_vs_you"] = round(scaled_monthly - GPU_MONTHLY_COST, 0)
        cp["savings_pct"] = round((1 - GPU_MONTHLY_COST / max(scaled_monthly, 1)) * 100, 1) if scaled_monthly > GPU_MONTHLY_COST else round((GPU_MONTHLY_COST / max(scaled_monthly, 1) - 1) * -100, 1)
    result["cloud_comparison"] = cloud_providers

    # Per-server breakdown
    for sk, d in all_recent.items():
        result["per_server"][sk] = {
            "server_name": SERVERS[sk]["name"],
            "gpu_count": d["gpu_count"],
            "avg_util": round(d["avg_util"], 1),
            "avg_calls": round(d["avg_calls"], 1),
            "peak_calls": d["peak_calls"],
            "avg_kv": round(d["avg_kv"], 1),
            "avg_vram": round(d["avg_vram"], 1),
        }

    return result


@app.get("/api/proactive")
async def get_proactive_alerts():
    now = time.time()
    result = {
        "predictive_alerts": [],
        "error_rates": {},
        "process_health": {},
        "service_uptime": {},
        "cluster_latency": {},
        "thermal_throttling": {},
        "call_forecast": {},
        "disk_trends": {},
    }

    # ── 1. Trend-Based Predictive Alerts ──
    predictions = []
    for sk in SERVERS:
        sk_hist = list(history[sk])
        if len(sk_hist) < 10:
            continue
        recent = sk_hist[-60:]  # last ~60 samples

        # KV Cache trend prediction
        kv_vals = [p.get("kv_cache", 0) for p in recent if p.get("kv_cache", 0) > 0]
        if len(kv_vals) >= 5:
            kv_rate = (kv_vals[-1] - kv_vals[0]) / max(len(kv_vals), 1)  # change per sample
            if kv_rate > 0 and kv_vals[-1] < 90:
                samples_to_90 = (90 - kv_vals[-1]) / kv_rate
                hours_to_90 = samples_to_90 * 3 / 3600  # 3s per sample
                if hours_to_90 < 24:
                    predictions.append({
                        "server": sk, "metric": "KV Cache", "current": round(kv_vals[-1], 1),
                        "threshold": 90, "eta_hours": round(hours_to_90, 1),
                        "severity": "critical" if hours_to_90 < 2 else "warning",
                        "message": f"KV Cache at {kv_vals[-1]:.1f}%, will hit 90% in ~{hours_to_90:.1f}h",
                    })

        # VRAM trend prediction
        vram_vals = [p.get("vram_pct", 0) for p in recent]
        if len(vram_vals) >= 5:
            vram_rate = (vram_vals[-1] - vram_vals[0]) / max(len(vram_vals), 1)
            if vram_rate > 0 and vram_vals[-1] < 95:
                samples_to_95 = (95 - vram_vals[-1]) / vram_rate
                hours_to_95 = samples_to_95 * 3 / 3600
                if hours_to_95 < 24:
                    predictions.append({
                        "server": sk, "metric": "VRAM", "current": round(vram_vals[-1], 1),
                        "threshold": 95, "eta_hours": round(hours_to_95, 1),
                        "severity": "critical" if hours_to_95 < 2 else "warning",
                        "message": f"VRAM at {vram_vals[-1]:.1f}%, will hit 95% in ~{hours_to_95:.1f}h",
                    })

        # GPU util trend (predicting saturation)
        util_vals = [p.get("avg_util", 0) for p in recent]
        if len(util_vals) >= 5:
            util_rate = (util_vals[-1] - util_vals[0]) / max(len(util_vals), 1)
            if util_rate > 0 and util_vals[-1] < 95:
                samples_to_95 = (95 - util_vals[-1]) / util_rate
                hours_to_95 = samples_to_95 * 3 / 3600
                if hours_to_95 < 12:
                    predictions.append({
                        "server": sk, "metric": "GPU Utilization", "current": round(util_vals[-1], 1),
                        "threshold": 95, "eta_hours": round(hours_to_95, 1),
                        "severity": "warning",
                        "message": f"GPU util at {util_vals[-1]:.1f}%, may hit 95% in ~{hours_to_95:.1f}h",
                    })

        # Temperature trend
        temp_vals = [p.get("max_temp", 0) for p in recent]
        if len(temp_vals) >= 5:
            temp_rate = (temp_vals[-1] - temp_vals[0]) / max(len(temp_vals), 1)
            if temp_rate > 0 and temp_vals[-1] < 85:
                samples_to_85 = (85 - temp_vals[-1]) / temp_rate
                hours_to_85 = samples_to_85 * 3 / 3600
                if hours_to_85 < 6:
                    predictions.append({
                        "server": sk, "metric": "Temperature", "current": round(temp_vals[-1], 1),
                        "threshold": 85, "eta_hours": round(hours_to_85, 1),
                        "severity": "warning",
                        "message": f"Temp at {temp_vals[-1]}°C, rising toward 85°C in ~{hours_to_85:.1f}h",
                    })

    # Disk prediction
    for sk in SERVERS:
        dh = list(disk_history[sk])
        if len(dh) >= 3:
            disk_rate = (dh[-1]["pct"] - dh[0]["pct"]) / max(len(dh), 1)
            if disk_rate > 0 and dh[-1]["pct"] < 90:
                samples_to_90 = (90 - dh[-1]["pct"]) / disk_rate
                # disk tracked every ~20 samples (~1 min each)
                hours_to_90 = samples_to_90 * 60 / 3600
                days_to_90 = hours_to_90 / 24
                if days_to_90 < 30:
                    predictions.append({
                        "server": sk, "metric": "Disk Space", "current": dh[-1]["pct"],
                        "threshold": 90, "eta_hours": round(hours_to_90, 1),
                        "severity": "critical" if days_to_90 < 3 else "warning",
                        "message": f"Disk at {dh[-1]['pct']}%, will hit 90% in ~{days_to_90:.1f} days",
                    })

    # Zombie growth prediction
    ph = list(process_health_history)
    if len(ph) >= 5:
        zombie_vals = [p["zombies"] for p in ph]
        z_rate = (zombie_vals[-1] - zombie_vals[0]) / max(len(zombie_vals), 1)
        if z_rate > 0.01:  # growing
            predictions.append({
                "server": "rtx5090", "metric": "Zombie Processes",
                "current": zombie_vals[-1], "threshold": 100,
                "eta_hours": round((100 - zombie_vals[-1]) / z_rate * 3 / 3600, 1) if zombie_vals[-1] < 100 else 0,
                "severity": "warning",
                "message": f"Zombies at {zombie_vals[-1]}, growing at {z_rate*20:.1f}/min — may need restart",
            })
        if zombie_vals[-1] >= 50:
            predictions.append({
                "server": "rtx5090", "metric": "Zombie Processes",
                "current": zombie_vals[-1], "threshold": 50,
                "eta_hours": 0, "severity": "critical",
                "message": f"{zombie_vals[-1]} zombie processes — consider restarting services",
            })

    result["predictive_alerts"] = sorted(predictions, key=lambda x: x.get("eta_hours", 999))

    # ── 2. Error Rate Tracker ──
    for sk in SERVERS:
        eh = list(error_history[sk])
        if not eh:
            continue
        recent_errors = eh[-60:]  # last 60 samples
        current_errors = recent_errors[-1]["errors"] if recent_errors else 0
        current_total = recent_errors[-1]["total"] if recent_errors else 0
        current_rate = recent_errors[-1]["rate"] if recent_errors else 0

        # Compute delta (errors in this window vs start of window)
        if len(recent_errors) >= 2:
            err_delta = recent_errors[-1]["errors"] - recent_errors[0]["errors"]
            req_delta = recent_errors[-1]["total"] - recent_errors[0]["total"]
            window_rate = round(err_delta / max(req_delta, 1) * 100, 2) if req_delta > 0 else 0
        else:
            err_delta = 0
            req_delta = 0
            window_rate = 0

        error_sparkline = [e["errors"] for e in recent_errors]
        result["error_rates"][sk] = {
            "server_name": SERVERS[sk]["name"],
            "total_errors": current_errors,
            "total_requests": current_total,
            "cumulative_rate": current_rate,
            "window_errors": err_delta,
            "window_requests": req_delta,
            "window_rate": window_rate,
            "sparkline": error_sparkline[-30:],
            "status": "critical" if window_rate > 5 else "warning" if window_rate > 1 else "healthy",
        }

    # ── 3. Zombie & Memory Leak Monitor ──
    if ph:
        zombie_spark = [p["zombies"] for p in ph[-60:]]
        rss_spark = [p["gunicorn_rss_mb"] for p in ph[-60:]]
        worker_spark = [p["gunicorn_workers"] for p in ph[-60:]]
        bot_spark = [p["active_bots"] for p in ph[-60:]]

        # Detect memory leak (RSS growing consistently)
        rss_growing = False
        if len(rss_spark) >= 10:
            first_half = sum(rss_spark[:len(rss_spark)//2]) / max(len(rss_spark)//2, 1)
            second_half = sum(rss_spark[len(rss_spark)//2:]) / max(len(rss_spark) - len(rss_spark)//2, 1)
            rss_growing = second_half > first_half * 1.05  # 5% growth

        result["process_health"] = {
            "current_zombies": ph[-1]["zombies"],
            "zombie_trend": zombie_spark,
            "gunicorn_rss_mb": ph[-1]["gunicorn_rss_mb"],
            "rss_trend": rss_spark,
            "rss_leak_detected": rss_growing,
            "gunicorn_workers": ph[-1]["gunicorn_workers"],
            "worker_trend": worker_spark,
            "voicemail_workers": ph[-1]["voicemail_workers"],
            "active_bots": ph[-1]["active_bots"],
            "bot_trend": bot_spark,
            "samples": len(ph),
        }

    # ── 4. Service Downtime Log ──
    for name, su in service_uptime.items():
        uptime_pct = round(su["up_samples"] / max(su["total_samples"], 1) * 100, 3)
        incidents = [{"t": i["t"], "from": i["from"], "to": i["to"],
                      "time_ago": f"{(now - i['t'])/60:.0f}m ago"} for i in su["incidents"]]
        result["service_uptime"][name] = {
            "status": su["status"],
            "uptime_pct": uptime_pct,
            "total_samples": su["total_samples"],
            "last_change_t": su["last_change_t"],
            "last_change_ago": f"{(now - su['last_change_t'])/60:.0f}m ago",
            "incidents": incidents[-10:],  # last 10
        }

    # ── 5. Inter-Cluster Latency ──
    # Measure RTX5090 → H200 round-trip by timing a lightweight SSH command
    try:
        rtx_server = SERVERS["rtx5090"]
        t0 = time.time()
        await run_command(rtx_server, f"curl -s -o /dev/null -w '%{{time_total}}' http://146.88.194.12:8001/health --max-time 5")
        rtt = (time.time() - t0) * 1000
        cluster_latency.append({"t": now, "rtt_ms": round(rtt, 1)})
    except Exception:
        pass

    lat_data = list(cluster_latency)
    if lat_data:
        rtts = [l["rtt_ms"] for l in lat_data[-30:]]
        result["cluster_latency"] = {
            "current_rtt_ms": rtts[-1] if rtts else 0,
            "avg_rtt_ms": round(sum(rtts) / len(rtts), 1),
            "max_rtt_ms": round(max(rtts), 1),
            "min_rtt_ms": round(min(rtts), 1),
            "trend": rtts,
            "status": "critical" if rtts[-1] > 3000 else "warning" if rtts[-1] > 1500 else "healthy",
        }

    # ── 6. GPU Thermal Throttling Detection ──
    for sk in SERVERS:
        throttle_gpus = []
        for idx, clock_deque in gpu_clock_history[sk].items():
            clocks = list(clock_deque)
            if len(clocks) < 5:
                continue
            temps = [c["temp"] for c in clocks[-20:]]
            powers = [c["power"] for c in clocks[-20:]]
            avg_temp = sum(temps) / len(temps)
            max_temp_gpu = max(temps)
            # Detect throttling: high temp + reduced power relative to peak
            peak_power = max(powers) if powers else 0
            current_power = powers[-1] if powers else 0
            power_drop_pct = round((1 - current_power / max(peak_power, 1)) * 100, 1) if peak_power > 0 else 0

            if max_temp_gpu >= 80 or (power_drop_pct > 15 and avg_temp > 70):
                throttle_gpus.append({
                    "gpu": idx,
                    "current_temp": temps[-1],
                    "avg_temp": round(avg_temp, 1),
                    "peak_temp": max_temp_gpu,
                    "current_power": current_power,
                    "peak_power": peak_power,
                    "power_drop_pct": power_drop_pct,
                    "likely_throttling": max_temp_gpu >= 83 or power_drop_pct > 20,
                })

        if throttle_gpus:
            result["thermal_throttling"][sk] = {
                "server_name": SERVERS[sk]["name"],
                "gpus": throttle_gpus,
                "any_throttling": any(g["likely_throttling"] for g in throttle_gpus),
            }

    # ── 7. Call Volume Forecasting ──
    from datetime import datetime, timezone
    dt_now = datetime.fromtimestamp(now, tz=timezone.utc)
    current_dow = dt_now.weekday()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    forecast_hours = []
    for h in range(24):
        key = f"{current_dow}_{h}"
        samples = call_volume_patterns.get(key, [])
        if samples:
            avg_calls = sum(samples) / len(samples)
            peak_calls = max(samples)
        else:
            avg_calls = 0
            peak_calls = 0
        forecast_hours.append({
            "hour": h,
            "label": f"{h:02d}:00",
            "avg_calls": round(avg_calls, 1),
            "peak_calls": peak_calls,
            "samples": len(samples),
            "is_current": h == dt_now.hour,
        })

    result["call_forecast"] = {
        "day": day_names[current_dow],
        "current_hour": dt_now.hour,
        "hours": forecast_hours,
        "peak_hour": max(forecast_hours, key=lambda x: x["avg_calls"]) if forecast_hours else None,
    }

    # ── 8. Disk & Log Growth ──
    for sk in SERVERS:
        dh = list(disk_history[sk])
        if not dh:
            continue
        current = dh[-1]
        growth_rate_pct_per_day = 0
        days_until_full = None
        if len(dh) >= 2:
            time_span_hours = (dh[-1]["t"] - dh[0]["t"]) / 3600
            pct_change = dh[-1]["pct"] - dh[0]["pct"]
            if time_span_hours > 0:
                growth_rate_pct_per_day = round(pct_change / time_span_hours * 24, 2)
                if growth_rate_pct_per_day > 0:
                    remaining = 100 - current["pct"]
                    days_until_full = round(remaining / growth_rate_pct_per_day, 1)

        result["disk_trends"][sk] = {
            "server_name": SERVERS[sk]["name"],
            "current_pct": current["pct"],
            "used_bytes": current["used_bytes"],
            "total_bytes": current["total_bytes"],
            "growth_rate_pct_per_day": growth_rate_pct_per_day,
            "days_until_full": days_until_full,
            "trend": [d["pct"] for d in dh[-30:]],
            "status": "critical" if current["pct"] >= 90 else "warning" if current["pct"] >= 80 else "healthy",
        }

    return result


# ── Cost & Analytics Config ───────────────────────────────────
GPU_MONTHLY_COST = 25000  # $25,000/month total
TOTAL_GPUS = 16  # 8 H200 + 8 RTX 5090
COST_PER_GPU_HOUR = round(GPU_MONTHLY_COST / 30 / 24 / TOTAL_GPUS, 2)
COST_PER_HOUR = round(GPU_MONTHLY_COST / 30 / 24, 2)

# Quality thresholds for alerts
ALERT_THRESHOLDS = {
    "ttft_ms": {"warn": 300, "crit": 600, "label": "TTFT"},
    "itl_ms": {"warn": 60, "crit": 120, "label": "Inter-Token Latency"},
    "e2e_s": {"warn": 3.0, "crit": 6.0, "label": "E2E Latency"},
    "kv_cache": {"warn": 75, "crit": 90, "label": "KV Cache Usage"},
    "gpu_util_avg": {"warn": 80, "crit": 95, "label": "Avg GPU Utilization"},
    "temp": {"warn": 75, "crit": 85, "label": "GPU Temperature"},
    "vram_pct": {"warn": 85, "crit": 95, "label": "VRAM Usage"},
    "queue_s": {"warn": 0.5, "crit": 2.0, "label": "Queue Time"},
}


@app.get("/api/analytics")
async def get_analytics():
    result = {
        "cost": {},
        "concurrency_quality": [],
        "bottleneck": {},
        "headroom": {},
        "alerts": [],
        "efficiency": {},
        "quality_scorecard": {},
    }

    # ── 1. Cost Breakdown ──
    # Compute cost based on actual utilization
    all_history = []
    for sk in SERVERS:
        all_history.extend(list(history[sk]))

    total_samples = len(all_history)
    avg_concurrent_calls = 0
    avg_gpu_util_all = 0
    if total_samples > 0:
        avg_concurrent_calls = sum(p.get("calls", 0) for p in all_history) / total_samples
        avg_gpu_util_all = sum(p.get("avg_util", 0) for p in all_history) / total_samples

    # Time window of data
    if total_samples >= 2:
        data_hours = (all_history[-1]["t"] - all_history[0]["t"]) / 3600
    else:
        data_hours = 0

    cost_per_call_minute = 0
    if avg_concurrent_calls > 0:
        # Cost per minute = hourly cost / 60 / avg concurrent calls
        cost_per_call_minute = round(COST_PER_HOUR / 60 / avg_concurrent_calls, 4)

    result["cost"] = {
        "monthly": GPU_MONTHLY_COST,
        "daily": round(GPU_MONTHLY_COST / 30, 2),
        "hourly": COST_PER_HOUR,
        "per_gpu_hour": COST_PER_GPU_HOUR,
        "per_call_minute": cost_per_call_minute,
        "per_call_hour": round(cost_per_call_minute * 60, 2),
        "avg_concurrent_calls": round(avg_concurrent_calls, 1),
        "avg_gpu_util": round(avg_gpu_util_all, 1),
        "total_gpus": TOTAL_GPUS,
        "data_hours": round(data_hours, 1),
        "idle_gpu_cost_hourly": round(COST_PER_HOUR * max(0, 1 - avg_gpu_util_all / 100), 2),
        "utilized_gpu_cost_hourly": round(COST_PER_HOUR * min(1, avg_gpu_util_all / 100), 2),
    }

    # ── 2. Concurrency vs Quality Correlation ──
    # Bucket history points by concurrent call count, compute avg latency per bucket
    call_buckets = {}
    for sk in SERVERS:
        for p in history[sk]:
            c = p.get("calls", 0)
            ttft = p.get("ttft_ms", 0)
            itl = p.get("itl_ms", 0)
            e2e = p.get("e2e_s", 0)
            queue = p.get("queue_s", 0)
            kv = p.get("kv_cache", 0)
            util = p.get("avg_util", 0)
            if c not in call_buckets:
                call_buckets[c] = {"ttft": [], "itl": [], "e2e": [], "queue": [], "kv": [], "util": [], "count": 0}
            call_buckets[c]["count"] += 1
            if ttft > 0:
                call_buckets[c]["ttft"].append(ttft)
            if itl > 0:
                call_buckets[c]["itl"].append(itl)
            if e2e > 0:
                call_buckets[c]["e2e"].append(e2e)
            if queue > 0:
                call_buckets[c]["queue"].append(queue)
            if kv > 0:
                call_buckets[c]["kv"].append(kv)
            call_buckets[c]["util"].append(util)

    for c in sorted(call_buckets.keys()):
        b = call_buckets[c]
        result["concurrency_quality"].append({
            "concurrent_calls": c,
            "samples": b["count"],
            "avg_ttft_ms": round(sum(b["ttft"]) / len(b["ttft"]), 1) if b["ttft"] else 0,
            "avg_itl_ms": round(sum(b["itl"]) / len(b["itl"]), 1) if b["itl"] else 0,
            "avg_e2e_s": round(sum(b["e2e"]) / len(b["e2e"]), 3) if b["e2e"] else 0,
            "avg_queue_s": round(sum(b["queue"]) / len(b["queue"]), 4) if b["queue"] else 0,
            "avg_kv_cache": round(sum(b["kv"]) / len(b["kv"]), 1) if b["kv"] else 0,
            "avg_gpu_util": round(sum(b["util"]) / len(b["util"]), 1) if b["util"] else 0,
        })

    # ── 3. Bottleneck Analysis ──
    # From H200 vLLM metrics: what fraction of pipeline is LLM, TTS, queue
    h200_hist = list(history.get("h200", []))
    llm_time = []
    tts_time = []
    queue_time = []
    for p in h200_hist:
        if p.get("e2e_s", 0) > 0:
            ttft = p.get("ttft_ms", 0) / 1000  # convert to seconds
            itl_total = p.get("e2e_s", 0) - ttft  # generation time
            q = p.get("queue_s", 0)
            total = p.get("e2e_s", 0) + q
            if total > 0:
                llm_time.append(ttft / total)
                tts_time.append(itl_total / total)
                queue_time.append(q / total)

    if llm_time:
        avg_llm = sum(llm_time) / len(llm_time)
        avg_tts = sum(tts_time) / len(tts_time)
        avg_queue = sum(queue_time) / len(queue_time)
        avg_other = max(0, 1 - avg_llm - avg_tts - avg_queue)
        result["bottleneck"] = {
            "llm_pct": round(avg_llm * 100, 1),
            "tts_generation_pct": round(avg_tts * 100, 1),
            "queue_pct": round(avg_queue * 100, 1),
            "other_pct": round(avg_other * 100, 1),
            "primary_bottleneck": "LLM (TTFT)" if avg_llm >= avg_tts else "TTS/Generation",
            "samples": len(llm_time),
        }
    else:
        result["bottleneck"] = {"llm_pct": 0, "tts_generation_pct": 0, "queue_pct": 0, "other_pct": 100, "primary_bottleneck": "Insufficient data", "samples": 0}

    # ── 4. GPU Headroom & Capacity Planning ──
    for sk in SERVERS:
        sk_hist = list(history[sk])
        if not sk_hist:
            result["headroom"][sk] = {"status": "no data"}
            continue
        recent = sk_hist[-min(60, len(sk_hist)):]  # last ~60 samples
        avg_util = sum(p.get("avg_util", 0) for p in recent) / len(recent)
        avg_vram = sum(p.get("vram_pct", 0) for p in recent) / len(recent)
        avg_kv = sum(p.get("kv_cache", 0) for p in recent) / len(recent)
        avg_power = sum(p.get("total_power", 0) for p in recent) / len(recent)
        peak_util = max(p.get("avg_util", 0) for p in recent)
        peak_calls = max(p.get("calls", 0) for p in recent)
        avg_calls = sum(p.get("calls", 0) for p in recent) / len(recent)
        gpu_count = recent[-1].get("gpu_count", 8)

        # Estimate max concurrent calls before GPU saturation
        if avg_calls > 0 and avg_util > 0:
            util_per_call = avg_util / avg_calls
            max_calls_est = int(90 / util_per_call) if util_per_call > 0 else 999
        else:
            util_per_call = 0
            max_calls_est = 0

        # KV cache capacity estimate
        if avg_calls > 0 and avg_kv > 0:
            kv_per_call = avg_kv / avg_calls
            max_calls_kv = int(85 / kv_per_call) if kv_per_call > 0 else 999
        else:
            kv_per_call = 0
            max_calls_kv = 0

        result["headroom"][sk] = {
            "server_name": SERVERS[sk]["name"],
            "gpu_count": gpu_count,
            "avg_util": round(avg_util, 1),
            "peak_util": round(peak_util, 1),
            "util_headroom": round(100 - avg_util, 1),
            "avg_vram_pct": round(avg_vram, 1),
            "vram_headroom": round(100 - avg_vram, 1),
            "avg_kv_cache": round(avg_kv, 1),
            "kv_headroom": round(100 - avg_kv, 1) if avg_kv > 0 else 100,
            "avg_power_w": round(avg_power, 0),
            "avg_calls": round(avg_calls, 1),
            "peak_calls": peak_calls,
            "util_per_call": round(util_per_call, 2),
            "est_max_calls_gpu": max_calls_est,
            "est_max_calls_kv": max_calls_kv,
            "est_max_calls": min(max_calls_est, max_calls_kv) if max_calls_kv > 0 else max_calls_est,
            "scale_factor": round(min(max_calls_est, max_calls_kv) / max(avg_calls, 1), 1) if max_calls_est > 0 else 0,
        }

    # ── 5. Call Quality Scorecard ──
    # Aggregate quality metrics from recent history
    all_ttft = []
    all_itl = []
    all_e2e = []
    all_queue = []
    all_kv = []
    for sk in SERVERS:
        for p in list(history[sk])[-300:]:  # last 300 samples
            if p.get("ttft_ms", 0) > 0:
                all_ttft.append(p["ttft_ms"])
            if p.get("itl_ms", 0) > 0:
                all_itl.append(p["itl_ms"])
            if p.get("e2e_s", 0) > 0:
                all_e2e.append(p["e2e_s"])
            if p.get("queue_s", 0) > 0:
                all_queue.append(p["queue_s"])
            if p.get("kv_cache", 0) > 0:
                all_kv.append(p["kv_cache"])

    def percentile(arr, p):
        if not arr:
            return 0
        s = sorted(arr)
        k = (len(s) - 1) * p / 100
        f = int(k)
        c = f + 1
        if c >= len(s):
            return s[f]
        return s[f] + (k - f) * (s[c] - s[f])

    result["quality_scorecard"] = {
        "ttft": {
            "avg": round(sum(all_ttft) / len(all_ttft), 1) if all_ttft else 0,
            "p50": round(percentile(all_ttft, 50), 1),
            "p95": round(percentile(all_ttft, 95), 1),
            "p99": round(percentile(all_ttft, 99), 1),
            "samples": len(all_ttft),
        },
        "itl": {
            "avg": round(sum(all_itl) / len(all_itl), 1) if all_itl else 0,
            "p50": round(percentile(all_itl, 50), 1),
            "p95": round(percentile(all_itl, 95), 1),
            "p99": round(percentile(all_itl, 99), 1),
            "samples": len(all_itl),
        },
        "e2e": {
            "avg": round(sum(all_e2e) / len(all_e2e), 3) if all_e2e else 0,
            "p50": round(percentile(all_e2e, 50), 3),
            "p95": round(percentile(all_e2e, 95), 3),
            "p99": round(percentile(all_e2e, 99), 3),
            "samples": len(all_e2e),
        },
        "queue": {
            "avg": round(sum(all_queue) / len(all_queue), 4) if all_queue else 0,
            "p50": round(percentile(all_queue, 50), 4),
            "p95": round(percentile(all_queue, 95), 4),
            "p99": round(percentile(all_queue, 99), 4),
            "samples": len(all_queue),
        },
        "kv_cache": {
            "avg": round(sum(all_kv) / len(all_kv), 1) if all_kv else 0,
            "peak": round(max(all_kv), 1) if all_kv else 0,
            "samples": len(all_kv),
        },
    }

    # ── 6. Alerts ──
    alerts = []
    for sk in SERVERS:
        sk_hist = list(history[sk])
        if not sk_hist:
            continue
        latest = sk_hist[-1]
        checks = {
            "ttft_ms": latest.get("ttft_ms", 0),
            "itl_ms": latest.get("itl_ms", 0),
            "e2e_s": latest.get("e2e_s", 0),
            "kv_cache": latest.get("kv_cache", 0),
            "gpu_util_avg": latest.get("avg_util", 0),
            "temp": latest.get("max_temp", 0),
            "vram_pct": latest.get("vram_pct", 0),
            "queue_s": latest.get("queue_s", 0),
        }
        for metric, value in checks.items():
            if value <= 0:
                continue
            th = ALERT_THRESHOLDS.get(metric)
            if not th:
                continue
            if value >= th["crit"]:
                alerts.append({"server": sk, "metric": th["label"], "value": round(value, 2), "threshold": th["crit"], "level": "critical"})
            elif value >= th["warn"]:
                alerts.append({"server": sk, "metric": th["label"], "value": round(value, 2), "threshold": th["warn"], "level": "warning"})
    result["alerts"] = alerts

    # ── 7. Model Efficiency ──
    for sk in SERVERS:
        sk_hist = list(history[sk])
        if len(sk_hist) < 2:
            continue
        recent = sk_hist[-min(60, len(sk_hist)):]
        avg_power = sum(p.get("total_power", 0) for p in recent) / len(recent)

        # Get current vLLM throughput from hourly stats
        hs = hourly_stats.get(sk, {})
        if hs:
            latest_hour = max(hs.values(), key=lambda x: x["hour_ts"])
            duration_s = max(latest_hour["samples"] * 3, 1)  # ~3s per sample
            reqs_delta = latest_hour["_vllm_reqs_end"] - latest_hour["_vllm_reqs_start"]
            prompt_delta = latest_hour["_vllm_prompt_end"] - latest_hour["_vllm_prompt_start"]
            gen_delta = latest_hour["_vllm_gen_end"] - latest_hour["_vllm_gen_start"]
            total_tokens = prompt_delta + gen_delta
            tokens_per_sec = total_tokens / duration_s if duration_s > 0 else 0
            reqs_per_min = reqs_delta / (duration_s / 60) if duration_s > 0 else 0
            tokens_per_watt = tokens_per_sec / avg_power if avg_power > 0 else 0

            result["efficiency"][sk] = {
                "server_name": SERVERS[sk]["name"],
                "tokens_per_sec": round(tokens_per_sec, 1),
                "reqs_per_min": round(reqs_per_min, 1),
                "tokens_per_watt": round(tokens_per_watt, 3),
                "avg_power_w": round(avg_power, 0),
                "prompt_tokens_hour": prompt_delta,
                "gen_tokens_hour": gen_delta,
                "total_tokens_hour": total_tokens,
                "cost_per_million_tokens": round((COST_PER_HOUR / max(total_tokens / 1e6, 0.001)), 2) if total_tokens > 0 else 0,
            }

    return result


@app.get("/api/{server_key}")
async def get_server_data(server_key: str):
    if server_key not in SERVERS:
        return {"error": "Unknown server", "status": "error"}
    try:
        return await asyncio.wait_for(fetch_gpu_data(server_key), timeout=15)
    except Exception as e:
        return {
            "server_name": SERVERS[server_key]["name"],
            "host": SERVERS[server_key]["host"],
            "status": "offline",
            "error": str(e),
        }


@app.get("/api/{server_key}/history")
async def get_history(server_key: str):
    if server_key not in SERVERS:
        return {"error": "Unknown server"}
    return {"history": list(history[server_key])}


@app.get("/api/{server_key}/daily")
async def get_daily(server_key: str):
    if server_key not in SERVERS:
        return {"error": "Unknown server"}
    hs = hourly_stats.get(server_key, {})
    hours_raw = sorted(hs.values(), key=lambda x: x["hour_ts"])
    # Compute derived fields for each hour
    hours = []
    for hr in hours_raw:
        s = hr["samples"] or 1
        hours.append({
            "hour_label": hr["hour_label"],
            "hour_ts": hr["hour_ts"],
            "samples": hr["samples"],
            "gpu_count": hr.get("gpu_count", 8),
            # Calls
            "max_calls": hr["max_calls"],
            "avg_calls": round(hr["sum_calls"] / s, 1),
            # GPU — sum of all GPUs and per-GPU average
            "max_gpu_util_sum": hr.get("max_gpu_util_sum", 0),
            "max_gpu_util_avg": hr.get("max_gpu_util_avg", 0),
            "avg_gpu_util": round(hr["sum_gpu_util"] / s, 1),
            "calls_at_max_gpu": hr["calls_at_max_gpu_util"],
            # Temp & Power
            "max_temp": hr["max_temp"],
            "avg_temp": round(hr["sum_temp"] / s, 1),
            "max_power": hr["max_power"],
            "avg_power": round(hr["sum_power"] / s, 1),
            # VRAM
            "max_vram_pct": hr.get("max_vram_pct", 0),
            # vLLM
            "max_kv_cache": hr.get("max_kv_cache", 0),
            "max_ttft_ms": hr.get("max_ttft_ms", 0),
            "avg_ttft_ms": round(hr["sum_ttft_ms"] / hr["ttft_samples"], 1) if hr.get("ttft_samples", 0) > 0 else 0,
            "max_e2e_s": hr.get("max_e2e_s", 0),
            "max_waiting": hr.get("max_waiting", 0),
            # Deltas (requests/tokens/traffic during this hour)
            "hour_requests": max(0, hr.get("_vllm_reqs_end", 0) - hr.get("_vllm_reqs_start", 0)),
            "hour_prompt_tokens": max(0, hr.get("_vllm_prompt_end", 0) - hr.get("_vllm_prompt_start", 0)),
            "hour_gen_tokens": max(0, hr.get("_vllm_gen_end", 0) - hr.get("_vllm_gen_start", 0)),
            "hour_net_rx": max(0, hr.get("_net_rx_end", 0) - hr.get("_net_rx_start", 0)),
            "hour_net_tx": max(0, hr.get("_net_tx_end", 0) - hr.get("_net_tx_start", 0)),
        })
    return {"hours": hours, "server_name": SERVERS[server_key]["name"]}


# ── Agent Status (RTX 5090 / AgentV2) ────────────────────────
AGENT_SERVICES = [
    {"name": "agentv2-main-server", "port": 8003, "health": "http://localhost:8003/health", "role": "Call Router"},
    {"name": "agentv2-voicemail-processor", "port": 8001, "health": None, "role": "Voicemail/Intent Classifier"},
    {"name": "agentv2-health-checker", "port": 9000, "health": "http://localhost:9000/health", "role": "Process Monitor"},
    {"name": "agentv2-failed-report", "port": 8006, "health": None, "role": "Retry Queue"},
    {"name": "agentv2-server-old", "port": None, "health": None, "role": "Legacy Server"},
]

EXTERNAL_DEPS = [
    {"name": "vLLM LLM (Qwen3-VL-8B)", "url": "http://146.88.194.12:8001/health", "host": "146.88.194.12:8001"},
    {"name": "vLLM TTS (recruite-tts-fp8)", "url": "http://146.88.194.12:8002/health", "host": "146.88.194.12:8002"},
    {"name": "DCGM GPU Exporter", "url": "http://localhost:9400/metrics", "host": "localhost:9400"},
]


async def fetch_agent_status() -> dict:
    server = SERVERS["rtx5090"]

    # Build a combined command to gather all agent data in one SSH call
    cmd = (
        # Health checker status (watched PIDs = active workers)
        "curl -s --max-time 2 localhost:9000/status 2>/dev/null; echo '===SEP==='; "
        # Health checker health (process count)
        "curl -s --max-time 2 localhost:9000/health 2>/dev/null; echo '===SEP==='; "
        # Main server health
        "curl -s --max-time 2 localhost:8003/health 2>/dev/null; echo '===SEP==='; "
        # Failed reports queue
        "curl -s --max-time 2 localhost:8006/v1/reports/failure 2>/dev/null; echo '===SEP==='; "
        # Active bot child processes (spawned per call, not gunicorn workers)
        "ps aux | grep -E 'python.*bot_' | grep -v grep | wc -l; echo '===SEP==='; "
        # Bot process details (PID, elapsed time, RSS memory, command)
        "ps aux | grep -E 'python.*bot_' | grep -v grep; echo '===SEP==='; "
        # Active TCP connections to port 8003
        "ss -tnp state established sport = :8003 2>/dev/null | tail -n +2 | wc -l; echo '===SEP==='; "
        # Gunicorn worker count and memory
        "ps aux | grep 'gunicorn.*server:app' | grep -v grep | wc -l; echo '===SEP==='; "
        # Total RSS memory of gunicorn workers (KB)
        "ps aux | grep 'gunicorn.*server:app' | grep -v grep | awk '{sum+=$6} END {print sum+0}'; echo '===SEP==='; "
        # Voicemail processor workers
        "ps aux | grep 'gunicorn.*voicemail' | grep -v grep | wc -l; echo '===SEP==='; "
        # Zombie processes
        "ps aux | grep defunct | grep -v grep | wc -l; echo '===SEP==='; "
        # External service checks (vLLM LLM, vLLM TTS)
        "curl -s --max-time 3 http://146.88.194.12:8001/health 2>/dev/null; echo '===SEP==='; "
        "curl -s --max-time 3 http://146.88.194.12:8002/health 2>/dev/null; echo '===SEP==='; "
        # DCGM GPU metrics
        "curl -s --max-time 2 localhost:9400/metrics 2>/dev/null | grep -E 'DCGM_FI_DEV_GPU_UTIL|DCGM_FI_DEV_GPU_TEMP|DCGM_FI_DEV_POWER_USAGE|DCGM_FI_DEV_SM_CLOCK' | grep -v '^#'; echo '===SEP==='; "
        # Service systemd status
        "systemctl is-active agentv2-main-server agentv2-voicemail-processor agentv2-health-checker agentv2-failed-report agentv2-server-old 2>/dev/null"
    )

    raw = await run_command(server, cmd)
    parts = raw.split("===SEP===")

    def safe(i):
        return parts[i].strip() if i < len(parts) else ""

    # Parse health checker status
    watched_pids = []
    try:
        hc_status = json.loads(safe(0))
        watched_pids = hc_status.get("watching", [])
    except Exception:
        pass

    # Health checker health
    working_processes = 0
    try:
        hc_health = json.loads(safe(1))
        working_processes = hc_health.get("working_processes", 0)
    except Exception:
        pass

    # Main server health
    main_healthy = False
    try:
        main_h = json.loads(safe(2))
        main_healthy = main_h.get("status", False)
    except Exception:
        pass

    # Failed reports
    failed_reports = []
    try:
        failed_reports = json.loads(safe(3))
    except Exception:
        pass

    # Active bot processes
    active_bots = int(safe(4)) if safe(4).isdigit() else 0

    # Bot details
    bot_details = []
    for line in safe(5).split("\n"):
        line = line.strip()
        if not line:
            continue
        parts_line = line.split()
        if len(parts_line) >= 11:
            bot_details.append({
                "pid": parts_line[1],
                "cpu": parts_line[2],
                "mem_pct": parts_line[3],
                "rss_kb": int(parts_line[5]) if parts_line[5].isdigit() else 0,
                "elapsed": parts_line[9],
                "cmd": " ".join(parts_line[10:])[:80],
            })

    # TCP connections
    tcp_connections = int(safe(6)) if safe(6).isdigit() else 0

    # Gunicorn workers
    gunicorn_workers = int(safe(7)) if safe(7).isdigit() else 0
    gunicorn_rss_kb = int(safe(8)) if safe(8).isdigit() else 0

    # Voicemail workers
    voicemail_workers = int(safe(9)) if safe(9).isdigit() else 0

    # Zombies
    zombie_count = int(safe(10)) if safe(10).isdigit() else 0

    # External deps
    vllm_llm_ok = "true" in safe(11).lower() or len(safe(11)) > 0
    vllm_tts_ok = "true" in safe(12).lower() or len(safe(12)) > 0

    # DCGM GPU metrics
    dcgm_gpus = {}
    for line in safe(13).split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'(DCGM_FI_DEV_\w+)\{gpu="(\d+)".*?\}\s+([\d.]+)', line)
        if m:
            metric, gpu_id, val = m.group(1), int(m.group(2)), float(m.group(3))
            if gpu_id not in dcgm_gpus:
                dcgm_gpus[gpu_id] = {}
            dcgm_gpus[gpu_id][metric] = val

    gpu_details = []
    for gpu_id in sorted(dcgm_gpus.keys()):
        g = dcgm_gpus[gpu_id]
        gpu_details.append({
            "index": gpu_id,
            "sm_clock": g.get("DCGM_FI_DEV_SM_CLOCK", 0),
            "temp": g.get("DCGM_FI_DEV_GPU_TEMP", 0),
            "power": round(g.get("DCGM_FI_DEV_POWER_USAGE", 0), 1),
            "util": g.get("DCGM_FI_DEV_GPU_UTIL", 0),
        })

    # Service statuses
    svc_lines = safe(14).split("\n")
    svc_names = ["agentv2-main-server", "agentv2-voicemail-processor", "agentv2-health-checker", "agentv2-failed-report", "agentv2-server-old"]
    services = []
    for i, name in enumerate(svc_names):
        status = svc_lines[i].strip() if i < len(svc_lines) else "unknown"
        services.append({"name": name, "status": status})

    return {
        "timestamp": time.time(),
        "main_server_healthy": main_healthy,
        "health_checker_workers": working_processes,
        "watched_pids": len(watched_pids),
        "active_bots": active_bots,
        "bot_details": bot_details,
        "tcp_connections": tcp_connections,
        "gunicorn_workers": gunicorn_workers,
        "gunicorn_memory_mb": round(gunicorn_rss_kb / 1024, 1),
        "voicemail_workers": voicemail_workers,
        "zombie_processes": zombie_count,
        "failed_reports_queue": len(failed_reports),
        "failed_reports": failed_reports[:10],
        "services": services,
        "external_deps": [
            {"name": "vLLM LLM (Qwen3-VL-8B)", "host": "146.88.194.12:8001", "ok": vllm_llm_ok},
            {"name": "vLLM TTS (recruite-tts-fp8)", "host": "146.88.194.12:8002", "ok": vllm_tts_ok},
        ],
        "dcgm_gpus": gpu_details,
    }


# ── Software Discovery ────────────────────────────────────────
AI_PACKAGES = {
    "vllm", "torch", "torchaudio", "torchvision", "transformers",
    "flash-attn", "flashinfer-python", "triton", "safetensors",
    "sentencepiece", "tokenizers", "huggingface-hub", "openai",
    "anthropic", "accelerate", "bitsandbytes", "deepspeed",
    "onnxruntime-gpu", "onnxruntime", "sentence-transformers",
    "nvidia-nccl-cu12", "nvidia-cudnn-cu12", "nvidia-cublas-cu12",
    "tts-rust", "whisper", "openai-whisper", "faster-whisper",
    "nemo-toolkit", "langchain", "llama-cpp-python",
}

_software_cache = {}


async def fetch_software(server_key: str) -> dict:
    now = time.time()
    if server_key in _software_cache and now - _software_cache[server_key]["_ts"] < 300:
        return _software_cache[server_key]

    server = SERVERS[server_key]

    # System info
    sys_cmd = (
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1; echo '==='; "
        "nvcc --version 2>/dev/null | grep release; echo '==='; "
        "uname -r; echo '==='; "
        "python3 --version 2>&1; echo '==='; "
        "systemctl list-units --type=service --state=running --no-pager --no-legend 2>/dev/null | grep -E '"
        + "|".join(server.get("services", []))
        + "' || echo 'none'"
    )

    # Pip packages from each venv
    venv_cmds = {}
    for venv_name, pip_path in server.get("venvs", {}).items():
        venv_cmds[venv_name] = f"{pip_path} list --format=json 2>/dev/null"

    tasks = [run_command(server, sys_cmd)]
    venv_keys = list(venv_cmds.keys())
    for k in venv_keys:
        tasks.append(run_command(server, venv_cmds[k]))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    sys_out = results[0] if not isinstance(results[0], Exception) else ""
    sys_sections = sys_out.split("===")
    driver = sys_sections[0].strip() if len(sys_sections) > 0 else "unknown"
    cuda_line = sys_sections[1].strip() if len(sys_sections) > 1 else ""
    cuda_ver = ""
    if "release" in cuda_line:
        cuda_ver = cuda_line.split("release")[-1].split(",")[0].strip()
    kernel = sys_sections[2].strip() if len(sys_sections) > 2 else "unknown"
    python_ver = sys_sections[3].strip() if len(sys_sections) > 3 else "unknown"
    services_raw = sys_sections[4].strip() if len(sys_sections) > 4 else ""
    services = []
    for line in services_raw.split("\n"):
        line = line.strip()
        if line and line != "none":
            parts = line.split()
            svc_name = parts[0] if parts else line
            status = "running" if "running" in line else "active"
            services.append({"name": svc_name, "status": status})

    # Parse pip packages
    environments = {}
    for i, venv_name in enumerate(venv_keys):
        raw = results[i + 1] if not isinstance(results[i + 1], Exception) else "[]"
        try:
            pkgs = json.loads(raw)
        except Exception:
            pkgs = []
        ai_pkgs = []
        for p in pkgs:
            name_lower = p.get("name", "").lower()
            if name_lower in AI_PACKAGES or any(k in name_lower for k in ("nvidia", "cuda", "torch", "llm", "tts", "stt", "whisper", "triton")):
                ai_pkgs.append({"name": p["name"], "version": p.get("version", "?")})
        ai_pkgs.sort(key=lambda x: x["name"].lower())
        environments[venv_name] = ai_pkgs

    data = {
        "_ts": now,
        "server_name": server["name"],
        "driver": driver,
        "cuda": cuda_ver,
        "kernel": kernel,
        "python": python_ver,
        "services": services,
        "environments": environments,
    }
    _software_cache[server_key] = data
    return data


@app.get("/api/{server_key}/software")
async def get_software(server_key: str):
    if server_key not in SERVERS:
        return {"error": "Unknown server"}
    try:
        return await asyncio.wait_for(fetch_software(server_key), timeout=20)
    except Exception as e:
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTML_PAGE


# ── HTML Dashboard ─────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPU Admin Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  body { font-family: 'Inter', sans-serif; background: #0a0a0f; }
  .glass { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); backdrop-filter: blur(20px); }
  .glass-bright { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); }
  .ring-bg { stroke: rgba(255,255,255,0.06); }
  .ring-fill { transition: stroke-dashoffset 0.8s ease; stroke-linecap: round; }
  .bar-track { background: rgba(255,255,255,0.06); }
  .bar-fill { transition: width 0.8s ease; }
  @keyframes pulse-green { 0%,100%{box-shadow:0 0 0 0 rgba(34,197,94,0.4)} 50%{box-shadow:0 0 0 6px rgba(34,197,94,0)} }
  @keyframes pulse-red { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0.4)} 50%{box-shadow:0 0 0 6px rgba(239,68,68,0)} }
  .status-online { animation: pulse-green 2s infinite; }
  .status-offline { animation: pulse-red 2s infinite; }
  @keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
  .shimmer { background: linear-gradient(90deg, rgba(255,255,255,0.03) 25%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0.03) 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite; }
  .gpu-card:hover { border-color: rgba(255,255,255,0.15); transform: translateY(-1px); }
  .gpu-card { transition: all 0.2s ease; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
  canvas { image-rendering: pixelated; }
  .tab-active { border-bottom: 2px solid #22c55e; color: #fff; }
  .tab-inactive { border-bottom: 2px solid transparent; color: #6b7280; }
  .tab-inactive:hover { color: #9ca3af; }
  .kpi-card { transition: all 0.2s ease; }
  .kpi-card:hover { background: rgba(255,255,255,0.06); }
  .sparkline { display: inline-block; vertical-align: middle; }
</style>
<script>
tailwind.config = { theme: { extend: { colors: { surface: { 50:'#0a0a0f', 100:'#111118', 200:'#1a1a24' } } } } }
</script>
</head>
<body class="min-h-screen text-gray-200 p-4 md:p-6">

<!-- Header -->
<div class="max-w-[1900px] mx-auto mb-6">
  <div class="flex items-center justify-between flex-wrap gap-4">
    <div class="flex items-center gap-3">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center">
        <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"/></svg>
      </div>
      <div>
        <h1 class="text-xl font-semibold text-white">GPU Admin Dashboard</h1>
        <p class="text-xs text-gray-500">Real-time monitoring &middot; Traffic &middot; KPIs</p>
      </div>
    </div>
    <div class="flex items-center gap-4">
      <div class="text-xs text-gray-500">
        Refresh: <select id="refreshRate" onchange="updateRefreshRate()" class="bg-surface-100 border border-gray-700 rounded px-2 py-1 text-gray-300 text-xs">
          <option value="3000">3s</option><option value="5000">5s</option><option value="10000">10s</option><option value="30000">30s</option>
        </select>
      </div>
      <div id="lastUpdate" class="text-xs text-gray-500"></div>
    </div>
  </div>
</div>

<!-- Tabs -->
<div class="max-w-[1900px] mx-auto mb-4 flex gap-6 border-b border-gray-800 pb-0">
  <button class="tab-active pb-2 text-sm font-medium px-1" onclick="switchTab('overview')" id="tab-overview">Overview</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('traffic')" id="tab-traffic">Traffic & KPIs</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('history')" id="tab-history">History & Peaks</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('software')" id="tab-software">Software & Tools</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('daily')" id="tab-daily">24h Report</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('agent')" id="tab-agent">Agent</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('analytics')" id="tab-analytics">Analytics & Cost</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('proactive')" id="tab-proactive">Alerts & Ops</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('capacity')" id="tab-capacity">Capacity Planner</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('quality')" id="tab-quality">Call Quality</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('network')" id="tab-network">Network & I/O</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('executive')" id="tab-executive">Executive Summary</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('livecalls')" id="tab-livecalls">Live Calls</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('modelcompare')" id="tab-modelcompare">Model Compare</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('power')" id="tab-power">Power Mgmt</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('incidents')" id="tab-incidents">Incidents</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('alertsconfig')" id="tab-alertsconfig">Alert Config</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('sla')" id="tab-sla">SLA</button>
  <button class="tab-inactive pb-2 text-sm font-medium px-1" onclick="switchTab('anomalies')" id="tab-anomalies">Anomalies</button>
</div>

<!-- Content -->
<div class="max-w-[1900px] mx-auto">
  <div id="page-overview"></div>
  <div id="page-traffic" class="hidden"></div>
  <div id="page-history" class="hidden"></div>
  <div id="page-software" class="hidden"></div>
  <div id="page-daily" class="hidden"></div>
  <div id="page-agent" class="hidden"></div>
  <div id="page-analytics" class="hidden"></div>
  <div id="page-proactive" class="hidden"></div>
  <div id="page-capacity" class="hidden"></div>
  <div id="page-quality" class="hidden"></div>
  <div id="page-network" class="hidden"></div>
  <div id="page-executive" class="hidden"></div>
  <div id="page-livecalls" class="hidden"></div>
  <div id="page-modelcompare" class="hidden"></div>
  <div id="page-power" class="hidden"></div>
  <div id="page-incidents" class="hidden"></div>
  <div id="page-alertsconfig" class="hidden"></div>
  <div id="page-sla" class="hidden"></div>
  <div id="page-anomalies" class="hidden"></div>
</div>

<script>
const SERVERS = ['h200', 'rtx5090'];
let refreshInterval = null;
let refreshRate = 3000;
let serverData = {};
let historyData = {};
let prevNetData = {};
let activeTab = 'overview';
let softwareData = {};
let softwareLoaded = false;
let pypiCache = {};
let dailyData = {};
let dailyLoaded = false;
let agentData = null;
let analyticsData = null;
let analyticsLoaded = false;
let proactiveData = null;
let capacityData = null;
let qualityData = null;
let networkData = null;
let executiveData = null;
let livecallsData = null;
let modelcompareData = null;
let powerData = null;
let incidentsData = null;
let alertsconfigData = null;
let slaData = null;
let anomaliesData = null;

// ── Utilities ──
function fmt(n) { return n.toLocaleString(); }
function formatBytes(b) { if(!b) return '0 B'; const g=b/(1024**3); return g>=1024?(g/1024).toFixed(1)+' TiB':g.toFixed(1)+' GiB'; }
function formatMiB(m) { return m>=1024?(m/1024).toFixed(1)+' GiB':m+' MiB'; }
function formatUptime(s) { const d=Math.floor(s/86400),h=Math.floor((s%86400)/3600),m=Math.floor((s%3600)/60); return d>0?d+'d '+h+'h '+m+'m':h>0?h+'h '+m+'m':m+'m'; }
function formatRate(bps) { if(bps>=1e9) return (bps/1e9).toFixed(1)+' Gbps'; if(bps>=1e6) return (bps/1e6).toFixed(1)+' Mbps'; if(bps>=1e3) return (bps/1e3).toFixed(1)+' Kbps'; return bps.toFixed(0)+' bps'; }
function getColor(v,t) { return v>=t[1]?{ring:'#ef4444',bg:'rgba(239,68,68,0.15)',text:'text-red-400'}:v>=t[0]?{ring:'#f59e0b',bg:'rgba(245,158,11,0.15)',text:'text-amber-400'}:{ring:'#22c55e',bg:'rgba(34,197,94,0.12)',text:'text-emerald-400'}; }
function getTempColor(t){return getColor(t,[60,80]);}
function getUtilColor(u){return getColor(u,[50,85]);}
function getPowerColor(p){return getColor(p,[60,85]);}

function ringSVG(value,max,size,sw,color) {
  const r=(size-sw)/2, c=2*Math.PI*r, pct=Math.min(value/max,1), off=c*(1-pct);
  return `<svg width="${size}" height="${size}" class="transform -rotate-90"><circle cx="${size/2}" cy="${size/2}" r="${r}" fill="none" stroke-width="${sw}" class="ring-bg"/><circle cx="${size/2}" cy="${size/2}" r="${r}" fill="none" stroke-width="${sw}" stroke="${color}" stroke-dasharray="${c}" stroke-dashoffset="${off}" class="ring-fill"/></svg>`;
}
function barHTML(v,m,c) { const p=m>0?Math.min((v/m)*100,100):0; return `<div class="bar-track rounded-full h-2 w-full"><div class="bar-fill rounded-full h-2" style="width:${p}%;background:${c}"></div></div>`; }

function sparklineSVG(data, width, height, color) {
  if (!data || data.length < 2) return '';
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  const step = width / (data.length - 1);
  let path = '';
  data.forEach((v, i) => {
    const x = i * step;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    path += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1);
  });
  return `<svg width="${width}" height="${height}" class="sparkline"><path d="${path}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}

// ── Tab Switching ──
function switchTab(tab) {
  activeTab = tab;
  ['overview','traffic','history','software','daily','agent','analytics','proactive','capacity','quality','network','executive','livecalls','modelcompare','power','incidents','alertsconfig','sla','anomalies'].forEach(t => {
    document.getElementById('page-'+t).classList.toggle('hidden', t !== tab);
    document.getElementById('tab-'+t).className = t === tab ? 'tab-active pb-2 text-sm font-medium px-1' : 'tab-inactive pb-2 text-sm font-medium px-1';
  });
  renderAll();
  if (tab === 'software' && !softwareLoaded) loadSoftware();
  if (tab === 'daily' && !dailyLoaded) loadDaily();
  if (tab === 'agent') loadAgent();
  if (tab === 'analytics') loadAnalytics();
  if (tab === 'proactive') loadProactive();
  if (tab === 'capacity') loadCapacity();
  if (tab === 'quality') loadQuality();
  if (tab === 'network') loadNetwork();
  if (tab === 'executive') loadExecutive();
  if (tab === 'livecalls') loadLiveCalls();
  if (tab === 'modelcompare') loadModelCompare();
  if (tab === 'power') loadPower();
  if (tab === 'incidents') loadIncidents();
  if (tab === 'alertsconfig') loadAlertsConfig();
  if (tab === 'sla') loadSLA();
  if (tab === 'anomalies') loadAnomalies();
}

// ── Overview Tab ──
function renderGPUCard(g, hasEcc) {
  const uc=getUtilColor(g.gpu_util), tc=getTempColor(g.temp||0);
  const pp=g.power_limit>0?(g.power_draw/g.power_limit*100):0, pc=getPowerColor(pp);
  const mp=g.mem_total>0?(g.mem_used/g.mem_total*100):0, mc=getColor(mp,[60,90]);
  return `<div class="gpu-card glass-bright rounded-xl p-4">
    <div class="flex items-center justify-between mb-3">
      <span class="text-xs font-medium text-gray-400">GPU ${g.index}</span>
      <div class="flex items-center gap-2">
        ${g.fan!==null?`<span class="text-[10px] text-gray-500">${g.fan}% fan</span>`:''}
        ${hasEcc&&g.ecc_errors!==undefined?`<span class="text-[10px] ${g.ecc_errors>0?'text-red-400':'text-gray-600'}">ECC:${g.ecc_errors}</span>`:''}
      </div>
    </div>
    <div class="flex items-center gap-4 mb-3">
      <div class="relative flex-shrink-0">${ringSVG(g.gpu_util,100,64,5,uc.ring)}
        <div class="absolute inset-0 flex items-center justify-center"><span class="text-sm font-semibold ${uc.text}">${g.gpu_util}%</span></div>
      </div>
      <div class="flex-1 space-y-1.5">
        <div class="flex justify-between text-[11px]"><span class="text-gray-500">Temp</span><span class="${tc.text} font-medium">${g.temp!==null?g.temp+'°C':'N/A'}</span></div>
        <div class="flex justify-between text-[11px]"><span class="text-gray-500">Power</span><span class="${pc.text} font-medium">${g.power_draw.toFixed(0)}W/${g.power_limit.toFixed(0)}W</span></div>
      </div>
    </div>
    <div class="space-y-1.5">
      <div class="flex items-center justify-between text-[11px]"><span class="text-gray-500">VRAM</span><span class="text-gray-400">${formatMiB(g.mem_used)}/${formatMiB(g.mem_total)}</span></div>
      ${barHTML(g.mem_used,g.mem_total,mc.ring)}
    </div>
  </div>`;
}

function renderOverviewServer(key, d) {
  if (!d || d.status==='offline') {
    return `<div class="glass rounded-2xl p-6"><div class="flex items-center gap-3 mb-4"><div class="w-2.5 h-2.5 rounded-full bg-red-500 status-offline"></div><h2 class="text-lg font-semibold text-white">${d?.server_name||key}</h2><span class="text-xs text-gray-500">${d?.host||''}</span></div><div class="text-red-400 text-sm">Connection failed: ${d?.error||'Unknown'}</div></div>`;
  }
  const gpus=d.gpus||[], procs=d.processes||[];
  const tv=gpus.reduce((s,g)=>s+g.mem_total,0), uv=gpus.reduce((s,g)=>s+g.mem_used,0);
  const au=gpus.length?Math.round(gpus.reduce((s,g)=>s+g.gpu_util,0)/gpus.length):0;
  const tp=gpus.reduce((s,g)=>s+g.power_draw,0), tc=gpus.reduce((s,g)=>s+g.power_limit,0);
  const vp=tv>0?((uv/tv)*100).toFixed(0):0;

  // System bars
  let sys='';
  if(d.ram?.total){sys+=`<div class="flex items-center gap-2"><span class="text-gray-500 text-xs w-12">RAM</span><div class="flex-1">${barHTML(d.ram.used,d.ram.total,'#6366f1')}</div><span class="text-xs text-gray-400 w-28 text-right">${formatBytes(d.ram.used)} / ${formatBytes(d.ram.total)}</span></div>`;}
  if(d.disk?.total){sys+=`<div class="flex items-center gap-2"><span class="text-gray-500 text-xs w-12">Disk</span><div class="flex-1">${barHTML(d.disk.used,d.disk.total,'#8b5cf6')}</div><span class="text-xs text-gray-400 w-28 text-right">${formatBytes(d.disk.used)} / ${formatBytes(d.disk.total)}</span></div>`;}

  // Net throughput
  let netHtml = '';
  if (prevNetData[key]) {
    const dt = (d.timestamp - prevNetData[key].t) || 1;
    const rxRate = ((d.net_rx_bytes - prevNetData[key].rx) * 8) / dt;
    const txRate = ((d.net_tx_bytes - prevNetData[key].tx) * 8) / dt;
    if (rxRate >= 0 && txRate >= 0) {
      netHtml = `<div class="flex items-center gap-4 text-[11px] mt-1"><span class="text-gray-500">Net:</span><span class="text-cyan-400">↓ ${formatRate(rxRate)}</span><span class="text-emerald-400">↑ ${formatRate(txRate)}</span></div>`;
    }
  }
  prevNetData[key] = { t: d.timestamp, rx: d.net_rx_bytes, tx: d.net_tx_bytes };

  // Procs
  let procHTML='';
  if(procs.length>0){
    const up={};procs.forEach(p=>{const k=p.pid+p.name;if(!up[k])up[k]={...p,count:1};else{up[k].mem_used+=p.mem_used;up[k].count++;}});
    const pl=Object.values(up).sort((a,b)=>b.mem_used-a.mem_used).slice(0,8);
    procHTML=`<details class="mt-4"><summary class="text-xs text-gray-500 cursor-pointer hover:text-gray-300">Processes (${procs.length})</summary><div class="mt-2 space-y-1">${pl.map(p=>`<div class="flex justify-between text-[11px] text-gray-400 py-1 px-2 rounded hover:bg-white/[0.03]"><span class="truncate flex-1 mr-2" title="${p.name}">${p.name}</span><span class="text-gray-500 flex-shrink-0">PID ${p.pid} · ${formatMiB(p.mem_used)}</span></div>`).join('')}</div></details>`;
  }

  return `<div class="glass rounded-2xl p-6">
    <div class="flex items-center justify-between mb-4">
      <div class="flex items-center gap-3"><div class="w-2.5 h-2.5 rounded-full bg-emerald-500 status-online"></div><h2 class="text-lg font-semibold text-white">${d.server_name}</h2><span class="text-xs text-gray-500 font-mono">${d.host}</span></div>
      <div class="text-xs text-gray-500">up ${formatUptime(d.uptime_seconds)}</div>
    </div>
    <div class="grid grid-cols-4 gap-3 mb-4">
      <div class="glass-bright rounded-lg p-3 text-center"><div class="text-[10px] text-gray-500 uppercase tracking-wider mb-1">GPUs</div><div class="text-lg font-semibold text-white">${gpus.length}</div><div class="text-[10px] text-gray-500">${d.gpu_model.replace('NVIDIA ','').replace('GeForce ','')}</div></div>
      <div class="glass-bright rounded-lg p-3 text-center"><div class="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Avg Util</div><div class="text-lg font-semibold ${getUtilColor(au).text}">${au}%</div><div class="text-[10px] text-gray-500">compute</div></div>
      <div class="glass-bright rounded-lg p-3 text-center"><div class="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Total VRAM</div><div class="text-lg font-semibold text-white">${vp}%</div><div class="text-[10px] text-gray-500">${formatMiB(uv)} / ${formatMiB(tv)}</div></div>
      <div class="glass-bright rounded-lg p-3 text-center"><div class="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Power</div><div class="text-lg font-semibold ${getPowerColor(tc>0?tp/tc*100:0).text}">${tp.toFixed(0)}W</div><div class="text-[10px] text-gray-500">/ ${tc.toFixed(0)}W cap</div></div>
    </div>
    <div class="space-y-2 mb-4">${sys}${netHtml}</div>
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-3">${gpus.map(g=>renderGPUCard(g, d.gpu_model.includes('H200'))).join('')}</div>
    ${procHTML}
  </div>`;
}

function renderOverview() {
  const el = document.getElementById('page-overview');
  el.innerHTML = `<div class="grid grid-cols-1 2xl:grid-cols-2 gap-6">${SERVERS.map(k => renderOverviewServer(k, serverData[k])).join('')}</div>`;
}

// ── Traffic & KPIs Tab ──
function renderVllmService(v) {
  const shortModel = v.model_name.split('/').pop();
  return `<div class="glass-bright rounded-xl p-4">
    <div class="flex items-center justify-between mb-3">
      <div><span class="text-sm font-medium text-white">${shortModel}</span><span class="text-[10px] text-gray-500 ml-2">:${v.port}</span></div>
      <div class="flex items-center gap-2">
        <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium ${v.requests_running>0?'bg-emerald-500/20 text-emerald-400':'bg-gray-700/50 text-gray-500'}">${v.requests_running} running</span>
        <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium ${v.requests_waiting>0?'bg-amber-500/20 text-amber-400':'bg-gray-700/50 text-gray-500'}">${v.requests_waiting} queued</span>
      </div>
    </div>
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
      <div class="kpi-card rounded-lg p-2.5 bg-white/[0.02]"><div class="text-[10px] text-gray-500 mb-0.5">Total Requests</div><div class="text-base font-semibold text-white">${fmt(v.total_requests)}</div></div>
      <div class="kpi-card rounded-lg p-2.5 bg-white/[0.02]"><div class="text-[10px] text-gray-500 mb-0.5">Prompt Tokens</div><div class="text-base font-semibold text-cyan-400">${(v.prompt_tokens/1e6).toFixed(1)}M</div></div>
      <div class="kpi-card rounded-lg p-2.5 bg-white/[0.02]"><div class="text-[10px] text-gray-500 mb-0.5">Gen Tokens</div><div class="text-base font-semibold text-purple-400">${(v.generation_tokens/1e6).toFixed(1)}M</div></div>
      <div class="kpi-card rounded-lg p-2.5 bg-white/[0.02]"><div class="text-[10px] text-gray-500 mb-0.5">Success</div><div class="text-base font-semibold text-emerald-400">${fmt(v.success_count)}</div></div>
    </div>
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <div class="kpi-card rounded-lg p-2.5 bg-white/[0.02]"><div class="text-[10px] text-gray-500 mb-0.5">Avg TTFT</div><div class="text-sm font-semibold ${v.avg_ttft_ms>500?'text-red-400':v.avg_ttft_ms>200?'text-amber-400':'text-emerald-400'}">${v.avg_ttft_ms.toFixed(0)}ms</div></div>
      <div class="kpi-card rounded-lg p-2.5 bg-white/[0.02]"><div class="text-[10px] text-gray-500 mb-0.5">Avg ITL</div><div class="text-sm font-semibold ${v.avg_itl_ms>100?'text-red-400':v.avg_itl_ms>50?'text-amber-400':'text-emerald-400'}">${v.avg_itl_ms.toFixed(1)}ms</div></div>
      <div class="kpi-card rounded-lg p-2.5 bg-white/[0.02]"><div class="text-[10px] text-gray-500 mb-0.5">Avg E2E</div><div class="text-sm font-semibold text-gray-300">${v.avg_e2e_s.toFixed(2)}s</div></div>
      <div class="kpi-card rounded-lg p-2.5 bg-white/[0.02]"><div class="text-[10px] text-gray-500 mb-0.5">Avg Queue</div><div class="text-sm font-semibold ${v.avg_queue_s>1?'text-red-400':v.avg_queue_s>0.5?'text-amber-400':'text-emerald-400'}">${v.avg_queue_s.toFixed(3)}s</div></div>
    </div>
    <div class="grid grid-cols-2 gap-3 mt-3">
      <div class="kpi-card rounded-lg p-2.5 bg-white/[0.02]"><div class="text-[10px] text-gray-500 mb-0.5">KV Cache</div><div class="text-sm font-semibold text-white">${v.kv_cache_usage}%</div>${barHTML(v.kv_cache_usage,100,'#f59e0b')}</div>
      <div class="kpi-card rounded-lg p-2.5 bg-white/[0.02]"><div class="text-[10px] text-gray-500 mb-0.5">Prefix Cache Hit</div><div class="text-sm font-semibold text-white">${v.cache_hit_rate}%</div>${barHTML(v.cache_hit_rate,100,'#22c55e')}</div>
    </div>
  </div>`;
}

function renderTraffic() {
  const el = document.getElementById('page-traffic');
  let html = '';
  for (const key of SERVERS) {
    const d = serverData[key];
    if (!d || d.status === 'offline') continue;
    const vllm = d.vllm || [];
    if (vllm.length === 0) {
      html += `<div class="glass rounded-2xl p-6 mb-6"><h3 class="text-lg font-semibold text-white mb-2">${d.server_name}</h3><p class="text-gray-500 text-sm">No vLLM services detected</p></div>`;
      continue;
    }
    // Aggregate KPIs
    const totalReqs = vllm.reduce((s,v)=>s+v.total_requests,0);
    const totalPrompt = vllm.reduce((s,v)=>s+v.prompt_tokens,0);
    const totalGen = vllm.reduce((s,v)=>s+v.generation_tokens,0);
    const totalRunning = vllm.reduce((s,v)=>s+v.requests_running,0);
    const totalWaiting = vllm.reduce((s,v)=>s+v.requests_waiting,0);

    html += `<div class="glass rounded-2xl p-6 mb-6">
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center gap-3"><div class="w-2.5 h-2.5 rounded-full bg-emerald-500 status-online"></div><h3 class="text-lg font-semibold text-white">${d.server_name}</h3></div>
        <div class="flex gap-4 text-xs">
          <span class="text-gray-400">Total: <span class="text-white font-semibold">${fmt(totalReqs)}</span> reqs</span>
          <span class="text-gray-400">Prompt: <span class="text-cyan-400 font-semibold">${(totalPrompt/1e6).toFixed(1)}M</span> tok</span>
          <span class="text-gray-400">Gen: <span class="text-purple-400 font-semibold">${(totalGen/1e6).toFixed(1)}M</span> tok</span>
          <span class="text-gray-400">Active: <span class="text-emerald-400 font-semibold">${totalRunning}</span></span>
          <span class="text-gray-400">Queued: <span class="${totalWaiting>0?'text-amber-400':'text-gray-500'} font-semibold">${totalWaiting}</span></span>
        </div>
      </div>
      <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">${vllm.map(v=>renderVllmService(v)).join('')}</div>
    </div>`;
  }
  if (!html) html = '<div class="text-gray-500 text-center py-12">Loading traffic data...</div>';
  el.innerHTML = html;
}

// ── History & Peaks Tab ──
function renderHistory() {
  const el = document.getElementById('page-history');
  let html = '';
  for (const key of SERVERS) {
    const h = historyData[key] || [];
    const d = serverData[key];
    if (!d || d.status === 'offline') continue;

    if (h.length < 2) {
      html += `<div class="glass rounded-2xl p-6 mb-6"><h3 class="text-lg font-semibold text-white mb-2">${d.server_name}</h3><p class="text-gray-500 text-sm">Collecting data... (${h.length} samples so far)</p></div>`;
      continue;
    }

    const utils = h.map(p=>p.avg_util);
    const temps = h.map(p=>p.max_temp);
    const powers = h.map(p=>p.total_power);
    const running = h.map(p=>p.requests_running);
    const waiting = h.map(p=>p.requests_waiting);
    const connections = h.map(p=>p.active_connections||0);
    const hasConnections = connections.some(c=>c>0);

    // Peak analysis
    const peakUtil = Math.max(...utils);
    const peakUtilIdx = utils.indexOf(peakUtil);
    const peakUtilTime = new Date(h[peakUtilIdx].t * 1000).toLocaleTimeString();
    const peakTemp = Math.max(...temps);
    const peakPower = Math.max(...powers);
    const peakRunning = Math.max(...running);
    const peakRunIdx = running.indexOf(peakRunning);
    const peakRunTime = running.length > 0 ? new Date(h[peakRunIdx].t * 1000).toLocaleTimeString() : 'N/A';
    const peakWaiting = Math.max(...waiting);
    const peakConns = Math.max(...connections);
    const peakConnsIdx = connections.indexOf(peakConns);
    const peakConnsTime = peakConnsIdx >= 0 ? new Date(h[peakConnsIdx].t * 1000).toLocaleTimeString() : 'N/A';
    const currentConns = connections.length > 0 ? connections[connections.length-1] : 0;

    const avgUtil = (utils.reduce((a,b)=>a+b,0)/utils.length).toFixed(1);
    const avgTemp = (temps.reduce((a,b)=>a+b,0)/temps.length).toFixed(1);
    const avgConns = connections.length > 0 ? (connections.reduce((a,b)=>a+b,0)/connections.length).toFixed(1) : '0';

    const duration = h.length > 1 ? ((h[h.length-1].t - h[0].t) / 60).toFixed(0) : 0;

    // Concurrent calls section (for RTX 5090 / AgentV2)
    let callsHTML = '';
    if (hasConnections) {
      callsHTML = `
      <div class="glass-bright rounded-xl p-4 mb-5">
        <div class="flex items-center gap-2 mb-3">
          <svg class="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"/></svg>
          <span class="text-sm font-semibold text-white">Concurrent Calls (AgentV2)</span>
          <span class="ml-auto text-xs px-2 py-0.5 rounded-full ${currentConns>0?'bg-emerald-500/20 text-emerald-400':'bg-gray-700/50 text-gray-500'}">${currentConns} active now</span>
        </div>
        <div class="grid grid-cols-3 gap-3 mb-3">
          <div class="bg-white/[0.02] rounded-lg p-3 text-center">
            <div class="text-[10px] text-gray-500 uppercase mb-1">Peak Concurrent Calls</div>
            <div class="text-2xl font-bold text-cyan-400">${peakConns}</div>
            <div class="text-[10px] text-gray-500">at ${peakConnsTime}</div>
          </div>
          <div class="bg-white/[0.02] rounded-lg p-3 text-center">
            <div class="text-[10px] text-gray-500 uppercase mb-1">Avg Concurrent Calls</div>
            <div class="text-2xl font-bold text-gray-300">${avgConns}</div>
            <div class="text-[10px] text-gray-500">over session</div>
          </div>
          <div class="bg-white/[0.02] rounded-lg p-3 text-center">
            <div class="text-[10px] text-gray-500 uppercase mb-1">Current Active</div>
            <div class="text-2xl font-bold ${currentConns>0?'text-emerald-400':'text-gray-500'}">${currentConns}</div>
            <div class="text-[10px] text-gray-500">connections</div>
          </div>
        </div>
        <div class="text-xs text-gray-400 mb-1">Concurrent Calls Over Time</div>
        ${sparklineSVG(connections, 600, 50, '#06b6d4')}
      </div>`;
    }

    html += `<div class="glass rounded-2xl p-6 mb-6">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-white">${d.server_name}</h3>
        <span class="text-xs text-gray-500">${h.length} samples over ${duration} min</span>
      </div>

      ${callsHTML}

      <!-- Peak Cards -->
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-5">
        <div class="glass-bright rounded-lg p-3">
          <div class="text-[10px] text-gray-500 uppercase mb-1">Peak GPU Util</div>
          <div class="text-xl font-bold ${getUtilColor(peakUtil).text}">${peakUtil}%</div>
          <div class="text-[10px] text-gray-500">at ${peakUtilTime}</div>
          <div class="mt-1">${sparklineSVG(utils.slice(-60), 80, 20, getUtilColor(peakUtil).ring)}</div>
        </div>
        <div class="glass-bright rounded-lg p-3">
          <div class="text-[10px] text-gray-500 uppercase mb-1">Avg GPU Util</div>
          <div class="text-xl font-bold ${getUtilColor(parseFloat(avgUtil)).text}">${avgUtil}%</div>
          <div class="text-[10px] text-gray-500">over session</div>
        </div>
        <div class="glass-bright rounded-lg p-3">
          <div class="text-[10px] text-gray-500 uppercase mb-1">Peak Temp</div>
          <div class="text-xl font-bold ${getTempColor(peakTemp).text}">${peakTemp}°C</div>
          <div class="text-[10px] text-gray-500">avg ${avgTemp}°C</div>
          <div class="mt-1">${sparklineSVG(temps.slice(-60), 80, 20, getTempColor(peakTemp).ring)}</div>
        </div>
        <div class="glass-bright rounded-lg p-3">
          <div class="text-[10px] text-gray-500 uppercase mb-1">Peak Power</div>
          <div class="text-xl font-bold text-amber-400">${peakPower.toFixed(0)}W</div>
          <div class="text-[10px] text-gray-500">cluster total</div>
          <div class="mt-1">${sparklineSVG(powers.slice(-60), 80, 20, '#f59e0b')}</div>
        </div>
        <div class="glass-bright rounded-lg p-3">
          <div class="text-[10px] text-gray-500 uppercase mb-1">Peak vLLM Active</div>
          <div class="text-xl font-bold text-cyan-400">${peakRunning}</div>
          <div class="text-[10px] text-gray-500">at ${peakRunTime}</div>
          <div class="mt-1">${sparklineSVG(running.slice(-60), 80, 20, '#06b6d4')}</div>
        </div>
        <div class="glass-bright rounded-lg p-3">
          <div class="text-[10px] text-gray-500 uppercase mb-1">Peak Queued</div>
          <div class="text-xl font-bold ${peakWaiting>0?'text-amber-400':'text-gray-500'}">${peakWaiting}</div>
          <div class="text-[10px] text-gray-500">max waiting</div>
          <div class="mt-1">${sparklineSVG(waiting.slice(-60), 80, 20, '#f59e0b')}</div>
        </div>
      </div>

      <!-- Sparkline Charts -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="glass-bright rounded-xl p-4">
          <div class="text-xs text-gray-400 mb-2">GPU Utilization Over Time</div>
          ${sparklineSVG(utils, 400, 60, '#22c55e')}
        </div>
        <div class="glass-bright rounded-xl p-4">
          <div class="text-xs text-gray-400 mb-2">${hasConnections ? 'Concurrent Calls Over Time' : 'Active Requests Over Time'}</div>
          ${sparklineSVG(hasConnections ? connections : running, 400, 60, '#06b6d4')}
        </div>
        <div class="glass-bright rounded-xl p-4">
          <div class="text-xs text-gray-400 mb-2">Temperature Over Time</div>
          ${sparklineSVG(temps, 400, 60, '#ef4444')}
        </div>
        <div class="glass-bright rounded-xl p-4">
          <div class="text-xs text-gray-400 mb-2">Power Draw Over Time</div>
          ${sparklineSVG(powers, 400, 60, '#f59e0b')}
        </div>
      </div>
    </div>`;
  }
  if (!html) html = '<div class="text-gray-500 text-center py-12">Loading history data...</div>';
  el.innerHTML = html;
}

// ── Software & Tools Tab ──
async function checkPyPI(pkgName) {
  const normalized = pkgName.toLowerCase().replace(/_/g, '-');
  if (pypiCache[normalized]) return pypiCache[normalized];
  try {
    const r = await fetch(`https://pypi.org/pypi/${normalized}/json`);
    if (!r.ok) return null;
    const d = await r.json();
    const latest = d.info.version;
    const summary = d.info.summary || '';
    const projectUrl = d.info.project_url || d.info.package_url || '';
    const releaseUrl = `https://pypi.org/project/${normalized}/${latest}/`;
    pypiCache[normalized] = { latest, summary, releaseUrl };
    return pypiCache[normalized];
  } catch(e) { return null; }
}

function versionCompare(a, b) {
  const pa = a.replace(/[^0-9.]/g,'').split('.').map(Number);
  const pb = b.replace(/[^0-9.]/g,'').split('.').map(Number);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const na = pa[i]||0, nb = pb[i]||0;
    if (na < nb) return -1;
    if (na > nb) return 1;
  }
  return 0;
}

async function loadSoftware() {
  const el = document.getElementById('page-software');
  el.innerHTML = '<div class="text-gray-500 text-center py-12"><div class="shimmer inline-block w-48 h-4 rounded mb-2"></div><div class="text-sm mt-2">Loading software inventory...</div></div>';
  const [h200, rtx] = await Promise.all([
    fetch('/api/h200/software').then(r=>r.json()).catch(()=>null),
    fetch('/api/rtx5090/software').then(r=>r.json()).catch(()=>null),
  ]);
  softwareData.h200 = h200;
  softwareData.rtx5090 = rtx;
  softwareLoaded = true;
  renderSoftware();
}

async function checkAllUpdates(serverKey) {
  const sw = softwareData[serverKey];
  if (!sw || !sw.environments) return;
  for (const [envName, pkgs] of Object.entries(sw.environments)) {
    for (const pkg of pkgs) {
      const info = await checkPyPI(pkg.name);
      if (info) {
        pkg._latest = info.latest;
        pkg._releaseUrl = info.releaseUrl;
        pkg._hasUpdate = versionCompare(pkg.version, info.latest) < 0;
      }
    }
  }
  renderSoftware();
}

function renderSoftwareServer(key, sw) {
  if (!sw || sw.error) {
    return `<div class="glass rounded-2xl p-6 mb-6"><h3 class="text-lg font-semibold text-white">${key}</h3><p class="text-red-400 text-sm mt-2">${sw?.error||'Failed to load'}</p></div>`;
  }

  let sysHTML = `<div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
    <div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500 uppercase mb-1">NVIDIA Driver</div><div class="text-sm font-semibold text-white">${sw.driver}</div></div>
    <div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500 uppercase mb-1">CUDA</div><div class="text-sm font-semibold text-white">${sw.cuda||'N/A'}</div></div>
    <div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500 uppercase mb-1">Kernel</div><div class="text-sm font-semibold text-white">${sw.kernel}</div></div>
    <div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500 uppercase mb-1">Python</div><div class="text-sm font-semibold text-white">${sw.python}</div></div>
  </div>`;

  // Services
  let svcHTML = '';
  if (sw.services && sw.services.length > 0) {
    svcHTML = `<div class="mb-4"><div class="text-xs text-gray-400 mb-2 font-medium">Running Services</div><div class="flex flex-wrap gap-2">${sw.services.map(s=>`<span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"><span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>${s.name}</span>`).join('')}</div></div>`;
  }

  // Environments
  let envHTML = '';
  for (const [envName, pkgs] of Object.entries(sw.environments || {})) {
    const hasAnyUpdate = pkgs.some(p => p._hasUpdate);
    const checkedUpdates = pkgs.some(p => p._latest !== undefined);
    envHTML += `<div class="glass-bright rounded-xl p-4 mb-3">
      <div class="flex items-center justify-between mb-3">
        <div class="text-sm font-medium text-white">${envName}</div>
        <div class="flex items-center gap-2">
          ${hasAnyUpdate ? `<span class="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400">${pkgs.filter(p=>p._hasUpdate).length} updates available</span>` : ''}
          <span class="text-[10px] text-gray-500">${pkgs.length} packages</span>
        </div>
      </div>
      <table class="w-full text-xs">
        <thead><tr class="text-gray-500 border-b border-gray-700/50">
          <th class="text-left py-1.5 font-medium">Package</th>
          <th class="text-left py-1.5 font-medium">Installed</th>
          <th class="text-left py-1.5 font-medium">Latest</th>
          <th class="text-right py-1.5 font-medium">Actions</th>
        </tr></thead>
        <tbody>${pkgs.map(p => {
          const hasUpdate = p._hasUpdate;
          const latest = p._latest || (checkedUpdates ? p.version : '-');
          const rowClass = hasUpdate ? 'bg-amber-500/5' : '';
          return `<tr class="${rowClass} border-b border-gray-800/30 hover:bg-white/[0.02]">
            <td class="py-1.5 text-gray-300 font-medium">${p.name}</td>
            <td class="py-1.5 ${hasUpdate?'text-amber-400':'text-gray-400'}">${p.version}</td>
            <td class="py-1.5 ${hasUpdate?'text-emerald-400':'text-gray-500'}">${latest}</td>
            <td class="py-1.5 text-right">
              ${p._releaseUrl ? `<a href="${p._releaseUrl}" target="_blank" title="Release notes" class="inline-flex items-center justify-center w-6 h-6 rounded hover:bg-white/10 text-gray-500 hover:text-blue-400"><svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg></a>` : ''}
              ${hasUpdate ? `<span title="Update available — run: pip install ${p.name}==${p._latest}" class="inline-flex items-center justify-center w-6 h-6 rounded hover:bg-white/10 text-amber-400 hover:text-amber-300 cursor-pointer"><svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg></span>` : ''}
            </td>
          </tr>`;
        }).join('')}</tbody>
      </table>
    </div>`;
  }

  return `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-lg font-semibold text-white">${sw.server_name}</h3>
      <button onclick="checkAllUpdates('${key}')" class="text-xs px-3 py-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20 transition-colors">Check for Updates</button>
    </div>
    ${sysHTML}${svcHTML}${envHTML}
  </div>`;
}

function renderSoftware() {
  const el = document.getElementById('page-software');
  if (!softwareLoaded) {
    el.innerHTML = '<div class="text-gray-500 text-center py-12">Click the Software & Tools tab to load inventory</div>';
    return;
  }
  el.innerHTML = SERVERS.map(k => renderSoftwareServer(k, softwareData[k])).join('');
}

// ── 24h Report Tab ──
async function loadDaily() {
  const el = document.getElementById('page-daily');
  el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading 24h report...</div>';
  const [h200, rtx] = await Promise.all([
    fetch('/api/h200/daily').then(r=>r.json()).catch(()=>null),
    fetch('/api/rtx5090/daily').then(r=>r.json()).catch(()=>null),
  ]);
  dailyData.h200 = h200;
  dailyData.rtx5090 = rtx;
  dailyLoaded = true;
  renderDaily();
}

function fmtTime(ts) { return new Date(ts*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}); }
function fmtTok(n) { if(n>=1e6) return (n/1e6).toFixed(1)+'M'; if(n>=1e3) return (n/1e3).toFixed(1)+'K'; return n.toString(); }

function renderDailyServer(key, dd) {
  if (!dd || dd.error || !dd.hours) {
    return `<div class="glass rounded-2xl p-6 mb-6"><h3 class="text-lg font-semibold text-white">${dd?.server_name||key}</h3><p class="text-gray-500 text-sm mt-2">${dd?.error||'No data yet — collecting hourly stats'}</p></div>`;
  }

  const hours = dd.hours;
  if (hours.length === 0) {
    return `<div class="glass rounded-2xl p-6 mb-6"><h3 class="text-lg font-semibold text-white">${dd.server_name}</h3><p class="text-gray-500 text-sm mt-2">No hourly data collected yet. Data populates as the dashboard runs.</p></div>`;
  }

  const padded = hours.slice(-24);
  const mid = Math.ceil(padded.length / 2);
  const left = padded.slice(0, mid);
  const right = padded.slice(mid);
  const gpuCount = padded[0]?.gpu_count || 8;

  // Summary stats across all hours
  const allMaxCalls = Math.max(...hours.map(h=>h.max_calls), 0);
  const allMaxGpuSum = Math.max(...hours.map(h=>h.max_gpu_util_sum), 0);
  const allMaxTemp = Math.max(...hours.map(h=>h.max_temp), 0);
  const allMaxPower = Math.max(...hours.map(h=>h.max_power), 0);
  const totalHourReqs = hours.reduce((s,h)=>s+h.hour_requests, 0);
  const totalPromptTok = hours.reduce((s,h)=>s+h.hour_prompt_tokens, 0);
  const totalGenTok = hours.reduce((s,h)=>s+h.hour_gen_tokens, 0);
  const totalNetRx = hours.reduce((s,h)=>s+h.hour_net_rx, 0);
  const totalNetTx = hours.reduce((s,h)=>s+h.hour_net_tx, 0);
  const peakCallsHour = hours.reduce((best, h) => h.max_calls > (best?.max_calls||0) ? h : best, hours[0]);
  const peakGpuHour = hours.reduce((best, h) => h.max_gpu_util_sum > (best?.max_gpu_util_sum||0) ? h : best, hours[0]);

  // Sparkline data
  const callsSpark = padded.map(h=>h.max_calls);
  const gpuSpark = padded.map(h=>h.max_gpu_util_sum);
  const tempSpark = padded.map(h=>h.max_temp);
  const powerSpark = padded.map(h=>h.max_power);
  const reqsSpark = padded.map(h=>h.hour_requests);

  function hourRow(h) {
    const gpuPct = gpuCount > 0 ? Math.round(h.max_gpu_util_sum / gpuCount) : h.max_gpu_util_avg;
    const utilColor = gpuPct >= 85 ? 'text-red-400' : gpuPct >= 50 ? 'text-amber-400' : 'text-emerald-400';
    const callsColor = h.max_calls > 0 ? 'text-cyan-400' : 'text-gray-500';
    const callsBar = allMaxCalls > 0 ? (h.max_calls / allMaxCalls * 100) : 0;
    const gpuBar = Math.min(gpuPct, 100);
    return `<tr class="border-b border-gray-800/30 hover:bg-white/[0.03]">
      <td class="py-1.5 px-2 text-gray-300 font-mono text-[11px]">${h.hour_label}</td>
      <td class="py-1.5 px-2">
        <div class="flex items-center gap-1.5">
          <span class="${callsColor} font-semibold text-[11px] w-6 text-right">${h.max_calls}</span>
          <div class="bar-track rounded-full h-1 flex-1"><div class="bar-fill rounded-full h-1" style="width:${callsBar}%;background:#06b6d4"></div></div>
        </div>
      </td>
      <td class="py-1.5 px-2">
        <div class="flex items-center gap-1.5">
          <span class="${utilColor} font-semibold text-[11px] w-14 text-right">${h.max_gpu_util_sum}%<span class="text-gray-600 font-normal"> (${gpuPct})</span></span>
          <div class="bar-track rounded-full h-1 flex-1"><div class="bar-fill rounded-full h-1" style="width:${gpuBar}%;background:${gpuPct>=85?'#ef4444':gpuPct>=50?'#f59e0b':'#22c55e'}"></div></div>
        </div>
      </td>
      <td class="py-1.5 px-2 text-center text-[11px] text-gray-300">${h.calls_at_max_gpu}</td>
      <td class="py-1.5 px-2 text-center text-[11px] ${h.max_temp>=80?'text-red-400':h.max_temp>=60?'text-amber-400':'text-gray-400'}">${h.max_temp}°</td>
      <td class="py-1.5 px-2 text-right text-[11px] text-gray-400">${h.max_power>0?h.max_power.toFixed(0)+'W':'-'}</td>
      <td class="py-1.5 px-2 text-right text-[11px] text-gray-400">${h.hour_requests>0?fmt(h.hour_requests):'-'}</td>
    </tr>`;
  }

  function halfTable(rows, label) {
    if (rows.length === 0) return `<div class="flex-1 glass-bright rounded-xl p-4"><div class="text-xs text-gray-500 text-center py-8">No data</div></div>`;
    return `<div class="flex-1 glass-bright rounded-xl p-3 overflow-x-auto">
      <div class="text-xs text-gray-400 mb-2 font-medium">${label}</div>
      <table class="w-full text-xs">
        <thead><tr class="text-gray-500 border-b border-gray-700/50">
          <th class="text-left py-1 px-2 font-medium text-[10px]">Hour</th>
          <th class="text-left py-1 px-2 font-medium text-[10px]">Peak Calls</th>
          <th class="text-left py-1 px-2 font-medium text-[10px]">GPU Sum% (avg)</th>
          <th class="text-center py-1 px-2 font-medium text-[10px]">Calls@GPU</th>
          <th class="text-center py-1 px-2 font-medium text-[10px]">Temp</th>
          <th class="text-right py-1 px-2 font-medium text-[10px]">Power</th>
          <th class="text-right py-1 px-2 font-medium text-[10px]">Reqs</th>
        </tr></thead>
        <tbody>${rows.map(hourRow).join('')}</tbody>
      </table>
    </div>`;
  }

  const firstHour = padded[0]?.hour_label || '?';
  const lastHour = padded[padded.length-1]?.hour_label || '?';

  return `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center justify-between mb-4">
      <div class="flex items-center gap-3">
        <h3 class="text-lg font-semibold text-white">${dd.server_name}</h3>
        <span class="text-xs text-gray-500">${firstHour} — ${lastHour} UTC &middot; ${padded.length}h &middot; ${gpuCount} GPUs</span>
      </div>
      <button onclick="loadDaily()" class="text-xs px-3 py-1.5 rounded-lg bg-white/[0.05] text-gray-400 border border-gray-700 hover:bg-white/10 transition-colors">Refresh</button>
    </div>

    <!-- Summary KPIs -->
    <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2 mb-5">
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Peak Calls</div>
        <div class="text-lg font-bold text-cyan-400">${allMaxCalls}</div>
        <div class="text-[9px] text-gray-500">${peakCallsHour?.hour_label||'?'} UTC</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Peak GPU Sum</div>
        <div class="text-lg font-bold ${allMaxGpuSum/gpuCount>=85?'text-red-400':allMaxGpuSum/gpuCount>=50?'text-amber-400':'text-emerald-400'}">${allMaxGpuSum}%</div>
        <div class="text-[9px] text-gray-500">avg ${Math.round(allMaxGpuSum/gpuCount)}% per GPU</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Busiest Hour</div>
        <div class="text-lg font-bold text-white">${peakCallsHour?.hour_label||'—'}</div>
        <div class="text-[9px] text-gray-500">${peakCallsHour?.max_calls||0} calls</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Peak Temp</div>
        <div class="text-lg font-bold ${allMaxTemp>=80?'text-red-400':allMaxTemp>=60?'text-amber-400':'text-emerald-400'}">${allMaxTemp}°C</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Peak Power</div>
        <div class="text-lg font-bold text-amber-400">${allMaxPower.toFixed(0)}W</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Total Requests</div>
        <div class="text-lg font-bold text-white">${fmtTok(totalHourReqs)}</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Tokens In/Out</div>
        <div class="text-sm font-bold text-purple-400">${fmtTok(totalPromptTok)}/${fmtTok(totalGenTok)}</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Network I/O</div>
        <div class="text-sm font-bold text-gray-300">${formatBytes(totalNetRx)}/${formatBytes(totalNetTx)}</div>
      </div>
    </div>

    <!-- Sparkline trends -->
    <div class="grid grid-cols-2 lg:grid-cols-5 gap-3 mb-5">
      <div class="glass-bright rounded-lg p-3">
        <div class="text-[10px] text-gray-500 mb-1">Calls per Hour</div>
        ${sparklineSVG(callsSpark, 140, 30, '#06b6d4')}
      </div>
      <div class="glass-bright rounded-lg p-3">
        <div class="text-[10px] text-gray-500 mb-1">GPU Sum% per Hour</div>
        ${sparklineSVG(gpuSpark, 140, 30, '#22c55e')}
      </div>
      <div class="glass-bright rounded-lg p-3">
        <div class="text-[10px] text-gray-500 mb-1">Temp per Hour</div>
        ${sparklineSVG(tempSpark, 140, 30, '#ef4444')}
      </div>
      <div class="glass-bright rounded-lg p-3">
        <div class="text-[10px] text-gray-500 mb-1">Power per Hour</div>
        ${sparklineSVG(powerSpark, 140, 30, '#f59e0b')}
      </div>
      <div class="glass-bright rounded-lg p-3">
        <div class="text-[10px] text-gray-500 mb-1">Requests per Hour</div>
        ${sparklineSVG(reqsSpark, 140, 30, '#8b5cf6')}
      </div>
    </div>

    <!-- Two-column hourly breakdown -->
    <div class="flex gap-4">
      ${halfTable(left, left.length > 0 ? left[0].hour_label + ' — ' + left[left.length-1].hour_label + ' UTC' : 'Earlier')}
      ${halfTable(right, right.length > 0 ? right[0].hour_label + ' — ' + right[right.length-1].hour_label + ' UTC' : 'Later')}
    </div>
  </div>`;
}

function renderDaily() {
  const el = document.getElementById('page-daily');
  if (!dailyLoaded) {
    el.innerHTML = '<div class="text-gray-500 text-center py-12">Click 24h Report tab to load data</div>';
    return;
  }
  el.innerHTML = SERVERS.map(k => renderDailyServer(k, dailyData[k])).join('');
}

// ── Agent Tab ──
async function loadAgent() {
  const el = document.getElementById('page-agent');
  if (!agentData) el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading AgentV2 status...</div>';
  try {
    const r = await fetch('/api/agent');
    agentData = await r.json();
  } catch(e) {
    agentData = { error: e.message };
  }
  renderAgent();
}

function renderAgent() {
  const el = document.getElementById('page-agent');
  const d = agentData;
  if (!d || d.error) {
    el.innerHTML = `<div class="glass rounded-2xl p-6"><h3 class="text-lg font-semibold text-white">AgentV2 Status</h3><p class="text-red-400 text-sm mt-2">${d?.error||'Failed to load'}</p></div>`;
    return;
  }

  // Status indicator
  const allUp = d.services?.every(s => s.status === 'active');
  const statusBadge = allUp
    ? '<span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"><span class="w-2 h-2 rounded-full bg-emerald-500 status-online"></span>All Services Running</span>'
    : '<span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20"><span class="w-2 h-2 rounded-full bg-amber-500"></span>Some Services Down</span>';

  // KPI cards
  let kpiHTML = `<div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3 mb-5">
    <div class="glass-bright rounded-lg p-3 text-center">
      <div class="text-[9px] text-gray-500 uppercase mb-1">Active Calls</div>
      <div class="text-2xl font-bold ${d.active_bots>0?'text-cyan-400':'text-gray-500'}">${d.active_bots}</div>
      <div class="text-[9px] text-gray-500">bot processes</div>
    </div>
    <div class="glass-bright rounded-lg p-3 text-center">
      <div class="text-[9px] text-gray-500 uppercase mb-1">TCP Connections</div>
      <div class="text-2xl font-bold ${d.tcp_connections>0?'text-emerald-400':'text-gray-500'}">${d.tcp_connections}</div>
      <div class="text-[9px] text-gray-500">port 8003</div>
    </div>
    <div class="glass-bright rounded-lg p-3 text-center">
      <div class="text-[9px] text-gray-500 uppercase mb-1">Gunicorn Workers</div>
      <div class="text-2xl font-bold text-white">${d.gunicorn_workers}</div>
      <div class="text-[9px] text-gray-500">${d.gunicorn_memory_mb} MB RSS</div>
    </div>
    <div class="glass-bright rounded-lg p-3 text-center">
      <div class="text-[9px] text-gray-500 uppercase mb-1">Voicemail Workers</div>
      <div class="text-2xl font-bold text-white">${d.voicemail_workers}</div>
      <div class="text-[9px] text-gray-500">classifier</div>
    </div>
    <div class="glass-bright rounded-lg p-3 text-center">
      <div class="text-[9px] text-gray-500 uppercase mb-1">Health Watcher</div>
      <div class="text-2xl font-bold text-white">${d.health_checker_workers}</div>
      <div class="text-[9px] text-gray-500">${d.watched_pids} PIDs watched</div>
    </div>
    <div class="glass-bright rounded-lg p-3 text-center">
      <div class="text-[9px] text-gray-500 uppercase mb-1">Failed Queue</div>
      <div class="text-2xl font-bold ${d.failed_reports_queue>0?'text-red-400':'text-emerald-400'}">${d.failed_reports_queue}</div>
      <div class="text-[9px] text-gray-500">pending retries</div>
    </div>
    <div class="glass-bright rounded-lg p-3 text-center">
      <div class="text-[9px] text-gray-500 uppercase mb-1">Zombies</div>
      <div class="text-2xl font-bold ${d.zombie_processes>0?'text-red-400':'text-emerald-400'}">${d.zombie_processes}</div>
      <div class="text-[9px] text-gray-500">defunct procs</div>
    </div>
  </div>`;

  // Services
  let svcHTML = '<div class="glass-bright rounded-xl p-4 mb-5"><div class="text-xs text-gray-400 mb-3 font-medium">Services</div><div class="grid grid-cols-2 md:grid-cols-5 gap-2">';
  const roles = {"agentv2-main-server":"Call Router :8003","agentv2-voicemail-processor":"Intent Classifier :8001","agentv2-health-checker":"Process Monitor :9000","agentv2-failed-report":"Retry Queue :8006","agentv2-server-old":"Legacy Server"};
  for (const s of (d.services||[])) {
    const ok = s.status === 'active';
    svcHTML += `<div class="rounded-lg p-2.5 ${ok?'bg-emerald-500/5 border border-emerald-500/10':'bg-red-500/5 border border-red-500/10'}">
      <div class="flex items-center gap-1.5 mb-1"><span class="w-1.5 h-1.5 rounded-full ${ok?'bg-emerald-500':'bg-red-500'}"></span><span class="text-[11px] font-medium ${ok?'text-emerald-400':'text-red-400'}">${ok?'Active':'Down'}</span></div>
      <div class="text-[11px] text-white font-medium">${s.name.replace('agentv2-','')}</div>
      <div class="text-[9px] text-gray-500">${roles[s.name]||''}</div>
    </div>`;
  }
  svcHTML += '</div></div>';

  // External Dependencies
  let extHTML = '<div class="glass-bright rounded-xl p-4 mb-5"><div class="text-xs text-gray-400 mb-3 font-medium">External Dependencies</div><div class="grid grid-cols-1 md:grid-cols-2 gap-2">';
  for (const dep of (d.external_deps||[])) {
    extHTML += `<div class="flex items-center justify-between rounded-lg p-2.5 ${dep.ok?'bg-emerald-500/5':'bg-red-500/5'} border ${dep.ok?'border-emerald-500/10':'border-red-500/10'}">
      <div><div class="text-[11px] text-white font-medium">${dep.name}</div><div class="text-[9px] text-gray-500 font-mono">${dep.host}</div></div>
      <span class="text-[10px] px-2 py-0.5 rounded-full ${dep.ok?'bg-emerald-500/20 text-emerald-400':'bg-red-500/20 text-red-400'}">${dep.ok?'Reachable':'Unreachable'}</span>
    </div>`;
  }
  extHTML += '</div></div>';

  // Active bot details
  let botsHTML = '';
  if (d.bot_details && d.bot_details.length > 0) {
    botsHTML = `<div class="glass-bright rounded-xl p-4 mb-5">
      <div class="text-xs text-gray-400 mb-3 font-medium">Active Call Bots (${d.bot_details.length})</div>
      <table class="w-full text-xs"><thead><tr class="text-gray-500 border-b border-gray-700/50">
        <th class="text-left py-1.5 font-medium">PID</th><th class="text-left py-1.5 font-medium">Elapsed</th><th class="text-right py-1.5 font-medium">CPU%</th><th class="text-right py-1.5 font-medium">Memory</th><th class="text-left py-1.5 font-medium pl-4">Command</th>
      </tr></thead><tbody>${d.bot_details.map(b=>`<tr class="border-b border-gray-800/30">
        <td class="py-1.5 text-gray-300 font-mono">${b.pid}</td><td class="py-1.5 text-cyan-400">${b.elapsed}</td><td class="py-1.5 text-right text-gray-400">${b.cpu}%</td><td class="py-1.5 text-right text-gray-400">${(b.rss_kb/1024).toFixed(0)} MB</td><td class="py-1.5 text-gray-500 pl-4 truncate max-w-xs" title="${b.cmd}">${b.cmd}</td>
      </tr>`).join('')}</tbody></table>
    </div>`;
  }

  // DCGM GPU details
  let gpuHTML = '';
  if (d.dcgm_gpus && d.dcgm_gpus.length > 0) {
    gpuHTML = `<div class="glass-bright rounded-xl p-4 mb-5">
      <div class="text-xs text-gray-400 mb-3 font-medium">GPU Telemetry (DCGM)</div>
      <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2">${d.dcgm_gpus.map(g=>{
        const tc = g.temp>=60?'text-red-400':g.temp>=40?'text-amber-400':'text-emerald-400';
        return `<div class="rounded-lg p-2.5 bg-white/[0.02] border border-gray-800/50 text-center">
          <div class="text-[9px] text-gray-500 mb-1">GPU ${g.index}</div>
          <div class="text-sm font-semibold ${tc}">${g.temp}°C</div>
          <div class="text-[9px] text-gray-500">${g.power}W &middot; ${g.sm_clock}MHz</div>
          <div class="text-[9px] text-gray-500">${g.util}% util</div>
        </div>`;
      }).join('')}</div>
    </div>`;
  }

  // Architecture info
  let archHTML = `<div class="glass-bright rounded-xl p-4">
    <div class="text-xs text-gray-400 mb-3 font-medium">Architecture</div>
    <div class="text-[11px] text-gray-500 space-y-1.5">
      <div class="flex gap-2"><span class="text-gray-400 w-24">Call Flow:</span><span>Telnyx/WebRTC → <span class="text-white">:8003 Main Server</span> → Daily.co Room → Bot Process (per call)</span></div>
      <div class="flex gap-2"><span class="text-gray-400 w-24">LLM:</span><span>Bot → <span class="text-white">H200 vLLM :8001</span> (Qwen3-VL-8B-Instruct)</span></div>
      <div class="flex gap-2"><span class="text-gray-400 w-24">TTS:</span><span>Bot → <span class="text-white">H200 vLLM :8002</span> (recruite-tts-fp8)</span></div>
      <div class="flex gap-2"><span class="text-gray-400 w-24">STT:</span><span>Bot → <span class="text-white">External Whisper</span> (GLM-ASR-Nano)</span></div>
      <div class="flex gap-2"><span class="text-gray-400 w-24">Intent:</span><span>Bot → <span class="text-white">:8001 Voicemail Processor</span> (OpenRouter/Ministral-14B)</span></div>
      <div class="flex gap-2"><span class="text-gray-400 w-24">Webhooks:</span><span>Bot → Caller's webhook URL (call events, transcripts, summaries)</span></div>
    </div>
  </div>`;

  el.innerHTML = `<div class="glass rounded-2xl p-6">
    <div class="flex items-center justify-between mb-5">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center">
          <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"/></svg>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-white">AgentV2 — Voice Call Agent</h3>
          <p class="text-[11px] text-gray-500">RTX 5090 Cluster &middot; 38.65.239.47</p>
        </div>
      </div>
      <div class="flex items-center gap-3">
        ${statusBadge}
        <button onclick="loadAgent()" class="text-xs px-3 py-1.5 rounded-lg bg-white/[0.05] text-gray-400 border border-gray-700 hover:bg-white/10 transition-colors">Refresh</button>
      </div>
    </div>
    ${kpiHTML}${svcHTML}${extHTML}${botsHTML}${gpuHTML}${archHTML}
  </div>`;
}

// ── Analytics & Cost Tab ──
async function loadAnalytics() {
  const el = document.getElementById('page-analytics');
  if (!analyticsData) el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading analytics...</div>';
  try {
    const r = await fetch('/api/analytics');
    analyticsData = await r.json();
  } catch(e) {
    analyticsData = { error: e.message };
  }
  analyticsLoaded = true;
  renderAnalytics();
}

function renderAnalytics() {
  const el = document.getElementById('page-analytics');
  const d = analyticsData;
  if (!d || d.error) {
    el.innerHTML = `<div class="glass rounded-2xl p-6"><h3 class="text-lg font-semibold text-white">Analytics</h3><p class="text-red-400 text-sm mt-2">${d?.error||'Failed to load'}</p></div>`;
    return;
  }

  const c = d.cost || {};
  const cq = d.concurrency_quality || [];
  const bn = d.bottleneck || {};
  const hd = d.headroom || {};
  const al = d.alerts || [];
  const qs = d.quality_scorecard || {};
  const ef = d.efficiency || {};

  // ── Section 1: Cost Breakdown ──
  let costHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center justify-between mb-5">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
          <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-white">Cost Breakdown</h3>
          <p class="text-[11px] text-gray-500">${c.total_gpus} GPUs &middot; Based on ${c.data_hours}h of data</p>
        </div>
      </div>
      <button onclick="loadAnalytics()" class="text-xs px-3 py-1.5 rounded-lg bg-white/[0.05] text-gray-400 border border-gray-700 hover:bg-white/10 transition-colors">Refresh</button>
    </div>
    <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-4">
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Monthly Cost</div>
        <div class="text-xl font-bold text-white">$${fmt(c.monthly)}</div>
        <div class="text-[9px] text-gray-500">total GPU infra</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Daily Cost</div>
        <div class="text-xl font-bold text-white">$${c.daily?.toFixed(0)}</div>
        <div class="text-[9px] text-gray-500">per day</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Per GPU/Hour</div>
        <div class="text-xl font-bold text-cyan-400">$${c.per_gpu_hour}</div>
        <div class="text-[9px] text-gray-500">${c.total_gpus} GPUs</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Cost/Call/Min</div>
        <div class="text-xl font-bold ${c.per_call_minute>0?'text-emerald-400':'text-gray-500'}">$${c.per_call_minute?.toFixed(4)||'—'}</div>
        <div class="text-[9px] text-gray-500">avg ${c.avg_concurrent_calls} concurrent</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Cost/Call/Hour</div>
        <div class="text-xl font-bold ${c.per_call_hour>0?'text-emerald-400':'text-gray-500'}">$${c.per_call_hour?.toFixed(2)||'—'}</div>
        <div class="text-[9px] text-gray-500">per active call</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Idle GPU Waste</div>
        <div class="text-xl font-bold text-red-400">$${c.idle_gpu_cost_hourly?.toFixed(2)}/hr</div>
        <div class="text-[9px] text-gray-500">${(100 - (c.avg_gpu_util||0)).toFixed(0)}% idle</div>
      </div>
    </div>
    <div class="glass-bright rounded-lg p-3">
      <div class="text-xs text-gray-400 mb-2">Cost Utilization</div>
      <div class="flex items-center gap-3">
        <div class="flex-1">
          <div class="bar-track rounded-full h-3 w-full flex overflow-hidden">
            <div class="h-3 rounded-l-full" style="width:${c.avg_gpu_util||0}%;background:#22c55e"></div>
            <div class="h-3 ${c.avg_gpu_util>=100?'':'rounded-r-full'}" style="width:${100-(c.avg_gpu_util||0)}%;background:rgba(239,68,68,0.3)"></div>
          </div>
        </div>
        <div class="flex gap-4 text-[10px]">
          <span class="text-emerald-400">Utilized: $${c.utilized_gpu_cost_hourly?.toFixed(2)}/hr</span>
          <span class="text-red-400">Idle: $${c.idle_gpu_cost_hourly?.toFixed(2)}/hr</span>
        </div>
      </div>
    </div>
  </div>`;

  // ── Section 2: Concurrency vs Quality ──
  let cqHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center gap-3 mb-4">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
      </div>
      <div>
        <h3 class="text-lg font-semibold text-white">Concurrency vs Quality</h3>
        <p class="text-[11px] text-gray-500">How latency changes as concurrent calls increase</p>
      </div>
    </div>`;

  if (cq.length === 0) {
    cqHTML += '<p class="text-gray-500 text-sm">Collecting data... Quality metrics will appear as calls are processed.</p>';
  } else {
    // Find the "sweet spot" — highest concurrency where TTFT stays under 300ms
    const sweetSpot = [...cq].reverse().find(r => r.avg_ttft_ms > 0 && r.avg_ttft_ms <= 300);
    const degradePoint = cq.find(r => r.avg_ttft_ms > 500 && r.concurrent_calls > 0);

    if (sweetSpot || degradePoint) {
      cqHTML += `<div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
        ${sweetSpot ? `<div class="glass-bright rounded-lg p-3 border-l-2 border-emerald-500">
          <div class="text-[10px] text-gray-500 uppercase mb-1">Sweet Spot</div>
          <div class="text-lg font-bold text-emerald-400">${sweetSpot.concurrent_calls} concurrent calls</div>
          <div class="text-[11px] text-gray-400">TTFT stays under 300ms (avg ${sweetSpot.avg_ttft_ms}ms)</div>
        </div>` : ''}
        ${degradePoint ? `<div class="glass-bright rounded-lg p-3 border-l-2 border-red-500">
          <div class="text-[10px] text-gray-500 uppercase mb-1">Degradation Point</div>
          <div class="text-lg font-bold text-red-400">${degradePoint.concurrent_calls} concurrent calls</div>
          <div class="text-[11px] text-gray-400">TTFT exceeds 500ms (avg ${degradePoint.avg_ttft_ms}ms)</div>
        </div>` : ''}
      </div>`;
    }

    const maxTTFT = Math.max(...cq.map(r=>r.avg_ttft_ms), 1);
    cqHTML += `<div class="glass-bright rounded-xl p-4 overflow-x-auto">
      <table class="w-full text-xs">
        <thead><tr class="text-gray-500 border-b border-gray-700/50">
          <th class="text-left py-1.5 px-2 font-medium">Concurrent Calls</th>
          <th class="text-left py-1.5 px-2 font-medium">TTFT (ms)</th>
          <th class="text-left py-1.5 px-2 font-medium">ITL (ms)</th>
          <th class="text-left py-1.5 px-2 font-medium">E2E (s)</th>
          <th class="text-left py-1.5 px-2 font-medium">Queue (s)</th>
          <th class="text-left py-1.5 px-2 font-medium">KV Cache %</th>
          <th class="text-left py-1.5 px-2 font-medium">GPU Util %</th>
          <th class="text-right py-1.5 px-2 font-medium">Samples</th>
        </tr></thead>
        <tbody>${cq.map(r => {
          const ttftColor = r.avg_ttft_ms > 500 ? 'text-red-400' : r.avg_ttft_ms > 300 ? 'text-amber-400' : 'text-emerald-400';
          const ttftBar = maxTTFT > 0 ? (r.avg_ttft_ms / maxTTFT * 100) : 0;
          return `<tr class="border-b border-gray-800/30 hover:bg-white/[0.03]">
            <td class="py-1.5 px-2 text-white font-semibold">${r.concurrent_calls}</td>
            <td class="py-1.5 px-2">
              <div class="flex items-center gap-2">
                <span class="${ttftColor} font-semibold">${r.avg_ttft_ms > 0 ? r.avg_ttft_ms.toFixed(0) : '—'}</span>
                <div class="bar-track rounded-full h-1 w-16"><div class="bar-fill rounded-full h-1" style="width:${ttftBar}%;background:${r.avg_ttft_ms>500?'#ef4444':r.avg_ttft_ms>300?'#f59e0b':'#22c55e'}"></div></div>
              </div>
            </td>
            <td class="py-1.5 px-2 text-gray-400">${r.avg_itl_ms > 0 ? r.avg_itl_ms.toFixed(1) : '—'}</td>
            <td class="py-1.5 px-2 text-gray-400">${r.avg_e2e_s > 0 ? r.avg_e2e_s.toFixed(2) : '—'}</td>
            <td class="py-1.5 px-2 text-gray-400">${r.avg_queue_s > 0 ? r.avg_queue_s.toFixed(3) : '—'}</td>
            <td class="py-1.5 px-2 text-gray-400">${r.avg_kv_cache > 0 ? r.avg_kv_cache.toFixed(1) : '—'}</td>
            <td class="py-1.5 px-2 text-gray-400">${r.avg_gpu_util.toFixed(1)}</td>
            <td class="py-1.5 px-2 text-right text-gray-500">${r.samples}</td>
          </tr>`;
        }).join('')}</tbody>
      </table>
    </div>`;
  }
  cqHTML += '</div>';

  // ── Section 3: Bottleneck Analysis ──
  let bnHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center gap-3 mb-4">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
      </div>
      <div>
        <h3 class="text-lg font-semibold text-white">Bottleneck Analysis</h3>
        <p class="text-[11px] text-gray-500">Where time is spent in the inference pipeline (${bn.samples||0} samples)</p>
      </div>
    </div>`;

  if (bn.samples > 0) {
    const segs = [
      { label: 'LLM (TTFT)', pct: bn.llm_pct, color: '#06b6d4' },
      { label: 'TTS/Generation', pct: bn.tts_generation_pct, color: '#8b5cf6' },
      { label: 'Queue Wait', pct: bn.queue_pct, color: '#f59e0b' },
      { label: 'Other/Network', pct: bn.other_pct, color: '#6b7280' },
    ];
    bnHTML += `<div class="glass-bright rounded-lg p-4 mb-3">
      <div class="flex items-center gap-2 mb-3">
        <span class="text-xs text-gray-400">Primary bottleneck:</span>
        <span class="text-sm font-semibold text-white">${bn.primary_bottleneck}</span>
      </div>
      <div class="bar-track rounded-full h-6 w-full flex overflow-hidden mb-3">
        ${segs.map(s => `<div class="h-6 flex items-center justify-center" style="width:${s.pct}%;background:${s.color}" title="${s.label}: ${s.pct}%">
          ${s.pct >= 10 ? `<span class="text-[9px] text-white font-medium">${s.pct}%</span>` : ''}
        </div>`).join('')}
      </div>
      <div class="flex flex-wrap gap-4">
        ${segs.map(s => `<div class="flex items-center gap-1.5">
          <div class="w-2.5 h-2.5 rounded-sm" style="background:${s.color}"></div>
          <span class="text-[11px] text-gray-400">${s.label}: <span class="text-white font-medium">${s.pct}%</span></span>
        </div>`).join('')}
      </div>
    </div>`;
  } else {
    bnHTML += '<p class="text-gray-500 text-sm">Collecting latency data...</p>';
  }
  bnHTML += '</div>';

  // ── Section 4: GPU Headroom & Capacity ──
  let hdHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center gap-3 mb-4">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
      </div>
      <div>
        <h3 class="text-lg font-semibold text-white">GPU Headroom & Capacity Planning</h3>
        <p class="text-[11px] text-gray-500">How much more load can each cluster handle</p>
      </div>
    </div>
    <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">`;

  for (const [sk, h] of Object.entries(hd)) {
    if (h.status === 'no data') {
      hdHTML += `<div class="glass-bright rounded-xl p-4"><span class="text-gray-500 text-sm">${sk}: No data yet</span></div>`;
      continue;
    }
    const utilColor = h.avg_util >= 80 ? 'text-red-400' : h.avg_util >= 50 ? 'text-amber-400' : 'text-emerald-400';
    hdHTML += `<div class="glass-bright rounded-xl p-4">
      <div class="flex items-center justify-between mb-3">
        <span class="text-sm font-semibold text-white">${h.server_name}</span>
        <span class="text-xs px-2 py-0.5 rounded-full ${h.scale_factor>=2?'bg-emerald-500/20 text-emerald-400':h.scale_factor>=1.3?'bg-amber-500/20 text-amber-400':'bg-red-500/20 text-red-400'}">${h.scale_factor}x headroom</span>
      </div>
      <div class="grid grid-cols-3 gap-2 mb-3">
        <div class="bg-white/[0.02] rounded-lg p-2 text-center">
          <div class="text-[9px] text-gray-500 uppercase">Avg GPU</div>
          <div class="text-base font-bold ${utilColor}">${h.avg_util}%</div>
          <div class="text-[9px] text-gray-500">${h.util_headroom}% free</div>
        </div>
        <div class="bg-white/[0.02] rounded-lg p-2 text-center">
          <div class="text-[9px] text-gray-500 uppercase">VRAM</div>
          <div class="text-base font-bold text-white">${h.avg_vram_pct}%</div>
          <div class="text-[9px] text-gray-500">${h.vram_headroom}% free</div>
        </div>
        <div class="bg-white/[0.02] rounded-lg p-2 text-center">
          <div class="text-[9px] text-gray-500 uppercase">KV Cache</div>
          <div class="text-base font-bold text-white">${h.avg_kv_cache}%</div>
          <div class="text-[9px] text-gray-500">${h.kv_headroom.toFixed(0)}% free</div>
        </div>
      </div>
      <div class="grid grid-cols-2 gap-2 mb-3">
        <div class="bg-white/[0.02] rounded-lg p-2">
          <div class="text-[9px] text-gray-500 uppercase mb-0.5">Current Calls</div>
          <div class="flex items-baseline gap-1">
            <span class="text-base font-bold text-cyan-400">${h.avg_calls}</span>
            <span class="text-[10px] text-gray-500">avg / ${h.peak_calls} peak</span>
          </div>
        </div>
        <div class="bg-white/[0.02] rounded-lg p-2">
          <div class="text-[9px] text-gray-500 uppercase mb-0.5">Est. Max Calls</div>
          <div class="flex items-baseline gap-1">
            <span class="text-base font-bold text-emerald-400">${h.est_max_calls}</span>
            <span class="text-[10px] text-gray-500">${h.util_per_call}% GPU/call</span>
          </div>
        </div>
      </div>
      <div class="text-[10px] text-gray-500">
        Limits: GPU → ${h.est_max_calls_gpu} calls &middot; KV Cache → ${h.est_max_calls_kv > 0 ? h.est_max_calls_kv : '∞'} calls
      </div>
    </div>`;
  }
  hdHTML += '</div></div>';

  // ── Section 5: Quality Scorecard ──
  let qsHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center gap-3 mb-4">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-pink-500 to-rose-600 flex items-center justify-center">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
      </div>
      <div>
        <h3 class="text-lg font-semibold text-white">Call Quality Scorecard</h3>
        <p class="text-[11px] text-gray-500">Latency percentiles from recent data</p>
      </div>
    </div>`;

  const metrics = [
    { key: 'ttft', label: 'Time to First Token', unit: 'ms', warn: 300, crit: 600 },
    { key: 'itl', label: 'Inter-Token Latency', unit: 'ms', warn: 60, crit: 120 },
    { key: 'e2e', label: 'End-to-End Latency', unit: 's', warn: 3.0, crit: 6.0 },
    { key: 'queue', label: 'Queue Wait Time', unit: 's', warn: 0.5, crit: 2.0 },
  ];

  qsHTML += '<div class="grid grid-cols-1 md:grid-cols-2 gap-3">';
  for (const m of metrics) {
    const v = qs[m.key] || {};
    if (!v.samples) continue;
    const p95Color = v.p95 >= m.crit ? 'text-red-400' : v.p95 >= m.warn ? 'text-amber-400' : 'text-emerald-400';
    const avgColor = v.avg >= m.crit ? 'text-red-400' : v.avg >= m.warn ? 'text-amber-400' : 'text-emerald-400';
    qsHTML += `<div class="glass-bright rounded-xl p-4">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs font-medium text-white">${m.label}</span>
        <span class="text-[10px] text-gray-500">${v.samples} samples</span>
      </div>
      <div class="grid grid-cols-4 gap-2">
        <div class="text-center"><div class="text-[9px] text-gray-500">Avg</div><div class="text-sm font-bold ${avgColor}">${v.avg}${m.unit}</div></div>
        <div class="text-center"><div class="text-[9px] text-gray-500">P50</div><div class="text-sm font-bold text-gray-300">${v.p50}${m.unit}</div></div>
        <div class="text-center"><div class="text-[9px] text-gray-500">P95</div><div class="text-sm font-bold ${p95Color}">${v.p95}${m.unit}</div></div>
        <div class="text-center"><div class="text-[9px] text-gray-500">P99</div><div class="text-sm font-bold text-gray-400">${v.p99}${m.unit}</div></div>
      </div>
      <div class="mt-2">${barHTML(v.avg, m.crit, v.avg >= m.crit ? '#ef4444' : v.avg >= m.warn ? '#f59e0b' : '#22c55e')}</div>
    </div>`;
  }
  // KV Cache
  if (qs.kv_cache?.samples) {
    const kv = qs.kv_cache;
    qsHTML += `<div class="glass-bright rounded-xl p-4">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs font-medium text-white">KV Cache Usage</span>
        <span class="text-[10px] text-gray-500">${kv.samples} samples</span>
      </div>
      <div class="grid grid-cols-2 gap-2">
        <div class="text-center"><div class="text-[9px] text-gray-500">Avg</div><div class="text-sm font-bold text-white">${kv.avg}%</div></div>
        <div class="text-center"><div class="text-[9px] text-gray-500">Peak</div><div class="text-sm font-bold ${kv.peak>=90?'text-red-400':kv.peak>=75?'text-amber-400':'text-emerald-400'}">${kv.peak}%</div></div>
      </div>
      <div class="mt-2">${barHTML(kv.avg, 100, kv.avg >= 90 ? '#ef4444' : kv.avg >= 75 ? '#f59e0b' : '#22c55e')}</div>
    </div>`;
  }
  qsHTML += '</div></div>';

  // ── Section 6: Alerts ──
  let alHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center gap-3 mb-4">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-red-500 to-rose-600 flex items-center justify-center">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
      </div>
      <div>
        <h3 class="text-lg font-semibold text-white">Alerts & Thresholds</h3>
        <p class="text-[11px] text-gray-500">Current threshold violations</p>
      </div>
    </div>`;

  if (al.length === 0) {
    alHTML += '<div class="glass-bright rounded-lg p-4 text-center"><span class="text-emerald-400 text-sm font-medium">All metrics within normal thresholds</span></div>';
  } else {
    alHTML += '<div class="space-y-2">';
    for (const a of al) {
      const isCrit = a.level === 'critical';
      alHTML += `<div class="glass-bright rounded-lg p-3 flex items-center justify-between ${isCrit?'border-l-2 border-red-500':'border-l-2 border-amber-500'}">
        <div class="flex items-center gap-3">
          <span class="w-2 h-2 rounded-full ${isCrit?'bg-red-500':'bg-amber-500'}"></span>
          <div>
            <span class="text-xs font-medium text-white">${a.metric}</span>
            <span class="text-[10px] text-gray-500 ml-2">${a.server}</span>
          </div>
        </div>
        <div class="flex items-center gap-3">
          <span class="text-sm font-bold ${isCrit?'text-red-400':'text-amber-400'}">${a.value}</span>
          <span class="text-[10px] text-gray-500">threshold: ${a.threshold}</span>
          <span class="text-[10px] px-2 py-0.5 rounded-full ${isCrit?'bg-red-500/20 text-red-400':'bg-amber-500/20 text-amber-400'}">${a.level}</span>
        </div>
      </div>`;
    }
    alHTML += '</div>';
  }
  alHTML += '</div>';

  // ── Section 7: Model Efficiency ──
  let efHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center gap-3 mb-4">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/></svg>
      </div>
      <div>
        <h3 class="text-lg font-semibold text-white">Model Efficiency</h3>
        <p class="text-[11px] text-gray-500">Throughput and cost-per-token metrics</p>
      </div>
    </div>
    <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">`;

  for (const [sk, e] of Object.entries(ef)) {
    efHTML += `<div class="glass-bright rounded-xl p-4">
      <div class="text-sm font-semibold text-white mb-3">${e.server_name}</div>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
        <div class="bg-white/[0.02] rounded-lg p-2 text-center">
          <div class="text-[9px] text-gray-500 uppercase">Tokens/sec</div>
          <div class="text-base font-bold text-cyan-400">${e.tokens_per_sec}</div>
        </div>
        <div class="bg-white/[0.02] rounded-lg p-2 text-center">
          <div class="text-[9px] text-gray-500 uppercase">Reqs/min</div>
          <div class="text-base font-bold text-white">${e.reqs_per_min}</div>
        </div>
        <div class="bg-white/[0.02] rounded-lg p-2 text-center">
          <div class="text-[9px] text-gray-500 uppercase">Tokens/Watt</div>
          <div class="text-base font-bold text-emerald-400">${e.tokens_per_watt}</div>
        </div>
        <div class="bg-white/[0.02] rounded-lg p-2 text-center">
          <div class="text-[9px] text-gray-500 uppercase">$/1M Tokens</div>
          <div class="text-base font-bold text-amber-400">$${e.cost_per_million_tokens}</div>
        </div>
      </div>
      <div class="grid grid-cols-3 gap-2 mt-2">
        <div class="text-center text-[10px]"><span class="text-gray-500">Prompt:</span> <span class="text-gray-300">${(e.prompt_tokens_hour/1e3).toFixed(1)}K tok/hr</span></div>
        <div class="text-center text-[10px]"><span class="text-gray-500">Gen:</span> <span class="text-gray-300">${(e.gen_tokens_hour/1e3).toFixed(1)}K tok/hr</span></div>
        <div class="text-center text-[10px]"><span class="text-gray-500">Power:</span> <span class="text-gray-300">${e.avg_power_w}W avg</span></div>
      </div>
    </div>`;
  }
  if (Object.keys(ef).length === 0) {
    efHTML += '<div class="glass-bright rounded-xl p-4 text-gray-500 text-sm col-span-2">Collecting throughput data...</div>';
  }
  efHTML += '</div></div>';

  el.innerHTML = costHTML + cqHTML + bnHTML + hdHTML + qsHTML + alHTML + efHTML;
}

// ── Proactive Alerts & Ops Tab ──
async function loadProactive() {
  const el = document.getElementById('page-proactive');
  if (!proactiveData) el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading proactive alerts...</div>';
  try {
    const r = await fetch('/api/proactive');
    proactiveData = await r.json();
  } catch(e) {
    proactiveData = { error: e.message };
  }
  renderProactive();
}

function renderProactive() {
  const el = document.getElementById('page-proactive');
  const d = proactiveData;
  if (!d || d.error) {
    el.innerHTML = `<div class="glass rounded-2xl p-6"><h3 class="text-lg font-semibold text-white">Proactive Alerts</h3><p class="text-red-400 text-sm mt-2">${d?.error||'Failed to load'}</p></div>`;
    return;
  }

  // ── 1. Predictive Alerts ──
  const pa = d.predictive_alerts || [];
  let paHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center justify-between mb-4">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-red-500 to-orange-600 flex items-center justify-center">
          <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
        </div>
        <div>
          <h3 class="text-lg font-semibold text-white">Predictive Alerts</h3>
          <p class="text-[11px] text-gray-500">Trend-based predictions — issues before they happen</p>
        </div>
      </div>
      <button onclick="loadProactive()" class="text-xs px-3 py-1.5 rounded-lg bg-white/[0.05] text-gray-400 border border-gray-700 hover:bg-white/10 transition-colors">Refresh</button>
    </div>`;

  if (pa.length === 0) {
    paHTML += '<div class="glass-bright rounded-lg p-4 text-center"><span class="text-emerald-400 text-sm font-medium">No predicted issues — all trends look healthy</span></div>';
  } else {
    paHTML += '<div class="space-y-2">';
    for (const a of pa) {
      const isCrit = a.severity === 'critical';
      paHTML += `<div class="glass-bright rounded-lg p-3 ${isCrit?'border-l-2 border-red-500':'border-l-2 border-amber-500'}">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <span class="w-2 h-2 rounded-full ${isCrit?'bg-red-500 status-offline':'bg-amber-500'}"></span>
            <div>
              <span class="text-xs font-medium text-white">${a.metric}</span>
              <span class="text-[10px] text-gray-500 ml-2">${a.server}</span>
            </div>
          </div>
          <div class="flex items-center gap-3">
            <span class="text-[10px] text-gray-400">Current: <span class="text-white font-medium">${a.current}</span></span>
            <span class="text-[10px] text-gray-400">Threshold: <span class="text-white font-medium">${a.threshold}</span></span>
            ${a.eta_hours > 0 ? `<span class="text-[10px] px-2 py-0.5 rounded-full ${isCrit?'bg-red-500/20 text-red-400':'bg-amber-500/20 text-amber-400'}">ETA: ${a.eta_hours < 1 ? (a.eta_hours*60).toFixed(0)+'m' : a.eta_hours.toFixed(1)+'h'}</span>` : ''}
          </div>
        </div>
        <div class="text-[11px] text-gray-400 mt-1.5 ml-5">${a.message}</div>
      </div>`;
    }
    paHTML += '</div>';
  }
  paHTML += '</div>';

  // ── 2. Error Rate Tracker ──
  const er = d.error_rates || {};
  let erHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center gap-3 mb-4">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-rose-500 to-pink-600 flex items-center justify-center">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636"/></svg>
      </div>
      <div>
        <h3 class="text-lg font-semibold text-white">Error Rate Tracker</h3>
        <p class="text-[11px] text-gray-500">vLLM request failures and error trends</p>
      </div>
    </div>
    <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">`;

  for (const [sk, e] of Object.entries(er)) {
    const statusColor = e.status === 'critical' ? 'text-red-400' : e.status === 'warning' ? 'text-amber-400' : 'text-emerald-400';
    const statusBg = e.status === 'critical' ? 'bg-red-500/20' : e.status === 'warning' ? 'bg-amber-500/20' : 'bg-emerald-500/20';
    erHTML += `<div class="glass-bright rounded-xl p-4">
      <div class="flex items-center justify-between mb-3">
        <span class="text-sm font-semibold text-white">${e.server_name}</span>
        <span class="text-[10px] px-2 py-0.5 rounded-full ${statusBg} ${statusColor}">${e.status}</span>
      </div>
      <div class="grid grid-cols-3 gap-2 mb-3">
        <div class="bg-white/[0.02] rounded-lg p-2 text-center">
          <div class="text-[9px] text-gray-500 uppercase">Window Errors</div>
          <div class="text-base font-bold ${e.window_errors>0?'text-red-400':'text-emerald-400'}">${e.window_errors}</div>
        </div>
        <div class="bg-white/[0.02] rounded-lg p-2 text-center">
          <div class="text-[9px] text-gray-500 uppercase">Window Rate</div>
          <div class="text-base font-bold ${statusColor}">${e.window_rate}%</div>
        </div>
        <div class="bg-white/[0.02] rounded-lg p-2 text-center">
          <div class="text-[9px] text-gray-500 uppercase">Total Requests</div>
          <div class="text-base font-bold text-white">${fmt(e.total_requests)}</div>
        </div>
      </div>
      <div class="text-[10px] text-gray-500 mb-1">Error count trend</div>
      ${sparklineSVG(e.sparkline || [], 200, 25, e.status === 'healthy' ? '#22c55e' : '#ef4444')}
    </div>`;
  }
  if (Object.keys(er).length === 0) {
    erHTML += '<div class="glass-bright rounded-xl p-4 text-gray-500 text-sm col-span-2">Collecting error data...</div>';
  }
  erHTML += '</div></div>';

  // ── 3. Process Health Monitor ──
  const ph = d.process_health || {};
  let phHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center gap-3 mb-4">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-violet-600 flex items-center justify-center">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"/></svg>
      </div>
      <div>
        <h3 class="text-lg font-semibold text-white">Process Health Monitor</h3>
        <p class="text-[11px] text-gray-500">Zombie processes, memory leaks, worker counts (RTX 5090)</p>
      </div>
    </div>`;

  if (ph.samples) {
    const zombieColor = ph.current_zombies >= 50 ? 'text-red-400' : ph.current_zombies >= 20 ? 'text-amber-400' : 'text-emerald-400';
    phHTML += `<div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Zombies</div>
        <div class="text-2xl font-bold ${zombieColor}">${ph.current_zombies}</div>
        <div class="text-[9px] text-gray-500">defunct processes</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Gunicorn RSS</div>
        <div class="text-2xl font-bold ${ph.rss_leak_detected?'text-red-400':'text-white'}">${(ph.gunicorn_rss_mb/1024).toFixed(1)}G</div>
        <div class="text-[9px] ${ph.rss_leak_detected?'text-red-400':'text-gray-500'}">${ph.rss_leak_detected?'LEAK DETECTED':'stable'}</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Workers</div>
        <div class="text-2xl font-bold text-white">${ph.gunicorn_workers}</div>
        <div class="text-[9px] text-gray-500">gunicorn</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Active Bots</div>
        <div class="text-2xl font-bold ${ph.active_bots>0?'text-cyan-400':'text-gray-500'}">${ph.active_bots}</div>
        <div class="text-[9px] text-gray-500">call processes</div>
      </div>
    </div>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
      <div class="glass-bright rounded-lg p-3">
        <div class="text-[10px] text-gray-500 mb-1">Zombie Trend</div>
        ${sparklineSVG(ph.zombie_trend || [], 140, 30, ph.current_zombies >= 50 ? '#ef4444' : '#f59e0b')}
      </div>
      <div class="glass-bright rounded-lg p-3">
        <div class="text-[10px] text-gray-500 mb-1">Memory (RSS) Trend</div>
        ${sparklineSVG(ph.rss_trend || [], 140, 30, ph.rss_leak_detected ? '#ef4444' : '#8b5cf6')}
      </div>
      <div class="glass-bright rounded-lg p-3">
        <div class="text-[10px] text-gray-500 mb-1">Worker Count</div>
        ${sparklineSVG(ph.worker_trend || [], 140, 30, '#22c55e')}
      </div>
      <div class="glass-bright rounded-lg p-3">
        <div class="text-[10px] text-gray-500 mb-1">Active Bots</div>
        ${sparklineSVG(ph.bot_trend || [], 140, 30, '#06b6d4')}
      </div>
    </div>`;
  } else {
    phHTML += '<p class="text-gray-500 text-sm">Visit the Agent tab first to start collecting process health data.</p>';
  }
  phHTML += '</div>';

  // ── 4. Service Uptime ──
  const su = d.service_uptime || {};
  let suHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center gap-3 mb-4">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
      </div>
      <div>
        <h3 class="text-lg font-semibold text-white">Service Uptime</h3>
        <p class="text-[11px] text-gray-500">Availability tracking and incident history</p>
      </div>
    </div>`;

  if (Object.keys(su).length > 0) {
    suHTML += '<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">';
    for (const [name, s] of Object.entries(su)) {
      const ok = s.status === 'active';
      const uptimeColor = s.uptime_pct >= 99.9 ? 'text-emerald-400' : s.uptime_pct >= 99 ? 'text-amber-400' : 'text-red-400';
      suHTML += `<div class="glass-bright rounded-xl p-4">
        <div class="flex items-center justify-between mb-2">
          <div class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full ${ok?'bg-emerald-500':'bg-red-500'}"></span>
            <span class="text-xs font-medium text-white">${name.replace('agentv2-','')}</span>
          </div>
          <span class="text-[10px] px-2 py-0.5 rounded-full ${ok?'bg-emerald-500/20 text-emerald-400':'bg-red-500/20 text-red-400'}">${ok?'UP':'DOWN'}</span>
        </div>
        <div class="flex items-baseline gap-2 mb-2">
          <span class="text-xl font-bold ${uptimeColor}">${s.uptime_pct}%</span>
          <span class="text-[10px] text-gray-500">uptime</span>
        </div>
        <div class="text-[10px] text-gray-500">Last change: ${s.last_change_ago}</div>
        ${s.incidents && s.incidents.length > 0 ? `<div class="mt-2 space-y-1">${s.incidents.slice(-3).map(i => `<div class="text-[10px] text-gray-400 flex gap-2"><span class="${i.to==='active'?'text-emerald-400':'text-red-400'}">${i.from} → ${i.to}</span><span class="text-gray-500">${i.time_ago}</span></div>`).join('')}</div>` : ''}
      </div>`;
    }
    suHTML += '</div>';
  } else {
    suHTML += '<p class="text-gray-500 text-sm">Visit the Agent tab first to start tracking service uptime.</p>';
  }
  suHTML += '</div>';

  // ── 5. Inter-Cluster Latency ──
  const cl = d.cluster_latency || {};
  let clHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center gap-3 mb-4">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
      </div>
      <div>
        <h3 class="text-lg font-semibold text-white">Inter-Cluster Latency</h3>
        <p class="text-[11px] text-gray-500">RTX 5090 → H200 round-trip time (LLM/TTS call path)</p>
      </div>
    </div>`;

  if (cl.current_rtt_ms) {
    const latColor = cl.status === 'critical' ? 'text-red-400' : cl.status === 'warning' ? 'text-amber-400' : 'text-emerald-400';
    clHTML += `<div class="grid grid-cols-4 gap-3 mb-4">
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Current RTT</div>
        <div class="text-xl font-bold ${latColor}">${cl.current_rtt_ms}ms</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Average</div>
        <div class="text-xl font-bold text-white">${cl.avg_rtt_ms}ms</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Min</div>
        <div class="text-xl font-bold text-emerald-400">${cl.min_rtt_ms}ms</div>
      </div>
      <div class="glass-bright rounded-lg p-3 text-center">
        <div class="text-[9px] text-gray-500 uppercase mb-1">Max</div>
        <div class="text-xl font-bold ${cl.max_rtt_ms>500?'text-red-400':'text-amber-400'}">${cl.max_rtt_ms}ms</div>
      </div>
    </div>
    <div class="glass-bright rounded-lg p-3">
      <div class="text-[10px] text-gray-500 mb-1">Latency Over Time</div>
      ${sparklineSVG(cl.trend || [], 400, 40, cl.status === 'healthy' ? '#22c55e' : '#f59e0b')}
    </div>`;
  } else {
    clHTML += '<p class="text-gray-500 text-sm">Measuring inter-cluster latency on next refresh...</p>';
  }
  clHTML += '</div>';

  // ── 6. Thermal Throttling ──
  const tt = d.thermal_throttling || {};
  let ttHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center gap-3 mb-4">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z"/></svg>
      </div>
      <div>
        <h3 class="text-lg font-semibold text-white">GPU Thermal Throttling</h3>
        <p class="text-[11px] text-gray-500">Detects when GPUs reduce performance due to heat</p>
      </div>
    </div>`;

  if (Object.keys(tt).length === 0) {
    ttHTML += '<div class="glass-bright rounded-lg p-4 text-center"><span class="text-emerald-400 text-sm font-medium">No thermal throttling detected</span></div>';
  } else {
    for (const [sk, t] of Object.entries(tt)) {
      ttHTML += `<div class="glass-bright rounded-xl p-4 mb-3">
        <div class="flex items-center justify-between mb-3">
          <span class="text-sm font-semibold text-white">${t.server_name}</span>
          ${t.any_throttling ? '<span class="text-[10px] px-2 py-0.5 rounded-full bg-red-500/20 text-red-400">THROTTLING</span>' : '<span class="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400">HIGH TEMP</span>'}
        </div>
        <div class="grid grid-cols-4 md:grid-cols-8 gap-2">
          ${t.gpus.map(g => `<div class="rounded-lg p-2 text-center ${g.likely_throttling?'bg-red-500/10 border border-red-500/20':'bg-amber-500/5 border border-amber-500/10'}">
            <div class="text-[9px] text-gray-500">GPU ${g.gpu}</div>
            <div class="text-sm font-bold ${g.current_temp>=83?'text-red-400':g.current_temp>=75?'text-amber-400':'text-emerald-400'}">${g.current_temp}°C</div>
            <div class="text-[9px] text-gray-500">${g.power_drop_pct > 0 ? '-'+g.power_drop_pct+'% pwr' : 'OK'}</div>
          </div>`).join('')}
        </div>
      </div>`;
    }
  }
  ttHTML += '</div>';

  // ── 7. Call Volume Forecast ──
  const cf = d.call_forecast || {};
  let cfHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center gap-3 mb-4">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-sky-600 flex items-center justify-center">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"/></svg>
      </div>
      <div>
        <h3 class="text-lg font-semibold text-white">Call Volume Forecast</h3>
        <p class="text-[11px] text-gray-500">${cf.day || ''} — hourly call pattern (builds over time)</p>
      </div>
    </div>`;

  const hours = cf.hours || [];
  if (hours.length > 0 && hours.some(h => h.avg_calls > 0)) {
    const maxAvg = Math.max(...hours.map(h => h.avg_calls), 1);
    const peakHour = cf.peak_hour;
    if (peakHour && peakHour.avg_calls > 0) {
      cfHTML += `<div class="glass-bright rounded-lg p-3 mb-4 border-l-2 border-cyan-500">
        <div class="text-[10px] text-gray-500 uppercase mb-1">Expected Peak Hour</div>
        <div class="text-lg font-bold text-cyan-400">${peakHour.label} UTC</div>
        <div class="text-[11px] text-gray-400">Avg ${peakHour.avg_calls} calls &middot; Peak ${peakHour.peak_calls}</div>
      </div>`;
    }
    cfHTML += `<div class="glass-bright rounded-xl p-4 overflow-x-auto">
      <div class="flex items-end gap-1" style="height:100px">
        ${hours.map(h => {
          const pct = maxAvg > 0 ? (h.avg_calls / maxAvg * 100) : 0;
          const color = h.is_current ? '#06b6d4' : (h.avg_calls / maxAvg > 0.8 ? '#f59e0b' : '#22c55e');
          return `<div class="flex flex-col items-center flex-1 gap-0.5">
            <div class="w-full rounded-t" style="height:${Math.max(pct, 2)}%;background:${color};min-height:2px" title="${h.label}: avg ${h.avg_calls} calls"></div>
            <span class="text-[8px] text-gray-500 ${h.is_current?'text-cyan-400 font-bold':''}">${h.hour % 3 === 0 ? h.label.replace(':00','') : ''}</span>
          </div>`;
        }).join('')}
      </div>
    </div>`;
  } else {
    cfHTML += '<p class="text-gray-500 text-sm">Building call volume patterns... Data accumulates as calls are processed.</p>';
  }
  cfHTML += '</div>';

  // ── 8. Disk & Log Growth ──
  const dt = d.disk_trends || {};
  let dtHTML = `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center gap-3 mb-4">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-gray-500 to-slate-600 flex items-center justify-center">
        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4"/></svg>
      </div>
      <div>
        <h3 class="text-lg font-semibold text-white">Disk & Log Growth</h3>
        <p class="text-[11px] text-gray-500">Storage usage trends and fill-time estimates</p>
      </div>
    </div>
    <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">`;

  for (const [sk, disk] of Object.entries(dt)) {
    const diskColor = disk.status === 'critical' ? 'text-red-400' : disk.status === 'warning' ? 'text-amber-400' : 'text-emerald-400';
    dtHTML += `<div class="glass-bright rounded-xl p-4">
      <div class="flex items-center justify-between mb-3">
        <span class="text-sm font-semibold text-white">${disk.server_name}</span>
        <span class="text-[10px] px-2 py-0.5 rounded-full ${disk.status==='critical'?'bg-red-500/20 text-red-400':disk.status==='warning'?'bg-amber-500/20 text-amber-400':'bg-emerald-500/20 text-emerald-400'}">${disk.status}</span>
      </div>
      <div class="flex items-baseline gap-2 mb-2">
        <span class="text-2xl font-bold ${diskColor}">${disk.current_pct}%</span>
        <span class="text-[10px] text-gray-500">used</span>
      </div>
      ${barHTML(disk.current_pct, 100, disk.current_pct >= 90 ? '#ef4444' : disk.current_pct >= 80 ? '#f59e0b' : '#22c55e')}
      <div class="grid grid-cols-2 gap-2 mt-3">
        <div class="text-[10px]"><span class="text-gray-500">Growth:</span> <span class="text-gray-300">${disk.growth_rate_pct_per_day}%/day</span></div>
        <div class="text-[10px]"><span class="text-gray-500">Days until full:</span> <span class="${disk.days_until_full && disk.days_until_full < 7 ? 'text-red-400' : 'text-gray-300'}">${disk.days_until_full ? disk.days_until_full+'d' : '∞'}</span></div>
      </div>
      ${disk.trend && disk.trend.length > 1 ? `<div class="mt-2"><div class="text-[10px] text-gray-500 mb-1">Usage trend</div>${sparklineSVG(disk.trend, 200, 25, disk.current_pct >= 80 ? '#f59e0b' : '#22c55e')}</div>` : ''}
    </div>`;
  }
  if (Object.keys(dt).length === 0) {
    dtHTML += '<div class="glass-bright rounded-xl p-4 text-gray-500 text-sm col-span-2">Disk tracking starts after a few minutes of data collection.</div>';
  }
  dtHTML += '</div></div>';

  el.innerHTML = paHTML + erHTML + phHTML + suHTML + clHTML + ttHTML + cfHTML + dtHTML;
}

// ── Capacity Planner Tab ──
async function loadCapacity() {
  try {
    const r = await fetch('/api/capacity');
    capacityData = await r.json();
    renderCapacity();
  } catch(e) { console.error('Capacity load error:', e); }
}

function renderCapacity() {
  const el = document.getElementById('page-capacity');
  if (!capacityData) { el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading capacity data...</div>'; return; }
  const d = capacityData;

  // ── 1. What-If Simulator ──
  let wiHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">What-If Simulator</h3>';
  wiHTML += '<p class="text-xs text-gray-500 mb-4">Projected metrics at different concurrent call levels</p>';
  wiHTML += '<div class="overflow-x-auto"><table class="w-full text-xs">';
  wiHTML += '<thead><tr class="text-gray-500 border-b border-gray-800">';
  wiHTML += '<th class="text-left py-2 px-2">Calls</th><th class="text-right py-2 px-2">GPU Util</th><th class="text-right py-2 px-2">TTFT (ms)</th><th class="text-right py-2 px-2">E2E (s)</th><th class="text-right py-2 px-2">KV Cache</th><th class="text-right py-2 px-2">GPUs Needed</th><th class="text-center py-2 px-2">Quality</th><th class="text-center py-2 px-2">Feasible</th>';
  wiHTML += '</tr></thead><tbody>';
  for (const w of (d.what_if || [])) {
    const utilC = w.projected_gpu_util > 85 ? 'text-red-400' : w.projected_gpu_util > 60 ? 'text-amber-400' : 'text-emerald-400';
    const qC = w.quality === 'good' ? 'text-emerald-400' : w.quality === 'degraded' ? 'text-amber-400' : 'text-red-400';
    const fC = w.feasible ? 'text-emerald-400' : 'text-red-400';
    const ttftC = w.projected_ttft_ms > 600 ? 'text-red-400' : w.projected_ttft_ms > 300 ? 'text-amber-400' : 'text-emerald-400';
    const highlight = w.target_calls === Math.round(d.roi?.current_avg_calls || 0) ? 'bg-emerald-500/10' : '';
    wiHTML += `<tr class="border-b border-gray-800/50 ${highlight}">`;
    wiHTML += `<td class="py-2 px-2 text-white font-medium">${w.target_calls}</td>`;
    wiHTML += `<td class="py-2 px-2 text-right ${utilC}">${w.projected_gpu_util}%</td>`;
    wiHTML += `<td class="py-2 px-2 text-right ${ttftC}">${w.projected_ttft_ms.toFixed(0)}</td>`;
    wiHTML += `<td class="py-2 px-2 text-right text-gray-300">${w.projected_e2e_s.toFixed(2)}</td>`;
    wiHTML += `<td class="py-2 px-2 text-right text-gray-300">${w.projected_kv_cache}%</td>`;
    wiHTML += `<td class="py-2 px-2 text-right text-gray-300">${w.gpus_needed_for_80pct}</td>`;
    wiHTML += `<td class="py-2 px-2 text-center ${qC}">${w.quality}</td>`;
    wiHTML += `<td class="py-2 px-2 text-center ${fC}">${w.feasible ? 'Yes' : 'No'}</td>`;
    wiHTML += '</tr>';
  }
  wiHTML += '</tbody></table></div></div>';

  // ── 2. GPU ROI Calculator ──
  const roi = d.roi || {};
  let roiHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">GPU ROI Calculator</h3>';
  roiHTML += '<p class="text-xs text-gray-500 mb-4">Cost efficiency and break-even analysis</p>';
  roiHTML += '<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">';
  roiHTML += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500 mb-1">Monthly Cost</div><div class="text-lg font-bold text-white">$${(roi.monthly_cost||0).toLocaleString()}</div></div>`;
  roiHTML += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500 mb-1">Cost / Call-Min</div><div class="text-lg font-bold ${(roi.current_cost_per_call_min||0) > 0.10 ? 'text-red-400' : 'text-emerald-400'}">$${(roi.current_cost_per_call_min||0).toFixed(4)}</div></div>`;
  roiHTML += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500 mb-1">Idle Waste / Month</div><div class="text-lg font-bold text-amber-400">$${(roi.idle_waste_monthly||0).toLocaleString()}</div></div>`;
  roiHTML += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500 mb-1">Effective Utilization</div><div class="text-lg font-bold ${(roi.effective_util_pct||0) > 50 ? 'text-emerald-400' : 'text-amber-400'}">${roi.effective_util_pct||0}%</div></div>`;
  roiHTML += '</div>';
  // Break-even table
  roiHTML += '<div class="text-xs text-gray-400 mb-2 font-medium">Break-Even Analysis</div>';
  roiHTML += '<div class="overflow-x-auto"><table class="w-full text-xs">';
  roiHTML += '<thead><tr class="text-gray-500 border-b border-gray-800"><th class="text-left py-2 px-2">Target $/call-min</th><th class="text-right py-2 px-2">Min Concurrent Calls Needed</th></tr></thead><tbody>';
  for (const b of (roi.breakeven || [])) {
    const met = (roi.current_avg_calls || 0) >= b.min_concurrent_calls;
    roiHTML += `<tr class="border-b border-gray-800/50"><td class="py-1 px-2 text-white">$${b.target_cost_per_min}</td><td class="py-1 px-2 text-right ${met ? 'text-emerald-400' : 'text-gray-400'}">${b.min_concurrent_calls} ${met ? '&#10003;' : ''}</td></tr>`;
  }
  roiHTML += '</tbody></table></div></div>';

  // ── 3. Scale Recommendation ──
  const sr = d.scale_recommendation || {};
  const actionColor = sr.action === 'scale_up' ? 'text-red-400' : sr.action === 'scale_down' ? 'text-amber-400' : 'text-emerald-400';
  const actionBg = sr.action === 'scale_up' ? 'bg-red-500/10 border-red-500/30' : sr.action === 'scale_down' ? 'bg-amber-500/10 border-amber-500/30' : 'bg-emerald-500/10 border-emerald-500/30';
  let srHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Scale Recommendation</h3>';
  srHTML += '<p class="text-xs text-gray-500 mb-4">Based on P95/P99 utilization patterns</p>';
  srHTML += `<div class="rounded-lg border p-4 mb-4 ${actionBg}">`;
  srHTML += `<div class="text-sm font-semibold ${actionColor} mb-1">${sr.action === 'scale_up' ? 'SCALE UP' : sr.action === 'scale_down' ? 'SCALE DOWN' : 'OPTIMAL'}</div>`;
  srHTML += `<div class="text-xs text-gray-300">${sr.detail || ''}</div>`;
  srHTML += '</div>';
  srHTML += '<div class="grid grid-cols-2 md:grid-cols-5 gap-3">';
  srHTML += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500">Current GPUs</div><div class="text-white font-bold">${sr.current_gpus||0}</div></div>`;
  srHTML += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500">P95 Util</div><div class="text-white font-bold">${sr.p95_util||0}%</div></div>`;
  srHTML += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500">P99 Util</div><div class="text-white font-bold">${sr.p99_util||0}%</div></div>`;
  srHTML += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500">Min GPUs (P95)</div><div class="text-white font-bold">${sr.min_gpus_for_p95||0}</div></div>`;
  srHTML += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500">Savings if Downscale</div><div class="text-emerald-400 font-bold">$${(sr.monthly_savings_if_downscale||0).toLocaleString()}/mo</div></div>`;
  srHTML += '</div></div>';

  // ── 4. Cloud Cost Comparison ──
  let ccHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Cloud Cost Comparison</h3>';
  ccHTML += `<p class="text-xs text-gray-500 mb-4">Your ${SERVERS.reduce((a,s)=>a,0) || 16} GPUs at $${(roi.monthly_cost||25000).toLocaleString()}/mo vs cloud providers (scaled to ${sr.current_gpus||16} GPUs)</p>`;
  ccHTML += '<div class="overflow-x-auto"><table class="w-full text-xs">';
  ccHTML += '<thead><tr class="text-gray-500 border-b border-gray-800">';
  ccHTML += '<th class="text-left py-2 px-2">Provider</th><th class="text-left py-2 px-2">GPU Config</th><th class="text-right py-2 px-2">$/hr (base)</th><th class="text-right py-2 px-2">$/mo (scaled)</th><th class="text-right py-2 px-2">You Save</th><th class="text-left py-2 px-2">Note</th>';
  ccHTML += '</tr></thead><tbody>';
  // Your row first
  ccHTML += `<tr class="border-b border-gray-800/50 bg-emerald-500/10"><td class="py-2 px-2 text-emerald-400 font-semibold">Your Cluster</td><td class="py-2 px-2 text-gray-300">8x H200 + 8x RTX 5090</td><td class="py-2 px-2 text-right text-white">$${(roi.hourly_cost||0).toFixed(2)}</td><td class="py-2 px-2 text-right text-white font-bold">$${(roi.monthly_cost||0).toLocaleString()}</td><td class="py-2 px-2 text-right text-emerald-400">-</td><td class="py-2 px-2 text-gray-500">Dedicated</td></tr>`;
  for (const cp of (d.cloud_comparison || [])) {
    const saveC = cp.savings_vs_you > 0 ? 'text-emerald-400' : 'text-red-400';
    const saveTxt = cp.savings_vs_you > 0 ? `+$${cp.savings_vs_you.toLocaleString()}` : `-$${Math.abs(cp.savings_vs_you).toLocaleString()}`;
    ccHTML += `<tr class="border-b border-gray-800/50">`;
    ccHTML += `<td class="py-2 px-2 text-white">${cp.provider}</td>`;
    ccHTML += `<td class="py-2 px-2 text-gray-400">${cp.gpu}</td>`;
    ccHTML += `<td class="py-2 px-2 text-right text-gray-300">$${cp.hourly.toFixed(2)}</td>`;
    ccHTML += `<td class="py-2 px-2 text-right text-gray-300">$${cp.scaled_monthly.toLocaleString()}</td>`;
    ccHTML += `<td class="py-2 px-2 text-right ${saveC}">${saveTxt}</td>`;
    ccHTML += `<td class="py-2 px-2 text-gray-500">${cp.note}</td>`;
    ccHTML += '</tr>';
  }
  ccHTML += '</tbody></table></div></div>';

  el.innerHTML = wiHTML + roiHTML + srHTML + ccHTML;
}

// ── Render All ──
function renderAll() {
  if (activeTab === 'overview') renderOverview();
  else if (activeTab === 'traffic') renderTraffic();
  else if (activeTab === 'history') renderHistory();
  else if (activeTab === 'software') renderSoftware();
  else if (activeTab === 'daily') renderDaily();
  else if (activeTab === 'agent') renderAgent();
  else if (activeTab === 'analytics') renderAnalytics();
  else if (activeTab === 'proactive') renderProactive();
  else if (activeTab === 'capacity') renderCapacity();
  else if (activeTab === 'quality') renderQuality();
  else if (activeTab === 'network') renderNetwork();
  else if (activeTab === 'executive') renderExecutive();
  else if (activeTab === 'livecalls') renderLiveCalls();
  else if (activeTab === 'modelcompare') renderModelCompare();
  else if (activeTab === 'power') renderPower();
  else if (activeTab === 'incidents') renderIncidents();
  else if (activeTab === 'alertsconfig') renderAlertsConfig();
  else if (activeTab === 'sla') renderSLA();
  else if (activeTab === 'anomalies') renderAnomalies();
}

// ── Live Call Tracker Tab ──
async function loadLiveCalls() {
  try { const r = await fetch('/api/livecalls'); livecallsData = await r.json(); renderLiveCalls(); } catch(e) { console.error(e); }
}
function renderLiveCalls() {
  const el = document.getElementById('page-livecalls');
  if (!livecallsData) { el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading...</div>'; return; }
  const d = livecallsData;
  const s = d.summary || {};

  // Active calls cards
  let html = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Active Calls Now</h3>';
  html += '<p class="text-xs text-gray-500 mb-4">Real-time view of concurrent calls across servers</p>';
  html += '<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">';
  html += `<div class="glass-bright rounded-lg p-4 text-center"><div class="text-4xl font-bold text-emerald-400">${s.total_active_now||0}</div><div class="text-[10px] text-gray-500 mt-1">Active Now</div></div>`;
  html += `<div class="glass-bright rounded-lg p-4 text-center"><div class="text-4xl font-bold text-cyan-400">${s.peak_concurrent||0}</div><div class="text-[10px] text-gray-500 mt-1">Peak</div></div>`;
  html += `<div class="glass-bright rounded-lg p-4 text-center"><div class="text-4xl font-bold text-gray-300">${s.avg_concurrent||0}</div><div class="text-[10px] text-gray-500 mt-1">Avg</div></div>`;
  html += `<div class="glass-bright rounded-lg p-4 text-center"><div class="text-4xl font-bold text-amber-400">${s.zero_call_pct||0}%</div><div class="text-[10px] text-gray-500 mt-1">Idle Time</div></div>`;
  html += '</div>';

  // Per-server active calls
  for (const c of (d.calls||[])) {
    const utilC = c.current_gpu_util > 80 ? 'text-red-400' : c.current_gpu_util > 50 ? 'text-amber-400' : 'text-emerald-400';
    html += `<div class="glass-bright rounded-lg p-3 mb-2 flex items-center justify-between">`;
    html += `<div><span class="text-white text-sm font-medium">${c.server_name}</span>`;
    html += `<div class="text-[10px] text-gray-500">Running: ${c.requests_running} | Waiting: ${c.requests_waiting} | Connections: ${c.active_connections}</div></div>`;
    html += `<div class="flex gap-4 text-[10px]">`;
    html += `<div class="text-center"><div class="text-lg font-bold text-white">${c.active_calls}</div><div class="text-gray-500">Calls</div></div>`;
    html += `<div class="text-center"><div class="text-lg font-bold ${utilC}">${c.current_gpu_util}%</div><div class="text-gray-500">GPU</div></div>`;
    html += `<div class="text-center"><div class="text-lg font-bold text-cyan-400">${c.current_ttft_ms}ms</div><div class="text-gray-500">TTFT</div></div>`;
    html += `<div class="text-center"><div class="text-lg font-bold text-gray-300">${c.current_e2e_s}s</div><div class="text-gray-500">E2E</div></div>`;
    html += `<div class="text-center"><div class="text-lg font-bold text-gray-300">${c.current_kv_cache}%</div><div class="text-gray-500">KV</div></div>`;
    html += '</div></div>';
  }
  html += '</div>';

  // Call timeline
  const tl = s.timeline || [];
  if (tl.length > 0) {
    html += '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Call Timeline</h3>';
    html += '<p class="text-xs text-gray-500 mb-4">Concurrent calls over time</p>';
    const maxC = Math.max(...tl.map(t=>t.calls), 1);
    html += '<div class="space-y-0.5">';
    for (const t of tl.slice(-40)) {
      const time = new Date(t.t * 1000).toLocaleTimeString();
      const barW = t.calls / maxC * 100;
      const barC = t.calls > 5 ? 'bg-red-500/60' : t.calls > 2 ? 'bg-amber-500/60' : 'bg-emerald-500/60';
      html += `<div class="flex items-center gap-1 text-[9px]"><span class="text-gray-600 w-14">${time}</span><span class="text-gray-500 w-4">${t.server[0]}</span>`;
      html += `<div class="flex-1 h-3 rounded bg-gray-800"><div class="${barC} h-3 rounded" style="width:${barW}%"></div></div>`;
      html += `<span class="text-gray-400 w-20 text-right">${t.calls}c ${t.gpu_util.toFixed(0)}%</span></div>`;
    }
    html += '</div></div>';
  }

  // Duration distribution
  const dd = d.duration_distribution || {};
  if (Object.keys(dd).length > 0) {
    html += '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Call Duration Distribution</h3>';
    html += '<div class="grid grid-cols-5 gap-2">';
    const maxDD = Math.max(...Object.values(dd), 1);
    for (const [label, count] of Object.entries(dd)) {
      const pct = count / maxDD * 100;
      html += `<div class="text-center"><div class="h-20 flex items-end justify-center mb-1"><div class="w-8 rounded-t bg-cyan-500/50" style="height:${pct}%"></div></div>`;
      html += `<div class="text-[10px] text-gray-400">${label}</div><div class="text-[10px] text-white font-medium">${count}</div></div>`;
    }
    html += '</div></div>';
  }

  el.innerHTML = html;
}

// ── Model Compare Tab ──
async function loadModelCompare() {
  try { const r = await fetch('/api/modelcompare'); modelcompareData = await r.json(); renderModelCompare(); } catch(e) { console.error(e); }
}
function renderModelCompare() {
  const el = document.getElementById('page-modelcompare');
  if (!modelcompareData) { el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading...</div>'; return; }
  const d = modelcompareData;

  // Active models
  let html = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Active Models</h3>';
  html += '<p class="text-xs text-gray-500 mb-4">Currently loaded vLLM models and their metrics</p>';
  if ((d.models||[]).length === 0) {
    html += '<div class="text-gray-500 text-sm">No vLLM models detected.</div>';
  } else {
    html += '<div class="grid grid-cols-1 md:grid-cols-2 gap-4">';
    for (const m of d.models) {
      html += `<div class="glass-bright rounded-lg p-4">`;
      html += `<div class="flex justify-between items-center mb-2"><span class="text-white text-sm font-medium">${m.model_name}</span><span class="text-[10px] text-gray-500">${m.server_name} :${m.port}</span></div>`;
      html += '<div class="grid grid-cols-3 gap-2 text-[10px]">';
      html += `<div><span class="text-gray-500">TTFT</span><div class="${m.avg_ttft_ms > 300 ? 'text-amber-400' : 'text-emerald-400'} font-medium">${m.avg_ttft_ms}ms</div></div>`;
      html += `<div><span class="text-gray-500">ITL</span><div class="text-gray-300 font-medium">${m.avg_itl_ms}ms</div></div>`;
      html += `<div><span class="text-gray-500">E2E</span><div class="${m.avg_e2e_s > 3 ? 'text-amber-400' : 'text-emerald-400'} font-medium">${m.avg_e2e_s}s</div></div>`;
      html += `<div><span class="text-gray-500">Queue</span><div class="text-gray-300 font-medium">${m.avg_queue_s}s</div></div>`;
      html += `<div><span class="text-gray-500">KV Cache</span><div class="${m.kv_cache_usage > 80 ? 'text-red-400' : 'text-emerald-400'} font-medium">${m.kv_cache_usage}%</div></div>`;
      html += `<div><span class="text-gray-500">Cache Hit</span><div class="text-cyan-400 font-medium">${m.cache_hit_rate}%</div></div>`;
      html += `<div><span class="text-gray-500">Running</span><div class="text-white font-medium">${m.requests_running}</div></div>`;
      html += `<div><span class="text-gray-500">Total Reqs</span><div class="text-white font-medium">${(m.total_requests||0).toLocaleString()}</div></div>`;
      html += '</div></div>';
    }
    html += '</div>';
  }
  html += '</div>';

  // Time-windowed comparison
  for (const tw of (d.time_windows||[])) {
    html += `<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">${tw.server_name} — Performance Over Time</h3>`;
    html += '<p class="text-xs text-gray-500 mb-4">Data split into quarters to detect changes</p>';
    html += '<div class="overflow-x-auto"><table class="w-full text-xs">';
    html += '<thead><tr class="text-gray-500 border-b border-gray-800"><th class="text-left py-2 px-2">Window</th><th class="text-right py-2 px-2">TTFT</th><th class="text-right py-2 px-2">E2E</th><th class="text-right py-2 px-2">GPU Util</th><th class="text-right py-2 px-2">KV Cache</th><th class="text-right py-2 px-2">Avg Calls</th><th class="text-right py-2 px-2">Peak</th><th class="text-right py-2 px-2">Samples</th></tr></thead><tbody>';
    for (const w of (tw.windows||[])) {
      const isLatest = w.label.includes('latest');
      const rowClass = isLatest ? 'bg-emerald-500/10' : '';
      html += `<tr class="border-b border-gray-800/50 ${rowClass}">`;
      html += `<td class="py-2 px-2 text-white font-medium">${w.label}</td>`;
      html += `<td class="py-2 px-2 text-right ${w.avg_ttft_ms > 300 ? 'text-amber-400' : 'text-gray-300'}">${w.avg_ttft_ms}ms</td>`;
      html += `<td class="py-2 px-2 text-right text-gray-300">${w.avg_e2e_s}s</td>`;
      html += `<td class="py-2 px-2 text-right text-gray-300">${w.avg_util}%</td>`;
      html += `<td class="py-2 px-2 text-right text-gray-300">${w.avg_kv}%</td>`;
      html += `<td class="py-2 px-2 text-right text-gray-300">${w.avg_calls}</td>`;
      html += `<td class="py-2 px-2 text-right text-gray-300">${w.peak_calls}</td>`;
      html += `<td class="py-2 px-2 text-right text-gray-500">${w.samples}</td>`;
      html += '</tr>';
    }
    html += '</tbody></table></div></div>';
  }

  // Efficiency comparison
  const comp = d.comparison || {};
  if (Object.keys(comp).length > 0) {
    html += '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Token Efficiency Comparison</h3>';
    html += '<div class="grid grid-cols-1 md:grid-cols-2 gap-4">';
    for (const [sk, c] of Object.entries(comp)) {
      html += `<div class="glass-bright rounded-lg p-4"><div class="text-xs text-white font-medium mb-2">${c.server_name}</div>`;
      html += `<div class="grid grid-cols-3 gap-2 text-[10px]">`;
      html += `<div><span class="text-gray-500">Total Tokens</span><div class="text-white font-medium">${(c.total_tokens_processed||0).toLocaleString()}</div></div>`;
      html += `<div><span class="text-gray-500">Avg Power</span><div class="text-gray-300 font-medium">${c.avg_power_w}W</div></div>`;
      html += `<div><span class="text-gray-500">Tokens/Wh</span><div class="text-cyan-400 font-medium">${c.tokens_per_watt_hour}</div></div>`;
      html += '</div></div>';
    }
    html += '</div></div>';
  }

  el.innerHTML = html;
}

// ── Power Management Tab ──
async function loadPower() {
  try { const r = await fetch('/api/power'); powerData = await r.json(); renderPower(); } catch(e) { console.error(e); }
}
function renderPower() {
  const el = document.getElementById('page-power');
  if (!powerData) { el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading...</div>'; return; }
  const d = powerData;
  const fs = d.fleet_summary || {};

  // Fleet power summary
  let html = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Fleet Power Summary</h3>';
  html += '<div class="grid grid-cols-2 md:grid-cols-4 gap-4">';
  html += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500 mb-1">Current Draw</div><div class="text-lg font-bold text-white">${fs.total_current_kw||0} kW</div></div>`;
  html += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500 mb-1">Monthly kWh</div><div class="text-lg font-bold text-gray-300">${(fs.monthly_kwh||0).toLocaleString()}</div></div>`;
  html += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500 mb-1">Est. Power Cost/mo</div><div class="text-lg font-bold text-amber-400">$${(fs.power_cost_estimate||0).toLocaleString()}</div></div>`;
  html += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500 mb-1">Power % of GPU Cost</div><div class="text-lg font-bold text-gray-300">${fs.power_as_pct_of_cost||0}%</div></div>`;
  html += '</div></div>';

  // Per-server power details
  for (const [sk, ps] of Object.entries(d.per_server || {})) {
    if (ps.no_data) continue;
    html += `<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">${ps.server_name} — Power Profile</h3>`;
    html += '<div class="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">';
    html += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500">Current</div><div class="text-white font-bold">${ps.current_power_w}W</div></div>`;
    html += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500">Peak</div><div class="text-red-400 font-bold">${ps.peak_power_w}W</div></div>`;
    html += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500">Min</div><div class="text-emerald-400 font-bold">${ps.min_power_w}W</div></div>`;
    html += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500">Per GPU</div><div class="text-gray-300 font-bold">${ps.power_per_gpu_w}W</div></div>`;
    html += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500">Tokens/kWh</div><div class="text-cyan-400 font-bold">${(ps.tokens_per_kwh||0).toLocaleString()}</div></div>`;
    html += '</div>';

    // Efficiency curve
    const ec = ps.efficiency_curve || [];
    if (ec.length > 0) {
      html += '<div class="text-xs text-gray-400 mb-2 font-medium">Power vs Utilization Curve</div>';
      const maxPow = Math.max(...ec.map(e=>e.avg_power_w), 1);
      html += '<div class="space-y-1 mb-4">';
      for (const e of ec) {
        const barW = e.avg_power_w / maxPow * 100;
        const isOptimal = ps.optimal_point && e.util_bucket === ps.optimal_point.util_bucket;
        const barC = isOptimal ? 'bg-emerald-500/60' : 'bg-cyan-500/40';
        html += `<div class="flex items-center gap-2 text-[10px] ${isOptimal ? 'border-l-2 border-emerald-400 pl-1' : ''}">`;
        html += `<span class="text-gray-500 w-14">${e.util_bucket}-${e.util_bucket+9}%</span>`;
        html += `<div class="flex-1 h-4 rounded bg-gray-800"><div class="${barC} h-4 rounded" style="width:${barW}%"></div></div>`;
        html += `<span class="text-gray-400 w-32 text-right">${e.avg_power_w}W | ${e.avg_temp_c}C | eff:${e.efficiency}</span>`;
        if (isOptimal) html += '<span class="text-emerald-400 text-[9px] font-bold">OPTIMAL</span>';
        html += '</div>';
      }
      html += '</div>';
    }

    // Per-GPU power
    const pg = ps.per_gpu || [];
    if (pg.length > 0) {
      html += '<div class="text-xs text-gray-400 mb-2 font-medium">Per-GPU Power Distribution</div>';
      html += '<div class="grid grid-cols-4 md:grid-cols-8 gap-2">';
      for (const g of pg) {
        const pc = g.avg_power_w > 300 ? 'text-red-400' : g.avg_power_w > 200 ? 'text-amber-400' : 'text-emerald-400';
        html += `<div class="glass-bright rounded-lg p-2 text-center text-[10px]"><div class="text-gray-500">GPU ${g.gpu}</div><div class="${pc} font-bold">${g.avg_power_w}W</div><div class="text-gray-500">${g.avg_temp_c}C</div></div>`;
      }
      html += '</div>';
    }
    html += '</div>';
  }

  el.innerHTML = html;
}

// ── Incidents Tab ──
async function loadIncidents() {
  try { const r = await fetch('/api/incidents'); incidentsData = await r.json(); renderIncidents(); } catch(e) { console.error(e); }
}
function renderIncidents() {
  const el = document.getElementById('page-incidents');
  if (!incidentsData) { el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading...</div>'; return; }
  const d = incidentsData;
  const s = d.summary || {};

  let html = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Incident Summary</h3>';
  html += '<p class="text-xs text-gray-500 mb-4">Service events, error spikes, and anomalies</p>';
  html += '<div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">';
  html += `<div class="glass-bright rounded-lg p-3 text-center"><div class="text-2xl font-bold text-white">${s.total||0}</div><div class="text-[10px] text-gray-500">Total</div></div>`;
  html += `<div class="glass-bright rounded-lg p-3 text-center"><div class="text-2xl font-bold text-amber-400">${s.last_hour||0}</div><div class="text-[10px] text-gray-500">Last Hour</div></div>`;
  html += `<div class="glass-bright rounded-lg p-3 text-center"><div class="text-2xl font-bold text-gray-300">${s.last_24h||0}</div><div class="text-[10px] text-gray-500">Last 24h</div></div>`;
  html += `<div class="glass-bright rounded-lg p-3 text-center"><div class="text-2xl font-bold text-red-400">${s.critical_count||0}</div><div class="text-[10px] text-gray-500">Critical</div></div>`;
  html += `<div class="glass-bright rounded-lg p-3 text-center"><div class="text-2xl font-bold text-amber-400">${s.warning_count||0}</div><div class="text-[10px] text-gray-500">Warnings</div></div>`;
  html += '</div>';

  // By type breakdown
  const bt = s.by_type || {};
  html += '<div class="grid grid-cols-5 gap-2 mb-4">';
  for (const [type, count] of Object.entries(bt)) {
    const icon = type === 'restart' ? '🔄' : type === 'error_spike' ? '⚠' : type === 'thermal' ? '🌡' : type === 'fetch_error' ? '🔌' : '📡';
    html += `<div class="glass-bright rounded-lg p-2 text-center text-[10px]"><div class="text-sm">${icon}</div><div class="text-white font-medium">${count}</div><div class="text-gray-500">${type.replace('_',' ')}</div></div>`;
  }
  html += '</div></div>';

  // Timeline
  html += '<div class="glass rounded-xl p-6"><h3 class="text-white font-semibold mb-4">Timeline</h3>';
  html += '<div class="space-y-2">';
  for (const inc of (d.incidents || []).slice(0, 50)) {
    const time = new Date(inc.t * 1000).toLocaleString();
    const sevC = inc.severity === 'critical' ? 'border-red-500 bg-red-500/10' : inc.severity === 'warning' ? 'border-amber-500 bg-amber-500/10' : 'border-blue-500 bg-blue-500/10';
    const sevT = inc.severity === 'critical' ? 'text-red-400' : inc.severity === 'warning' ? 'text-amber-400' : 'text-blue-400';
    html += `<div class="border-l-2 ${sevC} rounded-r-lg p-3">`;
    html += `<div class="flex justify-between items-start"><div><span class="${sevT} text-xs font-medium">${inc.severity.toUpperCase()}</span> <span class="text-white text-sm ml-2">${inc.title}</span></div>`;
    html += `<span class="text-[10px] text-gray-500 flex-shrink-0">${time}</span></div>`;
    html += `<div class="text-[10px] text-gray-400 mt-1">${inc.detail}</div>`;
    html += '</div>';
  }
  if ((d.incidents || []).length === 0) {
    html += '<div class="text-gray-500 text-sm text-center py-4">No incidents recorded yet. Data builds up over time.</div>';
  }
  html += '</div></div>';

  el.innerHTML = html;
}

// ── Alerts Config Tab ──
let emailConfigData = null;
async function loadAlertsConfig() {
  try {
    const [r1, r2] = await Promise.all([fetch('/api/alerts-config'), fetch('/api/email-config')]);
    alertsconfigData = await r1.json();
    emailConfigData = await r2.json();
    renderAlertsConfig();
  } catch(e) { console.error(e); }
}
async function updateAlertThreshold(ruleId, field, value) {
  try {
    await fetch('/api/alerts-config', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({rule_id: ruleId, [field]: value}) });
    loadAlertsConfig();
  } catch(e) { console.error(e); }
}
async function updateEmailConfig(payload) {
  try {
    await fetch('/api/email-config', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
    loadAlertsConfig();
  } catch(e) { console.error(e); }
}
async function addEmailRecipient() {
  const input = document.getElementById('new-email-input');
  const email = input.value.trim();
  if (email && email.includes('@')) { await updateEmailConfig({add_recipient: email}); input.value = ''; }
}
async function sendTestEmail() {
  const btn = document.getElementById('test-email-btn');
  btn.textContent = 'Sending...'; btn.disabled = true;
  try {
    const r = await fetch('/api/email-test', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}' });
    const result = await r.json();
    btn.textContent = result.ok ? 'Sent!' : 'Failed';
    setTimeout(() => { btn.textContent = 'Send Test Email'; btn.disabled = false; }, 3000);
    loadAlertsConfig();
  } catch(e) { btn.textContent = 'Error'; setTimeout(() => { btn.textContent = 'Send Test Email'; btn.disabled = false; }, 3000); }
}
function renderAlertsConfig() {
  const el = document.getElementById('page-alertsconfig');
  if (!alertsconfigData) { el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading...</div>'; return; }
  const d = alertsconfigData;
  const ec = emailConfigData || {};

  // Email Notifications Section
  let html = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Email Notifications</h3>';
  html += '<p class="text-xs text-gray-500 mb-4">SendGrid-powered alerts sent when thresholds are breached</p>';

  html += '<div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">';
  // Left: Settings
  html += '<div class="glass-bright rounded-lg p-4">';
  html += '<div class="flex items-center justify-between mb-3">';
  html += `<span class="text-sm text-white font-medium">Email Alerts</span>`;
  html += `<label class="flex items-center gap-2 cursor-pointer"><input type="checkbox" ${ec.enabled?'checked':''} onchange="updateEmailConfig({enabled:this.checked})" class="accent-emerald-500" /><span class="text-xs ${ec.enabled?'text-emerald-400':'text-gray-500'}">${ec.enabled?'Enabled':'Disabled'}</span></label>`;
  html += '</div>';
  html += `<div class="text-[10px] text-gray-500 mb-2">SendGrid: ${ec.sendgrid_configured?'<span class="text-emerald-400">Configured</span>':'<span class="text-red-400">Not configured</span>'} · From: ${ec.mail_from||'N/A'}</div>`;
  html += '<div class="flex items-center gap-2 mb-3">';
  html += '<span class="text-xs text-gray-400">Cooldown:</span>';
  html += `<input type="number" value="${ec.cooldown_minutes||30}" min="1" max="120" onchange="updateEmailConfig({cooldown_minutes:parseInt(this.value)})" class="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-xs text-right" />`;
  html += '<span class="text-[10px] text-gray-500">min between repeat alerts</span>';
  html += '</div>';
  html += `<button id="test-email-btn" onclick="sendTestEmail()" class="w-full py-2 px-3 rounded-lg text-xs font-medium ${ec.sendgrid_configured&&ec.recipients?.length?'bg-cyan-600 hover:bg-cyan-500 text-white':'bg-gray-700 text-gray-400 cursor-not-allowed'}" ${ec.sendgrid_configured&&ec.recipients?.length?'':'disabled'}>Send Test Email</button>`;
  html += '</div>';

  // Right: Recipients
  html += '<div class="glass-bright rounded-lg p-4">';
  html += '<div class="text-sm text-white font-medium mb-2">Recipients</div>';
  html += '<div class="space-y-1 mb-3">';
  for (const r of (ec.recipients||[])) {
    html += `<div class="flex items-center justify-between bg-gray-800/50 rounded px-3 py-1.5">`;
    html += `<span class="text-xs text-gray-300">${r}</span>`;
    html += `<button onclick="updateEmailConfig({remove_recipient:'${r}'})" class="text-red-400 hover:text-red-300 text-xs px-1">✕</button>`;
    html += '</div>';
  }
  if (!(ec.recipients||[]).length) html += '<div class="text-[10px] text-gray-500">No recipients configured</div>';
  html += '</div>';
  html += '<div class="flex gap-2">';
  html += '<input id="new-email-input" type="email" placeholder="email@example.com" class="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-white text-xs" onkeydown="if(event.key===\'Enter\')addEmailRecipient()" />';
  html += '<button onclick="addEmailRecipient()" class="bg-emerald-600 hover:bg-emerald-500 text-white text-xs px-3 py-1.5 rounded">Add</button>';
  html += '</div>';
  html += '</div>';
  html += '</div>';

  // Email History
  const hist = ec.history || [];
  if (hist.length > 0) {
    html += '<div class="text-xs text-gray-400 font-medium mb-2">Notification History</div>';
    html += '<div class="space-y-1 max-h-40 overflow-y-auto">';
    for (const h of hist.slice(0, 20)) {
      const time = new Date(h.t * 1000).toLocaleString();
      const statusC = h.status === 'sent' ? 'text-emerald-400' : h.status === 'failed' ? 'text-red-400' : 'text-amber-400';
      html += `<div class="flex items-center justify-between text-[10px] py-1 px-2 rounded bg-white/[0.02]">`;
      html += `<span class="text-gray-500">${time}</span>`;
      html += `<span class="text-gray-400 flex-1 mx-2 truncate">${h.subject||''}</span>`;
      html += `<span class="${statusC} font-medium">${h.status}</span>`;
      html += '</div>';
    }
    html += '</div>';
  }
  html += '</div>';

  // Alert Rules Section
  html += '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Alert Rules</h3>';
  html += '<p class="text-xs text-gray-500 mb-4">Customize thresholds — changes take effect immediately</p>';
  html += `<div class="text-xs text-gray-400 mb-4">${d.active_count||0} active alert(s) right now</div>`;

  html += '<div class="space-y-3">';
  for (const [ruleId, cfg] of Object.entries(d.configs || {})) {
    const sevC = cfg.severity === 'critical' ? 'text-red-400' : cfg.severity === 'warning' ? 'text-amber-400' : 'text-blue-400';
    html += `<div class="glass-bright rounded-lg p-4 flex items-center justify-between">`;
    html += `<div class="flex-1"><span class="text-white text-sm font-medium">${cfg.name}</span> <span class="${sevC} text-[10px]">${cfg.severity}</span>`;
    html += `<div class="text-[10px] text-gray-500 mt-1">Rule: ${ruleId}</div></div>`;
    html += `<div class="flex items-center gap-3">`;
    html += `<div class="flex items-center gap-1"><input type="number" value="${cfg.threshold}" onchange="updateAlertThreshold('${ruleId}','threshold',this.value)" class="w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-xs text-right" />`;
    html += `<span class="text-[10px] text-gray-500">${cfg.unit}</span></div>`;
    html += `<label class="flex items-center gap-1 cursor-pointer"><input type="checkbox" ${cfg.enabled?'checked':''} onchange="updateAlertThreshold('${ruleId}','enabled',this.checked)" class="accent-emerald-500" /><span class="text-[10px] text-gray-400">On</span></label>`;
    html += '</div></div>';
  }
  html += '</div></div>';

  // Active alerts
  if ((d.active_alerts||[]).length > 0) {
    html += '<div class="glass rounded-xl p-6"><h3 class="text-white font-semibold mb-4">Currently Triggered Alerts</h3>';
    html += '<div class="space-y-2">';
    for (const a of d.active_alerts) {
      const cfg = (d.configs||{})[a.rule] || {};
      const sevC = cfg.severity === 'critical' ? 'text-red-400 border-red-500' : 'text-amber-400 border-amber-500';
      html += `<div class="border-l-2 ${sevC} bg-white/[0.02] rounded-r-lg p-3 flex justify-between">`;
      html += `<div><span class="${sevC} text-xs font-medium">${cfg.name||a.rule}</span>`;
      html += `<div class="text-[10px] text-gray-400">${a.server} · GPU ${a.gpu}</div></div>`;
      html += `<div class="text-right"><div class="text-white text-sm font-bold">${a.value}${cfg.unit||''}</div>`;
      html += `<div class="text-[10px] text-gray-500">threshold: ${a.threshold}${cfg.unit||''}</div></div>`;
      html += '</div>';
    }
    html += '</div></div>';
  }

  el.innerHTML = html;
}

// ── SLA Dashboard Tab ──
async function loadSLA() {
  try { const r = await fetch('/api/sla'); slaData = await r.json(); renderSLA(); } catch(e) { console.error(e); }
}
async function updateSLATarget(slaId, value) {
  try {
    await fetch('/api/sla', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sla_id: slaId, target: value})
    });
    loadSLA();
  } catch(e) { console.error(e); }
}
function renderSLA() {
  const el = document.getElementById('page-sla');
  if (!slaData) { el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading...</div>'; return; }
  const d = slaData;
  const ov = d.overall || {};

  let html = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">SLA Compliance Overview</h3>';
  html += '<p class="text-xs text-gray-500 mb-4">Track service level objectives across the fleet</p>';
  html += '<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">';
  const gradeC = ov.grade === 'A' ? 'text-emerald-400' : ov.grade === 'B' ? 'text-cyan-400' : ov.grade === 'C' ? 'text-amber-400' : 'text-red-400';
  html += `<div class="glass-bright rounded-lg p-4 text-center"><div class="text-5xl font-bold ${gradeC}">${ov.grade||'?'}</div><div class="text-[10px] text-gray-500 mt-1">Grade</div></div>`;
  html += `<div class="glass-bright rounded-lg p-4 text-center"><div class="text-3xl font-bold text-white">${ov.score||0}%</div><div class="text-[10px] text-gray-500 mt-1">Score</div></div>`;
  html += `<div class="glass-bright rounded-lg p-4 text-center"><div class="text-3xl font-bold text-emerald-400">${ov.met||0}</div><div class="text-[10px] text-gray-500 mt-1">SLAs Met</div></div>`;
  html += `<div class="glass-bright rounded-lg p-4 text-center"><div class="text-3xl font-bold text-gray-300">${ov.total||0}</div><div class="text-[10px] text-gray-500 mt-1">Total SLAs</div></div>`;
  html += '</div></div>';

  // Individual SLA cards
  html += '<div class="grid grid-cols-1 md:grid-cols-2 gap-4">';
  for (const [slaId, sla] of Object.entries(d.targets || {})) {
    const met = sla.compliant;
    const borderC = met ? 'border-emerald-500/30' : 'border-red-500/30';
    const bgC = met ? 'bg-emerald-500/5' : 'bg-red-500/5';
    const statusC = met ? 'text-emerald-400' : 'text-red-400';
    const statusIcon = met ? '✓' : '✗';

    html += `<div class="glass rounded-xl p-5 border ${borderC} ${bgC}">`;
    html += `<div class="flex justify-between items-center mb-3">`;
    html += `<div><span class="text-white text-sm font-medium">${sla.name}</span><div class="text-[10px] text-gray-500 mt-0.5">${sla.direction === 'above' ? '≥' : '≤'} target</div></div>`;
    html += `<span class="${statusC} text-xl font-bold">${statusIcon}</span>`;
    html += '</div>';

    html += '<div class="grid grid-cols-3 gap-3 mb-3">';
    html += `<div><div class="text-[10px] text-gray-500">Current</div><div class="text-lg font-bold ${statusC}">${sla.current_value !== null ? sla.current_value : '—'}${sla.unit}</div></div>`;
    html += `<div><div class="text-[10px] text-gray-500">Target</div><div class="text-lg font-bold text-gray-300">${sla.target}${sla.unit}</div></div>`;
    html += `<div><div class="text-[10px] text-gray-500">Set Target</div>`;
    html += `<input type="number" value="${sla.target}" step="any" onchange="updateSLATarget('${slaId}',this.value)" class="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-xs mt-1" /></div>`;
    html += '</div>';

    // Progress bar
    let pct;
    if (sla.direction === 'above') {
      pct = sla.target > 0 ? Math.min(sla.current_value / sla.target * 100, 100) : 0;
    } else {
      pct = sla.target > 0 ? Math.min((1 - sla.current_value / sla.target) * 100 + 50, 100) : 100;
    }
    const barC = met ? '#22c55e' : '#ef4444';
    html += `<div class="bar-track rounded-full h-2"><div class="bar-fill rounded-full h-2" style="width:${Math.max(pct,5)}%;background:${barC}"></div></div>`;
    html += '</div>';
  }
  html += '</div>';

  el.innerHTML = html;
}

// ── Anomaly Detection Tab ──
async function loadAnomalies() {
  try { const r = await fetch('/api/anomalies'); anomaliesData = await r.json(); renderAnomalies(); } catch(e) { console.error(e); }
}
function renderAnomalies() {
  const el = document.getElementById('page-anomalies');
  if (!anomaliesData) { el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading...</div>'; return; }
  const d = anomaliesData;
  const s = d.summary || {};

  // Summary cards
  let html = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Anomaly Detection</h3>';
  html += '<p class="text-xs text-gray-500 mb-4">Rolling z-score analysis across all GPU metrics — flags deviations > 2.5&sigma;</p>';
  html += '<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">';
  const statusC = s.total_anomalies === 0 ? 'text-emerald-400' : s.critical > 0 ? 'text-red-400' : 'text-amber-400';
  const statusText = s.total_anomalies === 0 ? 'ALL NORMAL' : s.critical > 0 ? 'CRITICAL' : 'WARNING';
  html += `<div class="glass-bright rounded-lg p-4 text-center"><div class="text-2xl font-bold ${statusC}">${statusText}</div><div class="text-[10px] text-gray-500 mt-1">Status</div></div>`;
  html += `<div class="glass-bright rounded-lg p-4 text-center"><div class="text-3xl font-bold text-white">${s.total_anomalies||0}</div><div class="text-[10px] text-gray-500 mt-1">Anomalies</div></div>`;
  html += `<div class="glass-bright rounded-lg p-4 text-center"><div class="text-3xl font-bold text-red-400">${s.critical||0}</div><div class="text-[10px] text-gray-500 mt-1">Critical</div></div>`;
  html += `<div class="glass-bright rounded-lg p-4 text-center"><div class="text-3xl font-bold text-gray-300">${s.metrics_monitored||0}</div><div class="text-[10px] text-gray-500 mt-1">Metrics Tracked</div></div>`;
  html += '</div>';

  // Type breakdown
  const bt = s.by_type || {};
  if (Object.keys(bt).length > 0) {
    html += '<div class="grid grid-cols-3 md:grid-cols-7 gap-2 mb-4">';
    const icons = {util_anomaly:'📊',temp_anomaly:'🌡',memory_leak:'💾',power_anomaly:'⚡',error_spike:'⚠',fleet_util_anomaly:'🏭',fleet_power_anomaly:'🔌'};
    const labels = {util_anomaly:'Utilization',temp_anomaly:'Temperature',memory_leak:'Memory Leak',power_anomaly:'Power',error_spike:'Error Rate',fleet_util_anomaly:'Fleet Util',fleet_power_anomaly:'Fleet Power'};
    for (const [type, count] of Object.entries(bt)) {
      const c = count > 0 ? 'text-amber-400 border-amber-500/30' : 'text-emerald-400 border-emerald-500/20';
      html += `<div class="glass-bright rounded-lg p-2 text-center text-[10px] border ${count>0?'border-amber-500/30':'border-transparent'}">`;
      html += `<div class="text-sm">${icons[type]||'📈'}</div><div class="${c} font-medium">${count}</div><div class="text-gray-500">${labels[type]||type}</div></div>`;
    }
    html += '</div>';
  }
  html += '</div>';

  // Active anomalies list
  if ((d.anomalies||[]).length > 0) {
    html += '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-4">Active Anomalies</h3>';
    html += '<div class="space-y-2">';
    for (const a of d.anomalies) {
      const sevC = a.severity === 'critical' ? 'border-red-500 bg-red-500/10' : 'border-amber-500 bg-amber-500/10';
      const sevT = a.severity === 'critical' ? 'text-red-400' : 'text-amber-400';
      const zAbs = Math.abs(a.z||0);
      const zBar = Math.min(zAbs / 5 * 100, 100);
      const zC = zAbs > 3.5 ? 'bg-red-500/60' : zAbs > 2.5 ? 'bg-amber-500/60' : 'bg-blue-500/40';
      html += `<div class="border-l-2 ${sevC} rounded-r-lg p-3">`;
      html += `<div class="flex justify-between items-start mb-1">`;
      html += `<div><span class="${sevT} text-xs font-medium">${a.severity.toUpperCase()}</span> <span class="text-white text-sm ml-2">${a.title}</span></div>`;
      html += `<span class="text-xs text-gray-400 font-mono">z=${a.z}</span></div>`;
      html += `<div class="text-[10px] text-gray-400 mb-2">${a.detail}</div>`;
      html += `<div class="flex items-center gap-2"><span class="text-[9px] text-gray-500 w-8">|z|</span>`;
      html += `<div class="flex-1 h-2 rounded bg-gray-800"><div class="${zC} h-2 rounded" style="width:${zBar}%"></div></div>`;
      html += `<span class="text-[9px] text-gray-500 w-8">${zAbs.toFixed(1)}</span></div>`;
      html += '</div>';
    }
    html += '</div></div>';
  }

  // Metric health grid
  const ms = d.metric_stats || {};
  const keys = Object.keys(ms);
  if (keys.length > 0) {
    html += '<div class="glass rounded-xl p-6"><h3 class="text-white font-semibold mb-1">Metric Health Grid</h3>';
    html += '<p class="text-xs text-gray-500 mb-4">Current z-score for each tracked metric — green is normal</p>';
    html += '<div class="grid grid-cols-2 md:grid-cols-4 gap-2">';
    for (const [key, m] of Object.entries(ms)) {
      const z = Math.abs(m.z||0);
      const borderC = m.is_anomaly ? 'border-red-500/50' : z > 1.5 ? 'border-amber-500/30' : 'border-emerald-500/20';
      const zC = m.is_anomaly ? 'text-red-400' : z > 1.5 ? 'text-amber-400' : 'text-emerald-400';
      html += `<div class="glass-bright rounded-lg p-3 border ${borderC}">`;
      html += `<div class="flex justify-between items-center mb-1"><span class="text-[10px] text-gray-400 truncate">${m.label}</span><span class="text-[9px] text-gray-600">${m.server}</span></div>`;
      html += `<div class="flex items-baseline gap-2"><span class="text-sm font-bold ${zC}">z=${m.z}</span><span class="text-[10px] text-gray-500">${m.current} (μ=${m.mean})</span></div>`;
      // Mini z-score sparkline
      const zs = m.recent_zs || [];
      if (zs.length > 1) {
        const maxZ = Math.max(...zs.map(v=>Math.abs(v)), 3);
        const w = 100; const h = 20;
        const step = w / (zs.length - 1);
        let path = '';
        zs.forEach((v, i) => {
          const x = i * step;
          const y = h/2 - (v / maxZ) * (h/2 - 1);
          path += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1);
        });
        html += `<svg width="${w}" height="${h}" class="mt-1"><line x1="0" y1="${h/2}" x2="${w}" y2="${h/2}" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>`;
        html += `<path d="${path}" fill="none" stroke="${m.is_anomaly?'#ef4444':'#22c55e'}" stroke-width="1.5" stroke-linecap="round"/></svg>`;
      }
      html += '</div>';
    }
    html += '</div></div>';
  }

  if ((d.anomalies||[]).length === 0 && keys.length === 0) {
    html += '<div class="glass rounded-xl p-12 text-center"><div class="text-gray-500">Not enough history data yet. Anomaly detection needs ~10 samples to begin analysis.</div></div>';
  }

  el.innerHTML = html;
}

// ── Executive Summary Tab ──
async function loadExecutive() {
  try {
    const r = await fetch('/api/executive');
    executiveData = await r.json();
    renderExecutive();
  } catch(e) { console.error('Executive load error:', e); }
}

function renderExecutive() {
  const el = document.getElementById('page-executive');
  if (!executiveData) { el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading executive summary...</div>'; return; }
  const d = executiveData;

  // ── 1. Fleet Health Score ──
  const hs = d.health_score || {};
  const fleet = hs.fleet || 0;
  const grade = hs.grade || 'N/A';
  const gradeC = fleet >= 90 ? 'text-emerald-400' : fleet >= 75 ? 'text-cyan-400' : fleet >= 60 ? 'text-amber-400' : 'text-red-400';
  const gradeBg = fleet >= 90 ? 'from-emerald-500/20' : fleet >= 75 ? 'from-cyan-500/20' : fleet >= 60 ? 'from-amber-500/20' : 'from-red-500/20';

  let hsHTML = '<div class="glass rounded-xl p-6 mb-6">';
  hsHTML += '<div class="flex items-center justify-between mb-4">';
  hsHTML += '<div><h3 class="text-white font-semibold">Fleet Health Score</h3><p class="text-xs text-gray-500">Composite score across all GPUs</p></div>';
  hsHTML += `<div class="text-center"><div class="text-5xl font-bold ${gradeC}">${fleet}</div><div class="text-lg font-semibold ${gradeC}">Grade ${grade}</div></div>`;
  hsHTML += '</div>';
  // Per-server breakdown
  hsHTML += '<div class="grid grid-cols-1 md:grid-cols-2 gap-4">';
  for (const [sk, s] of Object.entries(hs.servers || {})) {
    const sc = s.score || 0;
    const scC = sc >= 90 ? 'text-emerald-400' : sc >= 75 ? 'text-cyan-400' : sc >= 60 ? 'text-amber-400' : 'text-red-400';
    hsHTML += `<div class="glass-bright rounded-lg p-4">`;
    hsHTML += `<div class="flex justify-between items-center mb-3"><span class="text-sm text-white font-medium">${s.server_name||sk}</span><span class="text-2xl font-bold ${scC}">${sc}</span></div>`;
    // Score bars
    const bd = s.breakdown || {};
    for (const [k, v] of Object.entries(bd)) {
      const bC = v >= 90 ? '#22c55e' : v >= 75 ? '#06b6d4' : v >= 60 ? '#f59e0b' : '#ef4444';
      hsHTML += `<div class="flex items-center gap-2 mb-1 text-[10px]"><span class="text-gray-500 w-20 capitalize">${k}</span><div class="flex-1 h-1.5 rounded-full bg-gray-800"><div class="h-1.5 rounded-full" style="width:${v}%;background:${bC}"></div></div><span class="text-gray-400 w-8 text-right">${v}</span></div>`;
    }
    // Key metrics
    const m = s.metrics || {};
    hsHTML += '<div class="flex gap-3 mt-2 text-[10px] text-gray-500">';
    hsHTML += `<span>Util: ${m.avg_util||0}%</span>`;
    hsHTML += `<span>Temp: ${m.max_temp||0}C</span>`;
    hsHTML += `<span>VRAM: ${m.avg_vram||0}%</span>`;
    hsHTML += `<span>Err: ${m.error_rate||0}%</span>`;
    if (m.avg_ttft) hsHTML += `<span>TTFT: ${m.avg_ttft}ms</span>`;
    hsHTML += '</div></div>';
  }
  hsHTML += '</div></div>';

  // ── 2. Daily Summary ──
  const ds = d.daily_summary || {};
  let dsHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Daily Summary</h3>';
  dsHTML += `<p class="text-xs text-gray-500 mb-4">${ds.data_hours||0} hours of data collected</p>`;
  dsHTML += '<div class="grid grid-cols-2 md:grid-cols-5 gap-3">';
  const kpis = [
    {label: 'Requests Served', value: (ds.total_requests_served||0).toLocaleString(), color: 'text-white'},
    {label: 'Peak Concurrent', value: ds.peak_concurrent_calls||0, color: 'text-cyan-400'},
    {label: 'Avg Concurrent', value: ds.avg_concurrent_calls||0, color: 'text-gray-300'},
    {label: 'Avg GPU Util', value: (ds.avg_gpu_util||0)+'%', color: ds.avg_gpu_util > 80 ? 'text-amber-400' : 'text-emerald-400'},
    {label: 'Max Temp', value: (ds.max_temp||0)+'C', color: ds.max_temp > 80 ? 'text-red-400' : 'text-gray-300'},
    {label: 'Avg TTFT', value: (ds.avg_ttft_ms||0)+'ms', color: ds.avg_ttft_ms > 300 ? 'text-amber-400' : 'text-emerald-400'},
    {label: 'Avg E2E', value: (ds.avg_e2e_s||0)+'s', color: ds.avg_e2e_s > 3 ? 'text-amber-400' : 'text-emerald-400'},
    {label: "Today's Cost", value: '$'+(ds.cost_today||0).toLocaleString(), color: 'text-white'},
    {label: 'Cost/Request', value: '$'+(ds.cost_per_request||0).toFixed(4), color: 'text-gray-300'},
    {label: 'Uptime', value: (ds.uptime_pct||0)+'%', color: ds.uptime_pct >= 99.9 ? 'text-emerald-400' : 'text-amber-400'},
  ];
  for (const kpi of kpis) {
    dsHTML += `<div class="glass-bright rounded-lg p-3"><div class="text-[10px] text-gray-500 mb-1">${kpi.label}</div><div class="text-lg font-bold ${kpi.color}">${kpi.value}</div></div>`;
  }
  dsHTML += '</div></div>';

  // ── 3. H200 vs RTX 5090 Comparison ──
  const comp = d.comparison || {};
  let cpHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">H200 vs RTX 5090 Comparison</h3>';
  cpHTML += '<p class="text-xs text-gray-500 mb-4">Side-by-side performance metrics</p>';
  const servers = Object.entries(comp).filter(([k,v]) => !v.no_data);
  if (servers.length >= 2) {
    const [sk1, s1] = servers[0];
    const [sk2, s2] = servers[1];
    const metrics = [
      {label: 'GPU Model', v1: s1.gpu_model, v2: s2.gpu_model},
      {label: 'GPU Count', v1: s1.gpu_count, v2: s2.gpu_count},
      {label: 'Avg Util %', v1: s1.avg_util+'%', v2: s2.avg_util+'%', better: s1.avg_util < s2.avg_util ? 1 : 2},
      {label: 'Avg Calls', v1: s1.avg_calls, v2: s2.avg_calls, better: s1.avg_calls > s2.avg_calls ? 1 : 2},
      {label: 'Peak Calls', v1: s1.peak_calls, v2: s2.peak_calls, better: s1.peak_calls > s2.peak_calls ? 1 : 2},
      {label: 'Avg Temp', v1: s1.avg_temp+'C', v2: s2.avg_temp+'C', better: s1.avg_temp < s2.avg_temp ? 1 : 2},
      {label: 'Avg Power', v1: s1.avg_power_w+'W', v2: s2.avg_power_w+'W', better: s1.avg_power_w < s2.avg_power_w ? 1 : 2},
      {label: 'VRAM Used', v1: s1.avg_vram_pct+'%', v2: s2.avg_vram_pct+'%', better: s1.avg_vram_pct < s2.avg_vram_pct ? 1 : 2},
      {label: 'Avg TTFT', v1: s1.avg_ttft_ms?s1.avg_ttft_ms+'ms':'N/A', v2: s2.avg_ttft_ms?s2.avg_ttft_ms+'ms':'N/A', better: (s1.avg_ttft_ms||999) < (s2.avg_ttft_ms||999) ? 1 : 2},
      {label: 'Avg E2E', v1: s1.avg_e2e_s?s1.avg_e2e_s+'s':'N/A', v2: s2.avg_e2e_s?s2.avg_e2e_s+'s':'N/A', better: (s1.avg_e2e_s||999) < (s2.avg_e2e_s||999) ? 1 : 2},
    ];
    cpHTML += '<div class="overflow-x-auto"><table class="w-full text-xs">';
    cpHTML += `<thead><tr class="border-b border-gray-800"><th class="text-left py-2 px-2 text-gray-500">Metric</th><th class="text-center py-2 px-2 text-white">${s1.server_name}</th><th class="text-center py-2 px-2 text-white">${s2.server_name}</th></tr></thead><tbody>`;
    for (const m of metrics) {
      const c1 = m.better === 1 ? 'text-emerald-400 font-bold' : 'text-gray-300';
      const c2 = m.better === 2 ? 'text-emerald-400 font-bold' : 'text-gray-300';
      cpHTML += `<tr class="border-b border-gray-800/50"><td class="py-2 px-2 text-gray-400">${m.label}</td><td class="py-2 px-2 text-center ${c1}">${m.v1}</td><td class="py-2 px-2 text-center ${c2}">${m.v2}</td></tr>`;
    }
    cpHTML += '</tbody></table></div>';
  } else {
    cpHTML += '<div class="text-gray-500 text-sm">Need data from both servers for comparison.</div>';
  }
  cpHTML += '</div>';

  // ── 4. Trends ──
  let trHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Trend Indicators</h3>';
  trHTML += '<p class="text-xs text-gray-500 mb-4">First half vs second half of collected data</p>';
  trHTML += '<div class="grid grid-cols-1 md:grid-cols-2 gap-4">';
  for (const [sk, tr] of Object.entries(d.trends || {})) {
    if (tr.no_data) continue;
    trHTML += `<div class="glass-bright rounded-lg p-4"><div class="text-xs text-white font-medium mb-3">${tr.server_name||sk}</div>`;
    trHTML += '<div class="grid grid-cols-2 gap-2">';
    const arrows = {up: '&#9650;', down: '&#9660;', stable: '&#9654;'};
    const colors = {up: 'text-red-400', down: 'text-emerald-400', stable: 'text-gray-500'};
    // For calls, up is good
    const callColors = {up: 'text-emerald-400', down: 'text-red-400', stable: 'text-gray-500'};
    const items = [
      {label: 'Utilization', data: tr.utilization, goodDown: true},
      {label: 'Temperature', data: tr.temperature, goodDown: true},
      {label: 'Calls', data: tr.calls, goodDown: false},
      {label: 'TTFT', data: tr.ttft, goodDown: true},
    ];
    for (const item of items) {
      const t = item.data || {};
      const trendDir = t.trend || 'stable';
      const tC = item.goodDown ? colors[trendDir] : callColors[trendDir];
      trHTML += `<div class="text-[10px]"><span class="text-gray-500">${item.label}</span>`;
      trHTML += `<div class="flex items-center gap-1"><span class="${tC}">${arrows[trendDir]||''}</span>`;
      trHTML += `<span class="text-gray-400">${t.old||0} → ${t.new||0}</span></div></div>`;
    }
    trHTML += '</div></div>';
  }
  trHTML += '</div></div>';

  el.innerHTML = hsHTML + dsHTML + cpHTML + trHTML;
}

// ── Network & I/O Tab ──
async function loadNetwork() {
  try {
    const r = await fetch('/api/network');
    networkData = await r.json();
    renderNetwork();
  } catch(e) { console.error('Network load error:', e); }
}

function renderNetwork() {
  const el = document.getElementById('page-network');
  if (!networkData) { el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading network data...</div>'; return; }
  const d = networkData;

  // ── 1. GPU Topology ──
  let topoHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">GPU-to-GPU Topology</h3>';
  topoHTML += '<p class="text-xs text-gray-500 mb-4">NVLink/PCIe interconnect matrix between GPUs</p>';
  topoHTML += '<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">';
  for (const [sk, topo] of Object.entries(d.gpu_topology || {})) {
    topoHTML += `<div class="glass-bright rounded-lg p-4"><div class="text-xs text-white font-medium mb-2">${topo.server_name || sk}</div>`;
    if (topo.error) {
      topoHTML += `<div class="text-xs text-red-400">${topo.error}</div>`;
    } else if ((topo.matrix||[]).length > 0) {
      topoHTML += '<div class="overflow-x-auto"><table class="text-[9px]">';
      if ((topo.headers||[]).length > 0) {
        topoHTML += '<thead><tr>';
        for (const h of topo.headers) topoHTML += `<th class="px-1 py-0.5 text-gray-500">${h}</th>`;
        topoHTML += '</tr></thead>';
      }
      topoHTML += '<tbody>';
      for (const row of topo.matrix) {
        topoHTML += '<tr>';
        topoHTML += `<td class="px-1 py-0.5 text-gray-400 font-medium">${row.label}</td>`;
        for (const conn of row.connections) {
          const color = conn === 'NV18' || conn.startsWith('NV') ? 'text-emerald-400' : conn === 'SYS' ? 'text-gray-500' : conn === 'X' ? 'text-white' : conn === 'PHB' ? 'text-cyan-400' : 'text-gray-400';
          topoHTML += `<td class="px-1 py-0.5 text-center ${color}">${conn}</td>`;
        }
        topoHTML += '</tr>';
      }
      topoHTML += '</tbody></table></div>';
    } else {
      topoHTML += '<div class="text-xs text-gray-500">No topology data available</div>';
    }
    topoHTML += '</div>';
  }
  topoHTML += '</div></div>';

  // ── 2. Network Throughput ──
  let netHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Network Throughput Timeline</h3>';
  netHTML += '<p class="text-xs text-gray-500 mb-4">RX/TX bandwidth with anomaly detection</p>';
  netHTML += '<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">';
  for (const [sk, net] of Object.entries(d.network_throughput || {})) {
    netHTML += `<div class="glass-bright rounded-lg p-4"><div class="flex justify-between items-center mb-2">`;
    netHTML += `<span class="text-xs text-white font-medium">${net.server_name || sk}</span>`;
    netHTML += `<span class="text-[10px] ${(net.anomalies||0) > 0 ? 'text-red-400' : 'text-gray-500'}">${net.anomalies||0} anomalies</span>`;
    netHTML += '</div>';
    netHTML += '<div class="grid grid-cols-4 gap-2 mb-3 text-[10px]">';
    netHTML += `<div><span class="text-gray-500">Avg RX</span><div class="text-cyan-400 font-medium">${net.avg_rx_mbps||0} Mbps</div></div>`;
    netHTML += `<div><span class="text-gray-500">Peak RX</span><div class="text-cyan-400 font-medium">${net.peak_rx_mbps||0} Mbps</div></div>`;
    netHTML += `<div><span class="text-gray-500">Avg TX</span><div class="text-emerald-400 font-medium">${net.avg_tx_mbps||0} Mbps</div></div>`;
    netHTML += `<div><span class="text-gray-500">Peak TX</span><div class="text-emerald-400 font-medium">${net.peak_tx_mbps||0} Mbps</div></div>`;
    netHTML += '</div>';
    // Mini timeline bars
    const tl = net.timeline || [];
    if (tl.length > 0) {
      const maxBps = Math.max(...tl.map(t => Math.max(t.rx_bps, t.tx_bps)), 1);
      netHTML += '<div class="space-y-0.5">';
      for (const t of tl.slice(-30)) {
        const time = new Date(t.t * 1000).toLocaleTimeString();
        const rxW = t.rx_bps / maxBps * 100;
        const txW = t.tx_bps / maxBps * 100;
        const anomClass = (t.rx_anomaly || t.tx_anomaly) ? 'border-l-2 border-red-400 pl-1' : '';
        netHTML += `<div class="flex items-center gap-1 text-[9px] ${anomClass}">`;
        netHTML += `<span class="text-gray-600 w-12">${time}</span>`;
        netHTML += `<div class="flex-1 flex gap-0.5">`;
        netHTML += `<div class="h-2 rounded bg-cyan-500/40" style="width:${rxW}%" title="RX: ${t.rx_mbps} Mbps"></div>`;
        netHTML += `<div class="h-2 rounded bg-emerald-500/40" style="width:${txW}%" title="TX: ${t.tx_mbps} Mbps"></div>`;
        netHTML += '</div>';
        netHTML += `<span class="text-gray-500 w-20 text-right">${t.rx_mbps}/${t.tx_mbps}</span>`;
        netHTML += '</div>';
      }
      netHTML += '</div>';
    } else {
      netHTML += '<div class="text-xs text-gray-500">No throughput data yet.</div>';
    }
    netHTML += '</div>';
  }
  netHTML += '</div></div>';

  // ── 3. Storage I/O ──
  let sioHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Storage I/O</h3>';
  sioHTML += '<p class="text-xs text-gray-500 mb-4">Disk usage trends and I/O activity per server</p>';
  sioHTML += '<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">';
  for (const [sk, sio] of Object.entries(d.storage_io || {})) {
    sioHTML += `<div class="glass-bright rounded-lg p-4"><div class="text-xs text-white font-medium mb-2">${sio.server_name || sk}</div>`;
    if (sio.error) {
      sioHTML += `<div class="text-xs text-red-400">${sio.error}</div>`;
    } else {
      const disk = sio.disk || {};
      if (disk.used_pct !== undefined) {
        const diskC = disk.used_pct > 90 ? 'text-red-400' : disk.used_pct > 80 ? 'text-amber-400' : 'text-emerald-400';
        sioHTML += `<div class="grid grid-cols-3 gap-2 mb-2 text-[10px]">`;
        sioHTML += `<div><span class="text-gray-500">Used</span><div class="${diskC} font-medium">${disk.used_pct}%</div></div>`;
        sioHTML += `<div><span class="text-gray-500">Growth/hr</span><div class="text-gray-300 font-medium">${(disk.growth_pct_hour||0).toFixed(3)}%</div></div>`;
        sioHTML += `<div><span class="text-gray-500">Days til Full</span><div class="${(disk.days_until_full||9999) < 30 ? 'text-red-400' : 'text-gray-300'} font-medium">${(disk.days_until_full||9999) > 999 ? '999+' : (disk.days_until_full||0).toFixed(0)}</div></div>`;
        sioHTML += '</div>';
        sioHTML += `<div class="h-2 rounded-full bg-gray-800 mb-3"><div class="h-2 rounded-full" style="width:${disk.used_pct}%;background:${disk.used_pct > 90 ? '#ef4444' : disk.used_pct > 80 ? '#f59e0b' : '#22c55e'}"></div></div>`;
      }
      // Device table
      const devs = sio.devices || [];
      if (devs.length > 0) {
        sioHTML += '<table class="w-full text-[10px]">';
        sioHTML += '<thead><tr class="text-gray-500"><th class="text-left py-0.5">Device</th><th class="text-right py-0.5">Reads</th><th class="text-right py-0.5">Read MB</th><th class="text-right py-0.5">Writes</th><th class="text-right py-0.5">Write MB</th></tr></thead>';
        sioHTML += '<tbody>';
        for (const dev of devs) {
          sioHTML += `<tr class="border-t border-gray-800/30"><td class="py-0.5 text-gray-300">${dev.device}</td><td class="py-0.5 text-right text-gray-400">${dev.reads.toLocaleString()}</td><td class="py-0.5 text-right text-cyan-400">${dev.read_mb.toLocaleString()}</td><td class="py-0.5 text-right text-gray-400">${dev.writes.toLocaleString()}</td><td class="py-0.5 text-right text-emerald-400">${dev.write_mb.toLocaleString()}</td></tr>`;
        }
        sioHTML += '</tbody></table>';
      }
    }
    sioHTML += '</div>';
  }
  sioHTML += '</div></div>';

  // ── 4. Model Loading & Warmup ──
  let mlHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Model Loading & KV Cache Warmup</h3>';
  mlHTML += '<p class="text-xs text-gray-500 mb-4">Active models and KV cache warmup timeline</p>';
  mlHTML += '<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">';
  for (const [sk, ml] of Object.entries(d.model_loading || {})) {
    mlHTML += `<div class="glass-bright rounded-lg p-4"><div class="text-xs text-white font-medium mb-2">${ml.server_name || sk}</div>`;
    // Models list
    const models = ml.models || [];
    if (models.length > 0) {
      for (const m of models) {
        mlHTML += `<div class="flex justify-between items-center text-[10px] mb-1">`;
        mlHTML += `<span class="text-gray-300">:${m.port} <span class="text-gray-500">${m.model_name}</span></span>`;
        mlHTML += `<span class="text-gray-400">KV: <span class="${m.kv_cache_usage > 80 ? 'text-red-400' : 'text-emerald-400'}">${m.kv_cache_usage}%</span> | ${m.requests_running} running | ${(m.total_requests||0).toLocaleString()} total</span>`;
        mlHTML += '</div>';
      }
    } else {
      mlHTML += '<div class="text-[10px] text-gray-500">No vLLM models on this server</div>';
    }
    // KV warmup
    mlHTML += `<div class="mt-2 text-[10px] text-gray-500">KV warmup: ${ml.kv_warmup_seconds||0}s (${ml.kv_warmup_samples||0} samples) | Current: ${(ml.current_kv||0).toFixed(1)}%</div>`;
    // Mini KV sparkline
    const kvH = ml.kv_history || [];
    if (kvH.length > 2) {
      mlHTML += '<div class="mt-1">' + sparklineSVG(kvH, 200, 20, '#22c55e') + '</div>';
    }
    mlHTML += '</div>';
  }
  mlHTML += '</div></div>';

  el.innerHTML = topoHTML + netHTML + sioHTML + mlHTML;
}

// ── Call Quality Deep Dive Tab ──
async function loadQuality() {
  try {
    const r = await fetch('/api/quality');
    qualityData = await r.json();
    renderQuality();
  } catch(e) { console.error('Quality load error:', e); }
}

function renderQuality() {
  const el = document.getElementById('page-quality');
  if (!qualityData) { el.innerHTML = '<div class="text-gray-500 text-center py-12">Loading quality data...</div>'; return; }
  const d = qualityData;

  // ── 1. Pipeline Waterfall ──
  let wfHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Pipeline Waterfall</h3>';
  wfHTML += '<p class="text-xs text-gray-500 mb-4">Call lifecycle breakdown: Queue → TTFT → Generation</p>';
  if ((d.waterfall||[]).length === 0) {
    wfHTML += '<div class="text-gray-500 text-sm">No waterfall data yet (needs vLLM metrics).</div>';
  } else {
    wfHTML += '<div class="space-y-1">';
    for (const w of d.waterfall.slice(-30)) {
      const time = new Date(w.t * 1000).toLocaleTimeString();
      const maxW = Math.max(w.total_s, 0.001);
      const qW = w.queue_s / maxW * 100;
      const tW = w.ttft_s / maxW * 100;
      const gW = w.generation_s / maxW * 100;
      wfHTML += `<div class="flex items-center gap-2 text-[10px]">`;
      wfHTML += `<span class="text-gray-500 w-16 flex-shrink-0">${time}</span>`;
      wfHTML += `<span class="text-gray-400 w-8 text-right">${w.calls}c</span>`;
      wfHTML += `<div class="flex-1 flex h-4 rounded overflow-hidden bg-gray-800">`;
      if (qW > 0.5) wfHTML += `<div style="width:${qW}%" class="bg-amber-500/60" title="Queue: ${w.queue_s.toFixed(3)}s"></div>`;
      if (tW > 0.5) wfHTML += `<div style="width:${tW}%" class="bg-cyan-500/60" title="TTFT: ${w.ttft_s.toFixed(3)}s"></div>`;
      if (gW > 0.5) wfHTML += `<div style="width:${gW}%" class="bg-purple-500/60" title="Gen: ${w.generation_s.toFixed(3)}s"></div>`;
      wfHTML += `</div>`;
      wfHTML += `<span class="text-gray-400 w-12 text-right">${w.total_s.toFixed(2)}s</span>`;
      wfHTML += `</div>`;
    }
    wfHTML += '</div>';
    wfHTML += '<div class="flex gap-4 mt-3 text-[10px]">';
    wfHTML += '<span class="flex items-center gap-1"><span class="w-3 h-3 rounded bg-amber-500/60"></span> Queue</span>';
    wfHTML += '<span class="flex items-center gap-1"><span class="w-3 h-3 rounded bg-cyan-500/60"></span> TTFT</span>';
    wfHTML += '<span class="flex items-center gap-1"><span class="w-3 h-3 rounded bg-purple-500/60"></span> Generation</span>';
    wfHTML += '</div>';
  }
  wfHTML += '</div>';

  // ── 2. Quality Degradation Heatmap ──
  let hmHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Quality Degradation Map</h3>';
  hmHTML += '<p class="text-xs text-gray-500 mb-4">TTFT by hour and GPU utilization — shows where quality drops</p>';
  const dmap = d.degradation_map || [];
  if (dmap.length === 0) {
    hmHTML += '<div class="text-gray-500 text-sm">No degradation data yet.</div>';
  } else {
    const hours = [...new Set(dmap.map(d=>d.hour))].sort((a,b)=>a-b);
    const buckets = [...new Set(dmap.map(d=>d.util_bucket))].sort((a,b)=>a-b);
    const lookup = {};
    let maxTtft = 1;
    for (const cell of dmap) {
      lookup[cell.hour+'_'+cell.util_bucket] = cell;
      if (cell.avg_ttft_ms > maxTtft) maxTtft = cell.avg_ttft_ms;
    }
    hmHTML += '<div class="overflow-x-auto"><table class="text-[10px]">';
    hmHTML += '<thead><tr><th class="px-1 py-1 text-gray-500">Util \\ Hour</th>';
    for (const h of hours) hmHTML += `<th class="px-1 py-1 text-gray-500 text-center">${String(h).padStart(2,'0')}</th>`;
    hmHTML += '</tr></thead><tbody>';
    for (const b of buckets) {
      hmHTML += `<tr><td class="px-1 py-1 text-gray-400 font-medium">${b}-${b+9}%</td>`;
      for (const h of hours) {
        const cell = lookup[h+'_'+b];
        if (cell && cell.avg_ttft_ms > 0) {
          const intensity = Math.min(cell.avg_ttft_ms / Math.max(maxTtft, 1), 1);
          const r = Math.round(intensity * 239);
          const g = Math.round((1-intensity) * 197);
          hmHTML += `<td class="px-1 py-1 text-center rounded" style="background:rgba(${r},${g},50,0.3)" title="${cell.avg_ttft_ms}ms TTFT, ${cell.samples} samples">${cell.avg_ttft_ms.toFixed(0)}</td>`;
        } else {
          hmHTML += '<td class="px-1 py-1 text-center text-gray-700">-</td>';
        }
      }
      hmHTML += '</tr>';
    }
    hmHTML += '</tbody></table></div>';
    hmHTML += '<div class="flex items-center gap-2 mt-2 text-[10px] text-gray-500"><span>Low TTFT</span><div class="w-24 h-2 rounded" style="background:linear-gradient(to right, rgba(50,197,50,0.3), rgba(239,50,50,0.3))"></div><span>High TTFT</span></div>';
  }
  hmHTML += '</div>';

  // ── 3. Latency Curve ──
  let lcHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">Concurrent Calls vs Latency</h3>';
  lcHTML += '<p class="text-xs text-gray-500 mb-4">Find the inflection point where quality degrades</p>';
  const curve = d.latency_curve || [];
  if (curve.length === 0) {
    lcHTML += '<div class="text-gray-500 text-sm">No latency curve data yet.</div>';
  } else {
    const maxE2E = Math.max(...curve.map(c=>c.avg_e2e_s), 0.001);
    lcHTML += '<div class="space-y-1">';
    for (const c of curve) {
      const barW = (c.avg_e2e_s / maxE2E * 100);
      const inflClass = c.inflection ? 'border-l-2 border-red-400 pl-2' : '';
      const barColor = c.avg_e2e_s > 6 ? 'bg-red-500/60' : c.avg_e2e_s > 3 ? 'bg-amber-500/60' : 'bg-emerald-500/60';
      lcHTML += `<div class="flex items-center gap-2 text-[11px] ${inflClass}">`;
      lcHTML += `<span class="text-gray-400 w-16 text-right">${c.concurrent_calls} calls</span>`;
      lcHTML += `<div class="flex-1 h-5 rounded bg-gray-800 relative overflow-hidden">`;
      lcHTML += `<div class="${barColor} h-full rounded" style="width:${barW}%"></div>`;
      lcHTML += `<span class="absolute inset-y-0 left-2 flex items-center text-white text-[10px]">${c.avg_e2e_s.toFixed(2)}s e2e &middot; ${c.avg_ttft_ms.toFixed(0)}ms ttft &middot; KV:${c.avg_kv_cache}%</span>`;
      lcHTML += `</div>`;
      lcHTML += `<span class="text-gray-500 w-12 text-right text-[10px]">${c.samples}s</span>`;
      if (c.inflection) lcHTML += '<span class="text-red-400 text-[10px] font-bold">INFLECTION</span>';
      lcHTML += '</div>';
    }
    lcHTML += '</div>';
  }
  lcHTML += '</div>';

  // ── 4. SLA Compliance ──
  let slaHTML = '<div class="glass rounded-xl p-6 mb-6"><h3 class="text-white font-semibold mb-1">SLA Compliance Tracker</h3>';
  slaHTML += '<p class="text-xs text-gray-500 mb-4">Track compliance against quality targets</p>';
  const sla = d.sla_compliance || {};
  if (Object.keys(sla).length === 0) {
    slaHTML += '<div class="text-gray-500 text-sm">No SLA data yet.</div>';
  } else {
    slaHTML += '<div class="grid grid-cols-1 md:grid-cols-2 gap-4">';
    for (const [metric, s] of Object.entries(sla)) {
      const compC = s.compliance_pct >= 99 ? 'text-emerald-400' : s.compliance_pct >= 95 ? 'text-amber-400' : 'text-red-400';
      slaHTML += `<div class="glass-bright rounded-lg p-4">`;
      slaHTML += `<div class="flex justify-between items-center mb-2">`;
      slaHTML += `<span class="text-xs text-white font-medium">${s.label}</span>`;
      slaHTML += `<span class="text-lg font-bold ${compC}">${s.compliance_pct}%</span>`;
      slaHTML += '</div>';
      slaHTML += `<div class="h-2 rounded-full bg-gray-800 mb-2"><div class="h-2 rounded-full" style="width:${Math.min(s.compliance_pct,100)}%;background:${s.compliance_pct >= 99 ? '#22c55e' : s.compliance_pct >= 95 ? '#f59e0b' : '#ef4444'}"></div></div>`;
      slaHTML += `<div class="flex justify-between text-[10px] text-gray-500 mb-2">`;
      slaHTML += `<span>${s.total_samples} samples</span>`;
      slaHTML += `<span class="${s.violations > 0 ? 'text-red-400' : 'text-gray-500'}">${s.violations} violations</span>`;
      slaHTML += '</div>';
      if ((s.worst_hours||[]).length > 0) {
        slaHTML += '<div class="text-[10px] text-gray-500">Worst hours: ';
        slaHTML += s.worst_hours.map(h => `<span class="text-red-400">${h.hour}</span> (${h.violations})`).join(', ');
        slaHTML += '</div>';
      }
      slaHTML += '</div>';
    }
    slaHTML += '</div>';
  }
  slaHTML += '</div>';

  el.innerHTML = wfHTML + hmHTML + lcHTML + slaHTML;
}

// ── Data Fetching ──
async function fetchServer(key) {
  try { const r = await fetch('/api/'+key); return await r.json(); }
  catch(e) { return { server_name: key, status:'offline', error: e.message }; }
}

async function fetchHistory(key) {
  try { const r = await fetch('/api/'+key+'/history'); const d = await r.json(); return d.history || []; }
  catch(e) { return []; }
}

async function refresh() {
  const [h200, rtx5090] = await Promise.all([fetchServer('h200'), fetchServer('rtx5090')]);
  serverData.h200 = h200;
  serverData.rtx5090 = rtx5090;

  // Fetch history less frequently
  if (!refresh._histCount) refresh._histCount = 0;
  if (refresh._histCount % 5 === 0) {
    const [hh, hr] = await Promise.all([fetchHistory('h200'), fetchHistory('rtx5090')]);
    historyData.h200 = hh;
    historyData.rtx5090 = hr;
  }
  refresh._histCount++;

  renderAll();
  document.getElementById('lastUpdate').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

function updateRefreshRate() {
  refreshRate = parseInt(document.getElementById('refreshRate').value);
  if (refreshInterval) clearInterval(refreshInterval);
  refreshInterval = setInterval(refresh, refreshRate);
}

// Init
renderOverview();
refresh();
refreshInterval = setInterval(refresh, refreshRate);
</script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
