import asyncio
import json
import time
import re
from collections import deque
from pathlib import Path
from fastapi import FastAPI
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

    history[server_key].append({
        "t": now,
        "avg_util": round(avg_util, 1),
        "max_temp": max_temp,
        "total_power": total_power,
        "vram_pct": round(sum(g["mem_used"] for g in gpus) / max(sum(g["mem_total"] for g in gpus), 1) * 100, 1),
        "requests_running": total_running,
        "requests_waiting": total_waiting,
        "active_connections": active_connections,
        "net_rx": net_rx,
        "net_tx": net_tx,
    })

    # ── Hourly aggregation ──
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(now, tz=timezone.utc)
    hour_key = dt.strftime("%Y-%m-%d-%H")
    hour_label = dt.strftime("%H:00")
    hour_ts = int(dt.replace(minute=0, second=0, microsecond=0).timestamp())

    hs = hourly_stats[server_key]
    if hour_key not in hs:
        hs[hour_key] = {
            "hour_key": hour_key,
            "hour_label": hour_label,
            "hour_ts": hour_ts,
            "max_calls": 0,
            "max_calls_time": now,
            "max_gpu_util": 0,
            "max_gpu_util_time": now,
            "calls_at_max_gpu_util": 0,
            "max_temp": 0,
            "max_power": 0,
            "samples": 0,
        }
        # Prune old hours
        cutoff = now - HOURLY_RETENTION * 3600
        for k in list(hs.keys()):
            if hs[k]["hour_ts"] < cutoff:
                del hs[k]

    h = hs[hour_key]
    h["samples"] += 1
    if calls > h["max_calls"]:
        h["max_calls"] = calls
        h["max_calls_time"] = now
    if avg_util > h["max_gpu_util"]:
        h["max_gpu_util"] = round(avg_util, 1)
        h["max_gpu_util_time"] = now
        h["calls_at_max_gpu_util"] = calls
    if max_temp > h["max_temp"]:
        h["max_temp"] = max_temp
    if total_power > h["max_power"]:
        h["max_power"] = total_power

    return data


# ── API Endpoints ──────────────────────────────────────────────
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
    hours = sorted(hs.values(), key=lambda x: x["hour_ts"])
    return {"hours": hours, "server_name": SERVERS[server_key]["name"]}


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
</div>

<!-- Content -->
<div class="max-w-[1900px] mx-auto">
  <div id="page-overview"></div>
  <div id="page-traffic" class="hidden"></div>
  <div id="page-history" class="hidden"></div>
  <div id="page-software" class="hidden"></div>
  <div id="page-daily" class="hidden"></div>
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
  ['overview','traffic','history','software','daily'].forEach(t => {
    document.getElementById('page-'+t).classList.toggle('hidden', t !== tab);
    document.getElementById('tab-'+t).className = t === tab ? 'tab-active pb-2 text-sm font-medium px-1' : 'tab-inactive pb-2 text-sm font-medium px-1';
  });
  renderAll();
  if (tab === 'software' && !softwareLoaded) loadSoftware();
  if (tab === 'daily' && !dailyLoaded) loadDaily();
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

function renderDailyServer(key, dd) {
  if (!dd || dd.error || !dd.hours) {
    return `<div class="glass rounded-2xl p-6 mb-6"><h3 class="text-lg font-semibold text-white">${dd?.server_name||key}</h3><p class="text-gray-500 text-sm mt-2">${dd?.error||'No data yet — collecting hourly stats'}</p></div>`;
  }

  const hours = dd.hours;
  if (hours.length === 0) {
    return `<div class="glass rounded-2xl p-6 mb-6"><h3 class="text-lg font-semibold text-white">${dd.server_name}</h3><p class="text-gray-500 text-sm mt-2">No hourly data collected yet. Data populates as the dashboard runs.</p></div>`;
  }

  // Pad to 24 entries if needed, split into two halves
  const padded = hours.slice(-24);
  const mid = Math.ceil(padded.length / 2);
  const left = padded.slice(0, mid);
  const right = padded.slice(mid);

  // Summary stats
  const allMaxCalls = Math.max(...hours.map(h=>h.max_calls), 0);
  const allMaxUtil = Math.max(...hours.map(h=>h.max_gpu_util), 0);
  const peakHour = hours.reduce((best, h) => h.max_calls > (best?.max_calls||0) ? h : best, hours[0]);

  function hourRow(h) {
    const utilColor = h.max_gpu_util >= 85 ? 'text-red-400' : h.max_gpu_util >= 50 ? 'text-amber-400' : 'text-emerald-400';
    const callsColor = h.max_calls > 0 ? 'text-cyan-400' : 'text-gray-500';
    const callsBar = allMaxCalls > 0 ? (h.max_calls / allMaxCalls * 100) : 0;
    const utilBar = h.max_gpu_util;
    return `<tr class="border-b border-gray-800/30 hover:bg-white/[0.03]">
      <td class="py-2 px-3 text-gray-300 font-mono text-xs">${h.hour_label}</td>
      <td class="py-2 px-3">
        <div class="flex items-center gap-2">
          <span class="${callsColor} font-semibold text-sm w-8 text-right">${h.max_calls}</span>
          <div class="bar-track rounded-full h-1.5 flex-1"><div class="bar-fill rounded-full h-1.5" style="width:${callsBar}%;background:#06b6d4"></div></div>
        </div>
      </td>
      <td class="py-2 px-3">
        <div class="flex items-center gap-2">
          <span class="${utilColor} font-semibold text-sm w-12 text-right">${h.max_gpu_util}%</span>
          <div class="bar-track rounded-full h-1.5 flex-1"><div class="bar-fill rounded-full h-1.5" style="width:${utilBar}%;background:${h.max_gpu_util>=85?'#ef4444':h.max_gpu_util>=50?'#f59e0b':'#22c55e'}"></div></div>
        </div>
      </td>
      <td class="py-2 px-3 text-center">
        <span class="text-gray-300 text-sm font-medium">${h.calls_at_max_gpu_util}</span>
      </td>
    </tr>`;
  }

  function halfTable(rows, label) {
    if (rows.length === 0) return `<div class="flex-1 glass-bright rounded-xl p-4"><div class="text-xs text-gray-500 text-center py-8">No data</div></div>`;
    return `<div class="flex-1 glass-bright rounded-xl p-4 overflow-hidden">
      <div class="text-xs text-gray-400 mb-3 font-medium">${label}</div>
      <table class="w-full text-xs">
        <thead><tr class="text-gray-500 border-b border-gray-700/50">
          <th class="text-left py-1.5 px-3 font-medium w-16">Time</th>
          <th class="text-left py-1.5 px-3 font-medium">Max Calls</th>
          <th class="text-left py-1.5 px-3 font-medium">Max GPU %</th>
          <th class="text-center py-1.5 px-3 font-medium w-24">Calls @ Peak GPU</th>
        </tr></thead>
        <tbody>${rows.map(hourRow).join('')}</tbody>
      </table>
    </div>`;
  }

  // Time range label
  const firstHour = padded[0]?.hour_label || '?';
  const lastHour = padded[padded.length-1]?.hour_label || '?';

  return `<div class="glass rounded-2xl p-6 mb-6">
    <div class="flex items-center justify-between mb-4">
      <div class="flex items-center gap-3">
        <h3 class="text-lg font-semibold text-white">${dd.server_name}</h3>
        <span class="text-xs text-gray-500">${firstHour} — ${lastHour} UTC &middot; ${padded.length}h tracked</span>
      </div>
      <button onclick="loadDaily()" class="text-xs px-3 py-1.5 rounded-lg bg-white/[0.05] text-gray-400 border border-gray-700 hover:bg-white/10 transition-colors">Refresh</button>
    </div>

    <!-- Summary -->
    <div class="grid grid-cols-3 gap-3 mb-5">
      <div class="glass-bright rounded-lg p-4 text-center">
        <div class="text-[10px] text-gray-500 uppercase mb-1">Peak Concurrent Calls</div>
        <div class="text-2xl font-bold text-cyan-400">${allMaxCalls}</div>
        <div class="text-[10px] text-gray-500">at ${peakHour?.hour_label||'?'} UTC</div>
      </div>
      <div class="glass-bright rounded-lg p-4 text-center">
        <div class="text-[10px] text-gray-500 uppercase mb-1">Peak GPU Utilization</div>
        <div class="text-2xl font-bold ${allMaxUtil>=85?'text-red-400':allMaxUtil>=50?'text-amber-400':'text-emerald-400'}">${allMaxUtil}%</div>
        <div class="text-[10px] text-gray-500">max across all hours</div>
      </div>
      <div class="glass-bright rounded-lg p-4 text-center">
        <div class="text-[10px] text-gray-500 uppercase mb-1">Busiest Hour</div>
        <div class="text-2xl font-bold text-white">${peakHour?.hour_label||'—'}</div>
        <div class="text-[10px] text-gray-500">${peakHour?.max_calls||0} calls, ${peakHour?.max_gpu_util||0}% GPU</div>
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

// ── Render All ──
function renderAll() {
  if (activeTab === 'overview') renderOverview();
  else if (activeTab === 'traffic') renderTraffic();
  else if (activeTab === 'history') renderHistory();
  else if (activeTab === 'software') renderSoftware();
  else if (activeTab === 'daily') renderDaily();
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
