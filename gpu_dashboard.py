import asyncio
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
    },
}

SSH_KEY = str(Path.home() / ".ssh" / "id_ed25519")

# ── History Store (in-memory, per server) ──────────────────────
MAX_HISTORY = 1440  # 24hrs at 1-min intervals, or ~72min at 3s
history = {k: deque(maxlen=MAX_HISTORY) for k in SERVERS}


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

    tasks = [
        run_command(server, gpu_query),
        run_command(server, proc_query),
        run_command(server, sys_query),
        run_command(server, net_query),
    ]
    if vllm_combined:
        tasks.append(run_command(server, vllm_combined))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    gpu_out = results[0] if not isinstance(results[0], Exception) else ""
    proc_out = results[1] if not isinstance(results[1], Exception) else ""
    sys_out = results[2] if not isinstance(results[2], Exception) else ""
    net_out = results[3] if not isinstance(results[3], Exception) else ""
    vllm_out = results[4] if len(results) > 4 and not isinstance(results[4], Exception) else ""

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
        "status": "online",
    }

    # Store in history
    avg_util = sum(g["gpu_util"] for g in gpus) / len(gpus) if gpus else 0
    total_running = sum(v["requests_running"] for v in vllm_services)
    total_waiting = sum(v["requests_waiting"] for v in vllm_services)
    history[server_key].append({
        "t": now,
        "avg_util": round(avg_util, 1),
        "max_temp": max((g["temp"] or 0 for g in gpus), default=0),
        "total_power": round(sum(g["power_draw"] for g in gpus), 1),
        "vram_pct": round(sum(g["mem_used"] for g in gpus) / max(sum(g["mem_total"] for g in gpus), 1) * 100, 1),
        "requests_running": total_running,
        "requests_waiting": total_waiting,
        "net_rx": net_rx,
        "net_tx": net_tx,
    })

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
</div>

<!-- Content -->
<div class="max-w-[1900px] mx-auto">
  <div id="page-overview"></div>
  <div id="page-traffic" class="hidden"></div>
  <div id="page-history" class="hidden"></div>
</div>

<script>
const SERVERS = ['h200', 'rtx5090'];
let refreshInterval = null;
let refreshRate = 3000;
let serverData = {};
let historyData = {};
let prevNetData = {};
let activeTab = 'overview';

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
  ['overview','traffic','history'].forEach(t => {
    document.getElementById('page-'+t).classList.toggle('hidden', t !== tab);
    document.getElementById('tab-'+t).className = t === tab ? 'tab-active pb-2 text-sm font-medium px-1' : 'tab-inactive pb-2 text-sm font-medium px-1';
  });
  renderAll();
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

    const avgUtil = (utils.reduce((a,b)=>a+b,0)/utils.length).toFixed(1);
    const avgTemp = (temps.reduce((a,b)=>a+b,0)/temps.length).toFixed(1);

    const duration = h.length > 1 ? ((h[h.length-1].t - h[0].t) / 60).toFixed(0) : 0;

    html += `<div class="glass rounded-2xl p-6 mb-6">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-white">${d.server_name}</h3>
        <span class="text-xs text-gray-500">${h.length} samples over ${duration} min</span>
      </div>

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
          <div class="text-[10px] text-gray-500 uppercase mb-1">Peak Concurrent</div>
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
          <div class="text-xs text-gray-400 mb-2">Active Requests Over Time</div>
          ${sparklineSVG(running, 400, 60, '#06b6d4')}
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

// ── Render All ──
function renderAll() {
  if (activeTab === 'overview') renderOverview();
  else if (activeTab === 'traffic') renderTraffic();
  else if (activeTab === 'history') renderHistory();
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
