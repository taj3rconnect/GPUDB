# GPUDB - GPU Dashboard

## Project Overview
Real-time GPU monitoring dashboard for two GPU clusters:
- **H200 Cluster** (146.88.194.12) — 8x NVIDIA H200, running vLLM (Qwen3-VL-8B, recruite-tts-fp8)
- **RTX 5090 Cluster** (38.65.239.47) — 8x NVIDIA GeForce RTX 5090, running AgentV2

## Architecture
- **Backend**: FastAPI + asyncssh, single-file `gpu_dashboard.py`
- **Frontend**: Inline HTML with Tailwind CSS CDN, client-side polling
- **Deployment**: API runs on H200 server (port 8080), static frontend on Vercel
- No software installed on GPU servers — uses only `nvidia-smi`, `/proc`, and existing Prometheus endpoints

## Key Files
- `gpu_dashboard.py` — Main app (API + HTML). Deployed to H200 at `~/gpu_dashboard.py`
- `public/index.html` — Vercel-hosted static frontend (mirrors the inline HTML with API_BASE set)
- `api/gpu.py` — Legacy Vercel serverless function (not used, kept for reference)

## Server Access
- H200: `ssh ubuntu@146.88.194.12` (runs API locally, SSHes to RTX 5090)
- RTX 5090: `ssh ubuntu@38.65.239.47`
- SSH key: `~/.ssh/id_ed25519` on local machine and H200

## Data Sources
- GPU metrics: `nvidia-smi --query-gpu` CSV output
- Processes: `nvidia-smi --query-compute-apps`
- System: `/proc/uptime`, `free -b`, `df -B1`, `/proc/net/dev`
- vLLM metrics: Prometheus endpoints on H200 ports 8001, 8002
- Concurrent calls: `ss -tnp` on RTX 5090 port 8003
- Software inventory: `pip list --format=json` from known venvs

## Running Locally
```bash
pip install fastapi uvicorn asyncssh
python gpu_dashboard.py
# Open http://localhost:8080
```

## Deploying to H200
```bash
scp gpu_dashboard.py ubuntu@146.88.194.12:~/gpu_dashboard.py
ssh ubuntu@146.88.194.12 'kill $(ps aux | grep gpu_dashboard | grep -v grep | awk "{print \$2}"); sleep 1; source ~/gpudb_env/bin/activate && nohup python ~/gpu_dashboard.py > ~/gpudb.log 2>&1 &'
```
