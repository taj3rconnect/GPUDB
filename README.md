# GPUDB — GPU Admin Dashboard

Real-time monitoring dashboard for multi-GPU clusters. Tracks GPU utilization, temperatures, power draw, VRAM, vLLM inference metrics, concurrent calls, and installed software — all without installing anything on GPU servers.

## Features

- **Overview** — Live GPU cards with utilization rings, temperature, power, VRAM bars, process lists, network throughput
- **Traffic & KPIs** — vLLM model metrics: request counts, token throughput, TTFT, ITL, E2E latency, KV cache usage, prefix cache hits
- **History & Peaks** — Time-series sparklines, peak analysis, concurrent call tracking (AgentV2)
- **Software & Tools** — Installed package inventory per environment, version comparison with PyPI, release notes links

## Monitored Servers

| Server | GPUs | VRAM | Workload |
|--------|------|------|----------|
| H200 Cluster | 8x NVIDIA H200 | 8x 141 GiB (1.1 TiB) | vLLM: Qwen3-VL-8B-Instruct, recruite-tts-fp8 |
| RTX 5090 Cluster | 8x NVIDIA RTX 5090 | 8x 32 GiB (256 GiB) | AgentV2 (call agent) |

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn asyncssh

# Run locally
python gpu_dashboard.py
# Open http://localhost:8080
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  Browser/Vercel │────>│ FastAPI on H200   │
│  (static HTML)  │     │ :8080             │
└─────────────────┘     └────────┬──────────┘
                                 │
                    ┌────────────┼────────────┐
                    │ local      │ SSH        │
                    v            v            v
              ┌──────────┐  ┌─────────┐  ┌──────────┐
              │ nvidia-smi│  │ vLLM    │  │ RTX 5090 │
              │ /proc/*   │  │ :8001   │  │ server   │
              │ free, df  │  │ :8002   │  │ (SSH)    │
              └──────────┘  └─────────┘  └──────────┘
```

- **No agents or daemons on GPU servers** — all data collected via SSH + existing CLI tools
- **Client-side polling** — browser refreshes every 3s (configurable), no server push overhead
- **Single-file app** — `gpu_dashboard.py` contains both API and HTML frontend

## Deployment

See [DEPLOY.md](DEPLOY.md) for detailed deployment instructions.

## Tech Stack

- **Backend**: Python 3.12, FastAPI, asyncssh
- **Frontend**: Tailwind CSS (CDN), vanilla JavaScript
- **Hosting**: H200 server (API), Vercel (static frontend)
