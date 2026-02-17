# Deployment Guide

## Architecture

The dashboard has two components:
1. **API Server** — Runs on the H200 server (`146.88.194.12:8080`), collects data from both GPU clusters
2. **Static Frontend** — Hosted on Vercel (`testssh.vercel.app`), polls the API

## Prerequisites

### H200 Server Setup
```bash
ssh ubuntu@146.88.194.12

# Create virtual environment (one-time)
python3 -m venv ~/gpudb_env
source ~/gpudb_env/bin/activate
pip install fastapi uvicorn asyncssh

# Ensure SSH key exists for connecting to RTX 5090
ls ~/.ssh/id_ed25519  # Should exist
# The public key must be in ubuntu@38.65.239.47:~/.ssh/authorized_keys
```

### Firewall
Port 8080 must be open on the H200 server:
```bash
sudo ufw allow 8080/tcp
```

## Deploy API to H200

```bash
# From local machine
scp gpu_dashboard.py ubuntu@146.88.194.12:~/gpu_dashboard.py

# SSH into H200 and restart
ssh ubuntu@146.88.194.12

# Find and kill existing process
ps aux | grep gpu_dashboard | grep -v grep
kill <PID>

# Start new process
source ~/gpudb_env/bin/activate
nohup python ~/gpu_dashboard.py > ~/gpudb.log 2>&1 &

# Verify
curl -s localhost:8080/api/h200 | python3 -c "import sys,json; d=json.load(sys.stdin); print('Status:', d['status'], 'GPUs:', len(d['gpus']))"
curl -s localhost:8080/api/rtx5090 | python3 -c "import sys,json; d=json.load(sys.stdin); print('Status:', d['status'], 'GPUs:', len(d['gpus']))"
```

## Deploy Frontend to Vercel

The Vercel project serves `public/index.html` as a static site. The HTML calls the H200 API at `http://146.88.194.12:8080`.

```bash
# From the project root
npx vercel --prod
```

### Vercel Configuration
- `vercel.json` — Routes and build config
- `public/index.html` — Static frontend with `API_BASE = 'http://146.88.194.12:8080'`

## Running as a Systemd Service (Optional)

To auto-start the dashboard on boot:

```bash
sudo tee /etc/systemd/system/gpudb.service << 'EOF'
[Unit]
Description=GPU Dashboard API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/home/ubuntu/gpudb_env/bin/python /home/ubuntu/gpu_dashboard.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable gpudb
sudo systemctl start gpudb
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | HTML dashboard (when accessing H200 directly) |
| `GET /api/{server}` | Live GPU + system data (`h200` or `rtx5090`) |
| `GET /api/{server}/history` | Time-series history (in-memory, resets on restart) |
| `GET /api/{server}/software` | Installed packages and system info (5-min cache) |

## Troubleshooting

- **Empty GPU data**: Check SSH connectivity from H200 to RTX 5090: `ssh ubuntu@38.65.239.47 nvidia-smi`
- **Connection refused on :8080**: Check `sudo ufw status` and ensure port 8080 is allowed
- **Vercel frontend offline**: API still accessible directly at `http://146.88.194.12:8080`
- **Check logs**: `tail -f ~/gpudb.log` on H200
