# Self-Hosting BrandonBot

## Overview

This guide explains how to self-host BrandonBot on your own server with Python 3.12. The self-hosted version uses Ollama with Llama 3.2 as the LLM judge for validation.

---

## System Requirements

### Minimum Requirements
- **OS**: Debian 13, Ubuntu 22.04+, or other Linux distribution
- **Python**: 3.12
- **RAM**: 8GB available (4GB for Llama, 4GB for system/embeddings)
- **Storage**: 15GB free space

### For LLM Judge (Validation)
- **Ollama**: Required for running Llama 3.2 locally
- **Model**: llama3.2:3b (~2GB)

---

## Quick Start

### 1. Install System Dependencies

```bash
sudo apt update
sudo apt install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    build-essential \
    curl
```

### 2. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama server:
```bash
ollama serve
```

### 3. Clone/Copy Project Files

```bash
git clone <your-repo-url>
cd brandonbot
```

Or copy these directories from Replit:
```
backend/           # All Python code
documents/         # Knowledge base documents
frontend/          # Web UI (if using)
```

### 4. Set Up Python Environment

```bash
cd backend
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Download Models

This downloads Ollama's Llama 3.2 (for LLM judge) and optional SLM safeguard models:

```bash
python download_models.py
```

**Ollama only** (for validation):
```bash
python download_models.py --ollama-only
```

**SLM models only** (for local safeguards):
```bash
python download_models.py --slm-only
```

### 6. Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
# Required for commercial LLM providers (Replit mode)
GOOGLE_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
NVIDIA_API_KEY=your_key_here

# Optional: SendGrid for email notifications
SENDGRID_API_KEY=your_key_here

# Database
DATABASE_PATH=./data/brandonbot.db
```

### 7. Start the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

Access at: http://localhost:5000

---

## Running Validation with Local LLM Judge

The validation suite uses Ollama + Llama 3.2 as the LLM judge:

```bash
# Ensure Ollama is running
ollama serve &

# Set environment variable
export USE_LOCAL_JUDGE=true

# Run validation
cd backend/validation
python validator.py
```

---

## Running as a System Service

### Create systemd service

```bash
sudo nano /etc/systemd/system/brandonbot.service
```

```ini
[Unit]
Description=BrandonBot AI Chatbot
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/backend
Environment="PATH=/path/to/backend/venv/bin"
ExecStart=/path/to/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 5000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable brandonbot
sudo systemctl start brandonbot
sudo systemctl status brandonbot
```

---

## Reverse Proxy with Nginx (Optional)

### Install and configure

```bash
sudo apt install nginx certbot python3-certbot-nginx
sudo nano /etc/nginx/sites-available/brandonbot
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Enable and get SSL

```bash
sudo ln -s /etc/nginx/sites-available/brandonbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
sudo certbot --nginx -d your-domain.com
```

---

## Troubleshooting

### Ollama not responding
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Pull Llama model if missing
ollama pull llama3.2:3b
```

### Port 5000 in use
```bash
sudo lsof -i :5000
sudo kill -9 <PID>

# Or use different port
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Out of memory
```bash
# Check available RAM
free -h

# Llama 3.2:3b requires ~4GB RAM
# Close other applications or use smaller model
ollama pull llama3.2:1b
```

---

## File Structure

```
brandonbot/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── agent_orchestrator.py   # LLM agent pipeline
│   ├── llm_providers.py        # Multi-provider LLM manager
│   ├── weaviate_manager.py     # Vector database
│   ├── database.py             # SQLite operations
│   ├── download_models.py      # Model downloader
│   ├── requirements.txt        # Full dependencies (self-hosted)
│   ├── requirements-replit.txt # Lightweight (Replit deployment)
│   └── validation/
│       ├── validator.py        # Validation suite
│       └── debug.db            # Debug logs
├── documents/                  # Knowledge base source docs
└── frontend/                   # Web UI
```

---

## Environment Modes

| Mode | LLM Provider | LLM Judge | Use Case |
|------|-------------|-----------|----------|
| Replit | Commercial APIs | Nvidia API | Development/Demo |
| Self-Hosted | Commercial APIs | Ollama/Llama | Production |
| Fully Local | Ollama | Ollama/Llama | Offline/Privacy |

---

## Next Steps

1. **Run validation** to test response quality
2. **Configure email notifications** via SendGrid
3. **Set up monitoring** (logs, health checks)
4. **Add SSL** via nginx + certbot
