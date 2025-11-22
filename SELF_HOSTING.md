# Self-Hosting BrandonBot on Debian 13

## Overview

This guide explains how to self-host BrandonBot on your own Debian 13 system, which will eliminate the CPU resource contention issues present in Replit's shared development environment.

**Expected Performance Improvement**:
- **Replit (shared)**: 1 token per 60-90 seconds (CPU starvation)
- **Self-hosted (dedicated)**: 10-30 tokens per second (60-180x faster)

---

## System Requirements

### Minimum Requirements
- **OS**: Debian 13 (or Ubuntu 22.04+, other Debian-based distros)
- **CPU**: 4+ cores (Intel/AMD x86_64)
- **RAM**: 6GB available (4GB for model, 1-2GB for system/embeddings)
- **Storage**: 10GB free space (2.6GB for Phi-3 model, ~5GB for Weaviate data, 2GB for dependencies)
- **Python**: 3.11 or higher (included in Debian 13)

### Recommended Requirements
- **CPU**: 6+ cores, 3.0+ GHz
- **RAM**: 8GB+ available
- **Storage**: SSD for faster model loading

---

## Installation Steps

### 1. Install System Dependencies

```bash
sudo apt update
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    libopenblas-dev \
    ffmpeg

# Verify installations
python3 --version  # Should show 3.11 or higher
gcc --version      # Should show GCC compiler
```

**Why these packages?**
- `python3-pip`, `python3-venv`: Python package management
- `build-essential`: C/C++ compilers needed by some Python packages
- `libopenblas-dev`: Optimized BLAS library for faster matrix operations (sentence-transformers)
- `ffmpeg`: Required by some audio/video processing dependencies

### 2. Clone/Copy Project Files

#### Option A: Clone from Git (if hosted)
```bash
git clone <your-repo-url>
cd <repo-directory>
```

#### Option B: Copy from Replit

**Files/Directories to Copy from Replit**:
```
/home/runner/workspace/
├── backend/               # All Python code (REQUIRED)
│   ├── main.py
│   ├── phi3_client.py
│   ├── rag_pipeline.py
│   ├── retrieval_orchestrator.py
│   ├── weaviate_manager.py
│   ├── database.py
│   ├── requirements.txt
│   └── data/             # Knowledge base documents (REQUIRED)
├── phi3_model/           # Phi-3 ONNX model files ~2.6GB (REQUIRED)
├── weaviate_data/        # Pre-computed embeddings (OPTIONAL - speeds up first start)
├── frontend/             # HTML/CSS/JS (REQUIRED if using web UI)
└── .env.example          # Environment variable template (if exists)
```

**Download methods**:
- Via Replit's download feature (right-click folders)
- Using `rsync` if you have SSH access
- Via Replit's deployment export

### 3. Set Up Python Environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected installation time**: 5-10 minutes (downloads PyTorch CPU, ONNX Runtime, sentence-transformers, etc.)

### 5. Download Phi-3 Model (if not copied)

If you didn't copy the `phi3_model/` directory from Replit:

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# Download Phi-3 model (~2.6GB)
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx \
    --include "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*" \
    --local-dir ./phi3_model
```

### 6. Configure Environment Variables (Optional)

Create a `.env` file in the `backend/` directory:

```bash
# Optional configurations
DATABASE_PATH=./data/brandonbot.db
WEAVIATE_DATA_PATH=../weaviate_data
LOG_LEVEL=INFO

# For commercial API migration (if using)
# OPENAI_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here
```

### 7. Initialize Database and Weaviate

```bash
# First run will initialize SQLite database and Weaviate embeddings
# If you copied weaviate_data/, this will be fast
# If starting fresh, expect 2-5 minutes for embedding generation

python3 main.py
```

**Expected startup logs**:
```
INFO:main:Starting BrandonBot (100% Open Source - No Docker Required)...
INFO:main:Initializing database...
INFO:database:Database initialized successfully
INFO:main:Initializing Weaviate (embedded mode)...
INFO:weaviate_manager:Starting Weaviate in embedded mode (no Docker required)...
INFO:weaviate_manager:Weaviate initialized successfully in embedded mode
INFO:main:Loading Phi-3 model (CPU-optimized)...
INFO:phi3_client:Loading Phi-3 model from ./phi3_model...
INFO:phi3_client:Phi-3 model loaded successfully
INFO:main:BrandonBot ready! Running entirely on open-source software.
INFO:     Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)
```

---

## Running the Server

### Development Mode (with auto-reload)
```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

### Production Mode (recommended)
```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 5000 --workers 1
```

**Important**: Use `--workers 1` because Phi-3 model cannot be shared across processes. Each worker would load its own copy (4GB RAM per worker).

### Access the Application
- **Local**: http://localhost:5000
- **Network**: http://<your-ip>:5000
- **API docs**: http://localhost:5000/docs

---

## Performance Tuning

### CPU Thread Configuration

Unlike Replit (where thread limits are needed to prevent contention), on dedicated hardware you can use **all available cores**:

**Option 1: Use all cores** (recommended for dedicated hardware)
```bash
# Do NOT set these environment variables
# Let ONNX Runtime auto-detect optimal thread count
```

**Option 2: Manual tuning** (if you want to limit CPU usage)
```bash
export OMP_NUM_THREADS=6
export ORT_INTRA_OP_NUM_THREADS=6
export ORT_INTER_OP_NUM_THREADS=1

# Then start server
uvicorn main:app --host 0.0.0.0 --port 5000
```

### Expected Performance Benchmarks

**Query Processing Time** (on typical 6-core CPU @ 3.0GHz):
- Retrieval (RAG): 0.3-0.8 seconds
- Phi-3 Generation: 3-10 seconds (100-300 tokens)
- **Total**: 3-11 seconds per query

**Token Generation Speed**:
- Expected: 10-30 tokens/second
- vs Replit shared: 0.01 tokens/second (1000-3000x faster!)

---

## Running as a System Service (Optional)

To run BrandonBot automatically on system boot:

### Create systemd service file
```bash
sudo nano /etc/systemd/system/brandonbot.service
```

### Service configuration
```ini
[Unit]
Description=BrandonBot AI Chatbot
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/your/backend
Environment="PATH=/path/to/your/backend/venv/bin"
# Optional: For API-based inference (OpenAI/Gemini)
# Environment="OPENAI_API_KEY=your_key_here"
# Environment="GOOGLE_API_KEY=your_key_here"
ExecStart=/path/to/your/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 5000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Note on environment variables for systemd:**
- For Phi-3 self-hosting: No API keys needed
- For commercial API: Add `Environment="OPENAI_API_KEY=..."` or `Environment="GOOGLE_API_KEY=..."`
- Alternatively, use an `.env` file and add `EnvironmentFile=/path/to/.env` to the `[Service]` section

### Enable and start service
```bash
sudo systemctl daemon-reload
sudo systemctl enable brandonbot
sudo systemctl start brandonbot
sudo systemctl status brandonbot
```

### Verify service is running correctly
```bash
# Check service status
sudo systemctl status brandonbot

# Should show: "active (running)"
# If failed, check logs:
sudo journalctl -u brandonbot -n 50

# Test API endpoint
curl http://localhost:5000/
# Should return HTML of the frontend

# Test query endpoint
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What are your positions?"}'
# Should return JSON with response and confidence
```

### View logs
```bash
# Follow logs in real-time
sudo journalctl -u brandonbot -f

# View last 100 lines
sudo journalctl -u brandonbot -n 100

# View logs from today
sudo journalctl -u brandonbot --since today
```

---

## Firewall Configuration (if needed)

Allow external access to port 5000:

```bash
# UFW (Uncomplicated Firewall)
sudo ufw allow 5000/tcp

# Or iptables
sudo iptables -A INPUT -p tcp --dport 5000 -j ACCEPT
```

---

## Reverse Proxy with Nginx (Optional)

For production deployment with HTTPS:

### Install Nginx
```bash
sudo apt install nginx certbot python3-certbot-nginx
```

### Configure Nginx
```bash
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
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Enable site and get SSL
```bash
sudo ln -s /etc/nginx/sites-available/brandonbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
sudo certbot --nginx -d your-domain.com
```

### Verify nginx setup
```bash
# Test nginx configuration
sudo nginx -t
# Should show: "configuration file /etc/nginx/nginx.conf test is successful"

# Check nginx status
sudo systemctl status nginx
# Should show: "active (running)"

# Test HTTP access
curl http://your-domain.com
# Should return HTML content

# Test HTTPS after certbot (if configured)
curl https://your-domain.com
# Should return HTML content with valid SSL

# Check SSL certificate
sudo certbot certificates
# Should show certificate details and expiry date
```

---

## Troubleshooting

### Issue: Model fails to load
**Symptoms**: `FileNotFoundError: phi3_model not found`

**Solution**:
```bash
# Verify model files exist
ls -lh phi3_model/

# Should see:
# - phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx (226KB)
# - phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data (2.6GB)
# - tokenizer.model, config.json, etc.

# If missing, download again
python3 download_phi3_model.py
```

### Issue: Out of memory errors
**Symptoms**: `RuntimeError: Unable to allocate tensor`

**Solution**:
```bash
# Check available RAM
free -h

# Close other applications
# Reduce system RAM usage
# Or upgrade to 8GB+ RAM
```

### Issue: Slow generation (<5 tokens/sec)
**Symptoms**: Still slow despite dedicated hardware

**Solution**:
```bash
# Check CPU usage
htop  # Should see uvicorn using 100-400% CPU during generation

# Verify no thread limits are set
env | grep -E "OMP|ORT"  # Should return nothing

# If set, unset them:
unset OMP_NUM_THREADS
unset ORT_INTRA_OP_NUM_THREADS
unset ORT_INTER_OP_NUM_THREADS

# Restart server
```

### Issue: Weaviate initialization fails
**Symptoms**: `Connection refused` or Weaviate errors

**Solution**:
```bash
# Check if Weaviate data is corrupted
rm -rf weaviate_data/

# Let it reinitialize (takes 2-5 minutes)
python3 main.py
```

### Issue: Port 5000 already in use
**Symptoms**: `Address already in use`

**Solution**:
```bash
# Find process using port 5000
sudo lsof -i :5000

# Kill it
sudo kill -9 <PID>

# Or use a different port
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Monitoring and Maintenance

### View real-time logs
```bash
tail -f backend/logs/*.log  # If logging to file
# Or use systemd journal if running as service
```

### Monitor resource usage
```bash
# Install htop
sudo apt install htop

# Run
htop
# Look for python3 process, observe CPU/RAM usage
```

### Backup important data
```bash
# Backup databases and logs
tar -czf brandonbot-backup-$(date +%Y%m%d).tar.gz \
    backend/data/brandonbot.db \
    backend/data/brandonbot_conversations.csv \
    weaviate_data/
```

---

## Migrating Back to Replit or Cloud

To move your self-hosted instance back to Replit:

1. Copy `weaviate_data/` to preserve embeddings
2. Copy `backend/data/` to preserve conversation logs
3. Upload to Replit
4. Re-add ONNX thread limits (see COMMERCIALAI_MIGRATION.md)

---

## Performance Comparison

| Metric | Replit (Shared) | Self-Hosted (Dedicated 6-core) |
|--------|----------------|-------------------------------|
| **Token generation** | 0.01 tokens/sec | 10-30 tokens/sec |
| **Query latency** | 60-90 seconds | 3-11 seconds |
| **CPU load** | 20+ (contention) | 1-4 (normal) |
| **Response quality** | Same | Same |
| **Reliability** | Timeouts common | Stable |
| **Cost** | Replit subscription | Self-hosted compute |

---

## Next Steps

After successful self-hosting, consider:

1. **Set up monitoring** (Prometheus + Grafana)
2. **Add HTTPS** (via nginx + certbot)
3. **Implement backup automation** (cron jobs)
4. **Scale horizontally** (multiple instances with load balancer)
5. **Migrate to commercial API** for even better performance (see COMMERCIALAI_MIGRATION.md)
