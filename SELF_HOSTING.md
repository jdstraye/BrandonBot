# Self-Hosting BrandonBot on Debian 13
- [Self-Hosting BrandonBot on Debian 13](#self-hosting-brandonbot-on-debian-13)
  - [Overview](#overview)
  - [System Requirements](#system-requirements)
    - [Minimum Requirements](#minimum-requirements)
    - [Recommended Requirements](#recommended-requirements)
  - [Installation Steps](#installation-steps)
    - [1. Install System Dependencies](#1-install-system-dependencies)
    - [2. Clone/Copy Project Files](#2-clonecopy-project-files)
      - [Option A: Clone from Git (if hosted)](#option-a-clone-from-git-if-hosted)
      - [Option B: Copy from Replit](#option-b-copy-from-replit)
    - [3. Set Up Python Environment](#3-set-up-python-environment)
    - [4. Install Python Dependencies](#4-install-python-dependencies)
    - [5. Download the local LLM and SLM models](#5-download-the-local-llm-and-slm-models)
    - [6. Verify Safeguard Models (Optional)](#6-verify-safeguard-models-optional)
    - [7. Configure Environment Variables](#7-configure-environment-variables)
    - [8. Initialize Database and Weaviate](#8-initialize-database-and-weaviate)
  - [Running the Server](#running-the-server)
    - [Production Mode (recommended)](#production-mode-recommended)
    - [Access the Application](#access-the-application)
  - [Performance Tuning](#performance-tuning)
    - [CPU Thread Configuration](#cpu-thread-configuration)
    - [Expected Performance Benchmarks](#expected-performance-benchmarks)
  - [Running as a System Service (Optional)](#running-as-a-system-service-optional)
    - [Create systemd service file](#create-systemd-service-file)
    - [Service configuration](#service-configuration)
    - [Enable and start service](#enable-and-start-service)
    - [Verify service is running correctly](#verify-service-is-running-correctly)
    - [View logs](#view-logs)
  - [Public Access](#public-access)
    - [Tailscale Setup (Private Network Access)](#tailscale-setup-private-network-access)
    - [Firewall Configuration (Tailscale Focus)](#firewall-configuration-tailscale-focus)
    - [Reverse Proxy (Not Required for Tailscale)](#reverse-proxy-not-required-for-tailscale)
      - [A word on https](#a-word-on-https)
      - [Reverse Proxy with Nginx (Future)](#reverse-proxy-with-nginx-future)
      - [Install Nginx](#install-nginx)
      - [Configure Nginx](#configure-nginx)
      - [Enable site and get SSL](#enable-site-and-get-ssl)
      - [Verify nginx setup](#verify-nginx-setup)
    - [Deployment Setup on GitHub Pages (Frontend for Tailscale)](#deployment-setup-on-github-pages-frontend-for-tailscale)
  - [Troubleshooting](#troubleshooting)
    - [Issue: Model fails to load](#issue-model-fails-to-load)
    - [Issue: Out of memory errors](#issue-out-of-memory-errors)
    - [Issue: Slow generation (\<5 tokens/sec)](#issue-slow-generation-5-tokenssec)
    - [Issue: Weaviate initialization fails](#issue-weaviate-initialization-fails)
    - [Issue: Port 5000 already in use](#issue-port-5000-already-in-use)
  - [Monitoring and Maintenance](#monitoring-and-maintenance)
    - [View real-time logs](#view-real-time-logs)
    - [Monitor resource usage](#monitor-resource-usage)
    - [Backup important data](#backup-important-data)
  - [Migrating Back to Replit or Cloud](#migrating-back-to-replit-or-cloud)
  - [Performance Comparison](#performance-comparison)
  - [Next Steps](#next-steps)

## Overview

This guide explains how to self-host BrandonBot on your own Debian 13 system, which will eliminate the CPU resource contention issues present in Replit's shared development environment. To keep the deployment free, it uses a the github.io free static domain with redirection to a TailScale private url.

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
- **Python**: 3.12 or higher (included in Debian 13)

### Recommended Requirements
- **CPU**: 6+ cores, 3.0+ GHz
- **RAM**: 8GB+ available
- **Storage**: SSD for faster model loading

On CustomJacob:
```
> free -h
               total        used        free      shared  buff/cache   available
Mem:            31Gi       9.7Gi        20Gi        51Mi       1.2Gi        21Gi
Swap:           24Gi       6.5Gi        17Gi
> lscpu | grep -E "Model name|CPU MHz|Core"
Model name:                              Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz
Core(s) per socket:                      8
> lsblk -o NAME,SIZE,TYPE,MODEL,ROTA
NAME          SIZE TYPE MODEL                ROTA
sda             0B disk USB HS-CF Card          0
sdb             0B disk USB HS-xD/SM            0
sdc             0B disk USB HS-MS Card          0
sdd             0B disk USB HS-SD Card          0
nvme1n1     476.9G disk INTEL HBRPEKNX0202A     0
├─nvme1n1p1 452.8G part                         0
├─nvme1n1p2     1K part                         0
└─nvme1n1p5  24.2G part                         0
nvme0n1      27.3G disk INTEL HBRPEKNX0202AO    0
└─nvme0n1p1  27.2G part                         0

```

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
├── weaviate_data/        # Pre-computed embeddings (OPTIONAL - speeds up first start)
├── frontend/             # HTML/CSS/JS (REQUIRED if using web UI)
└── .env                  # Environment variable template (if exists)
```

**Download methods**:
- Via Replit's download feature (right-click folders)
- Using `rsync` if you have SSH access
- Via Replit's deployment export

### 3. Set Up Python Environment
Python3.12 is the expected version, as it is common to Replit and Debian13.
```bash
cd backend

# Use the explicit python3.12 executable for a robust setup.
# If the python3.12 command is not found, use the generic python3 command.
python3.12 -m venv .venv_brandonbot

# Activate the environment
source venv/bin/activate

# Verify the version is 3.12.x
python --version
```

### 4. Install Python Dependencies

```bash
pip install --upgrade pip

# Get the CPU-only version of Torch, which is much smaller than the CUDA version:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install gunicorn 		# This is for production and may be missing from requirements.txt used in development
```
**Torch** can be HUGE if you get the CUDA version. If that is a problem, the command to get the CPU-only version is `pip install torch --extra-index-url https://download.pytorch.org/whl/cpu`

**Expected installation time**: 5-10 minutes (downloads PyTorch CPU, ONNX Runtime, transformers, sentence-transformers, etc.)

### 5. Download the local LLM and SLM models
BrandonBot uses 4 specialized SLM (Small Language Models) for the 6-safeguard validation pipeline and 1 LLM for validation to be the judge and user persona:

| Model | Purpose | Size |
|-------|---------|------|
| ME2-BERT | Ethics checking (Moral Foundations) | ~420MB |
| MS-MARCO Cross-Encoder | Intent/response alignment | ~120MB |
| DeBERTa-PII | PII detection | ~550MB |
| BERT-tiny | Confidence verification | ~15MB |

**Total: ~1.1GB disk space**

```bash
# Download all safeguard models
python download_models.py
```

**Expected output**:
```
============================================================
BrandonBot SLM Model Downloader
============================================================

Checking dependencies...
  torch: 2.x.x
  transformers: 4.x.x
  sentence-transformers: 2.x.x
  huggingface-hub: installed

Cache directory: /home/user/.cache/huggingface

[ETHICS] ME2-BERT (Ethics)
  Downloading tokenizer...
  Downloading model weights...
  Status: OK

[INTENT] MS-MARCO Cross-Encoder (Intent)
  Downloading cross-encoder...
  Status: OK

[PII] DeBERTa-PII (PII Detection)
  Downloading tokenizer...
  Downloading model weights...
  Status: OK

[CONFIDENCE] BERT-tiny (Confidence)
  Downloading tokenizer...
  Downloading model weights...
  Status: OK

============================================================
Summary
============================================================
Models ready: 4/4
Cache size: 1105.2MB

All models ready!
```

### 6. Verify Safeguard Models (Optional)

Run the smoke test suite to confirm all safeguards are working:

```bash
python -m pytest tests/test_ov_*.py tests/test_pq.py -v
```

**Expected**: 140+ tests pass (all 6 safeguards operational)

### 7. Configure Environment Variables

Create a `.env` file in the `backend/` directory and put all the secrets here:

```bash
# Optional configurations
DATABASE_PATH=./data/brandonbot.db
WEAVIATE_DATA_PATH=../weaviate_data
LOG_LEVEL=INFO

# For commercial API migration (if using)
# OPENAI_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here
```

### 8. Initialize Database and Weaviate

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
This concludes the local setup instructions.

## Running the Server
Now that the program is setup, we need to deploy it.

### Production Mode (recommended)
For production, we use **Gunicorn** as a process manager with its **UvicornWorker** class to ensure stability and reliable service management (like graceful restarts).

```bash
cd backend
source .venv_brandonbot/bin/activate
gunicorn main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:5000
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
gunicorn main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:5000
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
sudo emacs /etc/systemd/system/brandonbot.service
```

### Service configuration
```ini
[Unit]
Description=BrandonBot AI Chatbot
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/cana/cana/BrandonBot.git/backend
Environment="PATH=/home/cana/cana/BrandonBot.git/backend/.venv_brandonbot/bin"
EnvironmentFile=/home/cana/cana/BrandonBot.git/backend/.env
# Optional: For API-based inference (OpenAI/Gemini)
# Environment="OPENAI_API_KEY=your_key_here"
# Environment="GOOGLE_API_KEY=your_key_here"
ExecStart=gunicorn main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:5000

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

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

## Public Access
Solution to keep everything free. The public-facing address is going to be canaai.github.io/brandonbot. Going there simply redirects to a tailspace url, like debian13.<random>.ts.net, which provides https security and the other minimal services needed.

### Tailscale Setup (Private Network Access)

Tailscale provides a secure, zero-config VPN (a Tailnet) allowing your local frontend (or any authorized device) to securely connect to your self-hosted backend.

1.  **Install Tailscale on the Debian Host:**
    Follow the official Debian installation instructions:
    ```bash
    # 1. Add Tailscale signing key and repository
    curl -fsSL [https://pkgs.tailscale.com/stable/debian/bookworm.gpg](https://pkgs.tailscale.com/stable/debian/bookworm.gpg) | sudo gpg --dearmor -o /usr/share/keyrings/tailscale-archive-keyring.gpg
    curl -fsSL [https://pkgs.tailscale.com/stable/debian/bookworm.list](https://pkgs.tailscale.com/stable/debian/bookworm.list) | sudo tee /etc/apt/sources.list.d/tailscale.list

    # 2. Update and Install
    sudo apt update
    sudo apt install tailscale
    ```

2.  **Connect the Host to your Tailnet:**
    ```bash
    sudo tailscale up
    ```
    This command will provide a unique URL. Copy and paste this URL into your web browser and log in with your account to authorize the machine.

3. **Find the Server's Full Domain Name (FQDN):**
   
    For a stable, memorable URL, use the Magic DNS name. This name is composed of your server's hostname and your private Tailnet's domain.

    Access the Tailscale Admin Console: Go to https://login.tailscale.com/admin/.

    Navigate to the Machines page.

    Locate your server (it will likely be named debian13).

    The Full Domain Name will be displayed on the machine's detail page.

Example FQDN Format: debian13.yak-bebop.ts.net

This is the address you will use for the redirect.

**Next Step:** Once the service is running, you will access it using this Tailscale IP and the application port, e.g., `http://debian13.yak-bebop.ts.net:5000`.
### Firewall Configuration (Tailscale Focus)

**Crucial Note:** Since your service is accessed via **Tailscale**, you **DO NOT** need to open port 5000 to the public internet (WAN). The service should only be accessible locally or via the private Tailnet IP.

If you are using **UFW** (recommended on Debian/Ubuntu):
```bash
# Allow SSH access
sudo ufw allow ssh

# Tailscale runs on its own port and will handle security.
# To ensure the host can access the app locally:
sudo ufw allow from 127.0.0.1 to any port 5000

# Enable UFW
sudo ufw enable
```
---
### Reverse Proxy (Not Required for Tailscale)

Since the public site `canaai.github.io/brandonbot` will redirect/proxy to your private **Tailscale IP** (e.g., `http://100.x.x.x:5000` or `http://debian13.yak-bebop.ts.net:5000`), you **DO NOT** need to install a reverse proxy like Nginx or configure Certbot on the Debian server.

The connection will be:

**User Browser** $\rightarrow$ **canaai.github.io/brandonbot** $\rightarrow$ **Tailscale Network** $\rightarrow$ **Your Debian Server (Gunicorn on 127.0.0.1:5000)**

#### A word on https
We use HTTP for the application port (:5000) because of a technical separation of duties on the self-hosted server:
1. Tailscale's Security: Encryption and Identity

    The main security advantage of Tailscale is not that it automatically gives your app an HTTPS certificate. The main advantages are:
   - Network Encryption: All traffic between your client device and the Debian server is already encrypted by Tailscale's WireGuard protocol. It doesn't matter if your app uses HTTP or HTTPS; the network traffic itself is secured and invisible to anyone outside your Tailnet.
   - Access Control: Access is restricted to only the users you've authorized on your Tailscale account. No one from the public internet can even see your server's IP or FQDN.

2. The SSL Error Explained (The Application Layer)

    If you try to access it at **https://**`debian13.yak-bebop.ts.net:5000`, you will see the SSL_ERROR_RX_RECORD_TOO_LONG error.

    HTTPS (HTTP Secure) requires the application server to handle TLS/SSL encryption at the application layer.

    Your application (Gunicorn/Uvicorn running your FastAPI app) is only configured to speak plain HTTP on port 5000. It doesn't have an SSL certificate and doesn't know how to decrypt HTTPS requests.

    When your browser sends an encrypted HTTPS request, your application receives garbage data and crashes the connection with the error you saw.

3. How to Use HTTPS with Tailscale (If You Wanted To)

    If you absolutely needed the browser to display the green padlock (HTTPS), you would have two options, both of which require more setup than just running the app:

    |Option | Method | Server Setup Required |
    |-------|--------|-----------------------|
    | A | Tailscale | Funnel/Serve (Recommended) | Use the tailscale serve or tailscale funnel command on your Debian machine. Tailscale handles getting a .ts.net certificate and acts as a reverse proxy, translating HTTPS (on port 443) down to your app's HTTP (on port 5000).
    | B | Traditional Reverse Proxy	Install Nginx and use Tailscale's tailscale cert command to obtain certificate files. |	You would need to configure Nginx to listen on port 443, load the Tailscale certificate files, and proxy the request to your application on port 5000. 

Since the network is already secured by Tailscale, using the simple http://...:5000 is the easiest and most effective way to run your self-hosted application behind the Tailnet. Your connection is secure and private even without the "https" prefix.

#### Reverse Proxy with Nginx (Future)
If you decide to expose the server publicly later using a domain, you would re-introduce the Nginx/Certbot configuration.

#### Install Nginx
```bash
sudo apt install nginx certbot python3-certbot-nginx
```

#### Configure Nginx
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

#### Enable site and get SSL
```bash
sudo ln -s /etc/nginx/sites-available/brandonbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
sudo certbot --nginx -d your-domain.com
```

#### Verify nginx setup
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
### Deployment Setup on GitHub Pages (Frontend for Tailscale)

To connect your public canaai.github.io/brandonbot URL to your private Tailscale server, you need to create a simple HTML file in the frontend repository that redirects to the server's private Tailscale IP.

1. Update the Frontend Repository
This assumes you have a separate GitHub repository for your frontend that is published via GitHub Pages at the URL: https://canaai.github.io/brandonbot/ or, in this case, canaai.github.io with a a brandonbot subdirectory, containing a redirect index.html.
```html
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=https://debian13.tail4cde2b.ts.net" />
    <link rel="canonical" href="https://debian13.tail4cde2b.ts.net" />
    <title>Redirecting to BrandonBot...</title>
</head>
<body>
    <p>If you are not redirected automatically, follow this <a href="https://debian13.tail4cde2b.ts.net">link to BrandonBot</a>.</p>
</body>
</html>
```
Get the Server IP: On your Debian server, get the private Tailscale IP (e.g., 100.x.x.x):
```Bash
tailscale ip -4
#Example output: 100.10.10.10
```

Create/Edit index.html: In your frontend repository (the one hosted by GitHub Pages), create or update the index.html file inside the brandonbot/ directory (or wherever your application entry point is).

The file should contain a meta refresh tag that redirects the user's browser to the Tailscale IP and port 5000.

Example index.html content (Use your actual IP):
```HTML
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="0; url=http://100.10.10.10:5000">
        <title>Loading BrandonBot Self-Hosted Instance</title>
        <style>
            body { font-family: sans-serif; text-align: center; padding-top: 50px; }
        </style>
    </head>
    <body>
        <h1>Connecting to BrandonBot...</h1>
        <p>If you are not redirected, please ensure you are logged into your <a href="https://tailscale.com/download" target="_blank">Tailscale client</a> and visit <a href="http://100.10.10.10:5000">http://100.10.10.10:5000</a> directly.</p>
    </body>
    </html>
```
Commit and Push: Commit this change to your frontend repository's main branch (or gh-pages branch, depending on your GitHub Pages configuration) and push it to GitHub.

2. User Experience Flow

When a user visits https://canaai.github.io/brandonbot/:

1. GitHub Pages loads the index.html.
2. The <meta http-equiv="refresh" ...> tag immediately tells the browser to redirect to the Tailscale url.
3. If the user is logged into the same Tailnet (Tailscale must be running on their machine): The connection is successful and the app loads.
4. If the user is NOT logged into the same Tailnet: The connection will fail, and the user will see the message directing them to check their Tailscale client.

This approach effectively uses your public GitHub URL as a permanent, easy-to-remember entry point that points to the secure, but private, backend IP.

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
