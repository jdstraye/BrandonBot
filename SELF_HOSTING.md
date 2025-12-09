# Self-Hosting BrandonBot

## Overview

This guide explains how to self-host BrandonBot on your own server with Python 3.12. The self-hosted version uses:
- **SLM Models**: Required safeguards for vagueness detection, frustration detection, ethics, PII, and confidence
- **Weaviate**: Embedded vector database for RAG
- **FEC Compliance**: Mandatory RAG-based compliance checking (system fails closed without it)
- **Ollama + Llama 3.2**: Optional LLM judge for validation

---

## System Requirements

### Minimum Requirements
- **OS**: Debian 13, Ubuntu 22.04+, or other Linux distribution
- **Python**: 3.12
- **RAM**: 8GB available (4GB for models, 4GB for system)
- **Storage**: 15GB free space

### For LLM Judge (Validation Only)
- **Ollama**: Required for running Llama 3.2 locally
- **Model**: llama3.2:3b (~2GB)

---

## Quick Start (Two Steps)

After cloning and setting up Python, the system is ready in two commands:

```bash
# Step 1: Download all required models
python download_models.py

# Step 2: Initialize databases and load FEC data
python ingest_all.py
```

That's it! The system is now ready for operations.

---

## Detailed Setup

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

### 2. Install Ollama (Optional, for Validation)

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

### 5. Download Models (REQUIRED)

This downloads all SLM safeguard models and optionally sets up Ollama:

```bash
python download_models.py
```

**What gets downloaded:**
| Model | Size | Purpose |
|-------|------|---------|
| all-MiniLM-L6-v2 | 90MB | Text embeddings for RAG |
| ms-marco-MiniLM | 120MB | Intent/vagueness scoring |
| j-hartmann/emotion-english-distilroberta-base | 320MB | Frustration detection |
| ME2-BERT | 420MB | Ethics classification |
| deberta-pii | 550MB | PII detection |
| bert-tiny | 15MB | Confidence verification |
| llama3.2:3b | 2GB | LLM judge (optional) |

**Options:**
```bash
# Download only SLM models (skip Ollama)
python download_models.py --slm-only

# Download only Ollama model
python download_models.py --ollama-only

# Verify existing models
python download_models.py --verify-only
```

### 6. Initialize Databases (REQUIRED)

This creates the SQLite database and loads FEC compliance data into Weaviate:

```bash
python ingest_all.py
```

**What gets initialized:**
- **SQLite** (`data/brandonbot.db`): User consent, interactions, callbacks, volunteers, compliance logs
- **Weaviate** (embedded): FEC prohibited phrases, Brandon platform, previous Q&A

**CRITICAL**: FEC compliance data is MANDATORY. The system will refuse to operate without it.

**Options:**
```bash
# With custom documents directory
python ingest_all.py documents/

# Skip database initialization
python ingest_all.py --skip-db

# Custom database path
python ingest_all.py --db-path /path/to/database.db
```

### 7. Configure Environment Variables (Optional)

Create a `.env` file in the `backend/` directory:

```bash
# Required for commercial LLM providers (Replit mode)
GOOGLE_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
NVIDIA_API_KEY=your_key_here

# Optional: SendGrid for email notifications
SENDGRID_API_KEY=your_key_here

# Database (defaults to data/brandonbot.db)
DATABASE_PATH=./data/brandonbot.db
```

### 8. Start the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

Access at: http://localhost:5000

---

## Fail-Closed Design

BrandonBot is designed to fail closed for safety:

### SLM Safeguards
- **Required**: All 6 SLM models must be loaded for the system to operate
- **Fail behavior**: System refuses to start if models don't load

### FEC Compliance (RAG)
- **Required**: FECProhibited collection must exist and be populated
- **Fail behavior**: System refuses to process responses without FEC RAG
- **No fallback**: Pattern matching is NOT used as a fallback

### Verification
```bash
# Verify models are loaded
python download_models.py --verify-only

# Verify Weaviate collections
python -c "
import asyncio
from weaviate_manager import WeaviateManager

async def check():
    wm = WeaviateManager()
    await wm.initialize()
    count = await wm.get_collection_count('FECProhibited')
    print(f'FECProhibited: {count} documents')
    await wm.close()

asyncio.run(check())
"
```

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
python3.12 -m validation.validator --phase <phase>
python validator.py 
```
\<phase\> can be:
>
- all - run all the tests available, ~10 PQ + OV + ~150 prompts
- pq - Run the PreQualifier gray box tests, which would include irritation and vagueness detection.
- ov - Run the Output Validation gray box tests, which would include citation verification, DOS attack, PII redaction, and responding to the intent of the query.
- full - Run the prompts but not the pq and ov gray box tests.

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

### FEC RAG errors
```bash
# Re-run ingestion to load FEC data
python ingest_all.py

# Verify FEC collection
python -c "
import asyncio
from weaviate_manager import WeaviateManager

async def check():
    wm = WeaviateManager()
    await wm.initialize()
    count = await wm.get_collection_count('FECProhibited')
    if count == 0:
        print('ERROR: FECProhibited is empty!')
    else:
        print(f'OK: {count} FEC documents loaded')
    await wm.close()

asyncio.run(check())
"
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

# SLM models require ~2GB RAM
# Llama 3.2:3b requires ~4GB RAM additional
# Close other applications or use smaller Llama model
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
│   ├── fec_compliance_checker.py # FEC compliance (fail-closed)
│   ├── slm_manager.py          # SLM safeguard models
│   ├── ov_slm_models.py        # Output validation models
│   ├── download_models.py      # Model downloader
│   ├── ingest_all.py           # Database & Weaviate setup
│   ├── requirements.txt        # Full dependencies
│   └── validation/
│       ├── validator.py        # Validation suite
│       └── debug.db            # Debug logs
├── data/
│   └── brandonbot.db           # SQLite database
├── documents/                  # Knowledge base source docs
│   ├── brandon_platform/       # Brandon's policy documents
│   ├── party_platforms/        # Party platform documents
│   ├── previous_qa/            # Verified Q&A pairs
│   ├── market_gurus/           # Marketing guidance
│   └── fec_prohibited/         # Additional FEC docs (optional)
└── frontend/                   # Web UI
```

---

## Database Schema

The SQLite database (`data/brandonbot.db`) includes:

| Table | Purpose |
|-------|---------|
| user_consent | User consent tracking |
| interactions | Query/response pairs |
| callback_requests | Callback requests |
| new_questions | Unique questions tracking |
| conversation_history | Full conversation turns |
| request_logs | Complete request logging |
| model_performance | Model performance metrics |
| volunteers | Volunteer registrations |
| donation_interests | Donation interest (FEC compliant) |
| compliance_log | Compliance audit trail |

---

## Environment Modes

| Mode | LLM Provider | SLM Models | FEC RAG | Use Case |
|------|-------------|------------|---------|----------|
| Replit | Commercial APIs | Required | Required | Development/Demo |
| Self-Hosted | Commercial APIs | Required | Required | Production |
| Fully Local | Ollama | Required | Required | Offline/Privacy |

---

## Next Steps

1. **Run validation** to test response quality
2. **Add your documents** to `documents/` directories
3. **Configure email notifications** via SendGrid
4. **Set up monitoring** (logs, health checks)
5. **Add SSL** via nginx + certbot
