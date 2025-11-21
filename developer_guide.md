# BrandonBot - Git Setup & Deployment Guide

This document has a general flow:
Managing Github → Running (either in Replit or locally as a GitHub clone) → Test scripts → Open to testers & refine.

## Table of Contents
- [Part 1: Push to GitHub](#part-1-push-to-github)
- [Part 2: Running in Replit](#part-2-running-in-replit)
- [Part 3: Clone and Run Locally on Your Laptop](#part-3-clone-and-run-locally-on-your-laptop)
- [Part 4: Running Tests and Grading Responses](#part-4-running-tests-and-grading-responses)
- [Part 5: Deploy to Web (Low-Cost Options)](#part-5-deploy-to-web-low-cost-options)
- [Part 6: Ongoing Development Workflow](#part-6-ongoing-development-workflow)

## Current Status ✅
- Git repository has been reinitialized (you ran: `rm -fr .git`, `git init`, `git add .`, `git commit`)
- BrandonBot is running on Replit
- MarketGurus collection now has 3,481 marketing wisdom chunks (Hopkins + Caples)
- All core features working: Bible verses, callback detection, prompt formatting

---

## Part 1: Push to GitHub

### Step 1: Verify Your Current Git Status
```bash
git status
git log --oneline -5
```
Check that the log looks correct, reflecting recent actions.

### Step 2: Add GitHub Remote
```bash
git remote add origin https://github.com/jdstraye/BrandonBot.git
git remote -v
```
You should see:
```
gitsafe-backup  git://gitsafe:5418/backup.git (fetch)
gitsafe-backup  git://gitsafe:5418/backup.git (push)
origin  git@github.com:jdstraye/BrandonBot.git (fetch)
origin  git@github.com:jdstraye/BrandonBot.git (push)
```
The gitsafe-backup remotes are on a local Replit server and used by Replit as an emergency backup and restore.
The origin remote (git@github.com) is for your GitHub repos.

### Step 3: Remove Large Files from Git History

Before pushing, check for large files that might exceed GitHub's limits:

```bash
# Find largest objects in Git history
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  sed -n 's/^blob //p' | \
  sort -k2 -nr | \
  head -n 20 | \
  numfmt --field=2 --to=iec-i --suffix=B
```

If you see files over 100MB (or a total pack size approaching 2GB), you need to remove them from history:

```bash
# Install git-filter-repo (if not available, use filter-branch alternative below)
pip install git-filter-repo

# Remove large files (example: .onnx.data files)
git filter-repo --path 'backend/phi3_model/*.onnx.data' --invert-paths

# Alternative using git filter-branch (slower but works everywhere)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch backend/phi3_model/*.onnx.data" \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Verify the size is now manageable
git count-objects -vH
```

Add large files to `.gitignore` to prevent re-adding:
```bash
echo "backend/phi3_model/*.onnx.data" >> .gitignore
echo "backend/phi3_model/*.onnx" >> .gitignore
git add .gitignore
git commit -m "Add large model files to gitignore"
```

### Step 4: Push to GitHub
```bash
# First push requires -u flag to set upstream tracking
git push -u origin main
```

**If you get authentication errors:**
- GitHub no longer accepts password authentication
- You'll need to use a Personal Access Token (PAT) or SSH key

**Option A: SSH Key (Recommended)**

**1) Check for pre-existing keys**
```bash
ls -l ~/.ssh
```
Example output if no keys exist:
```
ls: cannot access '/home/runner/.ssh': No such file or directory
```

**2) Generate new SSH Keypairs**

Generate separate keypairs for Replit and GitHub:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/replit -q -N ""
ssh-keygen -t ed25519 -f ~/.ssh/github -q -N ""
```

The `ed25519` algorithm is a modern and secure elliptic-curve cryptography standard that offers better security than older RSA with shorter key lengths.

**3) Add the Public Keys to Your Accounts**

Display your public keys:
```bash
for account in replit github; do
    echo "=== ${account} public key ==="
    cat ~/.ssh/${account}.pub
    echo ""
done
```

**For Replit:**
1. Go to Replit Account Settings
2. Click on "SSH keys"
3. Select "Add SSH Key"
4. Paste the Replit public key (ending in `.pub`)

**For GitHub:**
1. Copy the GitHub public key output
2. Go to: https://github.com/settings/keys
3. Click "New SSH key"
4. Give it a title (e.g., "Replit SSH Key")
5. Paste the public key
6. Click "Add SSH key"

**4) Configure SSH**

Create or edit your SSH config file:
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/config && chmod 600 ~/.ssh/config
```

Add the configuration:
```bash
cat >> ~/.ssh/config << 'EOF'
Host *.replit.dev
    Port 22
    IdentityFile ~/.ssh/replit
    StrictHostKeyChecking accept-new

Host github.com
    IdentityFile ~/.ssh/github
    StrictHostKeyChecking accept-new
EOF
```

**5) Test the SSH Connection**

Test your GitHub connection:
```bash
ssh -T git@github.com
```

Expected output:
```
Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.
```

If successful, update your remote URL to use SSH and push:
```bash
git remote set-url origin git@github.com:jdstraye/BrandonBot.git
git push -u origin main
```

**Option B: Personal Access Tokens (PAT) (not recommended)**

1. Generate a Personal Access Token: Go to GitHub → Settings → Developer settings → Personal access tokens → Generate new token
2. Set Required Scopes: Choose `repo` for full control of private repositories
3. Copy the Token: Save it securely (you won't see it again)
4. Use the Token: When Git prompts for credentials, use your GitHub username and the PAT as the password

### Step 5: Verify Push Success
Go to https://github.com/jdstraye/BrandonBot and verify you see all your files.

---

## Part 2: Running in Replit

Replit provides a cloud-based development environment that's perfect for running BrandonBot without needing to set up local dependencies.

### Understanding Replit Tiers

**Free Tier (Starter)**
- **Cost**: $0/month
- **Resources**: 0.5 vCPU, 512MB RAM, 512MB disk
- **Limitations**:
  - Repls sleep after inactivity
  - Limited compute power
  - No always-on capability
  - Limited to public Repls
- **Best for**: Development, testing, demos

**Core Tier**
- **Cost**: $20/month
- **Resources**: 2 vCPU, 4GB RAM, 20GB disk
- **Benefits**:
  - Always-on Repls (deployments)
  - Private Repls
  - More compute power
  - Custom domains
- **Best for**: Production use with moderate traffic

### Setting Up BrandonBot on Replit (Free Tier)

#### Step 1: Create a New Repl

1. Go to https://replit.com
2. Sign up or log in
3. Click "Create Repl"
4. Choose "Import from GitHub"
5. Paste your repository URL: `https://github.com/jdstraye/BrandonBot.git`
6. Click "Import from GitHub"

Alternatively, if you already have a Repl:
1. Open the Repl
2. In the Shell, run:
   ```bash
   git clone https://github.com/jdstraye/BrandonBot.git .
   ```

#### Step 2: Configure the Repl

Replit should auto-detect Python. Verify the configuration:

1. Click on the `.replit` file in the file tree (create it if it doesn't exist)
2. Add this configuration:

```toml
run = "cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 5000"
language = "python3"

[nix]
channel = "stable-23_11"

[deployment]
run = ["sh", "-c", "cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 5000"]
deploymentTarget = "cloudrun"
```

#### Step 3: Install Dependencies

In the Shell tab, run:
```bash
pip install fastapi uvicorn sentence-transformers weaviate-client pypdf python-docx duckduckgo-search onnxruntime onnxruntime-genai
```

Or create a `requirements.txt` file in the root:
```txt
fastapi==0.104.1
uvicorn==0.24.0
sentence-transformers==2.2.2
weaviate-client==4.9.3
pypdf==3.17.1
python-docx==1.1.0
duckduckgo-search==4.1.1
onnxruntime==1.16.3
onnxruntime-genai==0.5.3
```

Then install with:
```bash
pip install -r requirements.txt
```

#### Step 4: Handle Large Model Files

The Phi-3 model files are too large to store in Git. You have two options:

**Option A: Download on first run (Recommended for Free Tier)**

Create a `download_phi3_model.py` script that downloads the model only if it doesn't exist:

```python
import os
import urllib.request

MODEL_DIR = "backend/phi3_model"
MODEL_URL = "https://your-storage-url/phi3-model.tar.gz"  # Use your own hosting

if not os.path.exists(f"{MODEL_DIR}/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx"):
    print("Downloading Phi-3 model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, f"{MODEL_DIR}/model.tar.gz")
    # Extract and cleanup
    os.system(f"tar -xzf {MODEL_DIR}/model.tar.gz -C {MODEL_DIR}")
    os.remove(f"{MODEL_DIR}/model.tar.gz")
    print("Model downloaded successfully!")
```

**Option B: Use Replit Storage (Better for persistence)**

Upload the model files to Replit Storage:
1. In your Repl, go to the "Storage" tab
2. Upload your model files
3. Access them in your code via the storage path

#### Step 5: Run BrandonBot

Click the green "Run" button at the top of the Repl, or in the Shell:
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 5000
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5000
```

#### Step 6: Access Your Application

Replit provides a webview. Click on the "Webview" tab or look for the URL at the top of the Repl window. It will look like:
```
https://brandonbot.yourusername.repl.co
```

Test the API:
```bash
curl -X POST https://brandonbot.yourusername.repl.co/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is your position on immigration?",
    "user_id": "test_user",
    "consent_given": false
  }'
```

### Managing Free Tier Limitations

**Preventing Sleep:**
- Free tier Repls sleep after inactivity
- Use a service like UptimeRobot (https://uptimerobot.com - free) to ping your Repl every 5 minutes
- Set up a monitor with your Repl URL: `https://brandonbot.yourusername.repl.co/health`

**Optimizing for Limited Resources:**
```python
# In your main.py, add memory-efficient settings
import gc

@app.on_event("startup")
async def startup_event():
    # Force garbage collection on startup
    gc.collect()
    
@app.middleware("http")
async def cleanup_middleware(request, call_next):
    response = await call_next(request)
    gc.collect()  # Clean up after each request
    return response
```

**Monitoring Resource Usage:**
```bash
# In Shell, check memory usage
free -h

# Check disk usage
df -h
```

### Upgrading to Core Tier (Optional)

When ready for production:
1. Click "Upgrade" in your Repl
2. Select "Core" plan ($20/month)
3. Enable "Always On" for your deployment
4. Configure custom domain if desired

---

## Part 3: Clone and Run Locally on Your Laptop

### Prerequisites
- **OS**: Linux or macOS (Weaviate embedded doesn't support Windows natively)
  - Windows users: Use WSL2 (Windows Subsystem for Linux)
- **Python**: 3.11 or higher
- **RAM**: At least 4GB available (Phi-3 model uses ~2GB)
- **Disk Space**: At least 5GB free (for model + dependencies)

### Step 1: Clone Repository
```bash
cd ~/Projects  # Or wherever you keep your code
git clone https://github.com/jdstraye/BrandonBot.git
cd BrandonBot
```

### Step 2: Set Up Python Environment
```bash
# Create virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# On Windows with WSL: source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies
```bash
# Install Python packages
pip install fastapi uvicorn sentence-transformers weaviate-client pypdf python-docx duckduckgo-search onnxruntime onnxruntime-genai

# Or if you have requirements.txt:
pip install -r requirements.txt
```

**Create requirements.txt** (if it doesn't exist):
```txt
fastapi==0.104.1
uvicorn==0.24.0
sentence-transformers==2.2.2
weaviate-client==4.9.3
pypdf==3.17.1
python-docx==1.1.0
duckduckgo-search==4.1.1
onnxruntime==1.16.3
onnxruntime-genai==0.5.3
```

### Step 4: Download Phi-3 Model (~2GB, one-time)
```bash
python download_phi3_model.py
```

This downloads the ONNX INT4 quantized model to `backend/phi3_model/`

### Step 5: Verify Data Files
```bash
# Check that documents exist
ls -la documents/brandon_platform/
ls -la documents/party_platforms/
ls -la documents/market_gurus/

# Check Weaviate database exists (created on first run)
ls -la weaviate_data/
```

**Note**: The `weaviate_data/` directory contains your vector database. If it doesn't exist, it will be created automatically when you first run the server.

### Step 6: Start BrandonBot
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 5000
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5000
```

### Step 7: Test It
Open browser to: http://localhost:5000

Or test via curl:
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is your position on immigration?",
    "user_id": "test_user",
    "consent_given": false
  }'
```

---

## Part 4: Running Tests and Grading Responses

### Create Test Script

Save this as `test_brandonbot.py` in your project root:

```python
#!/usr/bin/env python3
"""
BrandonBot Test Suite
Sends test questions, captures responses, and grades them
"""
import requests
import json
import time
from datetime import datetime
from typing import Dict, List

# Change this if testing a different URL
BASE_URL = "http://localhost:5000"
# For Replit: BASE_URL = "https://brandonbot.yourusername.repl.co"

TEST_QUESTIONS = [
    {
        "query": "What is your position on immigration?",
        "expected_type": "policy",
        "should_have_bible": False,
        "min_confidence": 0.4
    },
    {
        "query": "What does the Bible say about helping the poor?",
        "expected_type": "truth-seeking",
        "should_have_bible": True,
        "min_confidence": 0.3
    },
    {
        "query": "What is your tax policy?",
        "expected_type": "policy",
        "should_have_bible": False,
        "min_confidence": 0.4
    },
    {
        "query": "How do you feel about government spending?",
        "expected_type": "policy",
        "should_have_bible": False,
        "min_confidence": 0.3
    },
    {
        "query": "What does your faith say about abortion?",
        "expected_type": "truth-seeking",
        "should_have_bible": True,
        "min_confidence": 0.3
    }
]

def send_query(query: str) -> Dict:
    """Send a query to BrandonBot"""
    response = requests.post(
        f"{BASE_URL}/api/query",
        json={
            "query": query,
            "user_id": "test_user",
            "consent_given": False
        },
        timeout=120
    )
    return response.json()

def grade_response(response: Dict, expected: Dict) -> Dict:
    """Grade a response based on expectations"""
    score = 0
    max_score = 4
    issues = []
    
    # Check confidence
    confidence = response.get('confidence', 0)
    if confidence >= expected['min_confidence']:
        score += 1
    else:
        issues.append(f"Low confidence: {confidence:.2%} < {expected['min_confidence']:.2%}")
    
    # Check for Bible references
    response_text = response.get('response', '')
    bible_keywords = ['John', 'Proverbs', 'Ephesians', 'Matthew', 'Genesis', 
                     'Leviticus', 'Romans', 'verse', 'scripture', 'Bible', 
                     'Colossians', 'truth']
    has_bible = any(keyword in response_text for keyword in bible_keywords)
    
    if has_bible == expected['should_have_bible']:
        score += 1
    else:
        if expected['should_have_bible']:
            issues.append("Missing Bible reference")
        else:
            issues.append("Unexpected Bible reference in policy answer")
    
    # Check for prompt leakage
    leakage_patterns = ['Avoid political jargon', 'Solution=', 'system_prompt']
    has_leakage = any(pattern in response_text for pattern in leakage_patterns)
    if not has_leakage:
        score += 1
    else:
        issues.append("Prompt leakage detected")
    
    # Check response length (should be substantial)
    if len(response_text) > 100:
        score += 1
    else:
        issues.append(f"Response too short: {len(response_text)} chars")
    
    return {
        'score': score,
        'max_score': max_score,
        'percentage': (score / max_score) * 100,
        'issues': issues,
        'confidence': confidence,
        'response_length': len(response_text),
        'has_bible_reference': has_bible
    }

def run_tests():
    """Run all tests and generate report"""
    print("="*80)
    print("BrandonBot Test Suite")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    results = []
    total_score = 0
    total_max = 0
    
    for i, test in enumerate(TEST_QUESTIONS, 1):
        print(f"\nTest {i}/{len(TEST_QUESTIONS)}: {test['query'][:60]}...")
        
        start_time = time.time()
        try:
            response = send_query(test['query'])
            elapsed = time.time() - start_time
            
            grade = grade_response(response, test)
            total_score += grade['score']
            total_max += grade['max_score']
            
            print(f"  ⏱️  Response time: {elapsed:.1f}s")
            print(f"  📊 Score: {grade['score']}/{grade['max_score']} ({grade['percentage']:.0f}%)")
            print(f"  🎯 Confidence: {grade['confidence']:.2%}")
            print(f"  📝 Length: {grade['response_length']} chars")
            
            if grade['issues']:
                print(f"  ⚠️  Issues:")
                for issue in grade['issues']:
                    print(f"      - {issue}")
            else:
                print(f"  ✅ All checks passed!")
            
            results.append({
                'question': test['query'],
                'response': response.get('response', ''),
                'grade': grade,
                'elapsed_time': elapsed
            })
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({
                'question': test['query'],
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    overall_percentage = (total_score / total_max) * 100 if total_max > 0 else 0
    print(f"Overall Score: {total_score}/{total_max} ({overall_percentage:.1f}%)")
    
    avg_time = sum(r.get('elapsed_time', 0) for r in results) / len(results)
    print(f"Average Response Time: {avg_time:.1f}s")
    
    # Save detailed results
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'overall_score': total_score,
            'overall_max': total_max,
            'percentage': overall_percentage,
            'results': results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    run_tests()
```

### Run Tests

**Locally:**
```bash
chmod +x test_brandonbot.py
python test_brandonbot.py
```

**On Replit:**
```bash
# Make sure BASE_URL in the script points to your Repl URL
python test_brandonbot.py
```

---

## Part 5: Deploy to Web (Low-Cost Options)

### Option 1: Keep on Replit (Easiest)
**Cost**: $0-20/month depending on usage
- Free tier: Limited hours, sleeps when inactive
- Core plan: $20/month, always-on, custom domain

**Steps:**
1. Click "Deploy" button in Replit
2. Choose deployment type:
   - **Autoscale**: For standard web traffic (recommended)
   - **Reserved VM**: For always-on (if you need 24/7 availability)
3. Configure:
   - Build command: (leave empty)
   - Run command: `cd backend && uvicorn main:app --host 0.0.0.0 --port 5000`
4. Click "Deploy"

### Option 2: Railway.app (Simple PaaS)
**Cost**: ~$5-10/month (pay-as-you-go)
- 500 hours free per month
- $0.000231/GB-sec for memory

**Steps:**
1. Sign up at https://railway.app
2. Click "New Project" → "Deploy from GitHub"
3. Select your BrandonBot repository
4. Railway auto-detects Python
5. Add environment variables if needed
6. Click "Deploy"

### Option 3: DigitalOcean App Platform
**Cost**: $5/month (basic tier)
- Good performance for small apps

**Steps:**
1. Go to https://cloud.digitalocean.com/apps
2. Click "Create App"
3. Connect GitHub → Select BrandonBot
4. Configure:
   - Build: `pip install -r requirements.txt && python download_phi3_model.py`
   - Run: `cd backend && uvicorn main:app --host 0.0.0.0 --port 8080`
5. Choose $5/month tier
6. Deploy

### Option 4: Self-Host on VPS (Cheapest Long-Term)
**Cost**: $4-6/month
- Vultr, Hetzner, or DigitalOcean VPS

**Requirements:**
- 2GB RAM minimum (4GB recommended)
- Ubuntu 22.04 or Debian 12
- 20GB disk space

**Setup:**
```bash
# SSH into VPS
ssh root@your-vps-ip

# Update system
apt update && apt upgrade -y

# Install Python 3.11
apt install python3.11 python3.11-venv python3-pip nginx -y

# Clone repo
cd /opt
git clone https://github.com/jdstraye/BrandonBot.git
cd BrandonBot

# Set up virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download model
python download_phi3_model.py

# Create systemd service
cat > /etc/systemd/system/brandonbot.service << 'EOF'
[Unit]
Description=BrandonBot FastAPI Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/BrandonBot/backend
Environment="PATH=/opt/BrandonBot/venv/bin"
ExecStart=/opt/BrandonBot/venv/bin/uvicorn main:app --host 0.0.0.0 --port 5000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start service
systemctl daemon-reload
systemctl enable brandonbot
systemctl start brandonbot

# Configure Nginx as reverse proxy
cat > /etc/nginx/sites-available/brandonbot << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

ln -s /etc/nginx/sites-available/brandonbot /etc/nginx/sites-enabled/
systemctl restart nginx

# Install SSL certificate (free with Let's Encrypt)
apt install certbot python3-certbot-nginx -y
certbot --nginx -d your-domain.com
```

---

## Part 6: Ongoing Development Workflow

### Making Changes on Replit
```bash
# Make your changes in Replit
# Test them

# Commit and push
git add .
git commit -m "Description of changes"
git push origin main
```

### Pulling Changes to Laptop
```bash
cd ~/Projects/BrandonBot
git pull origin main

# Restart server
cd backend
uvicorn main:app --host 0.0.0.0 --port 5000
```

### Making Changes on Laptop
```bash
# Make changes
# Test them

# Commit and push
git add .
git commit -m "Description of changes"
git push origin main

# Pull into Replit (in Replit shell)
git pull origin main
```

---

## Troubleshooting

### Git Issues

**Large files preventing push:**
```bash
# Find large files in history
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  sed -n 's/^blob //p' | \
  sort -k2 -nr | \
  head -n 20

# Remove them using git filter-repo
pip install git-filter-repo
git filter-repo --path 'path/to/large/file' --invert-paths
```

**Merge conflicts:**
```bash
git status
# Manually resolve conflicts in files
git add .
git commit -m "Resolved merge conflicts"
git push origin main
```

**Force push (use with caution):**
```bash
# Only if you're sure and working alone
git push --force origin main
```

### Server Won't Start
```bash
# Check if port is in use
lsof -i :5000
# Kill process using port
kill -9 <PID>

# Check Python version
python --version  # Should be 3.11+

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Model Not Found
```bash
# Re-download Phi-3 model
python download_phi3_model.py

# Verify model exists
ls -la backend/phi3_model/
```

### Database Issues
```bash
# If Weaviate won't start, clear and rebuild
rm -rf weaviate_data/
python backend/ingest_documents.py documents/
```

### Replit-Specific Issues

**Repl keeps sleeping:**
- Use UptimeRobot to ping your Repl every 5 minutes
- Upgrade to Core tier for always-on

**Out of memory:**
```bash
# Check memory usage
free -h

# Restart the Repl
# Or upgrade to Core tier for more RAM
```

**Storage full:**
```bash
# Check disk usage
df -h

# Clear unnecessary files
rm -rf __pycache__
rm -rf .pytest_cache
rm -rf *.pyc
```

---

## Cost Comparison Summary

| Option | Monthly Cost | Pros | Cons |
|--------|-------------|------|------|
| Replit Free | $0 | Easy, no setup, good for development | Limited hours, sleeps when inactive |
| Replit