# BrandonBot - AI Political Chatbot with RAG

BrandonBot is an intelligent chatbot trained on Brandon's political positions and statements. It uses Retrieval-Augmented Generation (RAG) to provide accurate, confident answers based on a 4-tier knowledge base.

## Features

- **4-Tier Confidence System**: Prioritizes Brandon's direct statements, then Q&A history, party platforms, and internet sources
- **Smart Callback System**: Offers personal callbacks for low-confidence or complex questions
- **Full Interaction Logging**: Opt-in consent system for tracking and improving responses
- **SMS-like Interface**: Clean, simple chat UI for easy interaction
- **100% Open Source**: Uses Phi-3.5 AI model and Weaviate vector database - no paid APIs required
- **Cross-Platform**: Runs on Mac, Windows, and Linux via Docker

## System Requirements

- **macOS**: 10.15 or later, 4GB RAM minimum
- **Windows**: Windows 10/11, WSL2 enabled, 4GB RAM minimum
- **Linux**: Debian 12 or Ubuntu 20.04+, 4GB RAM minimum
- **Disk Space**: ~10GB for Docker images and AI models

## Quick Start

### For Mac Users (Customer Installation)

1. Download the BrandonBot package (ask your provider for the ZIP file)

2. Extract the ZIP file to your Desktop or preferred location

3. Open Terminal:
   - Press `Cmd + Space`
   - Type "Terminal" and press Enter

4. Navigate to the BrandonBot folder:
   ```bash
   cd ~/Desktop/BrandonBot
   ```

5. Run the installer:
   ```bash
   ./installers/install_mac.sh
   ```

6. Follow the on-screen instructions. If Docker Desktop isn't installed, the installer will guide you through installation.

7. Once complete, access BrandonBot at: **http://localhost:8000**

**To start BrandonBot later:**
```bash
cd ~/Desktop/BrandonBot
./start_brandonbot.sh
```

**To stop BrandonBot:**
```bash
docker compose down
```

### For Debian 12 (Developer Installation)

1. Clone or extract the BrandonBot package

2. Navigate to the directory:
   ```bash
   cd BrandonBot
   ```

3. Run the installer:
   ```bash
   ./installers/install_debian.sh
   ```

4. Access BrandonBot at: **http://localhost:8000**

### For Windows 11 (Testing Installation)

1. Extract the BrandonBot package to a folder (e.g., `C:\BrandonBot`)

2. Open PowerShell as Administrator:
   - Press `Win + X`
   - Select "Windows PowerShell (Admin)"

3. Navigate to the BrandonBot folder:
   ```powershell
   cd C:\BrandonBot
   ```

4. Run the installer:
   ```powershell
   .\installers\install_windows.ps1
   ```

5. Follow the on-screen instructions

6. Access BrandonBot at: **http://localhost:8000**

**To start BrandonBot later:**
```powershell
.\start_brandonbot.ps1
```

## Architecture

```
┌─────────────────┐
│   Frontend UI   │  (SMS-like chat interface)
│  (Port 8000)    │
└────────┬────────┘
         │
┌────────▼────────┐
│   FastAPI       │  (Backend API)
│   Backend       │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼────┐
│Weaviate│ │Ollama │  (Vector DB & AI Model)
│ (RAG)  │ │(Phi3.5│
└───┬────┘ └───────┘
    │
┌───▼────┐
│ SQLite │  (Interaction logging)
└────────┘
```

## Knowledge Base Collections

### 1. BrandonPlatform (Confidence Threshold: 0.8+)
- Brandon's own statements and speeches
- His synthesis of RNC and Independent platforms
- **Highest confidence** - direct from the source

### 2. PreviousQA (Confidence Threshold: 0.8+)
- Previously answered questions from town halls
- Documented Q&A sessions
- **Highest confidence** - Brandon's verified responses

### 3. PartyPlatform (Confidence Threshold: 0.6+)
- RNC platform
- Independent platform
- Arizona Republican platform
- **Medium confidence** - for comparison with Brandon's positions

### 4. InternetSources (Confidence Threshold: 0.4+)
- External information requiring citations
- **Lowest confidence** - always includes source attribution

## Adding Content

See [data_ingestion/README.md](data_ingestion/README.md) for detailed instructions on adding content to the knowledge base.

**Quick add example:**
```bash
cd data_ingestion
python ingest.py  # Loads sample data
python bulk_import.py your_data.json  # Bulk import custom data
```

## Privacy & Logging

BrandonBot implements ethical data collection:

1. **Opt-in Consent**: Users must explicitly consent to full logging
2. **Without Consent**: Only new questions are tracked (not user data or full conversations)
3. **With Consent**: Full interaction logging helps improve answer quality
4. **Callback Requests**: Contact information collected only when users request callbacks

View interaction stats:
```bash
curl http://localhost:8000/api/stats
```

## Troubleshooting

### Docker not starting
- **Mac**: Open Docker Desktop app, wait for whale icon in menu bar
- **Windows**: Check Docker Desktop is running in system tray
- **Linux**: Run `sudo systemctl start docker`

### Port 8000 already in use
```bash
docker compose down
# Wait 10 seconds
docker compose up -d
```

### Model download slow
The Phi-3.5 model is ~2GB. On first run, it may take 5-10 minutes to download depending on your internet connection.

### Containers won't start
```bash
# Check logs
docker compose logs

# Restart everything
docker compose down
docker compose up -d --build
```

### Chat not responding
1. Check all containers are running:
   ```bash
   docker compose ps
   ```

2. Check logs for errors:
   ```bash
   docker compose logs backend
   docker compose logs ollama
   docker compose logs weaviate
   ```

## Development

### Running locally for development

```bash
# Start all services
docker compose up -d

# Watch backend logs
docker compose logs -f backend

# Restart just the backend after code changes
docker compose restart backend
```

### Project Structure

```
BrandonBot/
├── backend/               # FastAPI application
│   ├── main.py           # API endpoints
│   ├── weaviate_manager.py  # Vector DB interface
│   ├── ollama_client.py  # LLM interface
│   ├── database.py       # SQLite logging
│   ├── rag_pipeline.py   # RAG logic
│   └── static/           # Frontend files
│       ├── index.html
│       ├── style.css
│       └── app.js
├── data_ingestion/       # Scripts to load content
│   ├── ingest.py         # Sample data loader
│   ├── bulk_import.py    # Bulk JSON importer
│   └── README.md         # Data ingestion guide
├── installers/           # Platform-specific installers
│   ├── install_mac.sh
│   ├── install_debian.sh
│   └── install_windows.ps1
├── docker-compose.yml    # Container orchestration
└── README.md            # This file
```

## Future Enhancements

This local version is designed for testing and development. Future enhancements include:

1. **Web Deployment**: Deploy to Replit or other cloud platforms
2. **SMS Integration**: Direct SMS responses via Twilio
3. **Advanced Reranking**: Improve answer accuracy with reranking models
4. **Multi-language Support**: Support for Spanish-speaking constituents
5. **Voice Interface**: Phone call integration
6. **Analytics Dashboard**: Visual analytics for most asked questions

## Support

For technical issues or questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review logs: `docker compose logs`
- Contact your system administrator

## License

Proprietary - All rights reserved.

Built for delivering AI products to SMBs.
