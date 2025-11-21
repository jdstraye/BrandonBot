# BrandonBot Project

## Overview
BrandonBot is a 100% open-source RAG-based AI chatbot for a political candidate. It's designed to answer questions about Brandon's political positions using a 5-tier confidence system with zero API costs.

## Project Type
**Replit-Native Open-Source Application** - This project runs entirely on Replit using embedded open-source components (no Docker, no API keys required). It includes:
- FastAPI backend
- Weaviate embedded vector database (no Docker!)
- Phi-3 Mini ONNX model (CPU-optimized, local inference)
- Sentence-Transformers embeddings (CPU-friendly)
- SQLite logging database
- Simple web UI

## Architecture
- **Backend**: FastAPI (Python) with 4-stage multi-dimensional RAG pipeline
- **Vector DB**: Weaviate Embedded (runs from Python, no Docker needed)
- **LLM**: Phi-3 Mini ONNX Runtime (3.8B params, INT4 quantized, ~2GB)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384-dim, CPU-optimized)
- **Storage**: SQLite for interaction logging
- **Frontend**: Vanilla HTML/CSS/JS (placeholder included)

### 4-Stage Response Pipeline
1. **Question Analysis** - Determine type (comparison, statistics, truth-seeking, policy, etc.), awareness level, emotional tone
2. **Multi-Source Retrieval** - Get Brandon's positions (RAG) + MarketGurus guidance + Bible verses + External search (when needed)
3. **Enhanced Response Generation** - Phi-3 with all contexts + proper framing based on question type
4. **Response Delivery** - Stay or break character appropriately

## Deployment Targets
1. **Replit** - Primary deployment platform (current)
2. **Local Debian/Ubuntu** - Can run with Python 3.11+
3. **Mac** - Compatible (Linux/macOS only for Weaviate embedded)
4. **Windows** - Not supported (Weaviate embedded is Linux/macOS only)

## Current State (November 20, 2025)
- ✅ 100% open-source architecture (no Docker, no API costs)
- ✅ Weaviate embedded mode integrated and fully functional
- ✅ Phi-3 Mini ONNX Runtime client with graceful degradation
- ✅ Sentence-transformers embeddings (CPU-only, 384-dim vectors)
- ✅ FastAPI backend with all endpoints operational
- ✅ Weaviate manager with 3-tier RAG + 1 guidance collection (all collections created)
- ✅ RAG pipeline with trust multiplier confidence scoring **TESTED & WORKING**
- ✅ **Phi-3 ONNX API Fixed**: Removed deprecated compute_logits() call - model now generates responses successfully
- ✅ **Token Limit Optimized**: Reduced max_output_tokens from 500 to 150 for 3x faster responses (~26-34 seconds)
- ✅ **Prompt Formatting Fixed**: Using Phi-3 chat template (<|system|><|user|><|assistant|>) prevents instruction leakage
- ✅ **Repetition Prevention**: Added repetition_penalty=1.2 to prevent repetitive output loops
- ✅ **Bible Verse Logic Fixed**: Truth-seeking questions now properly cite Scripture (only when explicitly asking about faith/morality)
- ✅ **Truth-Seeking Pattern Refinement**: Narrowed patterns to only match explicit faith/biblical/moral questions, not general policy questions
- ✅ **Chunking Strategy Optimized**: 128-char chunks with 40% overlap (51 chars) achieves 56.95% avg confidence (16% improvement over 512-char)
- ✅ **Lazy Loading Implementation**: Phi-3 model unloads after 5 min idle to free ~2GB RAM
- ✅ **LLM-Based Query Expansion**: Implemented but testing shows no improvement over dictionary expansion
- ✅ **DuckDuckGo Web Search**: Real search integration with domain trust scoring (2x boost for brandonsowers.com in search rankings)
- ✅ **BrandonSowers.com RAG Trust Multiplier**: External search results from brandonsowers.com get 0.8 confidence boost (safer than 1.0 in case of compromise)
- ✅ **Multi-dimensional analysis pipeline** - analyzes question type, awareness level, emotional tone
- ✅ **Callback Button**: Persistent callback button in UI for users to request personal contact
- ✅ **Callback Request Detection**: Auto-detects callback requests in chat messages (11 patterns: "call me back", "contact me", etc.)
- ✅ Vector search fully operational with trust-based weighting
- ✅ SQLite logging with opt-in consent
- ✅ Document ingestion scripts (PDF/DOCX/TXT) with configurable chunk sizes
- ✅ Replit workflow configured and running
- ✅ **MarketGurus collection populated with 14,058 chunks** - 6 comprehensive marketing resources:
  1. Gary Halbert "The Boron Letters" (25 chapters, 2,097 chunks)
  2. Claude Hopkins "Scientific Advertising" (classic public-domain book)
  3. John Caples "Tested Advertising Methods" (classic public-domain book)
  4. Eugene Schwartz "Lost Secrets of Breakthrough Advertising" (lecture transcript, 192 chunks)
  5. David Ogilvy "10 Commandments of Advertising" (principle summary)
  6. Robert Collier Letter Book principles (comprehensive summary)
  Plus: Schwartz's 5 Stages of Awareness framework (complete guide)
- ✅ **Bible verse collection (12 topics)** - immigration, stewardship, justice, truth, integrity, etc.
- ✅ **Character-breaking behavior** - stays/breaks character based on question type
- ✅ User uploaded documents to brandon_platform and party_platforms collections
- ⏳ **96-char chunk testing**: Deferred - user can manually run: `python backend/ingest_documents.py documents/ --chunk-size 96 --overlap 38`
- ⏳ **LLM expansion re-testing**: Needs validation with functioning Phi-3 on 512 and 96-char chunks
- ⏳ Frontend UI (functional chat interface, could use visual enhancements)
- ⏳ PreviousQA collection (awaiting historical Q&A data)

## Key Features
- **3-Tier Knowledge Base** (Weaviate Collections with Trust Multipliers):
  1. **BrandonPlatform** (1.0x trust) - Brandon's statements & platform (MOST TRUSTWORTHY)
  2. **PreviousQA** (1.0x trust) - Historical Q&A (VERY TRUSTWORTHY)
  3. **PartyPlatform** (0.6x trust) - Republican/RNC platforms (MODERATELY TRUSTWORTHY)

- **MarketGurus Collection**: Provides marketing wisdom (14,058 chunks from 6+ legendary copywriters) embedded in system prompts to guide HOW Brandon communicates, not WHAT he says. Includes:
  - **Gary Halbert**: The Boron Letters (prison letters to his son, 25 chapters on copywriting, life, business)
  - **Claude Hopkins**: Scientific Advertising (1923 classic, data-driven advertising principles)
  - **John Caples**: Tested Advertising Methods (proven headlines and techniques)
  - **Eugene Schwartz**: Lost Secrets lecture + 5 Stages of Awareness framework (prospect psychology)
  - **David Ogilvy**: 10 Commandments of Advertising (substance over style, research-driven)
  - **Robert Collier**: Letter Book principles (entering wedge, visualized desire, human motivators)
  
  Queried based on question analysis (awareness level, emotional tone) to retrieve relevant copywriting principles that match the situation.

- **Bible Verse Collection**: 12 topic-based collections (immigration, stewardship, justice, truth, integrity, compassion, family, authority, wealth, work, freedom, peace) for truth-seeking questions. Each verse includes reference, text, and application context. Only used for moral/spiritual questions, NOT trivial facts.

- **Web Search Integration**: MockWebSearchService for testing (ready for real search API). Provides external evidence for comparisons, statistics, and recent events. Returns structured SearchResult objects with full citations (footnote style with URLs).

- **Trust Multiplier System**: Confidence = Similarity × Trust Factor
  - Example: 0.7 similarity from BrandonPlatform = 0.7 × 1.0 = 0.7 confidence
  - Example: 0.7 similarity from PartyPlatform = 0.7 × 0.6 = 0.42 confidence
  - This reflects SOURCE TRUSTWORTHINESS, not just semantic similarity

- **InternetSources**: NOT a Weaviate collection - requires manual research/citation when knowledge base insufficient

- **Personality**: BrandonBot speaks AS Brandon (first person), encouraging direct engagement ("I believe", "my position", "feel free to ask me")

- **Marketing-Guided Communication**: Uses copywriting wisdom from 6 legendary copywriters (Halbert, Hopkins, Caples, Schwartz, Ogilvy, Collier) to guide communication style - direct, conversational, benefit-focused. Adapts to Schwartz's 5 levels of awareness (Unaware → Problem Aware → Solution Aware → Product Aware → Most Aware) and emotional tone analysis to match messaging to the prospect's current state.

- **Character-Breaking Behavior**:
  - **STAY in character**: High-confidence policy questions, statistics ("I had to use the internet to get latest data...")
  - **BREAK character**: Comparisons ("[Breaking character] This is challenging, let me get contact info..."), recent events, low-confidence questions
  
- **Question Type Detection**: Pattern-based analysis detects 6 types: comparison, statistics, truth-seeking, recent_event, policy, low_confidence. Each type gets appropriate response framing.

- **Callback System**: Offers personal callbacks for low-confidence answers (<0.5) and all external-search scenarios
- **Privacy-First**: Opt-in logging with ethical data collection
- **100% Open Source**: No paid APIs, no Docker, runs entirely on Replit server
- **CPU-Optimized**: INT4 quantized models, works on CPU-only environments
- **Zero Setup**: Weaviate embedded downloads binary automatically

## Important Notes
- This project **runs natively on Replit** - no Docker required!
- Weaviate embedded mode downloads the binary automatically on first run
- Phi-3 ONNX model (~2GB) must be downloaded once using `python download_phi3_model.py`
- Documents must be ingested using `python backend/ingest_documents.py documents/`
- System requirements: ~4GB RAM for model + database
- CPU-only inference (no GPU needed)

## Setup Instructions
1. **Download Phi-3 Model** (one-time, ~2GB):
   ```bash
   python download_phi3_model.py
   ```

2. **Prepare Documents** (organize in directories):
   ```
   documents/
     brandon_platform/      (Brandon's statements & platform)
     party_platforms/        (Republican, RNC platforms)
     market_gurus/           (Marketing/copywriting books)
     internet_sources/       (Research articles)
     previous_qa/            (Historical Q&A)
   ```

3. **Ingest Documents**:
   ```bash
   python backend/ingest_documents.py documents/
   ```

4. **Run Server** (automatic via Replit workflow)

5. **Access**: Click the webview in Replit or visit the public URL

## API Endpoints
- `GET /health` - Check system status
- `POST /api/query` - Ask BrandonBot a question
- `POST /api/consent` - Update logging consent  
- `POST /api/callback` - Request personal callback

## Future Enhancements
- Full chat UI frontend
- SMS integration via Twilio
- Advanced reranking for accuracy
- Analytics dashboard
- Voice/phone integration
- Streaming responses
