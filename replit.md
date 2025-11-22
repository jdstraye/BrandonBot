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

## Current State (November 22, 2025 - Alpha r0v2)
- ✅ 100% open-source architecture (no Docker, no API costs)
- ✅ Weaviate embedded mode integrated and fully functional
- ✅ Phi-3 Mini ONNX Runtime client with graceful degradation
- ✅ Sentence-transformers embeddings (CPU-only, 384-dim vectors)
- ✅ FastAPI backend with all endpoints operational
- ✅ Weaviate manager with 3-tier RAG + 1 guidance collection (all collections created)
- ✅ **Retrieval-First Architecture (November 22, 2025)**:
  - **FIXED BRITTLENESS BUG**: Eliminated confidence-based branching (0.2, 0.5 thresholds) that caused false "I don't have information" responses
  - **Single Code Path**: All questions now follow: retrieve → build context → Phi-3 generation (no more separate low/medium/high confidence paths)
  - **Simpler Prompts**: Compressed response guidelines from ~800 chars to ~350 chars, saving ~200 tokens per query
  - **Essential Patterns Preserved**: Callback detection, dual-source enforcement, truth-seeking, comparison detection still functional
  - **Validated Fix**: Abortion debate question now presents Brandon's position (confidence 0.61) instead of incorrectly claiming "no information"
  - **60% Code Reduction**: Removed _generate_low_confidence_response and all branching logic
- ✅ **RAG Pipeline Refactored (Content vs Style Separation)** - MarketGuru no longer pollutes content search:
  - Content search: Only queries BrandonPlatform, PreviousQA, PartyPlatform for WHAT to answer
  - Style guidance: Queries MarketGurus separately based on question analysis for HOW to answer
  - Confidence guardrails: Style guidance only applied when content confidence >= 0.7
  - Template directives: Marketing strategy converted to HOW-to-answer rules (no raw MarketGuru text)
  - **VALIDATED**: Policy questions now cite BrandonPlatform (conf 0.52), comparisons cite PartyPlatform (conf 0.61)
- ✅ **Prompt Formatting Fixed**: Using Phi-3 chat template (`<|system|>...<|end|><|user|>...<|end|><|assistant|>`) prevents instruction leakage
- ✅ **Token Limit Increased**: max_length raised to 4096 tokens to accommodate longer contexts
- ✅ **Confidence Thresholds Lowered**: BrandonPlatform 0.8→0.45, PartyPlatform 0.6→0.35 (more results pass through)
- ✅ **Results Sorted by Confidence**: Top sources shown first regardless of collection order
- ✅ **Bible Verse Logic Fixed**: Truth-seeking questions now properly cite Scripture (only when explicitly asking about faith/morality)
- ✅ **Truth-Seeking Pattern Refinement**: Narrowed patterns to only match explicit faith/biblical/moral questions, not general policy questions
- ✅ **Smart Boundary-Aware Chunking**: Enhanced ingest_documents.py with intelligent boundary detection (sections > paragraphs > sentences > characters) and CLI args --chunk-size/--overlap
- ✅ **Smoke Test Suite Created**: Comprehensive A/B testing (bash + Python versions) for RAG validation, comparison tests, MarketGuru impact, web search, and chunking optimization
- ✅ **Baseline Performance Measured** (1000-char chunks): Confidence scores 0.34-0.61 range across policy questions, MarketGuru queries, and comparisons
- ✅ **Lazy Loading Implementation**: Phi-3 model unloads after 5 min idle to free ~2GB RAM
- ✅ **LLM-Based Query Expansion**: Implemented but testing shows no improvement over dictionary expansion
- ✅ **DuckDuckGo Web Search**: Real search integration with domain trust scoring (2x boost for brandonsowers.com in search rankings)
- ✅ **BrandonSowers.com RAG Trust Multiplier**: External search results from brandonsowers.com get 0.8 confidence boost (safer than 1.0 in case of compromise)
- ✅ **Multi-dimensional analysis pipeline** - analyzes question type, awareness level, emotional tone
- ✅ **Callback Button**: Persistent callback button in UI for users to request personal contact
- ✅ **Callback Request Detection**: Auto-detects callback requests in chat messages (14 patterns: "call me back", "contact me", etc.)
- ✅ Vector search fully operational with trust-based weighting
- ✅ SQLite logging with opt-in consent
- ✅ Document ingestion scripts (PDF/DOCX/TXT) with configurable chunk sizes and smart boundary detection
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
- ⏳ **Dual-Source Retrieval for Comparisons (RECOMMENDED NEXT STEP)**:
  - Current: Comparison questions show highest-confidence sources (may be Brandon-only or Party-only)
  - Goal: Ensure comparison questions always include at least one Brandon and one Party source
  - Implementation: Balanced query logic or explicit dual-source enforcement before Phi-3 generation
- ⏳ **Wire Auxiliary Subsystems to Question Analysis**:
  - Current: QuestionAnalyzer detects needs_external_search and needs_bible_verse but doesn't trigger services
  - Goal: Actually invoke web search when detected, retrieve Bible verses when needed
  - Implementation: Connect analysis_pipeline outputs to web_search_service and Bible collection queries
- ⏳ **Chunk Size Optimization**:
  - Current baseline (1000-char chunks): 0.30-0.61 confidence range  
  - Architect recommendation: Test 256-char chunks with 80-char overlap FIRST (not 128-char - too granular)
  - Test command: `python backend/ingest_documents.py documents/brandon_platform/ --chunk-size 256 --overlap 80`
  - After validating improvement, batch-ingest full corpus with Weaviate bulk API (~64 vectors/payload) + asyncio.Semaphore
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
