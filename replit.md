# BrandonBot Project

## Overview
BrandonBot is an open-source, LLM-first agentic chatbot for a political candidate, designed to answer questions about Brandon's political positions. The system uses multi-provider LLM support with automatic failover, trust-based knowledge separation, and comprehensive intent/escalation detection for optimal user experience.

## User Preferences
I prefer iterative development with clear, concise communication. When making changes, please explain the "why" behind them, not just the "what." I value performance and cost-efficiency. Do not make changes to the `replit.md` file without explicit instruction.

## System Architecture

### LLM-First Multi-Provider Architecture (Nov 26, 2025)
The system now uses an LLM-first agentic architecture with 7 commercial LLM providers:

**Provider Priority (highest to lowest):**
1. Nvidia NIM (90) - 5 models (Llama-4, DeepSeek-R1, Qwen)
2. Z.ai (85) - 3 models (GLM-4.5 variants)
3. Google Gemini (80) - 2 models (2.0-flash, 2.5-flash)
4. Mistral AI (60) - 3 models (small, large, pixtral)
5. Cohere (55) - 3 models (command-r variants)
6. HuggingFace (50) - 2 models (Llama-3.3, DeepSeek-V3)
7. Replicate (40) - 1 model (Kimi-K2)

**Key Features:**
- **Session-Sticky Model Selection**: One model per conversation, switches only on failure
- **Automatic Failover**: Rate limits/errors trigger switch to next provider
- **Unified Interface**: All providers normalized to OpenAI-compatible interface
- **Performance Tracking**: SQLite logs provider, model, latency, success rate, tokens

### Core Components

**Backend (FastAPI):**
- `llm_providers.py` - Multi-provider LLM manager with failover
- `agent_orchestrator.py` - LLM-first agentic pipeline with tool calling
- `intent_detector.py` - Intent detection and escalation detection
- `security.py` - Input sanitization and rate limiting
- `agent_tools.py` - Tool definitions and schemas
- `weaviate_manager.py` - Embedded vector database integration
- `database.py` - SQLite for interactions, callbacks, model performance

### Trust-Based Knowledge Separation

**Authoritative Sources (Trust 1.0):**
- BrandonPlatform: Brandon's official statements and platform
- PreviousQA: Verified Q&A responses

**Supplementary Sources (Trust 0.6):**
- PartyPlatform: Republican and Independent platform context
- Clearly labeled as "party position" NOT Brandon's views

**Style Guidance (Trust 0.8):**
- MarketGurus: Copywriting principles for effective communication

### Intent Detection System
Detects underlying user intent beyond surface questions:
- `funding_sources` - "How will you pay for that?"
- `verification` - "Is that really true?"
- `scripture` - Faith/values-based questions
- `personal_values` - Moral/ethical questions
- `practical_impact` - "How does this affect me?"
- `comparison` - Candidate/party comparisons
- Plus: volunteer, donate, contact, timeline, etc.

### Escalation Detection
Monitors conversation patterns for frustration:
- Frustration indicators (repeated questions, strong punctuation)
- Urgency signals (need to talk to someone, time-sensitive)
- Automatically offers callbacks when escalation detected

### Security Features
- Input sanitization (XSS, SQL injection, prompt injection)
- Rate limiting per session (30 queries/minute, 5 web searches/minute)
- Callback rate limiting (3 requests per 5 minutes)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health with provider status |
| `/api/query` | POST | Main chat endpoint (rate limited) |
| `/api/callback` | POST | Request callback (rate limited) |
| `/api/consent` | POST | Update privacy consent |
| `/api/stats` | GET | Usage statistics |
| `/api/model-stats` | GET | Model performance metrics |
| `/api/volunteer` | POST | Volunteer registration |

## Environment Variables (Secrets)

Required API keys:
- `GOOGLE_API_KEY` - Gemini API
- `MISTRAL_API_KEY` - Mistral AI
- `COHERE_API_KEY` - Cohere
- `HUGGINGFACE_API_KEY` - HuggingFace Inference
- `REPLICATE_API_TOKEN` - Replicate
- `Z_API_KEY` - Z.ai (Zhipu)
- `NVIDIA_API_KEY_*` - 5 keys for different Nvidia models

## External Dependencies
- **FastAPI**: Python web framework
- **Weaviate Embedded**: Local vector database
- **Sentence-Transformers**: Text embeddings (all-MiniLM-L6-v2)
- **SQLite**: Interaction and performance logging
- **DuckDuckGo Search**: External web search
- **OpenAI/httpx**: Provider API clients

## File Structure
```
backend/
├── main.py              # FastAPI app with routes
├── llm_providers.py     # Multi-provider LLM manager
├── agent_orchestrator.py# Agentic pipeline with tools
├── intent_detector.py   # Intent and escalation detection
├── security.py          # Input sanitization, rate limiting
├── agent_tools.py       # Tool definitions
├── weaviate_manager.py  # Vector database
├── database.py          # SQLite operations
├── web_search_service.py# DuckDuckGo integration
└── query_expansion.py   # Question type detection

frontend/
├── index.html           # Chat interface
├── style.css            # Styling
└── app.js              # Frontend logic
```

## Recent Changes (Nov 27, 2025)
- Completed slot-based round-robin architecture: 9 API key slots across 5 providers, managing 17 unique models
- Removed Z.ai provider (no free tier available)
- Removed Replicate provider (no free tier available)
- Fixed Nvidia Maverick with correct API key (NVIDIA_LLAMA4_128e)
- Added Kimi K2 to HuggingFace slot (moonshotai/Kimi-K2-Instruct)
- All 9/9 slots operational (100% success rate)
- Session-sticky model selection with automatic failover on rate limits/errors

## Historical Changes (Nov 26, 2025)
- Migrated from Phi-3 local inference to multi-provider LLM architecture
- Added 7 LLM providers with automatic failover
- Implemented intent detection separate from question type
- Added escalation detection for frustrated users
- Added security hardening (sanitization, rate limiting)
- Added model performance tracking in database
- Removed dual deployment mode (now API-based only)
