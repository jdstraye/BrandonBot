# BrandonBot Project

## Overview
BrandonBot is an open-source, RAG-based AI chatbot for a political candidate, designed to answer questions about Brandon's political positions. Its core features include a 5-tier confidence system, zero API costs, and a commitment to transparent, accurate information reflecting Brandon's stated views, all while leveraging marketing principles for effective communication. The project is a Replit-native application, emphasizing open-source components and local inference for cost-efficiency and performance.

## User Preferences
I prefer iterative development with clear, concise communication. When making changes, please explain the "why" behind them, not just the "what." I value performance and cost-efficiency. Do not make changes to the `replit.md` file without explicit instruction.

## System Architecture
The system employs a FastAPI backend, an embedded Weaviate vector database, and a CPU-optimized Phi-3 Mini ONNX model for local inference. Embeddings are generated using Sentence-Transformers, and interaction logs are stored in SQLite. The frontend is built with vanilla HTML/CSS/JS.

**Key Architectural Decisions & Features:**
-   **4-Stage Multi-Dimensional RAG Pipeline**: Dynamically adapts based on context and confidence, encompassing Question Analysis, Multi-Source Retrieval, Enhanced Response Generation, and Response Delivery.
-   **Retrieval-First Architecture**: Ensures consistent confidence evaluation by routing all questions through a retrieve → build context → Phi-3 generation path.
-   **RAG Pipeline Refactored**: Separates content search (BrandonPlatform, PreviousQA, PartyPlatform) from style guidance (MarketGurus), applying style only when content confidence is high.
-   **Smart Boundary-Aware Chunking**: Optimizes document ingestion for efficient retrieval.
-   **Trust Multiplier System**: Calculates response confidence as `Similarity × Trust Factor`.
-   **3-Tier Knowledge Base (Weaviate Collections)**: Organizes information into BrandonPlatform (1.0x trust), PreviousQA (1.0x trust), and PartyPlatform (0.6x trust).
-   **Marketing-Guided Communication**: Influences communication style using copywriting principles from the MarketGurus collection, adapted to question analysis and prospect awareness levels.
-   **Character-Breaking Behavior**: The bot maintains character for high-confidence policy questions but can break character for specific scenarios like comparisons, recent events, or low-confidence responses.
-   **Question Type Detection**: Identifies and frames responses for six types: comparison, statistics, truth-seeking, recent_event, policy, and low_confidence.
-   **Callback System**: Offers personal callbacks for low-confidence answers and scenarios requiring external search.
-   **Privacy-First**: Implements opt-in logging for ethical data collection.
-   **CPU-Optimized**: Designed to run efficiently in CPU-only environments using INT4 quantized models.

## External Dependencies
-   **FastAPI**: Python web framework for the backend.
-   **Weaviate Embedded**: Local, embedded vector database for knowledge storage.
-   **Phi-3 Mini ONNX Runtime**: Local large language model for inference.
-   **Sentence-Transformers**: Used for generating text embeddings.
-   **SQLite**: Database for logging user interactions.
-   **DuckDuckGo Web Search**: Integrated for external search capabilities.
-   **OpenAI API (Planned)**: Future integration for enhanced performance and quality.

## Recent Changes & Known Issues (Nov 22, 2025)

### Production Issue: Phi-3 Performance Degradation
**Status**: Active issue in Replit shared development environment

**Symptoms**:
- Phi-3 generates only 1 token per 60-90 seconds (vs expected 10-30 tokens/sec)
- All queries timeout and show fallback message despite high retrieval confidence (70-80%)
- Users see: "I'm having trouble generating a complete response. Would you like Brandon to call you back?"

**Root Cause**: 
- CPU starvation due to resource contention in Replit's shared infrastructure
- Load average: 20.3 on 6 cores (extreme CPU contention)
- Python process competing with other Repls for CPU time

**Temporary Workarounds Applied**:
1. **ONNX Thread Limits** (Environment Variables):
   - `OMP_NUM_THREADS=4` - Limits OpenMP threads
   - `ORT_INTRA_OP_NUM_THREADS=4` - ONNX Runtime internal parallelism
   - `ORT_INTER_OP_NUM_THREADS=1` - Prevents operator-level parallelism
   - **Effect**: Reduced CPU usage from 220% → 109%, but speed issue persists
   - **Note**: These should be removed when self-hosting on dedicated hardware

2. **60-Second Timeout Safeguard** (`backend/phi3_client.py`):
   - Prevents infinite generation loops using `time.monotonic()` wall-clock timer
   - Aborts generation after 60 seconds and returns graceful fallback message
   - Returns structured error flags: `truncated`, `error` for debugging
   - Logs timing information for performance monitoring

**Solutions**:
- **Short-term**: Migrate to commercial LLM API (OpenAI/Gemini) - see `COMMERCIALAI_MIGRATION.md`
- **Long-term**: Self-host on dedicated hardware - see `SELF_HOSTING.md`

### Documentation Files
- **COMMERCIALAI_MIGRATION.md**: Complete guide for migrating from Phi-3 to OpenAI or Google Gemini API
- **SELF_HOSTING.md**: Instructions for running BrandonBot on Debian 13 or other dedicated hardware