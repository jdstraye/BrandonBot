# BrandonBot Project

## Overview
BrandonBot is an open-source RAG-based AI chatbot for a political candidate, designed to answer questions about Brandon's political positions using a 5-tier confidence system with zero API costs. The project aims to provide transparent and accurate information, directly reflecting Brandon's stated views and leveraging marketing principles to guide communication style. It's a Replit-native application, emphasizing open-source components and local inference.

## User Preferences
I prefer iterative development with clear, concise communication. When making changes, please explain the "why" behind them, not just the "what." I value performance and cost-efficiency. Do not make changes to the `replit.md` file without explicit instruction.

## System Architecture
The system employs a FastAPI backend, an embedded Weaviate vector database, and a CPU-optimized Phi-3 Mini ONNX model for local inference. It uses Sentence-Transformers for embeddings and SQLite for interaction logging. The frontend is a vanilla HTML/CSS/JS interface.

**Key Architectural Decisions & Features:**
-   **4-Stage Multi-Dimensional RAG Pipeline**:
    1.  **Question Analysis**: Determines question type (comparison, statistics, truth-seeking, policy), awareness level, and emotional tone.
    2.  **Multi-Source Retrieval**: Gathers information from Brandon's positions (RAG), MarketGurus guidance, and Bible verses, with external search as needed.
    3.  **Enhanced Response Generation**: Utilizes Phi-3 with combined contexts and framing based on question analysis.
    4.  **Response Delivery**: Adapts character (stay or break) based on context.
-   **Retrieval-First Architecture**: All questions follow a single path: retrieve → build context → Phi-3 generation, ensuring consistent confidence evaluation.
-   **RAG Pipeline Refactored**: Separates content search (BrandonPlatform, PreviousQA, PartyPlatform) from style guidance (MarketGurus), with style applied only when content confidence is high.
-   **Smart Boundary-Aware Chunking**: Document ingestion uses intelligent boundary detection for optimal chunking.
-   **Lazy Loading**: Phi-3 model unloads after 5 minutes of idle time to conserve RAM.
-   **Trust Multiplier System**: Confidence scores are calculated as `Similarity × Trust Factor`, reflecting source trustworthiness.
-   **3-Tier Knowledge Base (Weaviate Collections)**:
    -   **BrandonPlatform (1.0x trust)**: Brandon's statements & platform.
    -   **PreviousQA (1.0x trust)**: Historical Q&A.
    -   **PartyPlatform (0.6x trust)**: Republican/RNC platforms.
-   **Marketing-Guided Communication**: Leverages copywriting wisdom from the MarketGurus collection (Halbert, Hopkins, Caples, Schwartz, Ogilvy, Collier) to influence communication style based on question analysis and prospect awareness levels.
-   **Character-Breaking Behavior**: The bot stays in character for high-confidence policy questions but can break character for comparisons, recent events, or low-confidence responses.
-   **Question Type Detection**: Pattern-based analysis detects and frames responses for 6 types: comparison, statistics, truth-seeking, recent\_event, policy, and low\_confidence.
-   **Callback System**: Offers personal callbacks for low-confidence answers and external search scenarios.
-   **Privacy-First**: Opt-in logging for ethical data collection.
-   **CPU-Optimized**: Uses INT4 quantized models, designed for CPU-only environments.

## External Dependencies
-   **FastAPI**: Python web framework for the backend.
-   **Weaviate Embedded**: Local, embedded vector database.
-   **Phi-3 Mini ONNX Runtime**: Local large language model for inference.
-   **Sentence-Transformers**: Used for generating embeddings.
-   **SQLite**: Database for logging interactions.
-   **DuckDuckGo Web Search**: Integrated for external search capabilities (with domain trust scoring).
-   **OpenAI API (Planned Migration)**: Future integration for enhanced performance and quality, requiring `openai` library and `OPENAI_API_KEY`.

---

## OpenAI Migration Checklist

**Purpose**: Documents all Phi-3-specific workarounds that can be removed when migrating to OpenAI's API (10-30x faster responses, streaming support, better quality at ~$0.01-0.05 per query).

### Files to Remove (saves ~2.5GB)
- [ ] `download_phi3_model.py` - Model download script
- [ ] `phi3_model/` directory - 2GB ONNX model files
- [ ] `backend/phi3_client.py` - Replace with ~50 line OpenAI client

### Dependencies to Update
- [ ] Remove from requirements.txt: `onnxruntime`, `tiktoken`
- [ ] Add to requirements.txt: `openai`
- [ ] Optional: Keep `sentence-transformers` (zero cost) or switch to OpenAI embeddings

### Code Changes Required

**backend/phi3_client.py** - REPLACE ENTIRELY:
```python
# New openai_client.py (~50 lines)
from openai import OpenAI
class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
    async def generate_response(self, user_query, system_prompt, confidence):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            max_tokens=2048,
            temperature=0.7,
            stream=True
        )
```

**backend/rag_pipeline.py**:
- [ ] Remove Phi-3 chat template (`<|system|>...<|end|>` formatting)
- [ ] Expand compressed prompts (~350 chars → ~800 chars with 128k context)
- [ ] Remove token budget constraints (tiktoken encoding/counting)
- [ ] Simplify `_build_multi_section_prompt()` to return OpenAI messages

**backend/main.py**:
- [ ] Remove lazy loading system (5-min idle timeout, model unload/reload)
- [ ] Replace `Phi3Client()` with `OpenAIClient()`
- [ ] Remove model loading logic (instant startup)

### Configuration
- [ ] Add to Replit Secrets: `OPENAI_API_KEY`
- [ ] Optional: `OPENAI_MODEL=gpt-4o-mini` (default) or `gpt-4o` (premium)

### Migration Steps (90 minutes total)
1. [ ] Get OpenAI API key, add to Replit Secrets (10 min)
2. [ ] Create `openai_client.py`, update imports (30 min)
3. [ ] Remove Phi-3 formatting, expand prompts (30 min)
4. [ ] Test all scenarios: policy, comparison, debate, low-confidence (15 min)
5. [ ] Delete Phi-3 files, clean up dependencies (10 min)

### Expected Improvements
- **Speed**: 30-60 sec → 2-5 sec per response (10-30x faster)
- **Quality**: Better reasoning, fewer hallucinations
- **UX**: Streaming responses, instant startup
- **Resources**: 4GB RAM → 1GB RAM (no local model)
- **Cost**: $0 → ~$0.01-0.05 per query (ROI: massive UX improvement)

### Rollback Plan
Keep `phi3_client.py` in git history. To rollback: revert changes, re-run `python download_phi3_model.py`.