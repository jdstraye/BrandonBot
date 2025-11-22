# Commercial AI API Migration Guide

## Current Status (Nov 22, 2025)

**Production Issue**: Phi-3 local inference is experiencing severe performance degradation in Replit's shared development environment:
- **Observed Performance**: 1 token per 60-90 seconds (vs expected 10-30 tokens/second)
- **Root Cause**: CPU starvation due to resource contention (load average 20.3 on 6 cores in shared environment)
- **Temporary Workarounds Applied**:
  - ONNX thread limits: `OMP_NUM_THREADS=4`, `ORT_INTRA_OP_NUM_THREADS=4`, `ORT_INTER_OP_NUM_THREADS=1`
  - 60-second timeout safeguard in `phi3_client.py` (prevents infinite hangs)
  - Reduced CPU usage from 220% → 109%, but speed issue persists
- **User Impact**: All queries timeout and show fallback message despite high retrieval confidence (70-80%)

**Recommendation**: Migrate to commercial LLM API immediately for production, or self-host on dedicated hardware.

---

## Option 1: OpenAI API Migration

**Purpose**: Replace Phi-3 local inference with OpenAI's API for 10-30x faster responses, streaming support, and better quality.

**Expected Results**:
- **Speed**: 30-60 sec → 2-5 sec per response (10-30x faster)
- **Quality**: Better reasoning, fewer hallucinations, more nuanced responses
- **UX**: Streaming responses, instant startup (no model loading)
- **Resources**: 4GB RAM → 1GB RAM (no local model)
- **Cost**: $0 → ~$0.01-0.05 per query

### Files to Remove (saves ~2.5GB)
- [ ] `download_phi3_model.py` - Model download script
- [ ] `phi3_model/` directory - 2GB ONNX model files
- [ ] `backend/phi3_client.py` - Replace with ~50 line OpenAI client

### Dependencies to Update
- [ ] Remove from `requirements.txt`: `onnxruntime-genai`, `tiktoken` (if added)
- [ ] Add to `requirements.txt`: `openai`
- [ ] Optional: Keep `sentence-transformers` (zero cost) or switch to OpenAI embeddings

### Code Changes Required

**backend/openai_client.py** - NEW FILE:
```python
import os
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        logger.info(f"OpenAI client initialized with model: {self.model}")
    
    async def generate_response(self, query: str, context: str, confidence: float, 
                               system_prompt: str = None) -> dict:
        try:
            messages = [
                {"role": "system", "content": system_prompt or "You are BrandonBot."},
                {"role": "user", "content": f"Question: {query}\n\nContext: {context}"}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
                stream=False
            )
            
            return {
                "response": response.choices[0].message.content,
                "model": self.model,
                "tokens_used": response.usage.total_tokens
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {
                "response": "I'm having trouble processing your question right now. Would you like Brandon to call you back?",
                "model": "error",
                "error": str(e)
            }
```

**backend/rag_pipeline.py**:
- [ ] Remove Phi-3 chat template formatting (`<|system|>...<|end|>`)
- [ ] Expand compressed prompts (~350 chars → ~800 chars, OpenAI supports 128k context)
- [ ] Remove token budget constraints and tiktoken counting
- [ ] Update imports: `from phi3_client import Phi3Client` → `from openai_client import OpenAIClient`

**backend/main.py**:
- [ ] Remove lazy loading system (5-min idle timeout, model unload/reload logic)
- [ ] Replace `Phi3Client()` with `OpenAIClient()`
- [ ] Remove model loading delays (instant startup)
- [ ] Update startup logs to reflect OpenAI usage

**Environment Variables** (Replit Secrets):
- [ ] Add `OPENAI_API_KEY` (required)
- [ ] Optional: `OPENAI_MODEL=gpt-4o-mini` (default) or `gpt-4o` (premium, higher cost/quality)

### Migration Steps (90 minutes total)
1. [ ] Get OpenAI API key from platform.openai.com (10 min)
2. [ ] Add `OPENAI_API_KEY` to Replit Secrets (5 min)
3. [ ] Create `backend/openai_client.py`, update imports in pipeline and main (30 min)
4. [ ] Remove Phi-3 formatting, expand prompts in `rag_pipeline.py` (30 min)
5. [ ] Test all scenarios: policy, comparison, statistics, low-confidence (15 min)
6. [ ] Delete `phi3_model/` directory and `phi3_client.py` (5 min)
7. [ ] Update `requirements.txt`, remove ONNX dependencies (5 min)

### Rollback Plan
Keep `phi3_client.py` in git history. To rollback:
1. Revert code changes via git
2. Re-run `python download_phi3_model.py` to restore model files
3. Remove `openai` from requirements.txt

---

## Option 2: Google Gemini API Migration

**Purpose**: Alternative to OpenAI using Google's Gemini API with similar performance benefits and potentially lower cost.

**Expected Results**:
- **Speed**: Similar to OpenAI (2-5 sec per response)
- **Quality**: Competitive with GPT-4o-mini
- **Cost**: Competitive pricing, free tier available
- **Integration**: Replit has native Python Gemini blueprint

### Advantages Over OpenAI
- Free tier available (60 queries per minute)
- Competitive pricing on paid tiers
- Native Replit integration (handles API key management)
- Multimodal capabilities (if needed in future)

### Migration Steps (90 minutes total)
1. [ ] Get Google AI API key from https://makersuite.google.com/app/apikey (10 min)
2. [ ] Add `GOOGLE_API_KEY` to Replit Secrets OR use Replit's Gemini blueprint (10 min)
3. [ ] Add `google-generativeai` to requirements.txt, remove ONNX dependencies (5 min)
4. [ ] Create `backend/gemini_client.py` (see template below) (30 min)
5. [ ] Update `rag_pipeline.py` and `main.py` - same as OpenAI migration (30 min)
6. [ ] Test all scenarios: policy, comparison, statistics, low-confidence (15 min)

### Environment Variables (Replit Secrets)
- [ ] `GOOGLE_API_KEY` (required) - Get from Google AI Studio
- [ ] Optional: `GEMINI_MODEL=gemini-1.5-flash` (default, free tier) or `gemini-1.5-pro` (paid, higher quality)

### Dependencies
- [ ] Add to `requirements.txt`: `google-generativeai>=0.3.0`
- [ ] Remove from `requirements.txt`: `onnxruntime-genai`, `tiktoken` (if added)

### Code Template (backend/gemini_client.py)
```python
import os
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model)
        logger.info(f"Gemini client initialized with model: {model}")
    
    async def generate_response(self, query: str, context: str, confidence: float,
                               system_prompt: str = None) -> dict:
        try:
            prompt = f"{system_prompt}\n\nQuestion: {query}\n\nContext: {context}"
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 2048,
                }
            )
            
            return {
                "response": response.text,
                "model": "gemini",
                "tokens_used": response.usage_metadata.total_token_count
            }
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                "response": "I'm having trouble processing your question right now. Would you like Brandon to call you back?",
                "model": "error",
                "error": str(e)
            }
```

---

## Comparison: OpenAI vs Gemini vs Phi-3

| Feature | Phi-3 (Current) | OpenAI (gpt-4o-mini) | Google Gemini (1.5-flash) |
|---------|-----------------|----------------------|---------------------------|
| **Speed (Replit)** | 60-90s/response | 2-5s/response | 2-5s/response |
| **Speed (Self-hosted)** | 3-10s/response | 2-5s/response | 2-5s/response |
| **Cost** | $0 (compute only) | ~$0.01-0.05/query | ~$0.01-0.03/query (free tier available) |
| **Quality** | Good | Excellent | Excellent |
| **RAM Usage** | 4-5GB | <1GB | <1GB |
| **Startup Time** | 30s (model load) | Instant | Instant |
| **Streaming** | No | Yes | Yes |
| **Context Length** | 4k tokens | 128k tokens | 1M tokens |
| **Dependencies** | ONNX Runtime, model files | openai library | google-generativeai |
| **Replit Integration** | None | None | Native blueprint |

---

## Notes on Current Workarounds (Temporary)

These workarounds were added to keep Phi-3 functional in Replit's shared environment but should be **removed** after migrating to commercial API:

1. **ONNX Thread Limits** (in shared environment variables):
   - `OMP_NUM_THREADS=4`
   - `ORT_INTRA_OP_NUM_THREADS=4`
   - `ORT_INTER_OP_NUM_THREADS=1`
   - **Purpose**: Reduce CPU thread contention (220% → 109% CPU usage)
   - **Remove after migration**: Not needed with API-based inference

2. **60-Second Timeout Safeguard** (in `phi3_client.py`):
   - Prevents infinite generation loops
   - Uses `time.monotonic()` wall-clock timer
   - Returns fallback message after timeout
   - **Remove after migration**: APIs have their own timeout handling

---

## Recommendation

**For Production**: Migrate to **Google Gemini** first (free tier, Replit integration), then optionally try OpenAI if quality needs improvement.

**For Self-Hosting**: Keep Phi-3 and follow `SELF_HOSTING.md` guide to run on dedicated hardware where CPU resources aren't contested.
