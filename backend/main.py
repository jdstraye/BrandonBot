from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import os
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

weaviate_data_dir = os.getenv("WEAVIATE_DATA_DIR", "./weaviate_data")
database_path = os.getenv("DATABASE_PATH", "data/brandonbot.db")

weaviate_available = False
weaviate_manager = None

try:
    from weaviate_manager import WeaviateManager
    weaviate_manager = WeaviateManager(weaviate_data_dir)
    weaviate_available = True
    logger.info("Weaviate module loaded successfully")
except Exception as e:
    logger.warning(f"Weaviate not available: {e}")
    logger.info("Running without vector database")

from database import DatabaseManager
from web_search_service import WebSearchService
from security import input_sanitizer, rate_limiter

db_manager = DatabaseManager(database_path)
web_search_service = WebSearchService()

from agent_orchestrator import AgentOrchestrator
logger.info("Initializing LLM-First AgentOrchestrator (multi-provider mode)")
agent_orchestrator = AgentOrchestrator(weaviate_manager, web_search_service, db_manager)

app = FastAPI(title="BrandonBot API - LLM-First Agentic Architecture")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    consent_given: bool = False

class ConsentRequest(BaseModel):
    user_id: str
    consent_given: bool

class CallbackRequest(BaseModel):
    user_id: str
    name: str
    phone: str
    email: Optional[str] = None
    question: str

class VolunteerRequest(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    zip_code: Optional[str] = None
    interests: Optional[List[str]] = None
    availability: Optional[str] = "flexible"

class DonationRequest(BaseModel):
    amount: float
    donor_name: str
    donor_email: str
    employer: Optional[str] = None
    occupation: Optional[str] = None
    recurring: bool = False

@app.on_event("startup")
async def startup_event():
    logger.info("Starting BrandonBot - LLM-First Multi-Provider Architecture...")
    
    logger.info("Initializing database...")
    await db_manager.initialize()
    
    if weaviate_available and weaviate_manager:
        logger.info("Initializing Weaviate (embedded mode)...")
        await weaviate_manager.initialize()
    else:
        logger.warning("Weaviate not available - RAG features limited")
    
    provider_stats = agent_orchestrator.llm_manager.get_provider_stats()
    total_slots = provider_stats.get("total_slots", 0)
    available_slots = provider_stats.get("available_slots", 0)
    total_models = provider_stats.get("total_models", 0)
    logger.info(f"LLM Slots: {available_slots}/{total_slots} available, {total_models} unique models")
    
    logger.info("BrandonBot ready with multi-provider LLM support")

@app.on_event("shutdown")
async def shutdown_event():
    if weaviate_manager and hasattr(weaviate_manager, 'client') and weaviate_manager.client:
        weaviate_manager.client.close()
    await db_manager.close()

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("static/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="""
<!DOCTYPE html>
<html><head><title>BrandonBot API</title></head>
<body style="font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px;">
<h1>🤖 BrandonBot API - 100% Open Source</h1>
<p>The RAG-based political chatbot API is running on Replit with:</p>
<ul>
<li>✅ Weaviate Embedded (no Docker required)</li>
<li>✅ Phi-3 Mini ONNX (CPU-optimized)</li>
<li>✅ Sentence-Transformers embeddings</li>
<li>✅ Zero API costs - everything runs locally</li>
</ul>
<h2>API Endpoints:</h2>
<ul>
<li><code>GET /health</code> - Health check</li>
<li><code>POST /api/query</code> - Ask BrandonBot a question</li>
<li><code>POST /api/consent</code> - Update logging consent</li>
<li><code>POST /api/callback</code> - Request callback</li>
</ul>
<p><a href="/health">Check API Health →</a></p>
</body></html>
        """)

@app.get("/health")
async def health_check():
    weaviate_status = False
    if weaviate_manager and hasattr(weaviate_manager, 'client'):
        weaviate_status = weaviate_manager.client is not None
    
    provider_stats = agent_orchestrator.llm_manager.get_provider_stats()
    available_slots = provider_stats.get("available_slots", 0)
    total_slots = provider_stats.get("total_slots", 0)
    total_models = provider_stats.get("total_models", 0)
    
    providers_info = provider_stats.get("providers", {})
    available_providers = [name for name, info in providers_info.items() 
                          if any(slot.get("status") == "available" for slot in info.get("slots", []))]
    
    return {
        "status": "healthy" if available_slots > 0 else "degraded",
        "architecture": "llm_first_slot_based",
        "services": {
            "weaviate_embedded": "up" if weaviate_status else "down",
            "llm_slots": f"{available_slots}/{total_slots}",
            "llm_models": total_models,
            "llm_providers": available_providers,
            "database": "up",
            "agent_orchestrator": "up"
        },
        "slot_summary": agent_orchestrator.llm_manager.get_slot_rotation_summary(),
        "provider_stats": provider_stats,
        "note": f"Slot-based rotation with {available_slots} slots managing {total_models} models"
    }

@app.post("/api/query")
async def query_bot(request: QueryRequest):
    try:
        session_id = request.session_id or request.user_id or str(uuid.uuid4())
        
        is_allowed, wait_seconds = rate_limiter.check_rate_limit(session_id, "query")
        if not is_allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Too many requests. Please wait {wait_seconds} seconds."
            )
        
        sanitized = input_sanitizer.sanitize(request.query)
        if sanitized.issues_found:
            logger.warning(f"Input sanitized for {session_id}: {sanitized.issues_found}")
        
        response_text, metadata = await agent_orchestrator.process_message(
            user_message=sanitized.cleaned_text,
            session_id=session_id
        )
        
        if request.consent_given and request.user_id:
            await db_manager.log_interaction(
                user_id=request.user_id,
                query=request.query,
                response=response_text,
                confidence=metadata.get("confidence", 0.8),
                sources=metadata.get("sources", []),
                consent_given=request.consent_given,
                model_used=metadata.get("model_used")
            )
        
        return {
            "response": response_text,
            "session_id": session_id,
            "architecture": "llm_first",
            "metadata": {
                "tool_calls": metadata.get("tool_calls", []),
                "iterations": metadata.get("iterations", 1),
                "sources": metadata.get("sources", []),
                "model": metadata.get("model_used"),
                "provider": metadata.get("provider")
            }
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/consent")
async def update_consent(request: ConsentRequest):
    try:
        await db_manager.update_consent(request.user_id, request.consent_given)
        return {"status": "success", "message": "Consent updated"}
    except Exception as e:
        logger.error(f"Error updating consent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/callback")
async def request_callback(request: CallbackRequest):
    try:
        is_allowed, wait_seconds = rate_limiter.check_rate_limit(request.user_id, "callback")
        if not is_allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Too many callback requests. Please wait {wait_seconds} seconds."
            )
        
        await db_manager.log_callback_request(
            user_id=request.user_id,
            name=request.name,
            phone=request.phone,
            email=request.email,
            question=request.question
        )
        return {
            "status": "success", 
            "message": "Callback request received. Someone from the team will contact you soon."
        }
    except Exception as e:
        logger.error(f"Error logging callback request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    try:
        stats = await db_manager.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-stats")
async def get_model_stats():
    """Get model performance statistics for evaluation"""
    try:
        model_stats = await db_manager.get_model_stats()
        provider_stats = agent_orchestrator.llm_manager.get_provider_stats()
        return {
            "model_performance": model_stats,
            "provider_status": provider_stats
        }
    except Exception as e:
        logger.error(f"Error getting model stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/volunteer")
async def register_volunteer(request: VolunteerRequest):
    """Direct endpoint for volunteer registration (bypasses LLM)"""
    try:
        if agent_orchestrator:
            from agent_tools import ToolCall
            tool_call = ToolCall(
                name="register_volunteer",
                arguments={
                    "name": request.name,
                    "email": request.email,
                    "phone": request.phone or "",
                    "zip_code": request.zip_code or "",
                    "interests": request.interests or [],
                    "availability": request.availability or "flexible"
                }
            )
            result = await agent_orchestrator.tool_executor.execute(tool_call)
            
            if result.success:
                return {"status": "success", **result.data}
            else:
                raise HTTPException(status_code=400, detail=result.error_message)
        else:
            return {
                "status": "success",
                "message": f"Thank you, {request.name}! Volunteer registration recorded.",
                "note": "Legacy mode - CRM integration pending"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering volunteer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/donate")
async def make_donation(request: DonationRequest):
    """Direct endpoint for donation processing (bypasses LLM)"""
    try:
        if agent_orchestrator:
            from agent_tools import ToolCall
            tool_call = ToolCall(
                name="make_donation",
                arguments={
                    "amount": request.amount,
                    "donor_name": request.donor_name,
                    "donor_email": request.donor_email,
                    "employer": request.employer or "",
                    "occupation": request.occupation or "",
                    "recurring": request.recurring
                }
            )
            result = await agent_orchestrator.tool_executor.execute(tool_call)
            
            if result.success:
                return {"status": "success", **result.data}
            else:
                raise HTTPException(status_code=400, detail=result.error_message)
        else:
            return {
                "status": "success",
                "message": f"Thank you for your ${request.amount} contribution!",
                "note": "Legacy mode - payment processing pending"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing donation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/architecture")
async def get_architecture_info():
    """Return information about the current architecture mode"""
    return {
        "mode": architecture_mode,
        "llm_provider": llm_provider,
        "orchestrator_enabled": agent_orchestrator is not None,
        "available_tools": [
            "search_policy_collections",
            "perform_web_search", 
            "retrieve_answer_style",
            "register_volunteer",
            "make_donation"
        ] if agent_orchestrator else [],
        "description": "LLM-first agentic architecture where the LLM reasons and recommends tool calls, and the Orchestrator validates and executes them." if agent_orchestrator else "Legacy RAG-first pipeline"
    }

app.mount("/static", StaticFiles(directory="static"), name="static")
