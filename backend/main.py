from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import os
import uuid

# Load environment variables from .env file (before any env var access)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on platform env vars (Replit)

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
from email_service import email_service

db_manager = DatabaseManager(database_path)
web_search_service = WebSearchService()

slm_manager = None
try:
    from slm_manager import SLMManager
    slm_manager = SLMManager()
    logger.info("SLM Manager created (lazy loading - model loads on first use)")
except Exception as e:
    logger.warning(f"SLM Manager not available: {e}")
    logger.info("Running without local SLM - will use pattern-based fallbacks")

from agent_orchestrator import AgentOrchestrator
logger.info("Initializing LLM-First AgentOrchestrator (multi-provider mode)")
agent_orchestrator = AgentOrchestrator(weaviate_manager, web_search_service, db_manager, slm_manager=slm_manager)

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
    logging_consent_given: bool = False
    consent_given: bool = False  # Legacy field for backwards compatibility

class ConsentRequest(BaseModel):
    user_id: str
    ai_disclosure_accepted: bool = True
    logging_consent_given: bool = False
    consent_given: bool = False  # Legacy field for backwards compatibility

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

class DonateInterestRequest(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    message: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    logger.info("Starting BrandonBot - LLM-First Multi-Provider Architecture...")
    
    logger.info("Initializing database...")
    await db_manager.initialize()
    
    if weaviate_available and weaviate_manager:
        logger.info("Initializing Weaviate (embedded mode)...")
        await weaviate_manager.initialize()
    else:
        logger.error("FAIL-CLOSED: Weaviate not available - RAG is REQUIRED")
        raise RuntimeError("Weaviate/RAG is required for BrandonBot operation. Cannot start without vector database.")
    
    logger.info("Verifying SLM models for Output Validation (fail-closed)...")
    try:
        from ov_slm_models import msmarco_checker
        ms_marco_ready = await msmarco_checker.ensure_ready()
        if not ms_marco_ready:
            logger.error("FAIL-CLOSED: MS-MARCO model failed to load")
            raise RuntimeError("MS-MARCO intent checker is required for Output Validation. Cannot start without SLM models.")
        logger.info("MS-MARCO intent checker: READY")
    except ImportError as e:
        logger.error(f"FAIL-CLOSED: Cannot import SLM models: {e}")
        raise RuntimeError(f"SLM models import failed: {e}. Cannot start without Output Validation models.")
    except Exception as e:
        logger.error(f"FAIL-CLOSED: SLM initialization failed: {e}")
        raise RuntimeError(f"SLM initialization failed: {e}. Cannot start without Output Validation models.")
    
    provider_stats = agent_orchestrator.llm_manager.get_provider_stats()
    total_slots = provider_stats.get("total_slots", 0)
    available_slots = provider_stats.get("available_slots", 0)
    total_models = provider_stats.get("total_models", 0)
    logger.info(f"LLM Slots: {available_slots}/{total_slots} available, {total_models} unique models")
    
    logger.info("BrandonBot ready with multi-provider LLM support (fail-closed verification complete)")

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
        
        # Use logging_consent_given, fall back to legacy consent_given for backwards compatibility
        should_log = request.logging_consent_given or request.consent_given
        if should_log and request.user_id:
            await db_manager.log_interaction(
                user_id=request.user_id,
                query=request.query,
                response=response_text,
                confidence=metadata.get("confidence", 0.8),
                sources=metadata.get("sources", []),
                consent_given=should_log,
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
        # Use logging_consent_given, fall back to legacy consent_given for backwards compatibility
        logging_consent = request.logging_consent_given or request.consent_given
        await db_manager.update_consent(request.user_id, logging_consent)
        return {
            "status": "success", 
            "message": "Consent updated",
            "ai_disclosure_accepted": request.ai_disclosure_accepted,
            "logging_consent_given": logging_consent
        }
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
        
        email_result = await email_service.send_callback_notification(
            name=request.name,
            phone=request.phone,
            reason=request.question or "",
            preferred_time="",
            session_id=request.user_id
        )
        
        await db_manager.log_callback_request(
            user_id=request.user_id,
            name=request.name,
            phone=request.phone,
            email=request.email,
            question=request.question
        )
        
        logger.info(f"Callback request logged: {request.name} ({request.phone}), email sent: {email_result.success}")
        
        return {
            "status": "success", 
            "message": "Callback request received. Someone from the team will contact you soon.",
            "email_sent": email_result.success
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
    """
    Volunteer registration endpoint.
    Sends email notification to Brandon and logs to database.
    """
    try:
        email_result = await email_service.send_volunteer_notification(
            name=request.name,
            email=request.email,
            phone=request.phone or "",
            zip_code=request.zip_code or "",
            interests=request.interests or [],
            availability=request.availability or "flexible"
        )
        
        await db_manager.log_volunteer(
            name=request.name,
            email=request.email,
            phone=request.phone or "",
            zip_code=request.zip_code or "",
            interests=request.interests or [],
            availability=request.availability or "flexible"
        )
        
        logger.info(f"Volunteer registered: {request.name} ({request.email}), email sent: {email_result.success}")
        
        return {
            "status": "success",
            "message": f"Thank you, {request.name}! Your volunteer registration has been received. Someone from the team will be in touch soon.",
            "email_sent": email_result.success
        }
    except Exception as e:
        logger.error(f"Error registering volunteer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/donate-interest")
async def register_donation_interest(request: DonateInterestRequest):
    """
    Donation interest endpoint.
    
    IMPORTANT: This does NOT process donations. BrandonBot cannot and should not
    process donations directly to maintain FEC compliance.
    
    This endpoint:
    1. Captures the interested donor's contact info
    2. Sends email notification to Brandon
    3. Brandon follows up with secure, FEC-compliant donation methods
    """
    try:
        email_result = await email_service.send_donation_interest_notification(
            name=request.name,
            email=request.email,
            phone=request.phone or "",
            message=request.message or ""
        )
        
        await db_manager.log_donation_interest(
            name=request.name,
            email=request.email,
            phone=request.phone or "",
            message=request.message or ""
        )
        
        logger.info(f"Donation interest logged: {request.name} ({request.email}), email sent: {email_result.success}")
        
        return {
            "status": "success",
            "message": f"Thank you for your interest in supporting the campaign, {request.name}! Someone from the team will reach out with secure donation options.",
            "email_sent": email_result.success,
            "note": "For your security, we do not process donations through this chatbot."
        }
    except Exception as e:
        logger.error(f"Error logging donation interest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/architecture")
async def get_architecture_info():
    """Return information about the current architecture mode"""
    provider_stats = agent_orchestrator.llm_manager.get_provider_stats()
    return {
        "mode": "llm_first_slot_based",
        "orchestrator_enabled": agent_orchestrator is not None,
        "available_slots": provider_stats.get("available_slots", 0),
        "total_models": provider_stats.get("total_models", 0),
        "available_tools": [
            "search_policy_collections",
            "perform_web_search", 
            "retrieve_answer_style",
            "register_volunteer"
        ] if agent_orchestrator else [],
        "description": "LLM-first agentic architecture with slot-based multi-provider rotation"
    }

app.mount("/static", StaticFiles(directory="static"), name="static")
