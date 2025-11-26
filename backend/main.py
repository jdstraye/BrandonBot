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
phi3_model_path = os.getenv("PHI3_MODEL_PATH", "./phi3_model")
database_path = os.getenv("DATABASE_PATH", "data/brandonbot.db")
llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
architecture_mode = os.getenv("ARCHITECTURE_MODE", "llm_first").lower()
deployment_mode = os.getenv("DEPLOYMENT_MODE", "replit").lower()

weaviate_available = False
weaviate_manager = None
rag_pipeline = None

if deployment_mode != "replit":
    try:
        from weaviate_manager import WeaviateManager
        from rag_pipeline import RAGPipeline
        weaviate_manager = WeaviateManager(weaviate_data_dir)
        weaviate_available = True
        logger.info("Weaviate module loaded successfully (Debian 13 mode)")
    except Exception as e:
        logger.warning(f"Weaviate not available (protobuf conflict expected on Replit): {e}")
        logger.info("Running in Gemini-only mode without vector database")
else:
    logger.info("Replit mode: Skipping Weaviate (protobuf conflict with Gemini)")
    logger.info("For full RAG functionality, deploy to Debian 13")

llm_client = None
if llm_provider == "gemini":
    from gemini_client import GeminiClient
    logger.info("Using Gemini API for LLM inference")
    llm_client = GeminiClient()
elif llm_provider == "phi3":
    from phi3_client import Phi3Client
    logger.info("Using Phi-3 local model for LLM inference")
    llm_client = Phi3Client(phi3_model_path)

from database import DatabaseManager
from web_search_service import WebSearchService

db_manager = DatabaseManager(database_path)
web_search_service = WebSearchService()

if weaviate_available and weaviate_manager:
    rag_pipeline = RAGPipeline(weaviate_manager, llm_client, db_manager, web_search_service)

agent_orchestrator = None
if architecture_mode == "llm_first" and llm_provider == "gemini":
    from agent_orchestrator import AgentOrchestrator
    logger.info("Initializing LLM-First AgentOrchestrator (function-calling mode)")
    agent_orchestrator = AgentOrchestrator(llm_client, weaviate_manager, web_search_service)
else:
    logger.info(f"Using legacy RAG pipeline (architecture_mode={architecture_mode}, llm_provider={llm_provider})")

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
    logger.info("Starting BrandonBot - LLM-First Agentic Architecture...")
    logger.info(f"Deployment mode: {deployment_mode}")
    logger.info(f"Architecture mode: {architecture_mode}")
    
    logger.info("Initializing database...")
    await db_manager.initialize()
    
    if weaviate_available and weaviate_manager:
        logger.info("Initializing Weaviate (embedded mode)...")
        await weaviate_manager.initialize()
    else:
        logger.info("Weaviate disabled (Replit mode - use Debian 13 for full RAG)")
    
    if llm_client:
        logger.info(f"Configuring {llm_provider.upper()} API...")
        await llm_client.ensure_model_ready()
    
    logger.info(f"BrandonBot ready! Using {llm_provider.upper()} for inference.")
    if agent_orchestrator:
        logger.info("AgentOrchestrator enabled with function-calling tools")

@app.on_event("shutdown")
async def shutdown_event():
    if weaviate_manager and hasattr(weaviate_manager, 'client') and weaviate_manager.client:
        weaviate_manager.client.close()
    await db_manager.close()
    if llm_client and hasattr(llm_client, 'close'):
        await llm_client.close()

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
    
    llm_status = False
    llm_name = llm_provider
    if llm_client:
        if hasattr(llm_client, 'health_check'):
            llm_status = await llm_client.health_check()
        elif hasattr(llm_client, 'model') and llm_client.model is not None:
            llm_status = True
    
    all_ok = llm_status
    status = "healthy" if all_ok else "degraded"
    
    return {
        "status": status,
        "deployment_mode": deployment_mode,
        "architecture": architecture_mode,
        "services": {
            "weaviate_embedded": "up" if weaviate_status else ("disabled" if deployment_mode == "replit" else "down"),
            f"llm_{llm_name}": "up" if llm_status else "initializing",
            "database": "up",
            "agent_orchestrator": "up" if agent_orchestrator else "disabled"
        },
        "note": f"LLM-first agentic architecture with {llm_name.upper()}" + (" (Replit mode - Weaviate disabled)" if deployment_mode == "replit" else "")
    }

@app.post("/api/query")
async def query_bot(request: QueryRequest):
    try:
        session_id = request.session_id or request.user_id or str(uuid.uuid4())
        
        if agent_orchestrator:
            response_text, metadata = await agent_orchestrator.process_message(
                user_message=request.query,
                session_id=session_id
            )
            
            if request.consent_given and request.user_id:
                await db_manager.log_interaction(
                    user_id=request.user_id,
                    query=request.query,
                    response=response_text,
                    confidence=metadata.get("confidence", 0.8),
                    sources=metadata.get("sources", []),
                    consent_given=request.consent_given
                )
            
            return {
                "response": response_text,
                "session_id": session_id,
                "architecture": "llm_first",
                "metadata": {
                    "tool_calls": metadata.get("tool_calls", []),
                    "iterations": metadata.get("iterations", 1),
                    "sources": metadata.get("sources", [])
                }
            }
        elif rag_pipeline:
            response = await rag_pipeline.process_query(
                query=request.query,
                user_id=request.user_id,
                consent_given=request.consent_given
            )
            response["architecture"] = "rag_first"
            return response
        else:
            if llm_client:
                simple_response = await llm_client.generate_simple(
                    f"User question: {request.query}\n\nProvide a helpful response as BrandonBot, the political campaign assistant."
                )
                return {
                    "response": simple_response,
                    "session_id": session_id,
                    "architecture": "direct_llm",
                    "note": "Running in simplified mode (Weaviate disabled on Replit)"
                }
            else:
                return {
                    "response": "I'm sorry, the chatbot is not fully initialized. Please try again later.",
                    "architecture": "error"
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
        await db_manager.log_callback_request(
            user_id=request.user_id,
            name=request.name,
            phone=request.phone,
            email=request.email,
            question=request.question
        )
        return {
            "status": "success", 
            "message": "Callback request received. Brandon will contact you personally."
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
