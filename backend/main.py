from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import os

from weaviate_manager import WeaviateManager
from phi3_client import Phi3Client
from database import DatabaseManager
from rag_pipeline import RAGPipeline
from web_search_service import WebSearchService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BrandonBot API - 100% Open Source")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

weaviate_data_dir = os.getenv("WEAVIATE_DATA_DIR", "./weaviate_data")
phi3_model_path = os.getenv("PHI3_MODEL_PATH", "./phi3_model")
database_path = os.getenv("DATABASE_PATH", "data/brandonbot.db")

weaviate_manager = WeaviateManager(weaviate_data_dir)
phi3_client = Phi3Client(phi3_model_path)
db_manager = DatabaseManager(database_path)
web_search_service = WebSearchService()
rag_pipeline = RAGPipeline(weaviate_manager, phi3_client, db_manager, web_search_service)

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
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

@app.on_event("startup")
async def startup_event():
    logger.info("Starting BrandonBot (100% Open Source - No Docker Required)...")
    logger.info("Initializing database...")
    await db_manager.initialize()
    logger.info("Initializing Weaviate (embedded mode)...")
    await weaviate_manager.initialize()
    logger.info("Loading Phi-3 model (CPU-optimized)...")
    await phi3_client.ensure_model_ready()
    logger.info("BrandonBot ready! Running entirely on open-source software.")

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(weaviate_manager, 'client') and weaviate_manager.client:
        weaviate_manager.client.close()
    await db_manager.close()
    await phi3_client.close()

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
    weaviate_status = weaviate_manager.client is not None if hasattr(weaviate_manager, 'client') else False
    phi3_status = await phi3_client.health_check()
    
    return {
        "status": "healthy" if weaviate_status and phi3_status else "unhealthy",
        "services": {
            "weaviate_embedded": "up" if weaviate_status else "down",
            "phi3_onnx": "up" if phi3_status else "down",
            "database": "up"
        },
        "note": "100% open-source, running locally on Replit"
    }

@app.post("/api/query")
async def query_bot(request: QueryRequest):
    try:
        response = await rag_pipeline.process_query(
            query=request.query,
            user_id=request.user_id,
            consent_given=request.consent_given
        )
        return response
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

app.mount("/static", StaticFiles(directory="static"), name="static")
