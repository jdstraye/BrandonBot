import logging
from typing import Dict, List, Optional
from weaviate_manager import WeaviateManager
from phi3_client import Phi3Client
from database import DatabaseManager

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, weaviate_manager: WeaviateManager, 
                 phi3_client: Phi3Client, db_manager: DatabaseManager):
        self.weaviate = weaviate_manager
        self.phi3 = phi3_client
        self.db = db_manager
        
        self.confidence_thresholds = {
            "BrandonPlatform": 0.8,
            "PreviousQA": 0.8,
            "PartyPlatform": 0.6,
            "MarketGurus": 0.25
        }
        
        self.collection_order = [
            "BrandonPlatform",
            "PreviousQA", 
            "PartyPlatform",
            "MarketGurus"
        ]
    
    async def process_query(self, query: str, user_id: Optional[str] = None, 
                           consent_given: bool = False) -> Dict:
        try:
            all_results = []
            best_confidence = 0.0
            primary_source = None
            
            for collection in self.collection_order:
                results = await self.weaviate.search(
                    collection_name=collection,
                    query=query,
                    limit=10
                )
                
                threshold = self.confidence_thresholds[collection]
                
                for result in results:
                    if result["confidence"] >= threshold:
                        all_results.append({
                            **result,
                            "collection": collection
                        })
                        if result["confidence"] > best_confidence:
                            best_confidence = result["confidence"]
                            primary_source = collection
                
                if all_results and primary_source in ["BrandonPlatform", "PreviousQA"]:
                    break
            
            if not all_results or best_confidence < 0.5:
                response_text = self._generate_low_confidence_response(query)
                offer_callback = True
            else:
                context = self._build_context(all_results[:5])
                llm_response = await self.phi3.generate_response(
                    query=query,
                    context=context,
                    confidence=best_confidence
                )
                response_text = llm_response["response"]
                offer_callback = best_confidence < 0.7
            
            sources = [
                {
                    "collection": r["collection"],
                    "source": r["source"],
                    "confidence": r["confidence"]
                } for r in all_results[:3]
            ]
            
            if consent_given or (user_id and await self.db.get_consent(user_id)):
                await self.db.log_interaction(
                    user_id=user_id or "anonymous",
                    query=query,
                    response=response_text,
                    confidence=best_confidence,
                    sources=sources,
                    consent_given=True
                )
            else:
                await self.db._track_new_question(query)
            
            return {
                "response": response_text,
                "confidence": best_confidence,
                "sources": sources,
                "offer_callback": offer_callback,
                "primary_source": primary_source
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return {
                "response": "I'm having technical difficulties. Would you like Brandon to call you back?",
                "confidence": 0.0,
                "sources": [],
                "offer_callback": True,
                "error": str(e)
            }
    
    def _build_context(self, results: List[Dict]) -> str:
        context_parts = []
        for r in results:
            source_label = r['collection'].replace('Platform', ' Platform').replace('QA', ' Q&A')
            context_parts.append(f"[{source_label} - {r['source']}]\n{r['content']}")
        return "\n\n".join(context_parts)
    
    def _generate_low_confidence_response(self, query: str) -> str:
        return (
            "I don't have enough reliable information to answer that question confidently. "
            "This is an important topic that deserves an accurate answer directly from Brandon. "
            "Would you like him to call you back to discuss this personally?"
        )
