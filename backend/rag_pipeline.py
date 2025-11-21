import logging
from typing import Dict, List, Optional
from weaviate_manager import WeaviateManager
from phi3_client import Phi3Client
from database import DatabaseManager
from analysis_pipeline import QuestionAnalyzer

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, weaviate_manager: WeaviateManager, 
                 phi3_client: Phi3Client, db_manager: DatabaseManager):
        self.weaviate = weaviate_manager
        self.phi3 = phi3_client
        self.db = db_manager
        self.question_analyzer = QuestionAnalyzer(phi3_client)
        
        self.confidence_thresholds = {
            "BrandonPlatform": 0.3,
            "PreviousQA": 0.3,
            "PartyPlatform": 0.25
        }
        
        self.content_collections = [
            "BrandonPlatform",
            "PreviousQA", 
            "PartyPlatform"
        ]
    
    async def process_query(self, query: str, user_id: Optional[str] = None, 
                           consent_given: bool = False) -> Dict:
        try:
            try:
                question_analysis = self.question_analyzer.analyze_question(query)
            except Exception as e:
                logger.warning(f"Question analysis failed, using defaults: {e}")
                question_analysis = None
            
            content_results = []
            best_confidence = 0.0
            primary_source = None
            
            for collection in self.content_collections:
                results = await self.weaviate.search(
                    collection_name=collection,
                    query=query,
                    limit=10
                )
                
                threshold = self.confidence_thresholds[collection]
                
                for result in results:
                    if result["confidence"] >= threshold:
                        content_results.append({
                            **result,
                            "collection": collection
                        })
                        if result["confidence"] > best_confidence:
                            best_confidence = result["confidence"]
                            primary_source = collection
            
            style_guidance = []
            if question_analysis and best_confidence >= 0.7:
                style_guidance = await self._get_style_guidance(question_analysis)
            
            content_results.sort(key=lambda x: x["confidence"], reverse=True)
            all_results = content_results
            
            if not all_results or best_confidence < 0.5:
                response_text = self._generate_low_confidence_response(query)
                offer_callback = True
            else:
                content_context = self._build_context(all_results[:5])
                system_prompt = self._build_system_prompt(question_analysis, style_guidance)
                logger.info(f"System prompt length: {len(system_prompt)}, Question analysis: {question_analysis is not None}")
                llm_response = await self.phi3.generate_response(
                    query=query,
                    context=content_context,
                    system_prompt=system_prompt,
                    confidence=best_confidence
                )
                logger.info(f"LLM response length: {len(llm_response.get('response', ''))}, Model: {llm_response.get('model')}")
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
    
    async def _get_style_guidance(self, question_analysis) -> List[Dict]:
        """
        Query MarketGurus collection for communication style guidance based on question analysis.
        Returns copywriting principles that match the user's awareness level and emotional tone.
        """
        if not question_analysis.marketgurus_keywords:
            return []
        
        keywords_query = " ".join(question_analysis.marketgurus_keywords)
        
        try:
            results = await self.weaviate.search(
                collection_name="MarketGurus",
                query=keywords_query,
                limit=5
            )
            
            return [r for r in results if r["confidence"] >= 0.3]
        except Exception as e:
            logger.warning(f"Failed to get MarketGuru style guidance: {e}")
            return []
    
    def _build_system_prompt(self, question_analysis, style_guidance: List[Dict]) -> str:
        """
        Build system prompt that includes:
        - Marketing strategy (awareness level + emotional tone)
        - Stylistic directives ONLY (no raw MarketGuru content)
        """
        if not question_analysis:
            return "You are BrandonBot, speaking AS Brandon Sowers in first person. Be direct, conversational, and benefit-focused."
        
        prompt_parts = [
            "You are BrandonBot, speaking AS Brandon Sowers in first person.",
            f"\nCommunication Strategy: {question_analysis.marketing_strategy}"
        ]
        
        if style_guidance:
            style_directives = self._extract_style_directives(question_analysis, style_guidance)
            if style_directives:
                prompt_parts.append(f"\nStyle Instructions:\n{style_directives}")
        
        prompt_parts.extend([
            "\nKey Guidelines:",
            "- Speak in first person ('I believe', 'my position')",
            "- Be direct, conversational, benefit-focused",
            "- Match the user's awareness level and emotional tone"
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_style_directives(self, question_analysis, style_guidance: List[Dict]) -> str:
        """
        Extract ONLY stylistic directives from MarketGuru content.
        Returns template instructions, NOT raw text that could introduce factual claims.
        """
        awareness = question_analysis.awareness_level
        tone = question_analysis.emotional_tone
        
        directives = []
        
        if awareness == "unaware":
            directives.append("- Lead with the problem, not the solution")
            directives.append("- Build awareness before pitching")
        elif awareness == "problem_aware":
            directives.append("- Acknowledge their pain point immediately")
            directives.append("- Position your solution as the answer")
        elif awareness == "solution_aware":
            directives.append("- Show why your approach is superior")
            directives.append("- Use specific evidence and comparisons")
        elif awareness in ["product_aware", "most_aware"]:
            directives.append("- Reinforce their existing knowledge")
            directives.append("- Add depth and nuance")
        
        if tone == "concerned":
            directives.append("- Address concerns with empathy and evidence")
        elif tone == "frustrated":
            directives.append("- Acknowledge frustration, offer concrete solutions")
        elif tone == "curious":
            directives.append("- Satisfy curiosity with clear explanations")
        elif tone == "skeptical":
            directives.append("- Provide verifiable facts and citations")
        
        return "\n".join(directives) if directives else ""

