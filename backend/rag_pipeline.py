import logging
from typing import Dict, List, Optional
from weaviate_manager import WeaviateManager
from phi3_client import Phi3Client
from database import DatabaseManager
from analysis_pipeline import QuestionAnalyzer
from retrieval_orchestrator import RetrievalOrchestrator
from web_search_service import WebSearchService

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, weaviate_manager: WeaviateManager, 
                 phi3_client: Phi3Client, db_manager: DatabaseManager,
                 web_search_service: Optional[WebSearchService] = None):
        self.weaviate = weaviate_manager
        self.phi3 = phi3_client
        self.db = db_manager
        self.question_analyzer = QuestionAnalyzer(phi3_client)
        
        self.orchestrator = RetrievalOrchestrator(
            weaviate_manager=weaviate_manager,
            web_search_service=web_search_service
        )
    
    async def process_query(self, query: str, user_id: Optional[str] = None, 
                           consent_given: bool = False) -> Dict:
        try:
            try:
                question_analysis = self.question_analyzer.analyze_question(query)
                logger.info(f"Question type: {question_analysis.question_type}, " +
                           f"Needs dual sources: {question_analysis.needs_dual_sources}, " +
                           f"Comparison targets: {question_analysis.comparison_targets}")
            except Exception as e:
                logger.warning(f"Question analysis failed, using defaults: {e}")
                question_analysis = None
            
            retrieval_context = await self.orchestrator.retrieve(
                question=query,
                analysis=question_analysis,
                limit_per_collection=5
            )
            
            logger.info(f"Retrieved: {len(retrieval_context.content_results)} content, " +
                       f"{len(retrieval_context.marketguru_results)} MarketGuru, " +
                       f"{len(retrieval_context.bible_results)} Bible, " +
                       f"{len(retrieval_context.web_results)} web results. " +
                       f"Best confidence: {retrieval_context.best_confidence:.2f}, " +
                       f"Dual sources: {retrieval_context.has_dual_sources}")
            
            if not retrieval_context.content_results or retrieval_context.best_confidence < 0.3:
                response_text = self._generate_low_confidence_response(query, question_analysis)
                offer_callback = True
                best_confidence = retrieval_context.best_confidence
            else:
                facts_context = self._build_facts_context(retrieval_context.content_results[:5])
                external_context = self._build_external_context(
                    retrieval_context.bible_results,
                    retrieval_context.web_results
                )
                communication_strategy = self._build_communication_strategy(
                    question_analysis,
                    retrieval_context.marketguru_results
                )
                
                system_prompt = self._build_multi_section_prompt(
                    question_analysis,
                    facts_context,
                    external_context,
                    communication_strategy
                )
                
                logger.info(f"System prompt length: {len(system_prompt)} chars")
                
                llm_response = await self.phi3.generate_response(
                    query=query,
                    context="",
                    system_prompt=system_prompt,
                    confidence=retrieval_context.best_confidence
                )
                
                response_text = llm_response["response"]
                best_confidence = retrieval_context.best_confidence
                offer_callback = best_confidence < 0.5
                
                if question_analysis and question_analysis.needs_dual_sources and not retrieval_context.has_dual_sources:
                    logger.warning("Comparison question lacking dual sources - offering callback")
                    offer_callback = True
            
            sources = [
                {
                    "collection": r.collection,
                    "source": r.metadata.get("source", "Unknown"),
                    "confidence": r.confidence
                } for r in retrieval_context.content_results[:3]
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
                "primary_source": sources[0]["collection"] if sources else None,
                "has_dual_sources": retrieval_context.has_dual_sources if question_analysis and question_analysis.needs_dual_sources else None
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}", exc_info=True)
            return {
                "response": "I'm having technical difficulties. Would you like Brandon to call you back?",
                "confidence": 0.0,
                "sources": [],
                "offer_callback": True,
                "error": str(e)
            }
    
    def _build_facts_context(self, content_results: list) -> str:
        """Build the facts context section from content retrieval results"""
        if not content_results:
            return "No specific information available."
        
        context_parts = []
        for r in content_results:
            source_label = r.collection.replace('Platform', ' Platform').replace('QA', ' Q&A')
            confidence_pct = int(r.confidence * 100)
            context_parts.append(
                f"[{source_label} | Confidence: {confidence_pct}% | Source: {r.metadata.get('source', 'Unknown')}]\n{r.text}"
            )
        return "\n\n".join(context_parts)
    
    def _build_external_context(self, bible_results: list, web_results: list) -> str:
        """Build external supplements section (Bible verses + web search)"""
        sections = []
        
        if bible_results:
            bible_parts = []
            for r in bible_results:
                bible_parts.append(f"[{r.metadata.get('reference', 'Scripture')}]\n{r.text}")
            sections.append("=== Biblical References ===\n" + "\n\n".join(bible_parts))
        
        if web_results:
            web_parts = []
            for r in web_results:
                source_label = "Official Website" if r.collection == "brandonsowers_web" else "External Source"
                web_parts.append(
                    f"[{source_label} | {r.metadata.get('title', 'Web Result')}]\n" +
                    f"URL: {r.metadata.get('url', 'N/A')}\n{r.text}"
                )
            sections.append("=== Web Search Results ===\n" + "\n\n".join(web_parts))
        
        return "\n\n".join(sections) if sections else ""
    
    def _build_communication_strategy(self, question_analysis, marketguru_results: list) -> str:
        """
        Build communication strategy section from MarketGuru snippets.
        FULL SEMANTIC RICHNESS preserved (no template reduction).
        Labeled as style-only to prevent factual bleeding.
        """
        if not question_analysis:
            return "Be direct, conversational, and benefit-focused."
        
        strategy_parts = [
            f"Awareness Level: {question_analysis.awareness_level}",
            f"Emotional Tone: {question_analysis.emotional_tone}",
            f"Overall Strategy: {question_analysis.marketing_strategy}",
        ]
        
        if marketguru_results:
            strategy_parts.append("\n=== Copywriting Principles (Style Guidance Only - NOT Factual Claims) ===")
            for i, r in enumerate(marketguru_results[:5], 1):
                source = r.metadata.get('source', 'Marketing Expert')
                strategy_parts.append(f"\n[Principle #{i} from {source}]\n{r.text}")
            
            strategy_parts.append("\nIMPORTANT: These are COMMUNICATION STYLE guidelines, not factual content. " +
                                "Use them to shape HOW you deliver Brandon's message, not WHAT you say.")
        
        return "\n".join(strategy_parts)
    
    def _build_multi_section_prompt(
        self,
        question_analysis,
        facts_context: str,
        external_context: str,
        communication_strategy: str
    ) -> str:
        """
        Build multi-section system prompt with clear separation:
        1. Facts Context (content sources)
        2. External Supplements (Bible, web)
        3. Communication Strategy (MarketGuru style guidance)
        """
        prompt_parts = [
            "You are BrandonBot, an AI assistant speaking AS Brandon Sowers (first person).",
            "\n" + "="*80,
            "\n=== SECTION 1: FACTUAL CONTEXT ===",
            "This is the authoritative information about Brandon's positions and relevant evidence:",
            "\n" + facts_context,
        ]
        
        if external_context:
            prompt_parts.extend([
                "\n" + "="*80,
                "\n=== SECTION 2: EXTERNAL SUPPLEMENTS ===",
                "Additional supporting information from external sources:",
                "\n" + external_context,
            ])
        
        if communication_strategy:
            prompt_parts.extend([
                "\n" + "="*80,
                "\n=== SECTION 3: COMMUNICATION STRATEGY ===",
                "HOW to communicate (style guidance, not facts):",
                "\n" + communication_strategy,
            ])
        
        prompt_parts.extend([
            "\n" + "="*80,
            "\n=== RESPONSE GUIDELINES ===",
            "- Speak in first person ('I believe', 'my position', 'my plan')",
            "- Base your answer ONLY on Section 1 (Factual Context)",
            "- Use Section 2 for supporting evidence when relevant",
            "- Use Section 3 to guide communication style, NOT content",
            "- Be direct, conversational, and benefit-focused",
            "- If comparison question: contrast both perspectives fairly",
            "- If uncertain: be honest and offer personal callback"
        ])
        
        return "\n".join(prompt_parts)
    
    def _generate_low_confidence_response(self, query: str, question_analysis) -> str:
        """Generate response for low-confidence scenarios"""
        if question_analysis and question_analysis.needs_dual_sources and question_analysis.comparison_targets:
            targets = ", ".join(question_analysis.comparison_targets)
            return (
                f"I want to give you a thorough comparison regarding {targets}, but I don't have " +
                "enough reliable information from both perspectives to answer confidently. " +
                "This deserves a complete answer directly from Brandon. " +
                "Would you like him to call you back to discuss this personally?"
            )
        
        return (
            "I don't have enough reliable information to answer that question confidently. "
            "This is an important topic that deserves an accurate answer directly from Brandon. "
            "Would you like him to call you back to discuss this personally?"
        )

