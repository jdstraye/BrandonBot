import logging
import re
from typing import Dict, List, Optional
from weaviate_manager import WeaviateManager
from phi3_client import Phi3Client
from database import DatabaseManager
from analysis_pipeline import QuestionAnalyzer
from retrieval_orchestrator import RetrievalOrchestrator
from web_search_service import WebSearchService

logger = logging.getLogger(__name__)

class RAGPipeline:
    
    CALLBACK_REQUEST_PATTERNS = [
        r'\bcall me\b',
        r'\bcontact me\b',
        r'\breach out\b',
        r'\bget in touch\b',
        r'\btouch base\b',
        r'\bcallback\b',
        r'\bcall back\b',
        r'\btalk to brandon\b',
        r'\bspeak (?:with|to) brandon\b',
        r'\bhave brandon call\b',
        r'\bschedule a call\b',
        r'\bphone me\b',
        r'\bgive me a call\b',
        r'\bi (?:want|need) (?:a |to )(?:talk|speak|callback)',
        r'\bcan (?:you|brandon) call me\b'
    ]
    
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
    
    def _detect_callback_request(self, query: str) -> bool:
        """Detect if the user is requesting a callback in their message"""
        query_lower = query.lower()
        for pattern in self.CALLBACK_REQUEST_PATTERNS:
            if re.search(pattern, query_lower):
                logger.info(f"Callback request detected in query (pattern: {pattern})")
                return True
        return False
    
    async def process_query(self, query: str, user_id: Optional[str] = None, 
                           consent_given: bool = False) -> Dict:
        try:
            callback_requested = self._detect_callback_request(query)
            
            try:
                question_analysis = self.question_analyzer.analyze_question(query)
                logger.info(f"Question type: {question_analysis.question_type}, " +
                           f"Needs dual sources: {question_analysis.needs_dual_sources}, " +
                           f"Comparison targets: {question_analysis.comparison_targets}, " +
                           f"Callback requested: {callback_requested}")
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
            
            # SIMPLIFIED RETRIEVAL-FIRST ARCHITECTURE
            # Always build context from what we found and let Phi-3 decide how to respond
            facts_context = self._build_facts_context(retrieval_context.content_results[:10])
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
                communication_strategy,
                callback_requested=callback_requested
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
            
            # Determine callback offer
            offer_callback = callback_requested or (best_confidence < 0.4)
            
            # ESSENTIAL: Enforce dual-source requirement for comparisons
            if question_analysis and question_analysis.needs_dual_sources and not retrieval_context.has_dual_sources:
                logger.warning("Comparison question without dual sources - offering callback for complete information")
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
        communication_strategy: str,
        callback_requested: bool = False
    ) -> str:
        """
        Build multi-section system prompt with clear separation:
        1. Facts Context (content sources)
        2. External Supplements (Bible, web)
        3. Communication Strategy (MarketGuru style guidance)
        
        RETRIEVAL-FIRST PHILOSOPHY:
        - Always present what you found, even if incomplete
        - Trust Phi-3 to explain gaps naturally
        - No hard-coded confidence thresholds
        """
        prompt_parts = [
            "You are BrandonBot, an AI assistant speaking AS Brandon Sowers (first person).",
            "\n" + "="*80,
            "\n=== SECTION 1: FACTUAL CONTEXT ===",
            "This is the authoritative information about Brandon's positions and relevant evidence:",
            "\n" + facts_context if facts_context.strip() else "\n[No specific information found in Brandon's platform documents]",
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
            "1. PRESENT WHAT YOU FOUND - Even if limited, share Section 1 facts. For debates: present Brandon's stance. Never say 'no information' if Section 1 has content.",
            "2. HANDLE GAPS - Missing opponent view? Present Brandon's fully, note 'need to research opponent'. Incomplete? Explain what you know + what's missing.",
            "3. OFFER CALLBACKS - After debates, incomplete comparisons, complex topics. Natural: 'Would you like Brandon to call?'",
            "4. STYLE - First person, direct, conversational (use Section 3). For Scripture: cite Section 2.",
        ])
        
        if callback_requested:
            prompt_parts.append("\nNOTE: User has explicitly requested a callback. Acknowledge this and confirm Brandon will reach out.")
        
        return "\n".join(prompt_parts)
    

