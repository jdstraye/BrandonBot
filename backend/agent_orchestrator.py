"""
AgentOrchestrator for BrandonBot LLM-First Architecture

The Orchestrator is the "nerves and hands" that:
1. Validates LLM tool call requests
2. Executes tools in the correct order
3. Manages multi-turn conversation history
4. Controls execution flow and security boundaries
5. Returns results to the LLM for synthesis

The LLM is the "brain" that reasons and recommends actions.
The Orchestrator controls what actually gets executed.
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from agent_tools import (
    ToolCall, ToolResult, ToolName, TOOL_SCHEMAS,
    get_gemini_tool_declarations
)
from query_expansion import detect_question_type, get_topic_from_query
from llm_providers import LLMProviderManager
from intent_detector import intent_detector, UserIntent, escalation_detector
from prequalifier import prequalifier, PrequalifierResult, FrustrationDecision, VaguenessDecision
from output_validator import output_validator, OVValidationResult, ValidationStatus, RejectionReason, OVSafeguard
from validation_debug import get_debug_db
from structured_response import (
    parse_structured_response,
    get_structured_output_instructions,
    get_ov_regeneration_instructions,
    ParsedResponse
)

logger = logging.getLogger(__name__)


class ConversationRole(str, Enum):
    """Roles in a conversation turn"""
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass
class ConversationTurn:
    """A single turn in the conversation"""
    role: ConversationRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List[ToolResult]] = None
    metadata: Optional[Dict] = None


@dataclass
class Session:
    """Conversation session with history"""
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    user_context: Dict = field(default_factory=dict)
    
    def add_turn(self, role: ConversationRole, content: str, 
                 tool_calls: Optional[List[ToolCall]] = None,
                 tool_results: Optional[List[ToolResult]] = None):
        """Add a conversation turn"""
        self.turns.append(ConversationTurn(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_results=tool_results
        ))
        self.last_active = datetime.now()
    
    def get_history_for_llm(self, max_turns: int = 10) -> List[Dict]:
        """Get conversation history formatted for LLM context"""
        history = []
        for turn in self.turns[-max_turns:]:
            entry = {
                "role": turn.role.value,
                "content": turn.content
            }
            if turn.tool_calls:
                entry["tool_calls"] = [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in turn.tool_calls
                ]
            if turn.tool_results:
                entry["tool_results"] = [
                    tr.to_context_string() for tr in turn.tool_results
                ]
            history.append(entry)
        return history
    
    def get_context_summary(self) -> str:
        """Get a summary of the conversation for context"""
        if not self.turns:
            return "This is the start of a new conversation."
        
        user_turns = [t for t in self.turns if t.role == ConversationRole.USER]
        topics_discussed = []
        
        for turn in user_turns[-5:]:
            topics_discussed.append(turn.content[:100])
        
        return f"Previous topics in this conversation: {'; '.join(topics_discussed)}"


class SessionManager:
    """Manages conversation sessions"""
    
    def __init__(self, max_sessions: int = 1000, session_timeout_minutes: int = 60):
        self.sessions: Dict[str, Session] = {}
        self.max_sessions = max_sessions
        self.session_timeout_minutes = session_timeout_minutes
    
    def get_or_create_session(self, session_id: str) -> Session:
        """Get existing session or create new one"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_active = datetime.now()
            return session
        
        self._cleanup_old_sessions()
        
        session = Session(session_id=session_id)
        self.sessions[session_id] = session
        return session
    
    def _cleanup_old_sessions(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired = []
        
        for sid, session in self.sessions.items():
            age_minutes = (now - session.last_active).total_seconds() / 60
            if age_minutes > self.session_timeout_minutes:
                expired.append(sid)
        
        for sid in expired:
            del self.sessions[sid]
        
        while len(self.sessions) >= self.max_sessions:
            oldest = min(self.sessions.items(), key=lambda x: x[1].last_active)
            del self.sessions[oldest[0]]


class ToolExecutor:
    """Executes validated tool calls"""
    
    TRUST_LEVELS = {
        "BrandonPlatform": 1.0,
        "PreviousQA": 1.0,
        "PartyPlatform": 0.6,
        "MarketGurus": 0.8,
        "brandonsowers.com": 0.9,
        "web": 0.3
    }
    
    def __init__(self, weaviate_manager, web_search_service=None):
        self.weaviate = weaviate_manager
        self.web_search = web_search_service
    
    async def execute(self, tool_call: ToolCall, session_id: str = "default") -> ToolResult:
        """Execute a single tool call"""
        is_valid, error = tool_call.validate()
        if not is_valid:
            return ToolResult(
                tool_name=tool_call.name,
                success=False,
                data=None,
                error_message=error
            )
        
        try:
            if tool_call.name == ToolName.SEARCH_BRANDON_POSITIONS.value:
                return await self._execute_search_brandon_positions(tool_call.arguments)
            elif tool_call.name == ToolName.AUGMENT_BRANDON_WITH_PARTY.value:
                return await self._execute_augment_with_party(tool_call.arguments)
            elif tool_call.name == ToolName.GET_PARTY_COMPARISON.value:
                return await self._execute_party_comparison(tool_call.arguments)
            elif tool_call.name == ToolName.PERFORM_WEB_SEARCH.value:
                return await self._execute_web_search(tool_call.arguments, session_id)
            elif tool_call.name == ToolName.RETRIEVE_ANSWER_STYLE.value:
                return await self._execute_retrieve_answer_style(tool_call.arguments)
            elif tool_call.name == ToolName.REGISTER_VOLUNTEER.value:
                return await self._execute_register_volunteer(tool_call.arguments)
            elif tool_call.name == ToolName.MAKE_DONATION.value:
                return await self._execute_make_donation(tool_call.arguments)
            elif tool_call.name == ToolName.CHECK_FEC_RULES.value:
                return await self._execute_check_fec_rules(tool_call.arguments)
            else:
                return ToolResult(
                    tool_name=tool_call.name,
                    success=False,
                    data=None,
                    error_message=f"Tool not implemented: {tool_call.name}"
                )
        except Exception as e:
            logger.error(f"Tool execution error for {tool_call.name}: {e}")
            return ToolResult(
                tool_name=tool_call.name,
                success=False,
                data=None,
                error_message=str(e)
            )
    
    async def _execute_search_brandon_positions(self, args: Dict) -> ToolResult:
        """Search Brandon's AUTHORITATIVE positions (BrandonPlatform + PreviousQA only)"""
        query = args.get("query", "")
        limit_val = args.get("limit", 5)
        if isinstance(limit_val, float):
            limit_val = int(limit_val)
        limit = min(limit_val, 10)
        
        collections = ["BrandonPlatform", "PreviousQA"]
        all_results = []
        sources = []
        
        logger.info(f"Searching AUTHORITATIVE collections: {collections} for query: '{query}' limit: {limit}")
        
        for collection in collections:
            try:
                results = await self.weaviate.search(collection, query, limit=limit)
                logger.info(f"Collection {collection} returned {len(results)} results")
                for r in results:
                    r["collection"] = collection
                    r["trust_level"] = self.TRUST_LEVELS.get(collection, 1.0)
                    all_results.append(r)
                    if r.get("source"):
                        sources.append(r["source"])
            except Exception as e:
                logger.warning(f"Search failed for {collection}: {e}")
        
        all_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        top_results = all_results[:limit]
        
        avg_confidence = sum(r.get("confidence", 0) for r in top_results) / len(top_results) if top_results else 0
        
        return ToolResult(
            tool_name=ToolName.SEARCH_BRANDON_POSITIONS.value,
            success=True,
            data=top_results,
            confidence=avg_confidence,
            trust_level=1.0,
            sources=list(set(sources))
        )
    
    async def _execute_augment_with_party(self, args: Dict) -> ToolResult:
        """Augment Brandon's position with party platform context (clearly labeled)"""
        query = args.get("query", "")
        brandon_topic = args.get("brandon_topic", "")
        limit_val = args.get("limit", 3)
        if isinstance(limit_val, float):
            limit_val = int(limit_val)
        limit = min(limit_val, 5)
        
        all_results = []
        sources = []
        
        logger.info(f"Augmenting with PartyPlatform for query: '{query}'")
        
        try:
            results = await self.weaviate.search("PartyPlatform", query, limit=limit)
            logger.info(f"PartyPlatform returned {len(results)} results")
            for r in results:
                r["collection"] = "PartyPlatform"
                r["trust_level"] = self.TRUST_LEVELS.get("PartyPlatform", 0.6)
                r["is_party_position"] = True
                r["not_brandon_position"] = True
                all_results.append(r)
                if r.get("source"):
                    sources.append(r["source"])
        except Exception as e:
            logger.warning(f"PartyPlatform search failed: {e}")
        
        avg_confidence = sum(r.get("confidence", 0) for r in all_results) / len(all_results) if all_results else 0
        
        return ToolResult(
            tool_name=ToolName.AUGMENT_BRANDON_WITH_PARTY.value,
            success=True,
            data={
                "brandon_context": brandon_topic,
                "party_augmentation": all_results,
                "note": "These are PARTY PLATFORM positions, NOT Brandon's personal views. Present as 'Party platforms suggest...' or 'Republican/Independent positions include...'"
            },
            confidence=avg_confidence * 0.6,
            trust_level=0.6,
            sources=list(set(sources))
        )
    
    async def _execute_party_comparison(self, args: Dict) -> ToolResult:
        """Get Republican and Independent positions for explicit comparison questions"""
        topic = args.get("topic", "")
        include_republican = args.get("include_republican", True)
        include_independent = args.get("include_independent", True)
        
        republican_results = []
        independent_results = []
        sources = []
        
        query = f"{topic} policy position platform"
        
        logger.info(f"Getting party comparison for topic: '{topic}'")
        
        try:
            results = await self.weaviate.search("PartyPlatform", query, limit=6)
            for r in results:
                source = r.get("source", "").lower()
                if "republican" in source or "rnc" in source or "gop" in source:
                    if include_republican:
                        r["party"] = "Republican"
                        republican_results.append(r)
                elif "independent" in source:
                    if include_independent:
                        r["party"] = "Independent"
                        independent_results.append(r)
                else:
                    if include_republican:
                        r["party"] = "Republican"
                        republican_results.append(r)
                
                if r.get("source"):
                    sources.append(r["source"])
        except Exception as e:
            logger.warning(f"Party comparison search failed: {e}")
        
        contradictions = []
        if republican_results and independent_results:
            contradictions.append("Note: Republican and Independent platforms may have different positions on this topic. Present both clearly labeled.")
        
        return ToolResult(
            tool_name=ToolName.GET_PARTY_COMPARISON.value,
            success=True,
            data={
                "topic": topic,
                "republican_positions": republican_results[:3],
                "independent_positions": independent_results[:3],
                "contradictions": contradictions,
                "note": "These are PARTY positions for comparison purposes. NOT Brandon's personal views."
            },
            trust_level=0.6,
            sources=list(set(sources))
        )
    
    async def _execute_check_fec_rules(self, args: Dict) -> ToolResult:
        """Check FEC campaign finance rules"""
        query_type = args.get("query_type", "general_rules")
        amount = args.get("amount")
        donor_type = args.get("donor_type", "individual")
        
        fec_rules = {
            "individual_limit": {
                "limit_per_candidate_per_election": 3300,
                "limit_per_pac_per_year": 5000,
                "limit_per_party_per_year": 41300,
                "aggregate_limit": "No federal aggregate limit (McCutcheon v. FEC)",
                "note": "Limits are per election (primary and general are separate)"
            },
            "pac_limit": {
                "multicandidate_pac_to_candidate": 5000,
                "nonconnected_pac_to_candidate": 2900,
                "note": "PAC must receive contributions from more than 50 persons to qualify as multicandidate"
            },
            "corporate_rules": {
                "direct_contributions": "PROHIBITED - Corporations cannot contribute directly to federal candidates",
                "pac_contributions": "Corporations may establish a PAC using treasury funds for admin costs",
                "super_pac": "Corporations may make unlimited independent expenditures through super PACs"
            },
            "disclosure_requirements": {
                "threshold": 200,
                "required_info": ["Full name", "Mailing address", "Occupation", "Employer"],
                "note": "Required for aggregate contributions over $200 in a calendar year"
            },
            "foreign_national_rules": {
                "status": "ABSOLUTELY PROHIBITED",
                "applies_to": ["Federal elections", "State elections", "Local elections"],
                "note": "Foreign nationals may not make any contribution or donation, directly or indirectly"
            },
            "general_rules": {
                "cash_limit": 100,
                "anonymous_limit": 50,
                "earmarking": "Contributions earmarked for a candidate count against limits",
                "joint_fundraising": "Joint fundraising committees may collect larger amounts for distribution"
            }
        }
        
        result = fec_rules.get(query_type, fec_rules["general_rules"])
        
        validation_result = None
        if amount is not None:
            if donor_type == "individual":
                limit = 3300
                if amount > limit:
                    validation_result = f"EXCEEDS LIMIT: ${amount} exceeds individual limit of ${limit} per election"
                elif amount > 200:
                    validation_result = f"DISCLOSURE REQUIRED: Donations over $200 require donor occupation and employer"
                else:
                    validation_result = f"VALID: ${amount} is within legal limits"
            elif donor_type == "corporation":
                validation_result = "PROHIBITED: Corporations cannot make direct contributions to federal candidates"
        
        return ToolResult(
            tool_name=ToolName.CHECK_FEC_RULES.value,
            success=True,
            data={
                "query_type": query_type,
                "rules": result,
                "amount_validation": validation_result,
                "disclaimer": "This is informational only. Consult FEC.gov or campaign counsel for official guidance."
            },
            trust_level=1.0
        )
    
    async def _execute_web_search(self, args: Dict, session_id: str = "default") -> ToolResult:
        """Search the web using DuckDuckGo (rate limited)"""
        from security import rate_limiter
        
        is_allowed, wait_seconds = rate_limiter.check_rate_limit(session_id, "web_search")
        if not is_allowed:
            return ToolResult(
                tool_name=ToolName.PERFORM_WEB_SEARCH.value,
                success=False,
                data=None,
                error_message=f"Web search rate limit exceeded. Please wait {wait_seconds} seconds."
            )
        
        query = args.get("query", "")
        num_results = min(args.get("num_results", 5), 10)
        news_only = args.get("news_only", False)
        
        if self.web_search is None:
            from web_search_service import WebSearchService
            self.web_search = WebSearchService()
        
        try:
            if news_only:
                results = await self.web_search.search_news(query, max_results=num_results)
            else:
                results = await self.web_search.search(query, max_results=num_results)
            
            formatted_results = []
            sources = []
            
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", r.get("snippet", "")),
                    "url": r.get("href", r.get("url", "")),
                    "source": r.get("source", "web")
                })
                if r.get("href"):
                    sources.append(r["href"])
            
            return ToolResult(
                tool_name=ToolName.PERFORM_WEB_SEARCH.value,
                success=True,
                data=formatted_results,
                sources=sources
            )
        except Exception as e:
            return ToolResult(
                tool_name=ToolName.PERFORM_WEB_SEARCH.value,
                success=False,
                data=None,
                error_message=f"Web search failed: {e}"
            )
    
    async def _execute_retrieve_answer_style(self, args: Dict) -> ToolResult:
        """Retrieve copywriting style guidance from MarketGurus"""
        question_type = args.get("question_type", "simple_inquiry")
        topic = args.get("topic", "general")
        desired_tone = args.get("desired_tone", "educational")
        
        style_query = f"{question_type} {topic} {desired_tone} marketing communication style"
        
        try:
            results = await self.weaviate.search("MarketGurus", style_query, limit=3)
            
            filtered_results = []
            for r in results:
                metadata = r.get("metadata", "{}")
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                question_types = metadata.get("question_types", "[]")
                if isinstance(question_types, str):
                    try:
                        question_types = json.loads(question_types)
                    except:
                        question_types = []
                
                if question_type in question_types or not question_types:
                    filtered_results.append(r)
            
            if not filtered_results:
                filtered_results = results[:2]
            
            return ToolResult(
                tool_name=ToolName.RETRIEVE_ANSWER_STYLE.value,
                success=True,
                data=filtered_results,
                confidence=sum(r.get("confidence", 0) for r in filtered_results) / len(filtered_results) if filtered_results else 0
            )
        except Exception as e:
            return ToolResult(
                tool_name=ToolName.RETRIEVE_ANSWER_STYLE.value,
                success=False,
                data=None,
                error_message=f"Style retrieval failed: {e}"
            )
    
    async def _execute_register_volunteer(self, args: Dict) -> ToolResult:
        """Register a campaign volunteer (STUB - needs CRM integration)"""
        name = args.get("name", "")
        email = args.get("email", "")
        phone = args.get("phone", "")
        zip_code = args.get("zip_code", "")
        interests = args.get("interests", [])
        availability = args.get("availability", "flexible")
        
        if not name or not email:
            return ToolResult(
                tool_name=ToolName.REGISTER_VOLUNTEER.value,
                success=False,
                data=None,
                error_message="Name and email are required"
            )
        
        if "@" not in email:
            return ToolResult(
                tool_name=ToolName.REGISTER_VOLUNTEER.value,
                success=False,
                data=None,
                error_message="Invalid email format"
            )
        
        volunteer_id = f"VOL-{int(time.time())}"
        
        volunteer_data = {
            "volunteer_id": volunteer_id,
            "name": name,
            "email": email,
            "phone": phone,
            "zip_code": zip_code,
            "interests": interests,
            "availability": availability,
            "registered_at": datetime.now().isoformat(),
            "status": "pending_confirmation"
        }
        
        logger.info(f"VOLUNTEER REGISTRATION (STUB): {json.dumps(volunteer_data)}")
        
        return ToolResult(
            tool_name=ToolName.REGISTER_VOLUNTEER.value,
            success=True,
            data={
                "volunteer_id": volunteer_id,
                "message": f"Thank you, {name}! You've been registered as a volunteer.",
                "next_steps": [
                    "Check your email for a confirmation link",
                    "Complete your volunteer profile",
                    "Join our next volunteer orientation"
                ],
                "note": "STUB: In production, this would integrate with the campaign CRM"
            }
        )
    
    async def _execute_make_donation(self, args: Dict) -> ToolResult:
        """Process a donation (STUB - needs payment integration)"""
        amount = args.get("amount", 0)
        donor_name = args.get("donor_name", "")
        donor_email = args.get("donor_email", "")
        employer = args.get("employer", "")
        occupation = args.get("occupation", "")
        recurring = args.get("recurring", False)
        
        if amount <= 0:
            return ToolResult(
                tool_name=ToolName.MAKE_DONATION.value,
                success=False,
                data=None,
                error_message="Donation amount must be greater than $0"
            )
        
        if amount > 3300:
            return ToolResult(
                tool_name=ToolName.MAKE_DONATION.value,
                success=False,
                data=None,
                error_message="Individual contribution limit is $3,300 per election (FEC regulation)"
            )
        
        if amount > 200 and (not employer or not occupation):
            return ToolResult(
                tool_name=ToolName.MAKE_DONATION.value,
                success=False,
                data=None,
                error_message="For donations over $200, FEC requires employer and occupation information"
            )
        
        donation_id = f"DON-{int(time.time())}"
        
        donation_data = {
            "donation_id": donation_id,
            "amount": amount,
            "donor_name": donor_name,
            "donor_email": donor_email,
            "employer": employer,
            "occupation": occupation,
            "recurring": recurring,
            "created_at": datetime.now().isoformat(),
            "status": "pending_payment"
        }
        
        logger.info(f"DONATION REQUEST (STUB): {json.dumps(donation_data)}")
        
        secure_link = f"https://secure.brandonbot.com/donate/{donation_id}?amount={amount}"
        
        return ToolResult(
            tool_name=ToolName.MAKE_DONATION.value,
            success=True,
            data={
                "donation_id": donation_id,
                "amount": amount,
                "recurring": recurring,
                "secure_link": secure_link,
                "message": f"Thank you for your ${amount} {'monthly ' if recurring else ''}contribution!",
                "next_steps": [
                    f"Click the secure link to complete your donation: {secure_link}",
                    "You'll receive a tax receipt via email",
                    "Contributions are tax-deductible to the extent allowed by law"
                ],
                "note": "STUB: In production, this would integrate with ActBlue/Stripe"
            }
        )


class AgentOrchestrator:
    """
    The main orchestrator that controls the LLM-first agent flow.
    
    The Orchestrator is the "nerves and hands" - it:
    1. Receives user input and conversation history
    2. Sends to LLM for reasoning and tool recommendations
    3. Validates and executes recommended tool calls
    4. Returns tool results to LLM for synthesis
    5. Delivers final response to user
    
    The LLM is the "brain" - it reasons but doesn't execute.
    
    Multi-Provider Support:
    - Uses LLMProviderManager to handle multiple LLM providers
    - One provider+model per conversation (no mid-conversation switching)
    - Automatic failover on rate limits or errors
    - Tracks which model handled each query for evaluation
    """
    
    def __init__(self, weaviate_manager, web_search_service=None, db_manager=None, llm_client=None, slm_manager=None):
        """
        Initialize the orchestrator.
        
        Args:
            weaviate_manager: WeaviateManager for vector search
            web_search_service: Optional WebSearchService for web search
            db_manager: Optional DatabaseManager for logging model performance
            llm_client: Legacy parameter (ignored if LLMProviderManager is used)
            slm_manager: SLMManager for local classification tasks (prequalifier/validator)
        """
        self.llm_manager = LLMProviderManager()
        self.llm = llm_client
        self.db_manager = db_manager
        self.slm_manager = slm_manager
        self.tool_executor = ToolExecutor(weaviate_manager, web_search_service)
        self.session_manager = SessionManager()
        self.max_tool_iterations = 5
    
    def get_system_prompt(self, question_types: List[str] = None, topic: str = None, internal_hints_block: str = None) -> str:
        """Get the system prompt for the LLM with optional question type hints and internal context.
        
        Args:
            question_types: List of detected question types
            topic: Detected topic
            internal_hints_block: Pre-formatted internal hints block from PQ (InternalHints.to_system_prompt_block())
        """
        
        question_type_hint = ""
        if question_types:
            question_type_hint = f"\n\nDETECTED QUESTION TYPE: {', '.join(question_types)}"
            emotional_types = {"emotional", "emotional_appeal", "values_based", "faith"}
            if any(qt in emotional_types for qt in question_types):
                question_type_hint += "\nConsider using retrieve_answer_style with include_scripture=true"
            if "comparison" in question_types:
                question_type_hint += "\nUse get_party_comparison for explicit party position comparison"
            if "callback" in question_types:
                question_type_hint += "\nUser may want a callback - be ready to offer one"
        
        topic_hint = f"\nDETECTED TOPIC: {topic}" if topic else ""
        
        # Internal hints block goes at the END of system prompt (before structured output instructions)
        # These are sideband signals that should NEVER appear in user-facing output
        internal_context = ""
        if internal_hints_block:
            internal_context = f"\n\n{internal_hints_block}\nIMPORTANT: The INTERNAL_CONTEXT above is guidance for you. NEVER include any text from it in your response to the user."
        
        return f"""You are BrandonBot, an AI assistant for Brandon Sowers' political campaign.
{question_type_hint}{topic_hint}

YOUR ROLE:
- Answer questions about Brandon's policies, positions, and campaign
- Help users volunteer or donate
- Compare Brandon's positions to party platforms when asked
- Maintain a helpful, informative, and persuasive tone

QUESTION TYPE CLASSIFICATION (Schwartz Framework):
- unaware: User doesn't know they have a problem - use storytelling, don't push solutions
- problem_aware: User knows the problem but not Brandon - empathize, show understanding
- solution_aware: User knows solutions exist - differentiate Brandon's approach
- product_aware: User knows Brandon - provide specifics, address concerns
- most_aware: Strong supporter - call to action (donate, volunteer)
- oppositional: Disagrees with Brandon - acknowledge concerns, find common ground
- skeptical: Doubts claims - provide proof, cite sources
- emotional_appeal: Values/faith-based concern - include scripture when appropriate
- comparison: Explicitly comparing candidates/parties

AVAILABLE TOOLS (Trust-Based Separation):

AUTHORITATIVE SOURCES (Trust 1.0):
1. search_brandon_positions: Search Brandon's OWN statements and verified Q&A
   - Use FIRST for any question about Brandon's positions
   - Results are authoritative and should be quoted directly

SUPPLEMENTARY SOURCES (Trust 0.6):
2. augment_brandon_with_party: Get party platform context when Brandon's position is thin
   - Use ONLY if search_brandon_positions returns insufficient results
   - Results must be labeled as "party position" NOT Brandon's view
3. get_party_comparison: Compare Republican and Independent platforms
   - Use ONLY for explicit comparison questions

OTHER TOOLS:
4. perform_web_search: Search the internet for current events, competitor info, statistics, Brandon's positions not covered in search_brandon_positions, Brandon's public appearances and plans.
   - Use for competitor research, recent news, external claims to verify
5. retrieve_answer_style: Get copywriting guidance (use after gathering facts)
6. register_volunteer: Sign up volunteers
7. make_donation: Process donation requests
8. check_fec_rules: Verify FEC compliance for donations

GREETINGS AND SMALL TALK:
- For greetings like "Hi" or "How are you?", respond warmly and ask how you can help
- Do NOT fabricate Brandon's current activities or whereabouts
- If you want to share what Brandon is doing, USE perform_web_search first to find real information
- Prefer asking clarifying questions over inventing details

SENSITIVE TOPICS (offer callback):
- Abortion, gun rights, immigration enforcement
- Personal attacks on opponents
- Unverified claims or rumors
- Legal/financial advice beyond campaign info

CRITICAL RULES:
- NEVER conflate party positions with Brandon's personal views
- After receiving tool results, SYNTHESIZE into a final response
- If confidence is low, offer a callback from someone from the team
- Always cite sources when using information from tools
- For emotional/values questions, consider scripture inclusion
- NEVER invent specific dates, events, town halls, speeches, or quotes that are not in your retrieved context
- If you don't have a specific source for a claim, use general language like "Brandon has stated" rather than fabricating citations
{internal_context}
Remember: You're here to inform voters and build support for Brandon's campaign.
""" + get_structured_output_instructions()

    async def process_message(self, user_message: str, session_id: str) -> Tuple[str, Dict]:
        """
        Process a user message through the full 3-stage agent pipeline:
        
        Stage 1 - Prequalifier (PQ):
            - Rate limiting
            - Input sanitization
            - Hybrid frustration detection (pattern flags → SLM → ESCALATE/CONTINUE)
            - RAG-based vagueness detection (RAG confidence → SLM → CLEAR/VAGUE)
            - Prompt enrichment based on 2x2 matrix
        
        Stage 2 - LLM Agent:
            - Tool calling with enriched prompt from PQ
            - Multi-turn reasoning
        
        Stage 3 - Output Validator (OV):
            - Intent fulfillment check (SLM)
            - Ethics/morality check (SLM)
            - FEC compliance (RAG + SLM double-check)
            - De-escalation check (for frustrated users)
            - PII redaction (hybrid regex + SLM)
            - Citation verification
            - Regeneration loop on failure
        
        Args:
            user_message: The user's input
            session_id: Unique session identifier for conversation continuity
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        session = self.session_manager.get_or_create_session(session_id)
        history = [{"role": t.role.value, "content": t.content} for t in session.turns]
        
        # ===== STAGE 1: PREQUALIFIER =====
        # Set up PQ with dependencies
        prequalifier.set_weaviate_manager(self.tool_executor.weaviate)
        if self.slm_manager:
            prequalifier.set_slm_provider(self.slm_manager)
        
        pq_result = await prequalifier.analyze(user_message, session_id, history)
        
        # Handle blocked messages
        if pq_result.blocked:
            logger.warning(f"[{request_id}] Message blocked by prequalifier: {pq_result.block_reason}")
            return (
                "I'm not able to help with that particular request. Is there something else about Brandon's campaign I can help you with?",
                {
                    "request_id": request_id,
                    "blocked": True,
                    "block_reason": pq_result.block_reason,
                    "rate_limited": pq_result.rate_limited,
                    "duration_ms": int((time.time() - start_time) * 1000)
                }
            )
        
        # Handle rate limiting
        if pq_result.rate_limited:
            logger.warning(f"[{request_id}] Rate limited - wait {pq_result.rate_limit_wait_seconds}s")
            return (
                f"I'm receiving too many requests. Please wait a moment and try again.",
                {
                    "request_id": request_id,
                    "rate_limited": True,
                    "wait_seconds": pq_result.rate_limit_wait_seconds,
                    "duration_ms": int((time.time() - start_time) * 1000)
                }
            )
        
        # Use sanitized message from PQ
        sanitized_message = pq_result.sanitized_message or user_message
        
        # Detect question types and topic
        question_types = detect_question_type(sanitized_message)
        topic = get_topic_from_query(sanitized_message)
        
        # Get intent context from existing intent detector
        intent_result = intent_detector.detect(sanitized_message, history)
        intent_context = intent_detector.get_intent_context(intent_result)
        
        # Determine if user is frustrated based on PQ decision
        user_frustrated = pq_result.frustration_decision == FrustrationDecision.ESCALATE
        query_vague = pq_result.vagueness_decision == VaguenessDecision.VAGUE
        
        logger.info(f"[{request_id}] PQ Analysis - session: {session_id}, "
                   f"frustration: {pq_result.frustration_decision.value}, "
                   f"vagueness: {pq_result.vagueness_decision.value}, "
                   f"passthrough: {pq_result.passthrough}, "
                   f"sanitized: {pq_result.sanitization_applied}")
        
        session.add_turn(ConversationRole.USER, sanitized_message)
        
        metadata = {
            "request_id": request_id,
            "session_id": session_id,
            "tool_calls": [],
            "iterations": 0,
            "regeneration_attempts": 0,
            "total_tokens": 0,
            "sources": [],
            "model_used": None,
            "provider": None,
            "latency_ms": 0,
            "question_types": question_types,
            "topic": topic,
            "intent": intent_result.primary_intent.value,
            "needs_scripture": intent_result.needs_scripture,
            "needs_callback": intent_result.needs_callback or user_frustrated,
            "pq_frustration": pq_result.frustration_decision.value,
            "pq_vagueness": pq_result.vagueness_decision.value,
            "pq_passthrough": pq_result.passthrough,
            "sanitization_applied": pq_result.sanitization_applied,
            "sanitization_issues": pq_result.sanitization_issues,
            "user_frustrated": user_frustrated,
            "query_vague": query_vague,
            "detected_emotion": getattr(pq_result, 'detected_emotion', 'neutral'),
            "duration_ms": 0,
            "validation_status": None,
            "validation_rejections": [],
            "ov_modifications": []
        }
        
        try:
            messages = self._build_messages(session)
            
            # ===== STAGE 2: LLM AGENT WITH TOOLS =====
            # Build system prompt with PQ enrichment and internal hints
            internal_hints_block = pq_result.internal_hints.to_system_prompt_block() if pq_result.internal_hints else ""
            full_system_prompt = self.get_system_prompt(question_types, topic, internal_hints_block)
            
            # Log internal hints to debug.db for forensic analysis
            if pq_result.internal_hints and internal_hints_block:
                try:
                    debug_db = get_debug_db()
                    debug_db.log_internal_hints(
                        query=sanitized_message,
                        internal_hints=pq_result.internal_hints,
                        session_id=session_id,
                        request_id=request_id
                    )
                except Exception as e:
                    logger.warning(f"[{request_id}] Failed to log internal hints: {e}")
            
            # Add intent context
            if intent_context:
                full_system_prompt += f"\n\nINTENT ANALYSIS: {intent_context}"
            if intent_result.needs_scripture:
                full_system_prompt += "\n\nNote: User may appreciate faith-based perspective. Consider including relevant scripture if appropriate."
            if intent_result.needs_callback or user_frustrated:
                full_system_prompt += "\n\nNote: User appears to need personal attention. Offer a callback from someone on the team."
            
            # Add PQ enrichment if not passthrough
            if not pq_result.passthrough and pq_result.enriched_prompt:
                full_system_prompt += f"\n\n===== PREQUALIFIER INSTRUCTIONS =====\n{pq_result.pq_instructions or ''}\n\n{pq_result.enriched_prompt}"
            
            # Add escalation context if frustrated
            if user_frustrated:
                full_system_prompt += f"\n\nESCALATION DETECTED: User is frustrated (decision: {pq_result.frustration_decision.value}). Prioritize empathy and de-escalation."
            
            # Add detected emotion context for style adaptation
            detected_emotion = getattr(pq_result, 'detected_emotion', 'neutral') or 'neutral'
            if detected_emotion != 'neutral':
                emotion_guidance = self._get_emotion_style_guidance(detected_emotion)
                full_system_prompt += f"\n\nUSER EMOTION: The user appears to be feeling {detected_emotion}. {emotion_guidance}"
            
            # Add vagueness context
            if query_vague:
                full_system_prompt += f"\n\nVAGUE QUERY DETECTED: The user's question needs clarification. Ask clarifying questions before providing a detailed answer."
            
            iteration = 0
            final_response = None
            
            while iteration < self.max_tool_iterations:
                iteration += 1
                metadata["iterations"] = iteration
                
                llm_response = await self.llm_manager.generate_with_tools(
                    session_id=session_id,
                    messages=messages,
                    tools=get_gemini_tool_declarations(),
                    system_prompt=full_system_prompt
                )
                
                metadata["total_tokens"] += llm_response.tokens_used
                metadata["model_used"] = f"{llm_response.provider}/{llm_response.model}"
                metadata["provider"] = llm_response.provider
                metadata["latency_ms"] = llm_response.latency_ms
                
                tool_calls = llm_response.tool_calls or []
                
                if not tool_calls:
                    raw_response = llm_response.text or "I'm sorry, I couldn't process your request."
                    
                    # Parse structured response to separate reasoning from user-facing content
                    parsed = parse_structured_response(raw_response)
                    proposed_response = parsed.final_response or raw_response
                    
                    # Log reasoning to debug DB if present
                    if parsed.reasoning:
                        try:
                            debug_db = get_debug_db()
                            debug_db.log_reasoning(
                                session_id=session_id,
                                request_id=request_id,
                                reasoning=parsed.reasoning,
                                parse_method=parsed.parse_method,
                                raw_response=parsed.raw_response[:2000]
                            )
                        except Exception as db_err:
                            logger.debug(f"[{request_id}] Failed to log reasoning: {db_err}")
                    
                    metadata["response_parse_method"] = parsed.parse_method
                    
                    # Check for "intent to search" without actual tool call
                    # LLM sometimes says "I will search..." but doesn't call tools
                    # Only trigger if response is SHORT (< 200 chars) AND contains intent phrase ANYWHERE
                    import re
                    intent_to_search_patterns = [
                        r"I (?:will|shall|am going to|'ll) (?:now )?(?:search|check|verify|look up)",
                        r"I (?:need to|should) (?:search|check|verify)",
                        r"Let me (?:search|check|verify|look up|retrieve)",
                        r"(?:searching|search) (?:Brandon's|his|the) (?:official )?positions?",
                        r"check his (?:official )?position",
                        r"One moment (?:while|as) I (?:verify|check|search)",
                        r"verify Brandon's (?:official )?position",
                    ]
                    is_stub_response = len(proposed_response.strip()) < 200
                    intent_to_search = is_stub_response and any(
                        re.search(pattern, proposed_response, re.IGNORECASE) 
                        for pattern in intent_to_search_patterns
                    )
                    # Count how many times we've already tried to force action on intent stubs
                    intent_force_count = sum(1 for m in messages if "SYSTEM: You said you would search" in m.get("content", ""))
                    
                    if intent_to_search and iteration < self.max_tool_iterations:
                        if intent_force_count < 2:
                            logger.warning(f"[{request_id}] Intent-to-search stub detected (attempt {intent_force_count + 1}) - forcing action")
                            # Append the stub response to history so LLM has context
                            messages.append({
                                "role": "assistant",
                                "content": proposed_response
                            })
                            messages.append({
                                "role": "user",
                                "content": """SYSTEM: You said you would search but didn't actually call a tool.

Please either:
1. Call search_brandon_positions NOW to get the information, OR
2. Provide your final answer immediately without mentioning searching

Do NOT say you will search - either search or answer."""
                            })
                            continue
                        else:
                            # After 2 failed attempts, force a search ourselves
                            logger.warning(f"[{request_id}] Intent stub persisted after {intent_force_count} attempts - forcing search")
                            try:
                                # Extract topic from the original question for search
                                forced_search = ToolCall(
                                    name="search_brandon_positions",
                                    arguments={"query": sanitized_message, "limit": 5},
                                    call_id="forced_search"
                                )
                                forced_result = await self.tool_executor.execute(forced_search, session_id)
                                forced_context = forced_result.to_context_string()
                                
                                # Add the forced search results to messages
                                messages.append({
                                    "role": "assistant",
                                    "content": proposed_response
                                })
                                messages.append({
                                    "role": "tool",
                                    "content": f"FORCED SEARCH RESULTS (system auto-executed):\n\n{forced_context}\n\nNow provide your final answer based on these results."
                                })
                                metadata["tool_calls"].append({
                                    "name": "search_brandon_positions",
                                    "arguments": {"query": sanitized_message, "limit": 5},
                                    "forced": True
                                })
                                continue
                            except Exception as e:
                                logger.error(f"[{request_id}] Forced search failed: {e}")
                                final_response = "I apologize, but I'm having technical difficulties retrieving that information right now. Would you like someone from Brandon's team to call you back to discuss this in detail?"
                                break
                    
                    # Factual safeguard check - fires when THIS ITERATION had no tools
                    # (tool_calls is empty for this specific llm_response)
                    is_factual_policy = "policy" in question_types or topic not in ["general", "callback"]
                    this_iteration_no_tools = len(tool_calls) == 0  # Current llm_response had no tools
                    factual_force_count = sum(1 for m in messages if "SYSTEM CHECK:" in m.get("content", ""))
                    
                    if is_factual_policy and this_iteration_no_tools:
                        if factual_force_count < 2:
                            logger.warning(f"[{request_id}] Factual safeguard triggered (attempt {factual_force_count + 1}) - LLM answered policy without tools")
                            # Append the proposed answer to history so LLM has context
                            messages.append({
                                "role": "assistant",
                                "content": proposed_response
                            })
                            messages.append({
                                "role": "user",
                                "content": f"""SYSTEM CHECK: You answered a factual policy question without searching Brandon's knowledge base.

Please verify your answer by using search_brandon_positions to confirm Brandon's official position on this topic.

Your proposed answer was: {proposed_response[:200]}...

Either confirm with a search or explain why no search is needed."""
                            })
                            continue
                        else:
                            # After 2 failed attempts, force a search ourselves
                            logger.warning(f"[{request_id}] Factual safeguard exhausted after {factual_force_count} attempts - forcing search")
                            try:
                                forced_search = ToolCall(
                                    name="search_brandon_positions",
                                    arguments={"query": sanitized_message, "limit": 5},
                                    call_id="forced_factual_search"
                                )
                                forced_result = await self.tool_executor.execute(forced_search, session_id)
                                forced_context = forced_result.to_context_string()
                                
                                messages.append({
                                    "role": "assistant",
                                    "content": proposed_response
                                })
                                messages.append({
                                    "role": "tool",
                                    "content": f"FORCED VERIFICATION RESULTS (system auto-executed):\n\n{forced_context}\n\nNow provide your verified final answer based on these results."
                                })
                                metadata["tool_calls"].append({
                                    "name": "search_brandon_positions",
                                    "arguments": {"query": sanitized_message, "limit": 5},
                                    "forced": True
                                })
                                continue
                            except Exception as e:
                                logger.error(f"[{request_id}] Forced factual search failed: {e}")
                                # Accept the unverified response as last resort
                    
                    final_response = proposed_response
                    break
                
                # Execute tool calls
                tool_results = []
                for tc_data in tool_calls:
                    tool_call = ToolCall(
                        name=tc_data.get("name", ""),
                        arguments=tc_data.get("arguments", {}),
                        call_id=tc_data.get("id")
                    )
                    
                    metadata["tool_calls"].append({
                        "name": tool_call.name,
                        "arguments": tool_call.arguments
                    })
                    
                    import time as time_module
                    tool_start = time_module.time()
                    result = await self.tool_executor.execute(tool_call, session_id)
                    tool_duration_ms = int((time_module.time() - tool_start) * 1000)
                    tool_results.append(result)
                    
                    # Log tool call to debug.db
                    try:
                        debug_db = get_debug_db()
                        debug_db.log_tool_call(
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                            result=result.to_context_string()[:2000],
                            success=not result.error,
                            duration_ms=tool_duration_ms,
                            session_id=session_id
                        )
                    except Exception as db_err:
                        logger.debug(f"[{request_id}] Failed to log tool call: {db_err}")
                    
                    if result.sources:
                        metadata["sources"].extend(result.sources)
                
                tool_context = "\n\n".join(tr.to_context_string() for tr in tool_results)
                logger.info(f"Tool results context (first 500 chars): {tool_context[:500]}")
                
                messages.append({
                    "role": "assistant",
                    "content": f"Tool calls executed: {[tc['name'] for tc in metadata['tool_calls'][-len(tool_calls):]]}"
                })
                messages.append({
                    "role": "tool",
                    "content": f"""Here are the tool results. Use this information to answer the user's question:

{tool_context}

Now synthesize the above results into a helpful response. Do NOT call the same tool again."""
                })
            
            if final_response is None:
                final_response = "I apologize, but I'm having trouble completing this request. Would you like someone from the team to call you back to discuss this?"
            
            # ===== STAGE 3: OUTPUT VALIDATOR WITH REGENERATION LOOP =====
            # Set up FEC RAG for compliance checking
            if self.tool_executor and self.tool_executor.weaviate:
                output_validator.set_fec_rag(self.tool_executor.weaviate)
            
            regeneration_attempt = 0
            max_regenerations = 3
            pq_confidence = pq_result.avg_rag_confidence if pq_result.rag_results else 0.5
            
            while regeneration_attempt <= max_regenerations:
                validation_result = await output_validator.validate(
                    query=sanitized_message,
                    response=final_response,
                    pq_confidence=pq_confidence
                )
                
                metadata["validation_status"] = "passed" if validation_result.passed else "rejected"
                
                if validation_result.passed:
                    # All checks passed (score <= 3)
                    logger.info(f"[{request_id}] OV passed on attempt {regeneration_attempt}")
                    break
                else:
                    # Need to regenerate (score > 3)
                    regeneration_attempt += 1
                    metadata["regeneration_attempts"] = regeneration_attempt
                    metadata["validation_rejections"].append({
                        "attempt": regeneration_attempt,
                        "reason": validation_result.rejection_reason or "unknown",
                        "max_violation": validation_result.max_violation,
                        "failed_checks": [
                            {"safeguard": s.value, "score": r.score, "explanation": r.explanation}
                            for s, r in validation_result.results.items() if r.score > 3
                        ]
                    })
                    
                    logger.warning(f"[{request_id}] OV rejected (attempt {regeneration_attempt}): {validation_result.rejection_reason}")
                    
                    # Log to debug DB for investigation
                    try:
                        debug_db = get_debug_db()
                        debug_db.log_all_ov_failures(
                            query=sanitized_message,
                            original_response=final_response,
                            validation_result=validation_result,
                            test_id=metadata.get("test_id"),
                            session_id=session_id,
                            request_id=request_id
                        )
                    except Exception as db_err:
                        logger.warning(f"[{request_id}] Failed to log OV rejection to debug DB: {db_err}")
                    
                    if regeneration_attempt <= max_regenerations:
                        # Get feedback for regeneration with structured output instructions
                        regen_prompt = validation_result.get_feedback_for_retry()
                        
                        if regen_prompt:
                            # Add structured output reminder to prevent chatter
                            regen_prompt += get_ov_regeneration_instructions()
                            messages.append({
                                "role": "system",
                                "content": regen_prompt
                            })
                        
                        # Regenerate with LLM
                        regen_response = await self.llm_manager.generate_with_tools(
                            session_id=session_id,
                            messages=messages,
                            tools=[],  # No tools for regeneration
                            system_prompt=full_system_prompt
                        )
                        
                        # Parse structured response from regeneration
                        raw_regen = regen_response.text or ""
                        parsed_regen = parse_structured_response(raw_regen)
                        
                        # Log OV regeneration reasoning to debug DB
                        if parsed_regen.reasoning:
                            try:
                                debug_db = get_debug_db()
                                debug_db.log_reasoning(
                                    session_id=session_id,
                                    request_id=request_id,
                                    reasoning=f"[OV REGEN {regeneration_attempt}] {parsed_regen.reasoning}",
                                    parse_method=parsed_regen.parse_method,
                                    raw_response=parsed_regen.raw_response[:2000]
                                )
                            except Exception as db_err:
                                logger.debug(f"[{request_id}] Failed to log OV regen reasoning: {db_err}")
                        
                        final_response = parsed_regen.final_response or raw_regen
                        metadata["total_tokens"] += regen_response.tokens_used
                        metadata["ov_modifications"].append({
                            "attempt": regeneration_attempt,
                            "parse_method": parsed_regen.parse_method,
                            "had_reasoning": bool(parsed_regen.reasoning)
                        })
                    else:
                        # Max regenerations exceeded - use safe fallback
                        final_response = "I want to make sure I give you accurate information. Would you like someone from Brandon's team to call you back to discuss this personally?"
                        metadata["blocked_by_ov"] = True
                        logger.warning(f"[{request_id}] Max regenerations ({max_regenerations}) exceeded, using fallback. Last rejection: {validation_result.rejection_reason}")
                        break
            
            session.add_turn(ConversationRole.ASSISTANT, final_response)
            
            metadata["sources"] = list(set(metadata["sources"]))
            metadata["duration_ms"] = int((time.time() - start_time) * 1000)
            
            logger.info(f"[{request_id}] Complete - {metadata['iterations']} iterations, "
                       f"{metadata['regeneration_attempts']} regenerations, "
                       f"{metadata['total_tokens']} tokens, {metadata['duration_ms']}ms, "
                       f"model: {metadata['model_used']}")
            
            if self.db_manager:
                try:
                    await self.db_manager.log_model_performance(
                        provider=metadata.get("provider", "unknown"),
                        model=metadata.get("model_used", "unknown"),
                        success=True,
                        latency_ms=metadata.get("duration_ms", 0),
                        tokens_used=metadata.get("total_tokens", 0),
                        session_id=session_id,
                        request_id=request_id
                    )
                except Exception as e:
                    logger.warning(f"Failed to log model performance: {e}")
            
            return final_response, metadata
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[{request_id}] Error after {duration_ms}ms: {e}")
            error_response = "I encountered an unexpected issue. Please try again or ask a different question."
            session.add_turn(ConversationRole.ASSISTANT, error_response)
            
            if self.db_manager:
                try:
                    await self.db_manager.log_model_performance(
                        provider=metadata.get("provider", "unknown"),
                        model=metadata.get("model_used", "unknown"),
                        success=False,
                        latency_ms=duration_ms,
                        error=str(e),
                        session_id=session_id,
                        request_id=request_id
                    )
                except Exception as log_err:
                    logger.warning(f"Failed to log model performance: {log_err}")
            
            return error_response, {"error": str(e), "request_id": request_id, "duration_ms": duration_ms}
    
    def _get_emotion_style_guidance(self, emotion: str) -> str:
        """
        Get style guidance for the LLM based on detected user emotion.
        
        These guidelines integrate with the Ogilvy-style guidance from retrieve_answer_style
        to provide emotion-aware responses.
        
        Args:
            emotion: One of anger, disgust, fear, joy, neutral, sadness, surprise
        
        Returns:
            Style guidance string for the LLM
        """
        emotion_guidance = {
            "anger": (
                "Respond with calm, measured language. Acknowledge their frustration directly without being defensive. "
                "Focus on actionable solutions and next steps. Avoid dismissive language or minimizing their concerns. "
                "Consider offering to connect them with a real person if they prefer."
            ),
            "disgust": (
                "Acknowledge their strong feelings and validate their perspective. Avoid defensive responses. "
                "Focus on shared values and common ground. Be sincere and avoid corporate-speak. "
                "Show genuine understanding of why they might feel this way."
            ),
            "fear": (
                "Respond with reassurance and clarity. Provide specific, factual information to address their concerns. "
                "Avoid vague or dismissive responses. Use calming, confident language while respecting their worries. "
                "Explain what concrete steps are being taken to address their concerns."
            ),
            "sadness": (
                "Respond with empathy and understanding. Acknowledge the difficulty of the situation. "
                "Focus on hope and positive actions being taken. Avoid forced cheerfulness or dismissiveness. "
                "Show genuine care for their well-being and circumstances."
            ),
            "surprise": (
                "Provide clear context and background information. Explain the 'why' behind the situation. "
                "Be patient with follow-up questions. Help them understand the broader picture. "
                "Avoid assuming prior knowledge."
            ),
            "joy": (
                "Match their positive energy appropriately. Be warm and engaging. "
                "Build on their enthusiasm while staying informative. "
                "This is a great opportunity to deepen engagement and share the vision."
            ),
            "neutral": (
                "Respond in a balanced, informative manner. Focus on providing clear, helpful information. "
                "Be professional and approachable."
            )
        }
        
        return emotion_guidance.get(emotion, emotion_guidance["neutral"])
    
    def _build_messages(self, session: Session) -> List[Dict]:
        """Build the message list for the LLM including conversation history"""
        messages = []
        
        if len(session.turns) > 1:
            context_summary = session.get_context_summary()
            messages.append({
                "role": "system",
                "content": f"Conversation context: {context_summary}"
            })
        
        for turn in session.turns[-10:]:
            if turn.role == ConversationRole.USER:
                messages.append({"role": "user", "content": turn.content})
            elif turn.role == ConversationRole.ASSISTANT:
                messages.append({"role": "assistant", "content": turn.content})
        
        return messages
