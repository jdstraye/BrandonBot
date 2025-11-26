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
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from agent_tools import (
    ToolCall, ToolResult, ToolName, TOOL_SCHEMAS,
    get_gemini_tool_declarations
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
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
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
                return await self._execute_web_search(tool_call.arguments)
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
    
    async def _execute_web_search(self, args: Dict) -> ToolResult:
        """Search the web using DuckDuckGo"""
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
    """
    
    def __init__(self, llm_client, weaviate_manager, web_search_service=None):
        """
        Initialize the orchestrator.
        
        Args:
            llm_client: GeminiClient or similar LLM client with function calling support
            weaviate_manager: WeaviateManager for vector search
            web_search_service: Optional WebSearchService for web search
        """
        self.llm = llm_client
        self.tool_executor = ToolExecutor(weaviate_manager, web_search_service)
        self.session_manager = SessionManager()
        self.max_tool_iterations = 5
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return """You are BrandonBot, an AI assistant for Brandon Sowers' political campaign.

YOUR ROLE:
- Answer questions about Brandon's policies, positions, and campaign
- Help users volunteer or donate
- Compare Brandon's positions to party platforms when asked
- Maintain a helpful, informative, and persuasive tone

AVAILABLE TOOLS (Trust-Based Separation):

AUTHORITATIVE SOURCES (Trust 1.0):
1. search_brandon_positions: Search Brandon's OWN statements and verified Q&A (BrandonPlatform + PreviousQA)
   - Use FIRST for any question about Brandon's positions
   - Results are authoritative and should be quoted directly

SUPPLEMENTARY SOURCES (Trust 0.6):
2. augment_brandon_with_party: Get party platform context when Brandon's position is thin
   - Use ONLY if search_brandon_positions returns insufficient results
   - Results must be labeled as "party position" NOT Brandon's view
3. get_party_comparison: Compare Republican and Independent platforms
   - Use ONLY for explicit comparison questions
   - Clearly label each party's position

OTHER TOOLS:
4. perform_web_search: Search the internet for current events, competitor info
5. retrieve_answer_style: Get copywriting guidance from MarketGurus
6. register_volunteer: Sign up volunteers
7. make_donation: Process donation requests
8. check_fec_rules: Verify FEC compliance for donations

WORKFLOW:
1. For policy questions: Use search_brandon_positions FIRST
2. If results are thin: Use augment_brandon_with_party (clearly label as party position)
3. For comparisons: Use get_party_comparison
4. After gathering content: Use retrieve_answer_style for communication guidance
5. SYNTHESIZE the information into a response - don't just search repeatedly

CRITICAL RULES:
- NEVER conflate party positions with Brandon's personal views
- After receiving tool results, SYNTHESIZE into a final response
- If confidence is low, offer a callback from someone on the team
- Always cite sources when using information from tools
- Match response style to user's awareness level (Schwartz framework)

RESPONSE SYNTHESIS:
After receiving tool results, you MUST provide a synthesized answer. Do NOT call the same tool multiple times.
If you have search results, use them to construct your response immediately.

Remember: You're here to inform voters and build support for Brandon's campaign."""

    async def process_message(self, user_message: str, session_id: str) -> Tuple[str, Dict]:
        """
        Process a user message through the full agent pipeline.
        
        Args:
            user_message: The user's input
            session_id: Unique session identifier for conversation continuity
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        session = self.session_manager.get_or_create_session(session_id)
        
        session.add_turn(ConversationRole.USER, user_message)
        
        metadata = {
            "session_id": session_id,
            "tool_calls": [],
            "iterations": 0,
            "total_tokens": 0,
            "sources": []
        }
        
        try:
            messages = self._build_messages(session)
            
            iteration = 0
            final_response = None
            
            while iteration < self.max_tool_iterations:
                iteration += 1
                metadata["iterations"] = iteration
                
                llm_response = await self.llm.generate_with_tools(
                    messages=messages,
                    tools=get_gemini_tool_declarations(),
                    system_prompt=self.get_system_prompt()
                )
                
                if "usage" in llm_response:
                    metadata["total_tokens"] += llm_response["usage"].get("total_tokens", 0)
                
                tool_calls = llm_response.get("tool_calls", [])
                
                if not tool_calls:
                    final_response = llm_response.get("text", "I'm sorry, I couldn't process your request.")
                    break
                
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
                    
                    result = await self.tool_executor.execute(tool_call)
                    tool_results.append(result)
                    
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
                final_response = "I apologize, but I'm having trouble completing this request. Would you like Brandon to call you back to discuss this personally?"
            
            session.add_turn(ConversationRole.ASSISTANT, final_response)
            
            metadata["sources"] = list(set(metadata["sources"]))
            
            return final_response, metadata
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            error_response = "I encountered an unexpected issue. Please try again or ask a different question."
            session.add_turn(ConversationRole.ASSISTANT, error_response)
            return error_response, {"error": str(e)}
    
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
