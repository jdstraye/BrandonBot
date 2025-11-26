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
            if tool_call.name == ToolName.SEARCH_POLICY_COLLECTIONS.value:
                return await self._execute_search_policy_collections(tool_call.arguments)
            elif tool_call.name == ToolName.PERFORM_WEB_SEARCH.value:
                return await self._execute_web_search(tool_call.arguments)
            elif tool_call.name == ToolName.RETRIEVE_ANSWER_STYLE.value:
                return await self._execute_retrieve_answer_style(tool_call.arguments)
            elif tool_call.name == ToolName.REGISTER_VOLUNTEER.value:
                return await self._execute_register_volunteer(tool_call.arguments)
            elif tool_call.name == ToolName.MAKE_DONATION.value:
                return await self._execute_make_donation(tool_call.arguments)
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
    
    async def _execute_search_policy_collections(self, args: Dict) -> ToolResult:
        """Search Brandon's policy knowledge base"""
        query = args.get("query", "")
        collections = args.get("collections", ["BrandonPlatform", "PreviousQA", "PartyPlatform"])
        limit = min(args.get("limit", 5), 10)
        
        all_results = []
        sources = []
        
        for collection in collections:
            if collection == "MarketGurus":
                continue
            
            try:
                results = await self.weaviate.search(collection, query, limit=limit)
                for r in results:
                    r["collection"] = collection
                    all_results.append(r)
                    if r.get("source"):
                        sources.append(r["source"])
            except Exception as e:
                logger.warning(f"Search failed for {collection}: {e}")
        
        all_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        top_results = all_results[:limit]
        
        avg_confidence = sum(r.get("confidence", 0) for r in top_results) / len(top_results) if top_results else 0
        
        return ToolResult(
            tool_name=ToolName.SEARCH_POLICY_COLLECTIONS.value,
            success=True,
            data=top_results,
            confidence=avg_confidence,
            sources=list(set(sources))
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
        return """You are BrandonBot, an AI assistant for Brandon's political campaign.

YOUR ROLE:
- Answer questions about Brandon's policies, positions, and campaign
- Help users volunteer or donate
- Compare Brandon's positions to other candidates when asked
- Maintain a helpful, informative, and persuasive tone

AVAILABLE TOOLS:
1. search_policy_collections: Search Brandon's official positions and statements
2. perform_web_search: Search the internet for current information or competitor positions  
3. retrieve_answer_style: Get copywriting guidance on how to frame your response
4. register_volunteer: Sign up users who want to volunteer
5. make_donation: Process donation requests

WORKFLOW:
1. Analyze the user's question to understand their awareness level and intent
2. Use search_policy_collections FIRST for questions about Brandon's positions
3. Use perform_web_search for competitor info, current events, or external facts
4. Use retrieve_answer_style to get guidance on HOW to frame your response
5. Synthesize all information into a helpful, persuasive response
6. Include relevant calls-to-action (volunteer, donate, learn more)

IMPORTANT:
- Always cite sources when using information from tools
- If confidence is low (<0.5), acknowledge uncertainty and offer to have Brandon call them back
- For comparison questions, search both internal knowledge AND web for opponent positions
- Match your response style to the user's awareness level (per Schwartz framework)
- End with a clear next step or call-to-action when appropriate

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
                messages.append({
                    "role": "assistant",
                    "content": f"Tool calls executed: {[tc['name'] for tc in metadata['tool_calls'][-len(tool_calls):]]}"
                })
                messages.append({
                    "role": "tool",
                    "content": tool_context
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
