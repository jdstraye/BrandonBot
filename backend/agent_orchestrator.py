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
from backend.llm_providers import LLMProviderManager
from intent_detector import intent_detector, UserIntent, escalation_detector
from prequalifier import prequalifier, PrequalifierResult, FrustrationDecision, VaguenessDecision
from output_validator import output_validator, OVValidationResult, ValidationStatus, RejectionReason, OVSafeguard, SLMNotAvailableError
from validation_debug import get_debug_db
from structured_response import (
    parse_structured_response,
    get_structured_output_instructions,
    get_ov_regeneration_instructions,
    ParsedResponse
)

logger = logging.getLogger(__name__)

# Death-spiral detection defaults
SPIRAL_INTENT_THRESHOLD = 3  # consecutive intent-check failures
SPIRAL_REPEAT_WINDOW = 6     # number of OV attempts to inspect for repetition
SPIRAL_REPEAT_UNIQUE_LIMIT = 2  # <= unique sanitized responses considered repeating


def detect_death_spiral(metadata, test_id, session_id, debug_db, intent_threshold: int = SPIRAL_INTENT_THRESHOLD, repeat_window: int = SPIRAL_REPEAT_WINDOW, repeat_unique_limit: int = SPIRAL_REPEAT_UNIQUE_LIMIT) -> tuple[bool, str, dict]:
    """Detect a death-spiral based on consecutive intent-check failures
    or repeated identical sanitized responses in recent OV attempts.

    Returns (detected, reason, details)
    """
    # Check consecutive intent-checking rejections
    intent_streak = 0
    for entry in reversed(metadata.get("validation_rejections", [])):
        failed_checks = entry.get("failed_checks", [])
        if any(fc.get("safeguard") == OVSafeguard.INTENT_CHECKING.value and fc.get("score", 0) >= 4 for fc in failed_checks):
            intent_streak += 1
        else:
            break
    if intent_streak >= intent_threshold:
        return True, f"intent_rejection_streak={intent_streak}", {"intent_streak": intent_streak}

    # Check for repetition in recent OV attempts (use the debug DB)
    try:
        rows = []
        if test_id:
            conn = None
            try:
                conn = __import__('sqlite3').connect(debug_db.db_path)
                cur = conn.cursor()
                cur.execute('SELECT sanitized_response FROM ov_attempts WHERE test_id = ? ORDER BY id DESC LIMIT ?', (test_id, repeat_window))
                rows = [r[0] for r in cur.fetchall()]
            finally:
                if conn:
                    conn.close()

        if rows and len(rows) >= 3:
            unique_count = len(set(rows))
            if unique_count <= repeat_unique_limit:
                return True, f"repetition_detected_unique_count={unique_count}", {"recent_responses": rows}
    except Exception:
        # On DB errors, don't crash the regeneration flow
        pass

    return False, "", {}


class ConversationRole(str, Enum):
    """Roles in a conversation turn"""
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


def _build_regen_prompt(validation_result, original_query: str) -> str:
    """Build the regeneration prompt to send to the LLM when OV requests a retry.

    Adds OV feedback + structured regeneration instructions. If the failure
    includes intent checking, adds a short REMINDER with the original user
    query to keep the model focused on answering the user's question.
    """
    base = validation_result.get_feedback_for_retry() or ""
    base += get_ov_regeneration_instructions()
    failed_intent = any(
        (s == OVSafeguard.INTENT_CHECKING and r.score > 3)
        for s, r in validation_result.results.items()
    )
    if failed_intent:
        base = base + f"\n\nREMINDER: The user's original question was: '{original_query}'. Please answer that question directly (or ask ONE concise clarifying question if necessary)."
    return base


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
    last_callback_offered_turn: int = -1
    callback_offer_count: int = 0
    last_volunteer_offered_turn: int = -1
    volunteer_offer_count: int = 0
    last_donation_offered_turn: int = -1
    donation_offer_count: int = 0
    last_response_hash: Optional[str] = None  # Hash of last bot response to detect duplicates
    
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
    
    def get_previous_responses(self, count: int = 3) -> List[str]:
        """Get the last N assistant responses for repetition checking."""
        assistant_turns = [t for t in self.turns if t.role == ConversationRole.ASSISTANT]
        return [t.content for t in assistant_turns[-count:]]
    
    def get_tool_context(self) -> str:
        """
        Build context about what tools/actions have been offered in this conversation.
        Helps LLM understand what's already been proposed so it can move on.
        """
        context_parts = []
        
        if self.callback_offer_count > 0:
            context_parts.append(f"Callback has been offered {self.callback_offer_count} time(s) in this conversation.")
        
        if self.volunteer_offer_count > 0:
            context_parts.append(f"Volunteer signup has been offered {self.volunteer_offer_count} time(s) in this conversation.")
        
        if self.donation_offer_count > 0:
            context_parts.append(f"Donation/support signup has been offered {self.donation_offer_count} time(s) in this conversation.")
        
        if not context_parts:
            return ""
        
        return "CONVERSATION CONTEXT: " + " ".join(context_parts) + " If the user affirms interest, execute the appropriate tool. If already executed or user declines, move on to answering their question."
    
    def track_callback_offer(self, response: str) -> bool:
        """
        Check if response offers a callback and track it.
        
        Returns True if callback was offered, False otherwise.
        """
        callback_patterns = [
            "call you back",
            "callback",
            "call back",
            "someone from brandon's team",
            "someone from the team",
            "reach out to you",
            "personal call",
        ]
        response_lower = response.lower()
        
        if any(pattern in response_lower for pattern in callback_patterns):
            self.last_callback_offered_turn = len(self.turns)
            self.callback_offer_count += 1
            return True
        return False
    
    def track_volunteer_offer(self, response: str) -> bool:
        """
        Check if response offers volunteer signup and track it.
        
        Returns True if volunteer offer was made, False otherwise.
        """
        volunteer_patterns = [
            "volunteer",
            "sign up",
            "join our team",
            "get involved",
        ]
        response_lower = response.lower()
        
        if any(pattern in response_lower for pattern in volunteer_patterns):
            self.last_volunteer_offered_turn = len(self.turns)
            self.volunteer_offer_count += 1
            return True
        return False
    
    def track_donation_offer(self, response: str) -> bool:
        """
        Check if response offers donation signup and track it.
        
        Returns True if donation offer was made, False otherwise.
        """
        donation_patterns = [
            "donate",
            "contribute",
            "support the campaign",
            "financial support",
        ]
        response_lower = response.lower()
        
        if any(pattern in response_lower for pattern in donation_patterns):
            self.last_donation_offered_turn = len(self.turns)
            self.donation_offer_count += 1
            return True
        return False
        return False
    
    def get_response_hash(self, text: str) -> str:
        """Get hash of response to detect exact duplicates"""
        import hashlib
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
    def is_response_duplicate(self, response: str, window: int = 3) -> bool:
        """
        Check if this response duplicates recent responses.
        
        Args:
            response: The proposed response
            window: How many turns back to check
            
        Returns True if this exact response was recently given.
        """
        # Exact duplicate check (fast)
        response_hash = self.get_response_hash(response)
        assistant_turns = [t for t in self.turns if t.role == ConversationRole.ASSISTANT]
        for turn in assistant_turns[-window:]:
            if self.get_response_hash(turn.content) == response_hash:
                return True

        # Semantic duplicate check: normalized token overlap and fuzzy ratio
        try:
            from difflib import SequenceMatcher

            def normalize(text: str) -> str:
                import re
                txt = text.lower()
                txt = re.sub(r"[^a-z0-9\s]", " ", txt)
                txt = re.sub(r"\s+", " ", txt).strip()
                return txt

            def token_overlap(a: str, b: str) -> float:
                a_tokens = set(a.split())
                b_tokens = set(b.split())
                if not a_tokens or not b_tokens:
                    return 0.0
                inter = a_tokens.intersection(b_tokens)
                union = a_tokens.union(b_tokens)
                return len(inter) / len(union)

            norm_resp = normalize(response)
            for turn in assistant_turns[-window:]:
                norm_turn = normalize(turn.content)
                # quick token overlap
                overlap = token_overlap(norm_resp, norm_turn)
                if overlap >= 0.75:
                    return True
                # fuzzy sequence ratio as fallback
                seq = SequenceMatcher(None, norm_resp, norm_turn).ratio()
                if seq >= 0.85:
                    return True
        except Exception:
            # If semantic checks fail for any reason, fall back to exact only
            pass

        return False



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
            elif tool_call.name == ToolName.REQUEST_CALLBACK.value:
                return await self._execute_request_callback(tool_call.arguments, session_id)
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
        # Coerce to int - LLM sometimes passes string like "5" instead of 5
        if isinstance(limit_val, str):
            try:
                limit_val = int(limit_val)
            except ValueError:
                limit_val = 5
        elif isinstance(limit_val, float):
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
        # Coerce to int - LLM sometimes passes string like "3" instead of 3
        if isinstance(limit_val, str):
            try:
                limit_val = int(limit_val)
            except ValueError:
                limit_val = 3
        elif isinstance(limit_val, float):
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
    
    async def _execute_request_callback(self, args: Dict, session_id: str = "default") -> ToolResult:
        """Schedule a callback from the campaign team"""
        name = args.get("name", "")
        phone = args.get("phone", "")
        email = args.get("email", "")
        concern = args.get("concern", "")
        preferred_time = args.get("preferred_time", "anytime")
        urgency = args.get("urgency", "normal")
        
        if not name or not phone or not concern:
            return ToolResult(
                tool_name=ToolName.REQUEST_CALLBACK.value,
                success=False,
                data=None,
                error_message="Name, phone number, and concern description are required"
            )
        
        phone_digits = ''.join(c for c in phone if c.isdigit())
        if len(phone_digits) < 10:
            return ToolResult(
                tool_name=ToolName.REQUEST_CALLBACK.value,
                success=False,
                data=None,
                error_message="Please provide a valid phone number with at least 10 digits"
            )
        
        callback_id = f"CB-{int(time.time())}"
        
        callback_data = {
            "callback_id": callback_id,
            "name": name,
            "phone": phone,
            "email": email,
            "concern": concern,
            "preferred_time": preferred_time,
            "urgency": urgency,
            "session_id": session_id,
            "requested_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        logger.info(f"CALLBACK REQUEST: {json.dumps(callback_data)}")
        
        timeframe = "within 24-48 hours" if urgency == "normal" else "as soon as possible"
        
        return ToolResult(
            tool_name=ToolName.REQUEST_CALLBACK.value,
            success=True,
            data={
                "callback_id": callback_id,
                "message": f"Thank you, {name}! Your callback request has been submitted.",
                "confirmation": {
                    "phone": phone,
                    "preferred_time": preferred_time,
                    "concern_summary": concern[:100] + ("..." if len(concern) > 100 else ""),
                    "expected_timeframe": timeframe
                },
                "next_steps": [
                    f"A member of Brandon's team will call you {timeframe}",
                    "Keep your phone available during your preferred time",
                    "Check your email for confirmation if provided"
                ]
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

=== CANDIDATE IDENTITY (IMMUTABLE - NEVER CONTRADICT) ===
Brandon Sowers is running for U.S. Congress in ARIZONA.
- State: Arizona (AZ) - NOT Pennsylvania, NOT any other state
- Office: U.S. House of Representatives
- District: Arizona's 1st Congressional District (AZ-01)
- Party: Republican (running with Independent crossover appeal)
This is an Arizona campaign. Any reference to other states as Brandon's campaign location is WRONG.
=== END IDENTITY BLOCK ===
{question_type_hint}{topic_hint}

YOUR ROLE:
- Answer questions about Brandon's policies, positions, and campaign in ARIZONA
- Help Arizona voters volunteer or donate
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
6. register_volunteer: Sign up volunteers - ALWAYS use when user wants to volunteer
7. make_donation: Process donation requests - ALWAYS use when user wants to donate
8. check_fec_rules: Verify FEC compliance for donations
9. request_callback: Schedule a callback from Brandon's team

VOLUNTEER & DONATION CLOSING (CRITICAL):
When a user expresses interest in volunteering, donating, or helping the campaign:
1. ALWAYS provide the campaign website: brandonsowers.com
2. ALWAYS provide direct links for action:
   - Volunteer: "You can sign up at brandonsowers.com/volunteer"
   - Donate: "Visit brandonsowers.com/donate to contribute"
3. ASK for their contact info (name, email, phone) to register them
4. Once you have their info, CALL the appropriate tool (register_volunteer or make_donation)
5. NEVER just say "thank you" without providing the website and actionable next steps

VOLUNTEER TRIGGER PHRASES - Recognize when user says:
- "I want to volunteer", "How can I help?", "Sign me up"
- "I'd like to get involved", "Can I help with the campaign?"
- "Put me to work", "I want to support Brandon"

DONATION TRIGGER PHRASES - Recognize when user says:
- "I want to donate", "Where can I contribute?"
- "How do I give?", "Take my money", "I'd like to support financially"
   CALLBACK TRIGGER PHRASES - Recognize when user says:
   - "give me a call", "call me", "can you call me"
   - "can we talk", "I'd like to talk to someone"
   - "can I speak to someone", "talk to a person"
   - "have someone reach out", "get back to me"
   - "schedule a call", "set up a call"
   
   CRITICAL CALLBACK RULES:
   - NEVER call this tool with "Unknown", placeholder, or fake values
   - ALWAYS ask the user for their name and phone number FIRST
   - Only call the tool AFTER you have real name and phone from user
   - DO NOT OFFER CALLBACK if the user's question is CLEAR and you can answer it
   
   WHEN TO OFFER CALLBACK:
   1. User EXPLICITLY requests one ("call me", "give me a call")
   2. User is BOTH frustrated AND confused (vague+escalated sentiment)
   3. User provides phone number unprompted
   
   WHEN NOT TO OFFER CALLBACK:
   - User asks clear policy questions even if frustrated
   - User has already had a callback offered in this session (max 1 per session)
   - User is asking simple factual questions you can answer
   
   CORRECT FLOW when user asks for callback:
   1. User says "give me a call" or similar
   2. You respond: "I'd be happy to have someone from Brandon's team call you! To set that up, could you share your name and the best phone number to reach you?"
   3. User provides name and phone
   4. THEN you call the request_callback tool with real values
   
   FRUSTRATED+CLEAR QUERIES: Answer the question directly
   Example: User: "I've asked 3 times about taxes and keep getting same answer!!!"
   Response: Provide substantive new tax policy information, do NOT offer callback

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
- GEOGRAPHIC IDENTITY CHECK: Brandon is running in ARIZONA only. Never mention Pennsylvania, Ohio, or any other state as his campaign location.
{internal_context}
Remember: You're here to inform ARIZONA voters and build support for Brandon's Arizona campaign.
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
        # Quick-path: if the user explicitly provides volunteer signup info (email + affirmative)
        # then auto-execute the register_volunteer tool to avoid brittle LLM-only workflows
        try:
            import re
            email_match = re.search(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}", sanitized_message)
            zip_match = re.search(r"\b\d{5}\b", sanitized_message)
            volunteer_keywords = [
                "volunteer", "sign up", "sign me up", "join the",
                "i'd like to volunteer", "i would like to volunteer",
                "i'd like to join", "i want to join", "i'd like to help", "i want to help"
            ]
            lower_msg = sanitized_message.lower()
            looks_like_volunteer = False
            if email_match and any(k in lower_msg for k in volunteer_keywords):
                looks_like_volunteer = True

            name = None
            if email_match:
                start = max(0, sanitized_message.rfind('\n', 0, email_match.start()))
                name_candidate = sanitized_message[:email_match.start()].strip()
                # If comma-separated, take last segment
                if ',' in name_candidate:
                    name = name_candidate.split(',')[-1].strip()
                else:
                    # fallback to last two words
                    parts = name_candidate.split()
                    name = ' '.join(parts[-2:]) if len(parts) >= 2 else name_candidate
        except Exception:
            name = None

        if looks_like_volunteer:
            volunteer_args = {
                "name": name or "",
                "email": email_match.group(0) if email_match else "",
                "phone": "",
                "zip_code": zip_match.group(0) if zip_match else "",
                "interests": [],
                "availability": "flexible"
            }

            try:
                # Prepare idempotency key so repeated identical auto-exec attempts don't duplicate
                import hashlib, json
                id_source = json.dumps({"action": "register_volunteer", "email": volunteer_args.get("email", ""), "zip": volunteer_args.get("zip_code", ""), "session": session.session_id})
                id_key = hashlib.sha256(id_source.encode()).hexdigest()

                idmap = session.user_context.setdefault('idempotency', {})
                if id_key in idmap:
                    # Reuse the previous result (audit): don't execute twice
                    forced_result = idmap[id_key]['tool_result']
                    logger.info(f"Reusing idempotent volunteer registration for key {id_key}")
                else:
                    # Execute the register_volunteer tool and store the result for idempotency
                    forced_call = ToolCall(name=ToolName.REGISTER_VOLUNTEER.value, arguments=volunteer_args, call_id=f"auto_register_volunteer_{id_key[:8]}")
                    forced_result = await self.tool_executor.execute(forced_call, session_id)
                    # Persist minimal audit info in the session context
                    try:
                        tool_ctx = forced_result.to_context_string()
                    except Exception:
                        tool_ctx = str(forced_result.data) if forced_result.data is not None else ''
                    idmap[id_key] = {
                        'timestamp': datetime.now().isoformat(),
                        'action': 'register_volunteer',
                        'args': {k: volunteer_args.get(k) for k in ['name', 'email', 'zip_code']},
                        'tool_result': forced_result,
                        'tool_context': tool_ctx
                    }

                # Record in metadata-like structure so later code sees it
                # We append an assistant/tool pair into the session history so the LLM can synthesize confirmation
                session.add_turn(ConversationRole.ASSISTANT, "[System] Registering volunteer...", tool_calls=[ToolCall(name=ToolName.REGISTER_VOLUNTEER.value, arguments=volunteer_args, call_id="auto_register_volunteer")], tool_results=[forced_result])
                # Ensure we track that a volunteer offer/registration occurred so we don't re-offer
                session.last_volunteer_offered_turn = len(session.turns)
                session.volunteer_offer_count += 1
                # Store the tool result context for later inclusion in the system prompt
                session.user_context['last_forced_volunteer_result'] = idmap[id_key]['tool_context'] if id_key in idmap else tool_ctx
            except Exception as e:
                logger.warning(f"Failed to auto-register volunteer: {e}")
        
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
        # Attempt to extract a test_id from the session_id for validator runs
        try:
            import re
            match = re.search(r'_(?P<testid>[A-Z]_[A-Z0-9-]+)_', session_id or "")
            if match:
                metadata['test_id'] = match.group('testid')
        except Exception:
            pass
        
        try:
            messages = self._build_messages(session)
            
            # ===== STAGE 2: LLM AGENT WITH TOOLS =====
            # Build system prompt with PQ enrichment and internal hints
            internal_hints_block = pq_result.internal_hints.to_system_prompt_block() if pq_result.internal_hints else ""
            full_system_prompt = self.get_system_prompt(question_types, topic, internal_hints_block)
            # If we auto-registered a volunteer just above, include the tool result context
            try:
                last_vol_ctx = session.user_context.get('last_forced_volunteer_result') if session and session.user_context else None
                if last_vol_ctx:
                    full_system_prompt += f"\n\nVOLUNTEER REGISTRATION: The system registered this user with details:\n{last_vol_ctx}\nDo NOT offer another volunteer signup or callback for this user."
            except Exception:
                pass
            
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
            if intent_result.needs_callback or (user_frustrated and query_vague):
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
            
            # Add vagueness context with turn count awareness
            if query_vague:
                conversation_turn_count = len([t for t in session.turns if t.role == ConversationRole.USER])
                if conversation_turn_count <= 1:
                    full_system_prompt += f"""

VAGUE QUERY DETECTED (Turn 1): The user's question needs clarification. 
- Ask ONE clarifying question to understand their intent better
- ALWAYS include: "Visit brandonsowers.com to learn more about Brandon's positions."
- DO NOT offer a callback yet - first try to understand what they need"""
                elif conversation_turn_count == 2:
                    full_system_prompt += f"""

VAGUE QUERY (Turn 2): You've already asked for clarification once.
- Try a DIFFERENT approach than Turn 1 - offer 2-3 specific topic options to choose from
- Example: "Are you interested in: (1) border security, (2) economic policy, or (3) something else?"
- ALWAYS include brandonsowers.com
- DO NOT offer a callback yet - give them options first"""
                else:
                    full_system_prompt += f"""

VAGUE QUERY (Turn {conversation_turn_count}): Multiple clarifying attempts made.
- Make your BEST attempt to answer based on available context
- If still unclear, NOW you may offer a callback from the team
- ALWAYS include brandonsowers.com for more information
- DO NOT ask another clarifying question"""
            
            # Add meme/subcontext prompt if detected
            if pq_result.meme_detected and pq_result.meme_prompt:
                full_system_prompt += f"\n\n{pq_result.meme_prompt}"
            
            # Callback cooldown logic
            current_turn = len(session.turns)
            turns_since_callback = current_turn - session.last_callback_offered_turn
            callback_on_cooldown = session.last_callback_offered_turn >= 0 and turns_since_callback < 2
            
            if callback_on_cooldown and (intent_result.needs_callback or user_frustrated):
                if turns_since_callback < 2:
                    full_system_prompt += "\n\nCALLBACK COOLDOWN: You already offered a callback recently. Do NOT offer another callback yet. Focus on answering their question or providing helpful information."
                else:
                    full_system_prompt += "\n\nCALLBACK RE-OFFER: You offered a callback earlier but the user seems to still need help. Gently acknowledge you mentioned this before: 'I know I offered earlier, but I want to extend that olive branch again - would a personal call from our team be helpful?'"
            
            iteration = 0
            final_response = None
            
            while iteration < self.max_tool_iterations:
                iteration += 1
                metadata["iterations"] = iteration
                
                # Log exact messages sent to the LLM for debugging/traceability
                try:
                    debug_db = get_debug_db()
                    debug_db.log_llm_request(
                        system_prompt=full_system_prompt,
                        messages=messages,
                        tools=get_gemini_tool_declarations(),
                        test_id=metadata.get("test_id"),
                        session_id=session_id,
                        request_id=request_id,
                        extra={"query_vague": query_vague, "phase": "initial", "attempt": regeneration_attempt}
                    )
                except Exception:
                    logger.debug(f"[{request_id}] Failed to log LLM request; continuing")

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

                    # If the LLM returned a canned 'callback' fallback early,
                    # prefer a clarifying question when the *user query* is vague
                    # and we still have regeneration attempts left. This avoids
                    # prematurely offering a callback that then gets OV-rejected
                    # and starts a regeneration loop.
                    norm_proposed = (proposed_response or "").strip().lower()
                    if ("call you" in norm_proposed or "call you back" in norm_proposed or "having trouble completing" in norm_proposed):
                        if query_vague and regeneration_attempt < max_regenerations:
                            # Replace with an explicit clarifying question
                            proposed_response = "Hi — can you tell me what you'd like to discuss (policy, volunteering, or something else)?"
                            final_response = proposed_response
                        else:
                            # Use a neutral technical-difficulty response (no callback) until max regenerations are exhausted
                            proposed_response = "I'm having technical difficulties right now; please try again shortly."
                            final_response = proposed_response
                    
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
                                # Do not offer a callback here; prefer a neutral message and allow OV regen logic
                                final_response = "I'm having technical difficulties retrieving that information right now; please try again shortly."
                                break
                    
                    # Factual safeguard check - fires when THIS ITERATION had no tools
                    # (tool_calls is empty for this specific llm_response)
                    # Skip factual safeguard for meme-detected queries - the meme prompt provides witty pivot context
                    is_meme_query = pq_result.meme_detected
                    is_factual_policy = ("policy" in question_types or topic not in ["general", "callback"]) and not is_meme_query
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
                    
                    # ===== DUPLICATE RESPONSE DETECTION & PREVENTION =====
                    # Check if we're about to repeat the exact same response
                    if session.is_response_duplicate(proposed_response, window=5):
                        logger.warning(f"[{request_id}] Duplicate response detected! LLM repeating previous answer. Forcing variation or escalation.")
                        
                        # Track what action was last offered
                        previous_responses = session.get_previous_responses(count=2)
                        escalation_attempted = False
                        
                        if previous_responses and iteration < self.max_tool_iterations:
                            last_response = previous_responses[-1].lower()
                            
                            # If callback was offered, user is clearly still unsure - provide variation
                            if "callback" in last_response:
                                messages.append({
                                    "role": "assistant",
                                    "content": proposed_response
                                })
                                messages.append({
                                    "role": "user",
                                    "content": """SYSTEM: You've offered this same callback message multiple times. The user is affirming their interest (yes, please, absolutely, etc.).

Instead of repeating: Either:
1. ACTUALLY CALL the request_callback tool to schedule the callback, OR
2. Provide DIFFERENT helpful information about how they can get involved

Do NOT offer the same callback message again."""
                                })
                                escalation_attempted = True
                            
                            elif "volunteer" in last_response or "sign up" in last_response:
                                messages.append({
                                    "role": "assistant",
                                    "content": proposed_response
                                })
                                messages.append({
                                    "role": "user",
                                    "content": """SYSTEM: You've offered the volunteer signup message repeatedly. The user is affirming interest.

Instead of repeating: Either:
1. ACTUALLY CALL the register_volunteer tool to process their signup, OR
2. Ask for specific details they haven't provided (name, email, etc.)

Do NOT repeat the same volunteer message again."""
                                })
                                escalation_attempted = True
                        
                        if escalation_attempted and iteration < self.max_tool_iterations:
                            continue  # Try again with escalation message
                        else:
                            # After escalation attempts fail, accept and move on
                            logger.warning(f"[{request_id}] Duplicate detection exhausted after multiple attempts - accepting response")
                    
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
                    
                    # Block callback tool if on cooldown (within 2 turns of last callback)
                    if tool_call.name == ToolName.REQUEST_CALLBACK.value:
                        current_turn = len(session.turns)
                        turns_since_callback = current_turn - session.last_callback_offered_turn
                        if session.last_callback_offered_turn >= 0 and turns_since_callback < 2:
                            result = ToolResult(
                                tool_name=ToolName.REQUEST_CALLBACK.value,
                                success=False,
                                data=None,
                                error_message="Callback already scheduled recently. Focus on answering the user's current question instead of offering another callback."
                            )
                            tool_results.append(result)
                            metadata["tool_calls"].append({
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                                "blocked": "cooldown"
                            })
                            logger.info(f"[{request_id}] Callback tool blocked - on cooldown ({turns_since_callback} turns since last)")
                            continue
                    
                    metadata["tool_calls"].append({
                        "name": tool_call.name,
                        "arguments": tool_call.arguments
                    })
                    
                    import time as time_module
                    tool_start = time_module.time()
                    result = await self.tool_executor.execute(tool_call, session_id)
                    tool_duration_ms = int((time_module.time() - tool_start) * 1000)
                    tool_results.append(result)
                    
                    # Track callback tool execution for cooldown logic
                    if tool_call.name == ToolName.REQUEST_CALLBACK.value and result.success:
                        session.last_callback_offered_turn = len(session.turns)
                        session.callback_offer_count += 1
                        logger.info(f"[{request_id}] Callback tool executed (offer #{session.callback_offer_count})")
                    
                    # Log tool call to debug.db
                    try:
                        debug_db = get_debug_db()
                        debug_db.log_tool_call(
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                            result=result.to_context_string()[:2000],
                            success=result.success,
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
                if query_vague:
                    # Prefer to ask a clarifying question for vague queries instead of offering callback
                    final_response = "Hi — can you tell me what you'd like to discuss (policy, volunteering, or something else)?"
                else:
                    final_response = "I'm having technical difficulties right now; please try again shortly."
            
            # ===== STAGE 3: OUTPUT VALIDATOR WITH REGENERATION LOOP =====
            # Set up FEC RAG and weaviate manager for compliance checking
            # Fail-closed: Must always set weaviate_manager for repetition safeguard
            weaviate_mgr = self.tool_executor.weaviate if self.tool_executor else None
            if weaviate_mgr:
                output_validator.set_fec_rag(weaviate_mgr)
                output_validator.set_weaviate_manager(weaviate_mgr)
            else:
                # Try a best-effort local Weaviate init so OV has FEC RAG where possible
                try:
                    from weaviate_manager import WeaviateManager
                    logger.info(f"[{request_id}] No tool weaviate manager; attempting local WeaviateManager init")
                    local_wm = WeaviateManager()
                    await local_wm.initialize()
                    output_validator.set_fec_rag(local_wm)
                    output_validator.set_weaviate_manager(local_wm)
                    weaviate_mgr = local_wm
                    logger.info(f"[{request_id}] Local WeaviateManager initialized and wired to OutputValidator")
                except Exception as e:
                    # Fail-closed: If weaviate is unavailable, log error but continue
                    # The repetition check will raise SLMNotAvailableError which triggers fallback
                    logger.warning(f"[{request_id}] Weaviate unavailable - repetition safeguard will fail closed: {e}")
            
            regeneration_attempt = 0
            max_regenerations = 3
            pq_confidence = pq_result.avg_rag_confidence if pq_result.rag_results else 0.5
            
            callback_tool_invoked = any(
                tc.get("name") == ToolName.REQUEST_CALLBACK.value 
                for tc in metadata.get("tool_calls", [])
            )
            is_callback_flow = callback_tool_invoked
            
            if is_callback_flow:
                logger.info(f"[{request_id}] Callback flow detected - OV intent check will bypass "
                           f"(request_callback tool invoked)")
            
            # Constants for spiral detection
            SPIRAL_INTENT_THRESHOLD = 3  # consecutive intent-check failures
            SPIRAL_REPEAT_WINDOW = 6     # number of OV attempts to inspect for repetition
            SPIRAL_REPEAT_UNIQUE_LIMIT = 2  # <= unique sanitized responses considered repeating


            def detect_death_spiral(metadata, test_id, session_id, debug_db, intent_threshold: int = SPIRAL_INTENT_THRESHOLD, repeat_window: int = SPIRAL_REPEAT_WINDOW, repeat_unique_limit: int = SPIRAL_REPEAT_UNIQUE_LIMIT) -> tuple[bool, str, dict]:
                """Detect a death-spiral based on consecutive intent-check failures
                or repeated identical sanitized responses in recent OV attempts.

                Returns (detected, reason, details)
                """
                # Check consecutive intent-checking rejections
                intent_streak = 0
                for entry in reversed(metadata.get("validation_rejections", [])):
                    failed_checks = entry.get("failed_checks", [])
                    if any(fc.get("safeguard") == OVSafeguard.INTENT_CHECKING.value and fc.get("score", 0) >= 4 for fc in failed_checks):
                        intent_streak += 1
                    else:
                        break
                if intent_streak >= intent_threshold:
                    return True, f"intent_rejection_streak={intent_streak}", {"intent_streak": intent_streak}

                # Check for repetition in recent OV attempts (use the debug DB)
                try:
                    rows = []
                    if test_id:
                        conn = None
                        try:
                            conn = __import__('sqlite3').connect(debug_db.db_path)
                            cur = conn.cursor()
                            cur.execute('SELECT sanitized_response FROM ov_attempts WHERE test_id = ? ORDER BY id DESC LIMIT ?', (test_id, repeat_window))
                            rows = [r[0] for r in cur.fetchall()]
                        finally:
                            if conn:
                                conn.close()

                    if rows and len(rows) >= 3:
                        unique_count = len(set(rows))
                        if unique_count <= repeat_unique_limit:
                            return True, f"repetition_detected_unique_count={unique_count}", {"recent_responses": rows}
                except Exception:
                    # On DB errors, don't crash the regeneration flow
                    pass

                return False, "", {}

            # Use helper detect_death_spiral at module level

            while regeneration_attempt <= max_regenerations:
                validation_result = await output_validator.validate(
                    query=sanitized_message,
                    response=final_response,
                    pq_confidence=pq_confidence,
                    meme_detected=pq_result.meme_detected,
                    is_callback_flow=is_callback_flow,
                    is_vague_query=query_vague
                )
                
                # Add repetition check (try pattern fallback if embedding service missing)
                previous_responses = session.get_previous_responses(count=3)
                if previous_responses:
                    try:
                        repetition_result = await output_validator.check_repetition(
                            response=final_response,
                            previous_responses=previous_responses
                        )
                    except SLMNotAvailableError as e:
                        # Fall back to pattern-based repetition check instead of immediate hard fallback
                        logger.warning(f"[{request_id}] Repetition embedding unavailable; falling back to pattern check: {e}")
                        repetition_result = output_validator._check_repetition_pattern_fallback(final_response, previous_responses)

                    validation_result.results[OVSafeguard.REPETITION] = repetition_result
                    
                    # Update max_violation and passed status
                    validation_result.max_violation = max(r.score for r in validation_result.results.values())
                    validation_result.passed = validation_result.max_violation <= 3
                    
                    if not validation_result.passed:
                        failed = [f"{s.value}: {r.explanation}" 
                                 for s, r in validation_result.results.items() if r.score > 3]
                        validation_result.rejection_reason = "; ".join(failed)
                
                metadata["validation_status"] = "passed" if validation_result.passed else "rejected"

                # Log OV attempt for debug and traceability
                try:
                    debug_db = get_debug_db()
                    ov_results_map = {}
                    for s, r in validation_result.results.items():
                        ov_results_map[s.value] = {
                            "score": r.score,
                            "confidence": r.confidence,
                            "explanation": r.explanation,
                            "method": r.method
                        }
                    status_text = "passed" if validation_result.passed else "rejected"
                    from validation_debug import sanitize_bot_response
                    clean_resp = sanitize_bot_response(final_response or "")
                    debug_db.log_ov_attempt(
                        attempt_num=regeneration_attempt,
                        ov_results=ov_results_map,
                        final_status=status_text,
                        original_response=final_response or "",
                        sanitized_response=clean_resp,
                        aggregate_score=validation_result.max_violation,
                        test_id=metadata.get("test_id"),
                        session_id=session_id,
                        request_id=request_id
                    )
                except Exception:
                    pass
                
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

                    # Detect death-spiral patterns (consecutive intent failures or repeated identical responses)
                    try:
                        debug_db = get_debug_db()
                        detected, reason, details = _detect_death_spiral(metadata, metadata.get('test_id'), session_id, debug_db)
                        if detected:
                            # Persist the spiral event and break with deterministic clarifying question
                            try:
                                debug_db.log_spiral_event(test_id=metadata.get('test_id'), session_id=session_id, request_id=request_id, reason=reason, recent_rejections=metadata.get('validation_rejections', []))
                            except Exception:
                                logger.debug(f"[{request_id}] Failed to log spiral event to debug DB")

                            # Use a deterministic concise clarifying question to break the loop
                            final_response = "Hi — can you tell me what you'd like to discuss (policy, volunteering, or something else)?"
                            metadata['death_spiral'] = True
                            logger.warning(f"[{request_id}] Death-spiral detected ({reason}); switching to deterministic clarifying prompt and breaking regeneration loop")
                            break
                    except Exception as e:
                        logger.debug(f"[{request_id}] Spiral detection exception: {e}")
                    
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
                            # If intent checking failed, explicitly remind the model
                            # of the user's original question so it does not forget
                            # and avoids producing unrelated or callback responses.
                            def _build_regen_prompt(validation_result, original_query):
                                base = validation_result.get_feedback_for_retry() or ""
                                base += get_ov_regeneration_instructions()
                                # If intent_checking is one of the failures, add a short reminder
                                failed_intent = any(
                                    (s == OVSafeguard.INTENT_CHECKING and r.score > 3)
                                    for s, r in validation_result.results.items()
                                )
                                if failed_intent:
                                    base = base + f"\n\nREMINDER: The user's original question was: '{original_query}'. Please answer that question directly (or ask ONE concise clarifying question if necessary)."
                                return base

                            from backend.ov_utils import build_regen_prompt
                            regen_prompt = build_regen_prompt(validation_result, sanitized_message)
                            messages.append({
                                "role": "system",
                                "content": regen_prompt
                            })
                        
                        # Log OV regeneration request payload for debugging
                        try:
                            debug_db = get_debug_db()
                            debug_db.log_llm_request(
                                system_prompt=full_system_prompt,
                                messages=messages,
                                tools=[],
                                test_id=metadata.get("test_id"),
                                session_id=session_id,
                                request_id=request_id,
                                extra={"query_vague": query_vague, "phase": "regeneration", "attempt": regeneration_attempt}
                            )
                        except Exception:
                            logger.debug(f"[{request_id}] Failed to log OV regen LLM request; continuing")

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
            
            # Track callback/volunteer/donation offers for cooldown/duplicate prevention logic
            if session.track_callback_offer(final_response):
                logger.info(f"[{request_id}] Callback offer tracked (offer #{session.callback_offer_count})")
            
            if session.track_volunteer_offer(final_response):
                logger.info(f"[{request_id}] Volunteer offer tracked (offer #{session.volunteer_offer_count})")
            
            if session.track_donation_offer(final_response):
                logger.info(f"[{request_id}] Donation offer tracked (offer #{session.donation_offer_count})")
            
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
    
    async def process_query(self, message: str, session_id: str = "default") -> Tuple[str, Dict]:
        """
        Alias for process_message - used by validator and legacy code.
        
        Args:
            message: The user's input
            session_id: Unique session identifier
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        return await self.process_message(message, session_id)
