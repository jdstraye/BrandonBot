"""
StructuredResponse - Enforces separation of LLM reasoning from user-facing response.

This module provides:
1. Pydantic schema for structured LLM output (reasoning, final_response, metadata)
2. Parsing logic with JSON extraction and delimiter fallback
3. Debug logging to validation_debug.py
"""

import json
import re
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class StructuredResponseSchema(BaseModel):
    """
    Schema for structured LLM output.
    
    LLMs should return JSON with:
    - reasoning: Internal thought process (logged to debug.db, not shown to user)
    - final_response: User-facing response (shown in CSV and to user)
    - metadata: Optional additional info (tool_needed, confidence, etc.)
    """
    reasoning: str = Field(
        default="",
        description="Internal reasoning/thought process - NOT shown to user"
    )
    final_response: str = Field(
        description="The user-facing response - this is what the user sees"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata like confidence, sources, etc."
    )


@dataclass
class ParsedResponse:
    """Result of parsing an LLM response."""
    final_response: str
    reasoning: str = ""
    metadata: Optional[Dict[str, Any]] = None
    parse_method: str = "unknown"
    raw_response: str = ""


# Delimiter patterns for fallback parsing
DELIMITER_PATTERNS = [
    # <final_response>...</final_response>
    re.compile(r'<final_response>\s*(.*?)\s*</final_response>', re.DOTALL | re.IGNORECASE),
    # <response>...</response>
    re.compile(r'<response>\s*(.*?)\s*</response>', re.DOTALL | re.IGNORECASE),
    # ### Final Response\n...
    re.compile(r'###\s*Final\s*Response\s*:?\s*\n(.+?)(?:\n###|\Z)', re.DOTALL | re.IGNORECASE),
    # **Final Response:**\n...
    re.compile(r'\*\*Final\s*Response\s*:?\*\*\s*\n?(.+?)(?:\n\*\*|\Z)', re.DOTALL | re.IGNORECASE),
    # [RESPONSE]...[/RESPONSE]
    re.compile(r'\[RESPONSE\]\s*(.*?)\s*\[/RESPONSE\]', re.DOTALL | re.IGNORECASE),
]

# Patterns that indicate reasoning/chatter to strip
REASONING_PREFIXES = [
    re.compile(r"^You(?:'re| are) right,?\s*", re.IGNORECASE),
    re.compile(r"^I should (?:verify|check|search)\s*", re.IGNORECASE),
    re.compile(r"^(?:Let me|I'll) (?:verify|check|search|make sure)\s*", re.IGNORECASE),
    re.compile(r"^Here(?:'s| is) (?:the|my) (?:corrected|updated|revised) response:?\s*", re.IGNORECASE),
    re.compile(r"^(?:The|My) response is:?\s*", re.IGNORECASE),
    re.compile(r"^I(?:'ll| will) proceed\s*", re.IGNORECASE),
    re.compile(r"^No (?:search|tool|verification) is needed[^.]*[.,]\s*", re.IGNORECASE),
]

# Patterns to match reasoning blocks that should be fully removed
REASONING_BLOCK_PATTERNS = [
    # "You're right, ... Let me check Brandon's platform."
    re.compile(r"You(?:'re| are) right[^.]*\.[^\n]*(?:Let me|I'll)[^\n]*\.\s*", re.IGNORECASE | re.DOTALL),
    # "I'll search the knowledge base." and similar
    re.compile(r"I(?:'ll| will) (?:search|check|verify)[^.]*\.\s*", re.IGNORECASE),
    # "Here's the corrected response:" (anywhere, not just at start)
    re.compile(r"Here(?:'s| is) (?:the|my) (?:corrected|updated|revised) response:?\s*", re.IGNORECASE),
]

# Step heading patterns to strip
STEP_HEADING_PATTERNS = [
    re.compile(r'^#+\s*Step \d+:?[^\n]*\n?', re.MULTILINE),
    re.compile(r'\n#+\s*Step \d+:?[^\n]*', re.MULTILINE),
    re.compile(r'^##\s*(?:Corrected\s+)?Response:?\s*\n?', re.MULTILINE | re.IGNORECASE),
]


def parse_structured_response(raw_response: str) -> ParsedResponse:
    """
    Parse an LLM response into structured components.
    
    Parsing priority:
    1. JSON object with 'final_response' field
    2. Delimiter-based extraction (<final_response>, etc.)
    3. Strip reasoning prefixes/chatter from raw text
    
    Args:
        raw_response: Raw LLM output string
        
    Returns:
        ParsedResponse with separated reasoning and final_response
    """
    if not raw_response:
        return ParsedResponse(
            final_response="",
            parse_method="empty",
            raw_response=raw_response
        )
    
    # Try JSON parsing first
    json_result = _try_parse_json(raw_response)
    if json_result:
        return json_result
    
    # Try delimiter-based extraction
    delimiter_result = _try_parse_delimiters(raw_response)
    if delimiter_result:
        return delimiter_result
    
    # Fallback: strip reasoning chatter from raw text
    cleaned = _strip_reasoning_chatter(raw_response)
    
    return ParsedResponse(
        final_response=cleaned,
        reasoning="",
        parse_method="chatter_stripped",
        raw_response=raw_response
    )


def _fix_json_newlines(json_str: str) -> str:
    """
    Fix malformed JSON with literal newlines inside string values.
    
    LLMs sometimes output JSON like:
        {"final_response": "Line 1
    Line 2"}
    
    This converts literal newlines inside strings to proper \\n escapes.
    """
    result = []
    in_string = False
    escape_next = False
    
    for char in json_str:
        if escape_next:
            result.append(char)
            escape_next = False
        elif char == '\\':
            result.append(char)
            escape_next = True
        elif char == '"':
            result.append(char)
            in_string = not in_string
        elif char == '\n' and in_string:
            # Convert literal newline inside string to escape sequence
            result.append('\\n')
        else:
            result.append(char)
    
    return ''.join(result)


def _try_parse_json(raw_response: str) -> Optional[ParsedResponse]:
    """Try to parse response as JSON with StructuredResponseSchema."""
    
    # First, try direct JSON parse (most common case)
    try:
        data = json.loads(raw_response.strip())
        if isinstance(data, dict) and "final_response" in data:
            schema = StructuredResponseSchema(**data)
            return ParsedResponse(
                final_response=schema.final_response.strip(),
                reasoning=schema.reasoning,
                metadata=schema.metadata,
                parse_method="json",
                raw_response=raw_response
            )
    except (json.JSONDecodeError, ValidationError):
        pass
    
    # Try fixing malformed JSON with literal newlines in strings
    if '{' in raw_response and '"final_response"' in raw_response:
        fixed_json = _fix_json_newlines(raw_response.strip())
        try:
            data = json.loads(fixed_json)
            if isinstance(data, dict) and "final_response" in data:
                schema = StructuredResponseSchema(**data)
                return ParsedResponse(
                    final_response=schema.final_response.strip(),
                    reasoning=schema.reasoning,
                    metadata=schema.metadata,
                    parse_method="json",
                    raw_response=raw_response
                )
        except (json.JSONDecodeError, ValidationError):
            pass
    
    # Try to find JSON object with balanced braces
    json_start = raw_response.find('{')
    if json_start != -1 and '"final_response"' in raw_response:
        # Find the matching closing brace
        brace_count = 0
        json_end = -1
        for i, char in enumerate(raw_response[json_start:], json_start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end > json_start:
            json_str = raw_response[json_start:json_end]
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "final_response" in data:
                    schema = StructuredResponseSchema(**data)
                    return ParsedResponse(
                        final_response=schema.final_response.strip(),
                        reasoning=schema.reasoning,
                        metadata=schema.metadata,
                        parse_method="json",
                        raw_response=raw_response
                    )
            except (json.JSONDecodeError, ValidationError) as e:
                logger.debug(f"Balanced brace JSON parse failed: {e}")
    
    # Try code block extraction
    code_block_pattern = re.compile(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```')
    match = code_block_pattern.search(raw_response)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "final_response" in data:
                schema = StructuredResponseSchema(**data)
                return ParsedResponse(
                    final_response=schema.final_response.strip(),
                    reasoning=schema.reasoning,
                    metadata=schema.metadata,
                    parse_method="json",
                    raw_response=raw_response
                )
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Code block JSON parse failed: {e}")
    
    return None


def _try_parse_delimiters(raw_response: str) -> Optional[ParsedResponse]:
    """Try to extract final_response using delimiter patterns."""
    for pattern in DELIMITER_PATTERNS:
        match = pattern.search(raw_response)
        if match:
            final_response = match.group(1).strip()
            if final_response:
                # Extract reasoning as everything before the delimiter
                before = raw_response[:match.start()].strip()
                return ParsedResponse(
                    final_response=final_response,
                    reasoning=before,
                    parse_method="delimiter",
                    raw_response=raw_response
                )
    return None


def _strip_reasoning_chatter(text: str) -> str:
    """
    Strip reasoning prefixes and step headings from response text.
    
    This is the fallback when JSON/delimiters fail.
    """
    result = text.strip()
    
    # Strip leading/trailing quotes
    if (result.startswith('"') and result.endswith('"')) or \
       (result.startswith("'") and result.endswith("'")):
        result = result[1:-1].strip()
    
    # Strip reasoning block patterns (anywhere in text)
    for pattern in REASONING_BLOCK_PATTERNS:
        result = pattern.sub('', result)
    
    # Strip step heading patterns
    for pattern in STEP_HEADING_PATTERNS:
        result = pattern.sub('', result)
    
    # Strip reasoning prefixes (iterate since removing one may expose another)
    changed = True
    max_iterations = 5
    iteration = 0
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        for pattern in REASONING_PREFIXES:
            new_result = pattern.sub('', result, count=1)
            if new_result != result:
                result = new_result.strip()
                changed = True
    
    return result.strip()


def get_structured_output_instructions() -> str:
    """
    Get system prompt instructions for structured JSON output.
    
    These instructions should be appended to the system prompt to
    request structured output from the LLM.
    """
    return """

===== RESPONSE FORMAT =====
You MUST return your response as a JSON object with this structure:
{
  "reasoning": "Your internal thought process (not shown to user)",
  "final_response": "The actual response the user will see"
}

CRITICAL: 
- Put ALL self-reflection, verification thoughts, and corrections in "reasoning"
- Put ONLY the clean user-facing message in "final_response"
- The user will ONLY see the "final_response" content
- Do NOT include phrases like "Here's the corrected response" in final_response

Example:
{
  "reasoning": "The user asked about taxes. I should search Brandon's platform first. Found relevant info. The response addresses their concern.",
  "final_response": "Brandon supports tax reform that benefits working families. His plan includes..."
}
"""


def get_ov_regeneration_instructions() -> str:
    """
    Get instructions for OV regeneration that enforce structured output.
    """
    return """

IMPORTANT: When regenerating your response, return ONLY the JSON object.
Do NOT include:
- "You're right, I should..." 
- "Here's the corrected response:"
- Step-by-step corrections
- Any text before or after the JSON

Return exactly:
{
  "reasoning": "What I'm fixing and why",
  "final_response": "The corrected user-facing message"
}
"""
