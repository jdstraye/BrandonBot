"""
Tool Definitions for BrandonBot Agentic Architecture
Defines schemas for Gemini function calling: search_policy_collections, 
perform_web_search, retrieve_answer_style, register_volunteer, make_donation
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json


class ToolName(str, Enum):
    """Available tool names for the LLM to invoke"""
    SEARCH_POLICY_COLLECTIONS = "search_policy_collections"
    PERFORM_WEB_SEARCH = "perform_web_search"
    RETRIEVE_ANSWER_STYLE = "retrieve_answer_style"
    REGISTER_VOLUNTEER = "register_volunteer"
    MAKE_DONATION = "make_donation"


TOOL_SCHEMAS = {
    "search_policy_collections": {
        "name": "search_policy_collections",
        "description": """Search Brandon's policy knowledge base for official positions and statements.
        
This tool searches three collections:
- BrandonPlatform: Brandon's own statements, speeches, and official platform (highest trust)
- PreviousQA: Previously answered questions with verified responses (high trust)  
- PartyPlatform: RNC, Independent, and local Republican platforms (moderate trust)

Use this tool when you need factual information about Brandon's positions, voting record, 
or policy stances. The tool returns relevant documents with confidence scores.

DO NOT use this tool for copywriting advice or communication style - use retrieve_answer_style instead.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and include key policy terms. Example: 'Brandon healthcare position affordable care' or 'tax reform middle class'"
                },
                "collections": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["BrandonPlatform", "PreviousQA", "PartyPlatform"]},
                    "description": "Which collections to search. Defaults to all three if not specified."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-10). Default is 5."
                }
            },
            "required": ["query"]
        }
    },
    
    "perform_web_search": {
        "name": "perform_web_search",
        "description": """Search the internet for current information, competitor positions, or recent news.

Use this tool when:
- The user asks about competitor/opponent positions
- You need current news or recent events
- The internal knowledge base doesn't have the answer
- You need to verify or fact-check external claims

The tool uses DuckDuckGo and returns relevant web results with titles, snippets, and URLs.
Always cite sources when using information from web search results.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific. For opponent research, include their name and the topic. Example: 'Jane Doe healthcare policy position 2024'"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10). Default is 5."
                },
                "news_only": {
                    "type": "boolean",
                    "description": "If true, search only news sources. Useful for recent events."
                }
            },
            "required": ["query"]
        }
    },
    
    "retrieve_answer_style": {
        "name": "retrieve_answer_style",
        "description": """Retrieve copywriting and communication style guidance from the MarketGurus collection.

This tool provides advice from legendary copywriters (Ogilvy, Schwartz, Collier, etc.) 
on how to frame and communicate your response based on the type of question.

Use this tool AFTER you have the facts (from search_policy_collections or perform_web_search) 
to get guidance on HOW to present those facts persuasively.

Question types follow the Schwartz awareness stages and Ogilvy framework:
- unaware: Prospect doesn't know they have a problem
- problem_aware: Knows the problem, not the solution
- solution_aware: Knows solutions exist, not your specific solution
- product_aware: Knows about Brandon, needs convincing
- most_aware: Ready to support, needs a reason to act NOW
- oppositional: Disagrees or is hostile
- skeptical: Doubts claims, needs proof
- comparison: Comparing Brandon to other candidates
- trust_building: Building credibility and rapport""",
        "parameters": {
            "type": "object",
            "properties": {
                "question_type": {
                    "type": "string",
                    "enum": ["unaware", "problem_aware", "solution_aware", "product_aware", 
                            "most_aware", "oppositional", "skeptical", "seeking_proof",
                            "comparison", "simple_inquiry", "trust_building", "emotional_appeal"],
                    "description": "The classified type of the user's question based on their awareness level and intent."
                },
                "topic": {
                    "type": "string",
                    "enum": ["economy", "healthcare", "education", "immigration", "environment",
                            "foreign_policy", "taxes", "security", "infrastructure", "general", 
                            "values", "leadership"],
                    "description": "The policy topic being discussed. Helps retrieve topic-specific style advice."
                },
                "desired_tone": {
                    "type": "string", 
                    "enum": ["aspirational", "empathetic", "authoritative", "urgent", 
                            "reassuring", "educational", "persuasive", "storytelling", "direct"],
                    "description": "The desired emotional tone for the response."
                }
            },
            "required": ["question_type"]
        }
    },
    
    "register_volunteer": {
        "name": "register_volunteer",
        "description": """Register a user as a campaign volunteer.

Use this tool when:
- User explicitly says they want to volunteer
- User asks how they can help the campaign
- User wants to get involved

Collect their contact information and preferred volunteer activities.
After registration, thank them warmly and provide next steps.""",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Volunteer's full name"
                },
                "email": {
                    "type": "string",
                    "description": "Volunteer's email address"
                },
                "phone": {
                    "type": "string",
                    "description": "Volunteer's phone number (optional)"
                },
                "zip_code": {
                    "type": "string",
                    "description": "Volunteer's ZIP code for local event matching"
                },
                "interests": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["phone_banking", "door_knocking", 
                             "event_help", "social_media", "data_entry", "transportation", "other"]},
                    "description": "Types of volunteer activities they're interested in"
                },
                "availability": {
                    "type": "string",
                    "enum": ["weekdays", "weekends", "evenings", "flexible"],
                    "description": "When they're available to volunteer"
                }
            },
            "required": ["name", "email"]
        }
    },
    
    "make_donation": {
        "name": "make_donation",
        "description": """Initiate a donation to the campaign.

Use this tool when:
- User wants to donate or contribute financially
- User asks how they can support the campaign financially
- User is ready to make a contribution

This tool generates a secure donation link and provides information about contribution limits.
Note: This is a STUB - actual payment processing requires external integration.""",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Donation amount in USD. Must be between $1 and $3,300 (federal limit)."
                },
                "donor_name": {
                    "type": "string",
                    "description": "Donor's full name (required by FEC)"
                },
                "donor_email": {
                    "type": "string",
                    "description": "Donor's email for receipt"
                },
                "employer": {
                    "type": "string",
                    "description": "Donor's employer (required by FEC for donations over $200)"
                },
                "occupation": {
                    "type": "string",
                    "description": "Donor's occupation (required by FEC for donations over $200)"
                },
                "recurring": {
                    "type": "boolean",
                    "description": "Whether this is a recurring monthly donation"
                }
            },
            "required": ["amount", "donor_name", "donor_email"]
        }
    }
}


def get_gemini_tool_declarations() -> List[Dict]:
    """Get tool declarations in Gemini API format"""
    declarations = []
    for name, schema in TOOL_SCHEMAS.items():
        declarations.append({
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["parameters"]
        })
    return declarations


def get_openai_tool_declarations() -> List[Dict]:
    """Get tool declarations in OpenAI API format (for future use)"""
    declarations = []
    for name, schema in TOOL_SCHEMAS.items():
        declarations.append({
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"]
            }
        })
    return declarations


@dataclass
class ToolCall:
    """Represents a tool call from the LLM"""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None
    
    def validate(self) -> tuple[bool, str]:
        """Validate the tool call against its schema"""
        if self.name not in TOOL_SCHEMAS:
            return False, f"Unknown tool: {self.name}"
        
        schema = TOOL_SCHEMAS[self.name]
        required = schema["parameters"].get("required", [])
        properties = schema["parameters"].get("properties", {})
        
        for req in required:
            if req not in self.arguments:
                return False, f"Missing required parameter: {req}"
        
        for param, value in self.arguments.items():
            if param not in properties:
                continue
            
            prop_schema = properties[param]
            expected_type = prop_schema.get("type")
            
            if expected_type == "string" and not isinstance(value, str):
                return False, f"Parameter {param} must be a string"
            elif expected_type == "integer" and not isinstance(value, int):
                return False, f"Parameter {param} must be an integer"
            elif expected_type == "number" and not isinstance(value, (int, float)):
                return False, f"Parameter {param} must be a number"
            elif expected_type == "boolean" and not isinstance(value, bool):
                return False, f"Parameter {param} must be a boolean"
            elif expected_type == "array" and not isinstance(value, list):
                return False, f"Parameter {param} must be an array"
            
            if "enum" in prop_schema and value not in prop_schema["enum"]:
                return False, f"Parameter {param} must be one of: {prop_schema['enum']}"
        
        return True, "Valid"


@dataclass 
class ToolResult:
    """Result from executing a tool"""
    tool_name: str
    success: bool
    data: Any
    error_message: Optional[str] = None
    confidence: Optional[float] = None
    sources: Optional[List[str]] = None
    
    def to_context_string(self) -> str:
        """Convert result to a string for LLM context"""
        if not self.success:
            return f"[TOOL ERROR: {self.tool_name}] {self.error_message}"
        
        if isinstance(self.data, list):
            items = []
            for i, item in enumerate(self.data[:5]):
                if isinstance(item, dict):
                    content = item.get("content", item.get("snippet", str(item)))
                    source = item.get("source", item.get("url", "unknown"))
                    confidence = item.get("confidence", "N/A")
                    items.append(f"[Result {i+1}] (confidence: {confidence}, source: {source})\n{content[:500]}")
                else:
                    items.append(f"[Result {i+1}] {str(item)[:500]}")
            return f"[TOOL RESULT: {self.tool_name}]\n" + "\n\n".join(items)
        elif isinstance(self.data, dict):
            return f"[TOOL RESULT: {self.tool_name}]\n{json.dumps(self.data, indent=2)}"
        else:
            return f"[TOOL RESULT: {self.tool_name}]\n{str(self.data)}"
