"""
Tool Definitions for BrandonBot Agentic Architecture

Tool separation by trust level:
- search_brandon_positions: BrandonPlatform + PreviousQA only (authoritative, trust 1.0)
- augment_brandon_with_party: Supplements Brandon results with party context (clearly labeled, trust 0.6)
- get_party_comparison: For explicit comparison questions, shows R and I positions
- perform_web_search: Internet search for competitors, news, external info
- retrieve_answer_style: MarketGuru copywriting guidance
- register_volunteer: CRM stub for volunteer registration
- make_donation: CRM stub for donations
- check_fec_rules: FEC compliance validation
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json


class ToolName(str, Enum):
    """Available tool names for the LLM to invoke"""
    SEARCH_BRANDON_POSITIONS = "search_brandon_positions"
    AUGMENT_BRANDON_WITH_PARTY = "augment_brandon_with_party"
    GET_PARTY_COMPARISON = "get_party_comparison"
    PERFORM_WEB_SEARCH = "perform_web_search"
    RETRIEVE_ANSWER_STYLE = "retrieve_answer_style"
    REGISTER_VOLUNTEER = "register_volunteer"
    MAKE_DONATION = "make_donation"
    CHECK_FEC_RULES = "check_fec_rules"


TOOL_SCHEMAS = {
    "search_brandon_positions": {
        "name": "search_brandon_positions",
        "description": """Search Brandon's AUTHORITATIVE policy knowledge base for his official positions and statements.

This tool searches ONLY the most trusted sources:
- BrandonPlatform: Brandon's own statements, speeches, and official platform (trust: 1.0)
- PreviousQA: Previously answered questions with verified responses (trust: 1.0)

Use this tool FIRST for any question about Brandon's positions, voting record, or policy stances.
Results from this tool represent Brandon's actual views and should be treated as authoritative.

If results are thin or inconclusive, you may follow up with augment_brandon_with_party to get additional context.

DO NOT use this tool for copywriting advice - use retrieve_answer_style instead.
DO NOT search for party positions here - use get_party_comparison for explicit comparisons.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and include key policy terms. Example: 'Brandon healthcare position affordable care' or 'tax reform middle class'"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-10). Default is 5."
                }
            },
            "required": ["query"]
        }
    },
    
    "augment_brandon_with_party": {
        "name": "augment_brandon_with_party",
        "description": """Supplement Brandon's positions with party platform context when his official position is thin.

This tool searches PartyPlatform (Republican and Independent platforms) to provide additional context.
Results are CLEARLY LABELED as party positions, NOT Brandon's own statements.

ONLY use this tool AFTER search_brandon_positions returns insufficient results.
The augmented information should be presented as: "Party platforms suggest..." or "Republican/Independent positions include..."

Trust level: 0.6 (moderate - may not reflect Brandon's exact views)

NEVER present party platform information as Brandon's personal position.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for party platform context"
                },
                "brandon_topic": {
                    "type": "string",
                    "description": "Brief description of Brandon's position on this topic (from search_brandon_positions) for context"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-5). Default is 3."
                }
            },
            "required": ["query", "brandon_topic"]
        }
    },
    
    "get_party_comparison": {
        "name": "get_party_comparison",
        "description": """Get Republican and Independent party platform positions for EXPLICIT COMPARISON questions.

Use this tool ONLY when the user explicitly asks to compare:
- "How does Brandon compare to..."
- "What do Republicans/Democrats think about..."
- "What's the party position on..."

This tool returns BOTH Republican and Independent platform positions, with handling for contradictions.
Results should be framed as party positions, not Brandon's personal views.

For Brandon's own positions, use search_brandon_positions instead.""",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The policy topic to compare. Example: 'healthcare', 'immigration', 'taxes'"
                },
                "include_republican": {
                    "type": "boolean",
                    "description": "Include Republican platform positions. Default true."
                },
                "include_independent": {
                    "type": "boolean",
                    "description": "Include Independent platform positions. Default true."
                }
            },
            "required": ["topic"]
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
Always cite sources when using information from web search results.

Trust level: Variable (depends on source domain)
- brandonsowers.com: High trust (0.9)
- Major news outlets: Moderate trust (0.5)
- Other sources: Low trust (0.3) - verify before using""",
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

Use this tool AFTER you have the facts (from search_brandon_positions or perform_web_search) 
to get guidance on HOW to present those facts persuasively.

Question types follow the Schwartz awareness stages and Ogilvy framework.""",
        "parameters": {
            "type": "object",
            "properties": {
                "question_type": {
                    "type": "string",
                    "enum": ["unaware", "problem_aware", "solution_aware", "product_aware", 
                            "most_aware", "oppositional", "skeptical", "seeking_proof",
                            "comparison", "simple_inquiry", "trust_building", "emotional_appeal",
                            "values_based", "leadership_focused"],
                    "description": "The classified type of the user's question based on their awareness level and intent."
                },
                "topic": {
                    "type": "string",
                    "enum": ["economy", "healthcare", "education", "immigration", "environment",
                            "foreign_policy", "taxes", "security", "infrastructure", "general", 
                            "values", "leadership", "family", "faith"],
                    "description": "The policy topic being discussed. Helps retrieve topic-specific style advice."
                },
                "desired_tone": {
                    "type": "string", 
                    "enum": ["aspirational", "empathetic", "authoritative", "urgent", 
                            "reassuring", "educational", "persuasive", "storytelling", "direct"],
                    "description": "The desired emotional tone for the response."
                },
                "include_scripture": {
                    "type": "boolean",
                    "description": "Whether to include scripture/faith-based context. True for emotional_appeal, values_based, or trust_building question types."
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
    },
    
    "check_fec_rules": {
        "name": "check_fec_rules",
        "description": """Check FEC (Federal Election Commission) rules for campaign finance compliance.

Use this tool when:
- User asks about donation limits
- User asks "how much can I donate?"
- User asks about campaign finance rules
- Validating a donation amount before processing
- Any question about legal contribution limits

This tool provides accurate, up-to-date FEC rules to ensure campaign compliance.""",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["individual_limit", "pac_limit", "corporate_rules", "disclosure_requirements", 
                            "foreign_national_rules", "general_rules"],
                    "description": "The type of FEC rule to check"
                },
                "amount": {
                    "type": "number",
                    "description": "Optional: A specific donation amount to validate"
                },
                "donor_type": {
                    "type": "string",
                    "enum": ["individual", "pac", "corporation", "llc", "partnership"],
                    "description": "The type of donor asking about rules"
                }
            },
            "required": ["query_type"]
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
            elif expected_type == "integer":
                if isinstance(value, float):
                    self.arguments[param] = int(value)
                elif not isinstance(value, int):
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
    trust_level: Optional[float] = None
    sources: Optional[List[str]] = None
    
    def to_context_string(self) -> str:
        """Convert result to a string for LLM context"""
        if not self.success:
            return f"[TOOL ERROR: {self.tool_name}] {self.error_message}"
        
        trust_label = ""
        if self.trust_level:
            if self.trust_level >= 0.9:
                trust_label = " [AUTHORITATIVE]"
            elif self.trust_level >= 0.6:
                trust_label = " [MODERATE TRUST - verify context]"
            else:
                trust_label = " [LOW TRUST - use with caution]"
        
        if isinstance(self.data, list):
            items = []
            for i, item in enumerate(self.data[:5]):
                if isinstance(item, dict):
                    content = item.get("content", item.get("snippet", str(item)))
                    source = item.get("source", item.get("url", "unknown"))
                    collection = item.get("collection", "")
                    confidence = item.get("confidence", "N/A")
                    items.append(f"[Result {i+1}] (confidence: {confidence}, source: {source}, collection: {collection})\n{content[:500]}")
                else:
                    items.append(f"[Result {i+1}] {str(item)[:500]}")
            return f"[TOOL RESULT: {self.tool_name}]{trust_label}\n" + "\n\n".join(items)
        elif isinstance(self.data, dict):
            return f"[TOOL RESULT: {self.tool_name}]{trust_label}\n{json.dumps(self.data, indent=2)}"
        else:
            return f"[TOOL RESULT: {self.tool_name}]{trust_label}\n{str(self.data)}"
