import logging
import os
import json
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        self.model = None
        self.model_with_tools = None
        self.api_key = None
        
        self.system_prompt = """You are BrandonBot, an AI assistant trained on Brandon's political positions and statements.

IMPORTANT GUIDELINES:
- You represent Brandon's views based on his public statements and platform
- Always strive for accuracy, but acknowledge when you're uncertain
- If you don't have enough information or confidence is low, offer to have Brandon call back personally
- Keep responses clear, concise, and conversational (like texting)
- Avoid political jargon when possible
- For controversial topics, present Brandon's position thoughtfully

DISCLAIMER: While I've been trained on Brandon's positions, I may make mistakes. For critical questions or when I'm unsure, I'll offer to have Brandon contact you personally."""
    
    async def ensure_model_ready(self):
        try:
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                logger.error("GOOGLE_API_KEY not found in environment variables")
                logger.info("Please set GOOGLE_API_KEY in Replit Secrets")
                return False
            
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("Gemini API configured successfully (gemini-2.0-flash)")
            return True
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {str(e)}")
            return False
    
    async def generate_response(self, query: str, context: str, confidence: float, 
                               system_prompt: Optional[str] = None) -> dict:
        try:
            if not self.model:
                ready = await self.ensure_model_ready()
                if not ready:
                    return {
                        "response": "I'm having trouble connecting to the AI service. Would you like Brandon to call you back?",
                        "model": "error-model-not-loaded",
                        "error": "Gemini API not available"
                    }
            
            active_system_prompt = system_prompt if system_prompt else self.system_prompt
            
            if confidence < 0.5:
                prompt = f"""{active_system_prompt}

Based on the limited information available:
Question: {query}

Context: {context}

I don't have enough confidence in this information. Suggest offering a callback."""
            else:
                prompt = f"""{active_system_prompt}

Answer the following question based on Brandon's positions:

Question: {query}

Context from Brandon's statements and platform:
{context}

Provide a clear, conversational answer as BrandonBot. Keep it concise (2-4 sentences for simple questions, longer for complex topics)."""
            
            logger.info(f"Gemini prompt length: {len(prompt)}, confidence: {confidence}")
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
                top_p=0.9,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if not response or not response.text:
                logger.warning("Gemini returned empty response")
                return {
                    "response": "I'm having trouble generating a response. Would you like Brandon to call you back?",
                    "model": "gemini-2.0-flash",
                    "error": "empty_response"
                }
            
            response_text = response.text.strip()
            logger.info(f"Gemini response length: {len(response_text)}")
            
            return {
                "response": response_text,
                "model": "gemini-2.0-flash",
                "tokens_used": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None
            }
            
        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}")
            return {
                "response": "I encountered an error while generating a response. Would you like Brandon to call you back?",
                "model": "gemini-2.0-flash",
                "error": str(e)
            }
    
    async def generate_with_tools(self, messages: List[Dict], tools: List[Dict], 
                                   system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response with function calling support.
        
        This is the core method for the LLM-first agentic architecture.
        The LLM can recommend tool calls, which the Orchestrator will execute.
        
        Args:
            messages: Conversation history as list of {role, content} dicts
            tools: List of tool declarations in Gemini format
            system_prompt: Optional custom system prompt
            
        Returns:
            Dict with either 'text' (final response) or 'tool_calls' (tool recommendations)
        """
        try:
            if not self.model:
                ready = await self.ensure_model_ready()
                if not ready:
                    return {
                        "text": "I'm having trouble connecting. Would you like Brandon to call you back?",
                        "error": "API not available"
                    }
            
            function_declarations = []
            for tool in tools:
                fd = FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=tool["parameters"]
                )
                function_declarations.append(fd)
            
            gemini_tools = [Tool(function_declarations=function_declarations)]
            
            model_with_tools = genai.GenerativeModel(
                'gemini-2.0-flash',
                tools=gemini_tools,
                system_instruction=system_prompt or self.system_prompt
            )
            
            gemini_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "user":
                    gemini_messages.append({"role": "user", "parts": [content]})
                elif role == "assistant":
                    gemini_messages.append({"role": "model", "parts": [content]})
                elif role == "tool":
                    gemini_messages.append({"role": "user", "parts": [f"Tool results:\n{content}"]})
                elif role == "system":
                    gemini_messages.append({"role": "user", "parts": [f"System context: {content}"]})
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
                top_p=0.9,
            )
            
            response = model_with_tools.generate_content(
                gemini_messages,
                generation_config=generation_config
            )
            
            result = {
                "usage": {
                    "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                }
            }
            
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                
                tool_calls = []
                text_parts = []
                
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        args = {}
                        if fc.args:
                            for key, value in fc.args.items():
                                if hasattr(value, 'string_value'):
                                    args[key] = value.string_value
                                elif hasattr(value, 'number_value'):
                                    args[key] = value.number_value
                                elif hasattr(value, 'bool_value'):
                                    args[key] = value.bool_value
                                elif hasattr(value, 'list_value'):
                                    args[key] = [v for v in value.list_value.values]
                                elif isinstance(value, (str, int, float, bool)):
                                    args[key] = value
                                else:
                                    args[key] = str(value)
                        
                        tool_calls.append({
                            "name": fc.name,
                            "arguments": args,
                            "id": f"{fc.name}_{len(tool_calls)}"
                        })
                    elif hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                
                if tool_calls:
                    result["tool_calls"] = tool_calls
                    logger.info(f"Gemini recommended {len(tool_calls)} tool calls: {[tc['name'] for tc in tool_calls]}")
                
                if text_parts:
                    result["text"] = "\n".join(text_parts).strip()
                    logger.info(f"Gemini text response length: {len(result['text'])}")
            
            if "text" not in result and "tool_calls" not in result:
                result["text"] = "I'm processing your request..."
            
            return result
            
        except Exception as e:
            logger.error(f"Error in generate_with_tools: {str(e)}")
            return {
                "text": "I encountered an error. Would you like Brandon to call you back?",
                "error": str(e)
            }
    
    async def generate_simple(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Simple generation without tools - for quick responses"""
        try:
            if not self.model:
                ready = await self.ensure_model_ready()
                if not ready:
                    return "I'm having trouble connecting. Please try again."
            
            full_prompt = f"{system_prompt or self.system_prompt}\n\n{prompt}"
            
            response = self.model.generate_content(full_prompt)
            
            if response and response.text:
                return response.text.strip()
            return "I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Error in generate_simple: {e}")
            return f"Error: {str(e)}"
    
    async def health_check(self) -> bool:
        """Check if the Gemini API is ready"""
        if self.model:
            return True
        return await self.ensure_model_ready()
    
    async def close(self):
        """Cleanup (no-op for Gemini as it's stateless)"""
        self.model = None
        self.api_key = None
