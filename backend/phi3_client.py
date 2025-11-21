import logging
import os
from typing import Optional
import onnxruntime_genai as og

logger = logging.getLogger(__name__)

class Phi3Client:
    def __init__(self, model_path: str = "./phi3_model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
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
            if not os.path.exists(self.model_path):
                logger.error(f"Phi-3 model not found at {self.model_path}")
                logger.info("Please download the model using: huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx")
                return False
            
            logger.info(f"Loading Phi-3 model from {self.model_path}...")
            self.model = og.Model(self.model_path)
            self.tokenizer = og.Tokenizer(self.model)
            logger.info("Phi-3 model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load Phi-3 model: {str(e)}")
            return False
    
    async def generate_response(self, query: str, context: str, confidence: float, 
                               system_prompt: Optional[str] = None) -> dict:
        try:
            if not self.model or not self.tokenizer:
                ready = await self.ensure_model_ready()
                if not ready:
                    return {
                        "response": "I'm having trouble loading the AI model. Would you like Brandon to call you back?",
                        "model": "error-model-not-loaded",
                        "error": "Phi-3 model not available"
                    }
            
            active_system_prompt = system_prompt if system_prompt else self.system_prompt
            
            if confidence < 0.5:
                prompt = f"""Based on the limited information available:
Question: {query}

Context: {context}

I don't have enough confidence in this information. Suggest offering a callback."""
            else:
                prompt = f"""Answer the following question based on Brandon's positions:

Question: {query}

Context from Brandon's statements and platform:
{context}

Provide a clear, conversational answer as BrandonBot. Keep it concise (2-4 sentences for simple questions, longer for complex topics)."""
            
            full_prompt = f"<|system|>{active_system_prompt}<|end|><|user|>{prompt}<|end|><|assistant|>"
            logger.info(f"Phi-3 full prompt length: {len(full_prompt)}, confidence: {confidence}")
            
            tokens = self.tokenizer.encode(full_prompt)
            logger.info(f"Encoded {len(tokens)} tokens")
            
            params = og.GeneratorParams(self.model)
            params.input_ids = tokens
            params.set_search_options(
                max_length=2048,
                temperature=0.7,
                top_p=0.9
            )
            
            generator = og.Generator(self.model, params)
            
            response_text = ""
            token_count = 0
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                response_text += self.tokenizer.decode([new_token])
                token_count += 1
            
            logger.info(f"Generated {token_count} tokens, response length: {len(response_text)}")
            
            return {
                "response": response_text.strip(),
                "model": "phi-3-mini-onnx"
            }
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return {
                "response": "I'm having trouble processing your question right now. Would you like Brandon to call you back?",
                "model": "error",
                "error": str(e)
            }
    
    async def health_check(self):
        try:
            return self.model is not None and self.tokenizer is not None
        except:
            return False
    
    async def close(self):
        self.model = None
        self.tokenizer = None
        logger.info("Phi-3 client closed")
