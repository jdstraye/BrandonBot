import ollama
import logging
import asyncio

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, url: str):
        self.url = url
        self.model_name = "phi3.5:3.8b"
        self.client = ollama.Client(host=url)
        
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
            models = self.client.list()
            model_exists = any(self.model_name in model['name'] for model in models.get('models', []))
            
            if not model_exists:
                logger.info(f"Pulling model {self.model_name}... This may take several minutes.")
                self.client.pull(self.model_name)
                logger.info(f"Model {self.model_name} pulled successfully")
            else:
                logger.info(f"Model {self.model_name} already available")
            return True
        except Exception as e:
            logger.error(f"Failed to ensure model ready: {str(e)}")
            return False
    
    async def generate_response(self, query: str, context: str, confidence: float) -> dict:
        try:
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
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system=self.system_prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 300
                }
            )
            
            return {
                "response": response['response'].strip(),
                "model": self.model_name
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
            models = self.client.list()
            return True
        except:
            return False
