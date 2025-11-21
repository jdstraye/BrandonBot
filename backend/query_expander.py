"""
Query Expansion Module

Supports both LLM-based and dictionary-based query expansion.
LLM-based expansion uses Phi-3 to generate contextually relevant search terms.
Dictionary-based expansion uses a manual synonym dictionary as fallback.
"""

import logging
from typing import List, Optional
import re
import onnxruntime_genai as og

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Expands user queries into multiple semantically related search terms
    using either LLM-based expansion or a hardcoded synonym dictionary.
    """
    
    def __init__(self, phi3_client=None):
        """
        Initialize the query expander
        
        Args:
            phi3_client: Optional Phi3Client instance for LLM-based expansion
        """
        self.phi3_client = phi3_client
        # Manual synonym dictionary for common policy areas
        # Each key is a base term, each value is a list of related search terms
        self.policy_synonyms = {
            # Education
            "education": ["schools", "teachers", "students", "curriculum", "school choice", "public education", "education funding"],
            "school": ["education", "teachers", "students", "classroom", "learning", "public schools"],
            "teacher": ["educators", "schools", "education", "classroom", "teaching"],
            
            # Immigration
            "immigration": ["border", "immigrants", "asylum", "visa", "citizenship", "border security", "illegal immigration"],
            "border": ["immigration", "border security", "border control", "southern border", "illegal crossing"],
            "immigrant": ["immigration", "asylum", "refugee", "visa", "naturalization"],
            
            # Healthcare
            "healthcare": ["health insurance", "medical care", "doctors", "hospitals", "Medicare", "Medicaid", "Obamacare", "Affordable Care Act"],
            "health": ["healthcare", "medical", "insurance", "coverage", "hospitals", "doctors"],
            "insurance": ["health insurance", "coverage", "premiums", "deductibles", "Medicare", "Medicaid"],
            
            # Economy
            "economy": ["jobs", "employment", "wages", "inflation", "economic growth", "unemployment", "GDP"],
            "job": ["employment", "workers", "economy", "unemployment", "wages", "career"],
            "tax": ["taxes", "taxation", "tax cuts", "tax reform", "tax policy", "revenue"],
            "inflation": ["prices", "cost of living", "economy", "Federal Reserve", "monetary policy"],
            
            # Crime & Justice
            "crime": ["law enforcement", "police", "criminals", "justice", "public safety", "prosecution"],
            "police": ["law enforcement", "cops", "public safety", "crime", "policing"],
            "justice": ["courts", "legal system", "prosecution", "judges", "sentencing"],
            
            # Energy
            "energy": ["oil", "gas", "electricity", "renewable energy", "fossil fuels", "energy independence"],
            "oil": ["energy", "gas", "fossil fuels", "petroleum", "drilling", "energy production"],
            "renewable": ["solar", "wind", "clean energy", "green energy", "renewable energy"],
            
            # 2nd Amendment
            "gun": ["firearms", "Second Amendment", "2A", "gun rights", "gun control", "weapons"],
            "firearm": ["guns", "weapons", "Second Amendment", "2A", "gun rights"],
            
            # Abortion
            "abortion": ["pro-life", "pro-choice", "Roe v Wade", "reproductive rights", "unborn"],
            "life": ["pro-life", "abortion", "unborn", "right to life"],
            
            # Foreign Policy
            "foreign": ["international", "diplomacy", "allies", "adversaries", "defense"],
            "china": ["China", "Chinese", "CCP", "Beijing", "trade with China"],
            "russia": ["Russia", "Russian", "Putin", "Ukraine", "Moscow"],
            
            # Veterans
            "veteran": ["veterans", "vets", "VA", "military service", "armed forces"],
            "military": ["armed forces", "defense", "troops", "soldiers", "veterans"],
        }
    
    async def expand_query(self, query: str) -> List[str]:
        """
        Expand a user query into multiple semantically related search terms
        
        Strategy:
        1. Extract key terms from the query
        2. Look up synonyms for each term in our dictionary
        3. Combine original query + top synonyms
        
        Args:
            query: The original user query
            
        Returns:
            List of expanded search terms (includes original query + expansions)
        """
        # Always include the original query
        expanded_terms = [query]
        
        # Normalize query for matching (lowercase, remove punctuation)
        normalized_query = re.sub(r'[^\w\s]', '', query.lower())
        query_words = normalized_query.split()
        
        # Track which synonyms we've added (avoid duplicates)
        added_synonyms = set()
        
        # Look for matches in our synonym dictionary
        for word in query_words:
            if word in self.policy_synonyms:
                synonyms = self.policy_synonyms[word]
                # Add top 3 synonyms for this term
                for syn in synonyms[:3]:
                    if syn not in added_synonyms and syn.lower() not in normalized_query:
                        expanded_terms.append(syn)
                        added_synonyms.add(syn)
                        logger.info(f"Query expansion: '{word}' → '{syn}'")
        
        # Limit to 5 total terms (original + 4 expansions max)
        expanded_terms = expanded_terms[:5]
        
        logger.info(f"Query expansion: {len(expanded_terms)} terms generated from '{query}'")
        return expanded_terms
    
    async def expand_query_with_llm(self, query: str) -> List[str]:
        """
        Expand query using LLM (Phi-3) to generate contextually relevant search terms
        
        This method uses a lightweight prompt to generate 3 additional search terms
        that are semantically related to the original query. Falls back to dictionary
        expansion if LLM is unavailable.
        
        Args:
            query: The original user query
            
        Returns:
            List of expanded search terms (includes original + 3 LLM-generated terms)
        """
        # Always include original query
        expanded_terms = [query]
        
        # If no Phi3Client available, fall back to dictionary expansion
        if not self.phi3_client:
            logger.warning("No Phi3Client available, falling back to dictionary expansion")
            return await self.expand_query(query)
        
        try:
            # Ensure model is ready
            if not self.phi3_client.model or not self.phi3_client.tokenizer:
                ready = await self.phi3_client.ensure_model_ready()
                if not ready:
                    logger.warning("Phi-3 model not ready, falling back to dictionary expansion")
                    return await self.expand_query(query)
            
            # Update usage tracking
            self.phi3_client.last_used = __import__('time').time()
            await self.phi3_client._schedule_unload()
            
            # Minimal prompt for query expansion
            prompt = f"""Generate exactly 3 alternative search terms for this question about political positions:

Question: "{query}"

Return ONLY 3 short search terms (2-5 words each), one per line, no numbering or explanation.
Example format:
border security policy
immigration reform plans
asylum seeker regulations"""

            # Encode and generate with strict token limits
            tokens = self.phi3_client.tokenizer.encode(prompt)
            input_length = len(tokens)
            safe_max_length = min(2000, input_length + 100)  # Very short response expected
            
            params = og.GeneratorParams(self.phi3_client.model)
            params.set_search_options(
                max_length=safe_max_length,
                temperature=0.3,  # Lower temperature for more focused terms
                top_p=0.9
            )
            
            generator = og.Generator(self.phi3_client.model, params)
            generator.append_tokens(tokens)
            
            # Generate with strict limits
            response_text = ""
            max_output_tokens = 50  # Very short output for 3 search terms
            token_count = 0
            
            while not generator.is_done() and token_count < max_output_tokens:
                generator.compute_logits()
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                response_text += self.phi3_client.tokenizer.decode([new_token])
                token_count += 1
            
            # Parse response - extract up to 3 lines
            lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
            for line in lines[:3]:
                # Clean up any numbering or bullets
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
                cleaned = re.sub(r'^[-*•]\s*', '', cleaned)
                if cleaned and cleaned.lower() != query.lower():
                    expanded_terms.append(cleaned)
            
            logger.info(f"LLM query expansion: {len(expanded_terms)} terms from '{query}'")
            logger.debug(f"Expanded terms: {expanded_terms}")
            
            return expanded_terms[:4]  # Original + up to 3 expansions
            
        except Exception as e:
            logger.error(f"LLM query expansion failed: {str(e)}, falling back to dictionary")
            return await self.expand_query(query)
