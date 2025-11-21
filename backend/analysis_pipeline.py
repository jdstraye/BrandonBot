"""
Multi-Dimensional Question Analysis Pipeline
Analyzes user questions to determine response strategy
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import re


@dataclass
class QuestionAnalysis:
    """Structured analysis of a user question"""
    question_type: str  # comparison, recent_event, statistics, truth_seeking, policy, low_confidence
    needs_external_search: bool
    needs_bible_verse: bool
    search_queries: List[str]
    bible_topics: List[str]
    awareness_level: str  # unaware, problem_aware, solution_aware, product_aware, most_aware
    emotional_tone: str  # concerned, curious, skeptical, supportive, neutral
    marketing_strategy: str  # Direct description of how to communicate
    marketgurus_keywords: List[str]  # Keywords to query MarketGurus collection


class QuestionAnalyzer:
    """
    Analyzes questions using pattern matching and Phi-3 LLM
    """
    
    # Pattern detection for question types
    COMPARISON_PATTERNS = [
        r'how (?:do|does) (?:you|your) (?:differ|compare)',
        r'what.+difference between',
        r'versus|vs\.?',
        r'compared to',
        r'how are you different'
    ]
    
    STATISTICS_PATTERNS = [
        r'how (?:many|much|will)',
        r'what (?:are the|is the) (?:number|rate|percentage|stat)',
        r'data|statistics|numbers',
        r'how will.+affect'
    ]
    
    TRUTH_SEEKING_PATTERNS = [
        r'is it true',
        r'what (?:does|do) (?:you|your) (?:faith|religion|belief|values) say',
        r'(?:biblically|scripturally|spiritually|morally|ethically) speaking',
        r'from a (?:moral|ethical|spiritual|biblical|religious|faith) (?:perspective|standpoint|view)',
        r'what (?:does|do) (?:the bible|scripture|god|jesus|your faith) say',
        r'is (?:it|this|that) (?:morally|ethically|biblically|scripturally) (?:right|wrong|acceptable)',
        r'(?:moral|ethical|spiritual|biblical|religious) (?:stance|view|position|perspective) on',
        r'your (?:faith|religious|spiritual|moral|ethical) (?:view|belief|position|perspective)',
        r'(?:integrity|character|honesty|truthfulness) (?:in|as a)',
        r'(?:god|jesus|bible|scripture|faith|prayer) (?:and|in|on)',
        r'what (?:are|is) your (?:core )?(?:values|beliefs|principles|convictions)',
        r'(?:christian|faith-based|biblical) (?:perspective|view|approach)'
    ]
    
    RECENT_EVENT_PATTERNS = [
        r'recent|recently|latest|new',
        r'what.+(?:response|reaction) to',
        r'current|today|this week'
    ]
    
    def __init__(self, phi3_client=None):
        """
        Initialize analyzer
        
        Args:
            phi3_client: Optional Phi3Client for enhanced analysis
        """
        self.phi3_client = phi3_client
    
    def analyze_question(self, question: str, rag_confidence: float = 1.0) -> QuestionAnalysis:
        """
        Perform multi-dimensional analysis of a question
        
        Args:
            question: The user's question
            rag_confidence: RAG search confidence (used to detect low-confidence scenarios)
            
        Returns:
            QuestionAnalysis with all dimensions filled
        """
        question_lower = question.lower()
        
        # Step 1: Pattern-based type detection
        question_type = self._detect_question_type(question_lower, rag_confidence)
        
        # Step 2: Determine if external search is needed
        needs_external_search = self._needs_external_search(question_type, rag_confidence)
        
        # Step 3: Determine if Bible verse is needed
        needs_bible_verse = self._needs_bible_verse(question_type, question_lower)
        
        # Step 4: Generate search queries if needed
        search_queries = self._generate_search_queries(question, question_type) if needs_external_search else []
        
        # Step 5: Identify Bible topics if needed
        bible_topics = self._identify_bible_topics(question_lower) if needs_bible_verse else []
        
        # Step 6: Determine awareness level
        awareness_level = self._determine_awareness_level(question_lower)
        
        # Step 7: Detect emotional tone
        emotional_tone = self._detect_emotional_tone(question_lower)
        
        # Step 8: Create marketing strategy
        marketing_strategy = self._create_marketing_strategy(
            question_type, awareness_level, emotional_tone, rag_confidence
        )
        
        # Step 9: Generate MarketGurus keywords
        marketgurus_keywords = self._generate_marketgurus_keywords(
            question_type, awareness_level, emotional_tone
        )
        
        return QuestionAnalysis(
            question_type=question_type,
            needs_external_search=needs_external_search,
            needs_bible_verse=needs_bible_verse,
            search_queries=search_queries,
            bible_topics=bible_topics,
            awareness_level=awareness_level,
            emotional_tone=emotional_tone,
            marketing_strategy=marketing_strategy,
            marketgurus_keywords=marketgurus_keywords
        )
    
    def _detect_question_type(self, question_lower: str, rag_confidence: float) -> str:
        """Detect the primary type of question
        
        NOTE: Pattern matching happens FIRST, before checking RAG confidence.
        This ensures truth-seeking/moral questions get Bible verses even if RAG confidence is low.
        """
        # Check patterns in priority order FIRST (before confidence check)
        if self._matches_patterns(question_lower, self.TRUTH_SEEKING_PATTERNS):
            return "truth_seeking"
        
        if self._matches_patterns(question_lower, self.COMPARISON_PATTERNS):
            return "comparison"
        
        if self._matches_patterns(question_lower, self.STATISTICS_PATTERNS):
            return "statistics"
        
        if self._matches_patterns(question_lower, self.RECENT_EVENT_PATTERNS):
            return "recent_event"
        
        # Low confidence from RAG = needs external help (checked AFTER patterns)
        if rag_confidence < 0.3:
            return "low_confidence"
        
        return "policy"
    
    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the regex patterns"""
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _needs_external_search(self, question_type: str, rag_confidence: float) -> bool:
        """Determine if external web search is required"""
        # These types ALWAYS need external search
        if question_type in ["comparison", "recent_event", "low_confidence"]:
            return True
        
        # Statistics needs external search for current data
        if question_type == "statistics":
            return True
        
        return False
    
    def _needs_bible_verse(self, question_type: str, question_lower: str) -> bool:
        """Determine if Bible verse is needed"""
        if question_type != "truth_seeking":
            return False
        
        # Check if it's a factual question (dates, math, trivial facts)
        factual_patterns = [
            r'what (?:day|date|time)',
            r'when is',
            r'how many (?:days|years)',
            r'\d+',  # Contains numbers
        ]
        
        # If it's factual, no Bible verse needed
        if self._matches_patterns(question_lower, factual_patterns):
            return False
        
        # Moral/spiritual truth-seeking gets Bible verses
        return True
    
    def _generate_search_queries(self, question: str, question_type: str) -> List[str]:
        """Generate targeted search queries for external search"""
        queries = []
        
        if question_type == "comparison":
            # Extract opponent/entity being compared
            # Simple heuristic: look for "vs", "compared to", "differ from"
            comparison_match = re.search(
                r'(?:differ from|compare(?:d)? (?:to|with)|vs\.?|versus) (.+?)(?:\?|$)',
                question,
                re.IGNORECASE
            )
            if comparison_match:
                opponent = comparison_match.group(1).strip()
                queries.append(f"{opponent} political position policy platform")
        
        elif question_type == "statistics":
            # Use the question itself as search query
            queries.append(question + " latest data statistics")
        
        elif question_type == "recent_event":
            # Search for current news on the topic
            queries.append(question + " latest news 2024")
        
        elif question_type == "low_confidence":
            # General search for the topic
            queries.append(question)
        
        return queries if queries else [question]
    
    def _identify_bible_topics(self, question_lower: str) -> List[str]:
        """Identify relevant Bible topics from the question"""
        topic_keywords = {
            "immigration": ["immigra", "border", "foreigner", "stranger", "alien"],
            "stewardship": ["environment", "climate", "nature", "creation", "pollution", "earth"],
            "justice": ["justice", "fair", "equal", "discriminat", "rights"],
            "truth": ["truth", "honest", "lie", "lying", "deception"],
            "integrity": ["integrity", "character", "corrupt", "ethics", "moral"],
            "compassion": ["compassion", "mercy", "kind", "caring", "help"],
            "family": ["family", "parent", "child", "marriage", "father", "mother"],
            "authority": ["government", "authority", "leader", "rule"],
            "wealth": ["money", "wealth", "rich", "poor", "economy", "tax"],
            "work": ["work", "job", "employ", "labor"],
            "freedom": ["freedom", "liberty", "free"],
            "peace": ["peace", "war", "conflict", "violence"]
        }
        
        matched_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                matched_topics.append(topic)
        
        return matched_topics if matched_topics else ["truth", "integrity"]
    
    def _determine_awareness_level(self, question_lower: str) -> str:
        """Determine Schwartz's 5 levels of awareness"""
        # Most aware: Ready to take action, direct questions
        if any(word in question_lower for word in ["how do i", "where can i", "sign up", "join"]):
            return "most_aware"
        
        # Product aware: Knows about Brandon, asking specifics
        if any(word in question_lower for word in ["your position", "your plan", "you believe"]):
            return "product_aware"
        
        # Solution aware: Knows solutions exist, exploring options
        if any(word in question_lower for word in ["how does", "what", "who supports"]):
            return "solution_aware"
        
        # Problem aware: Focused on the problem
        if any(word in question_lower for word in ["problem", "issue", "concern", "worried"]):
            return "problem_aware"
        
        # Default to solution-aware for political questions
        return "solution_aware"
    
    def _detect_emotional_tone(self, question_lower: str) -> str:
        """Detect the emotional tone of the question"""
        # Concerned/worried
        if any(word in question_lower for word in ["worried", "concern", "afraid", "fear"]):
            return "concerned"
        
        # Skeptical/challenging
        if any(word in question_lower for word in ["really", "actually", "prove", "why should"]):
            return "skeptical"
        
        # Supportive
        if any(word in question_lower for word in ["support", "agree", "love", "great"]):
            return "supportive"
        
        # Curious/exploring
        if any(word in question_lower for word in ["how", "what", "why", "tell me"]):
            return "curious"
        
        return "neutral"
    
    def _create_marketing_strategy(
        self,
        question_type: str,
        awareness_level: str,
        emotional_tone: str,
        rag_confidence: float
    ) -> str:
        """Create a marketing strategy description"""
        strategies = []
        
        # Base strategy on awareness level (Schwartz)
        awareness_strategies = {
            "unaware": "Introduce the problem and create curiosity",
            "problem_aware": "Agitate the pain and highlight consequences",
            "solution_aware": "Explain why Brandon's solution is superior",
            "product_aware": "Provide proof, specifics, and differentiation",
            "most_aware": "Direct call to action with urgency"
        }
        strategies.append(awareness_strategies.get(awareness_level, "Direct and informative"))
        
        # Adjust for emotional tone
        if emotional_tone == "concerned":
            strategies.append("empathetic and reassuring")
        elif emotional_tone == "skeptical":
            strategies.append("proof-driven with facts")
        elif emotional_tone == "supportive":
            strategies.append("energizing and action-oriented")
        else:
            strategies.append("clear and conversational")
        
        # Adjust for confidence
        if rag_confidence < 0.5:
            strategies.append("humble and honest about limitations")
        elif rag_confidence > 0.7:
            strategies.append("confident and authoritative")
        
        # Add communication style
        strategies.append("first-person engagement")
        strategies.append("benefit-focused")
        
        return " + ".join(strategies)
    
    def _generate_marketgurus_keywords(
        self,
        question_type: str,
        awareness_level: str,
        emotional_tone: str
    ) -> List[str]:
        """Generate keywords to query MarketGurus collection"""
        keywords = []
        
        # Add awareness-based keywords
        awareness_keywords = {
            "unaware": ["curiosity", "attention"],
            "problem_aware": ["problem", "agitate"],
            "solution_aware": ["solution", "benefits"],
            "product_aware": ["proof", "specific", "features"],
            "most_aware": ["urgency", "action", "offer"]
        }
        keywords.extend(awareness_keywords.get(awareness_level, []))
        
        # Add tone-based keywords
        if emotional_tone == "concerned":
            keywords.extend(["empathy", "reassurance"])
        elif emotional_tone == "skeptical":
            keywords.extend(["proof", "credibility", "facts"])
        elif emotional_tone == "supportive":
            keywords.extend(["enthusiasm", "action"])
        
        # Add question-type keywords
        if question_type == "comparison":
            keywords.extend(["differentiation", "unique"])
        elif question_type == "statistics":
            keywords.extend(["specific", "numbers"])
        elif question_type == "truth_seeking":
            keywords.extend(["honesty", "integrity"])
        
        # Core copywriting principles that always apply
        keywords.extend(["benefits", "clear", "direct", "conversational"])
        
        # Remove duplicates, return unique keywords
        return list(set(keywords))
