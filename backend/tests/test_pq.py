"""
Pytest tests for Prequalifier (PQ) Pipeline

Tests the 3-stage PQ pipeline:
1. Vagueness detection (probabilistic)
2. Frustration detection (3-bucket classification)
3. RAG retrieval and confidence scoring
"""

import pytest
import asyncio

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


VAGUENESS_CLEAR_CASES = [
    ("What is Brandon's tax policy?", "clear"),
    ("How can I volunteer for the campaign?", "clear"),
    ("Where does Brandon stand on immigration?", "clear"),
    ("What is your healthcare plan?", "clear"),
    ("Tell me about Brandon's education proposals", "clear"),
]

VAGUENESS_VAGUE_CASES = [
    ("hi", "vague"),
    ("hello", "vague"),
    ("?", "vague"),
    ("what", "vague"),
    ("stuff", "vague"),
    ("things", "vague"),
]

FRUSTRATION_CALM_CASES = [
    ("What is your tax policy?", "calm"),
    ("Can you tell me about healthcare?", "calm"),
    ("I'd like to learn more about Brandon", "calm"),
]

FRUSTRATION_FRUSTRATED_CASES = [
    ("THIS IS RIDICULOUS!!! WHY WON'T YOU ANSWER ME???", "frustrated"),
    ("I've asked this 5 times already and you keep ignoring me!", "frustrated"),
    ("Your answers are useless and I'm fed up with this bot!", "frustrated"),
]


class TestPrequalifier:
    """Test suite for the Prequalifier pipeline."""
    
    @pytest.fixture
    def pq(self):
        """Get the prequalifier instance."""
        from prequalifier import Prequalifier
        return Prequalifier()
    
    @pytest.mark.parametrize("query,expected", VAGUENESS_CLEAR_CASES)
    def test_vagueness_clear_cases(self, pq, query, expected):
        """Test that clear queries are classified correctly."""
        async def run_test():
            result = await pq.analyze(query, session_id="test")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        from prequalifier import VaguenessDecision
        assert result.vagueness_decision != VaguenessDecision.VAGUE, f"Query '{query}' incorrectly classified as vague"
    
    @pytest.mark.parametrize("query,expected", VAGUENESS_VAGUE_CASES)
    def test_vagueness_vague_cases(self, pq, query, expected):
        """Test that vague queries are classified correctly."""
        async def run_test():
            result = await pq.analyze(query, session_id="test")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        from prequalifier import VaguenessDecision
        assert result.vagueness_decision in [VaguenessDecision.VAGUE, VaguenessDecision.NEEDS_CLARIFICATION], f"Query '{query}' should be vague"
    
    @pytest.mark.parametrize("query,expected", FRUSTRATION_CALM_CASES)
    def test_frustration_calm_cases(self, pq, query, expected):
        """Test that calm queries are classified correctly."""
        async def run_test():
            result = await pq.analyze(query, session_id="test")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        from prequalifier import FrustrationDecision
        assert result.frustration_decision == FrustrationDecision.CALM, f"Query '{query}' should be calm"
    
    @pytest.mark.parametrize("query,expected", FRUSTRATION_FRUSTRATED_CASES)
    def test_frustration_frustrated_cases(self, pq, query, expected):
        """Test that frustrated queries are classified correctly."""
        async def run_test():
            result = await pq.analyze(query, session_id="test")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        from prequalifier import FrustrationDecision
        assert result.frustration_decision in [FrustrationDecision.FRUSTRATED, FrustrationDecision.ANNOYED], f"Query '{query}' should show frustration"


class TestPatternFlags:
    """Test the pattern flags detection component."""
    
    @pytest.fixture
    def pq(self):
        from prequalifier import Prequalifier
        return Prequalifier()
    
    def test_all_caps_detection(self, pq):
        """Test detection of all-caps as frustration indicator."""
        all_caps = "THIS IS ALL CAPS AND CLEARLY FRUSTRATED"
        flags = pq._detect_patterns(all_caps)
        assert flags.excessive_caps, "All caps should be detected"
    
    def test_excessive_punctuation(self, pq):
        """Test detection of excessive punctuation."""
        excessive = "Why won't you answer me???!!!"
        flags = pq._detect_patterns(excessive)
        assert flags.excessive_punctuation, "Excessive punctuation should be detected"
    
    def test_calm_queries(self, pq):
        """Test that calm queries have low frustration flags."""
        calm = [
            "Could you please explain the tax policy?",
            "I would like to learn more about volunteering.",
            "Thank you for the information.",
        ]
        
        for q in calm:
            flags = pq._detect_patterns(q)
            assert not flags.any_high_risk(), f"Calm query should have no high risk flags: {q}"
    
    def test_profanity_detection(self, pq):
        """Test detection of profanity."""
        queries = [
            "This is bullshit!",
            "What the hell is going on?",
        ]
        
        for q in queries:
            flags = pq._detect_patterns(q)
            assert flags.profanity, f"Profanity should be detected: {q}"


class TestRAGConfidence:
    """Test RAG retrieval and confidence scoring."""
    
    @pytest.fixture
    def pq(self):
        from prequalifier import Prequalifier
        return Prequalifier()
    
    def test_result_has_confidence(self, pq):
        """Test that result includes confidence score."""
        async def run_test():
            result = await pq.analyze(
                "What is Brandon's position on tax reform?",
                session_id="test"
            )
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert hasattr(result, 'confidence')
        assert result.confidence >= 0.0 and result.confidence <= 1.0


class TestPrequalifierResult:
    """Test the PrequalifierResult structure."""
    
    @pytest.fixture
    def pq(self):
        from prequalifier import Prequalifier
        return Prequalifier()
    
    def test_result_structure(self, pq):
        """Test that result has all required fields."""
        async def run_test():
            result = await pq.analyze(
                "What is Brandon's tax policy?",
                session_id="test"
            )
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert hasattr(result, 'query')
        assert hasattr(result, 'vagueness_decision')
        assert hasattr(result, 'frustration_decision')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'rag_results')
    
    def test_result_to_dict(self, pq):
        """Test that result can be converted to dict."""
        async def run_test():
            result = await pq.analyze(
                "What is Brandon's tax policy?",
                session_id="test"
            )
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)


class TestPQEdgeCases:
    """Edge cases for the Prequalifier."""
    
    @pytest.fixture
    def pq(self):
        from prequalifier import Prequalifier
        return Prequalifier()
    
    def test_empty_query(self, pq):
        """Test handling of empty query."""
        async def run_test():
            result = await pq.analyze("", session_id="test")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        from prequalifier import VaguenessDecision
        assert result.vagueness_decision == VaguenessDecision.VAGUE
    
    def test_very_long_query(self, pq):
        """Test handling of very long query."""
        async def run_test():
            long_query = "What is Brandon's position on " + "policy " * 500
            result = await pq.analyze(long_query, session_id="test")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result is not None
    
    def test_unicode_query(self, pq):
        """Test handling of unicode characters."""
        async def run_test():
            result = await pq.analyze(
                "What about healthcare? 🏥💊",
                session_id="test"
            )
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
