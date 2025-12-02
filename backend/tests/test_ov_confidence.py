"""
Pytest tests for Confidence Verification Safeguard (BERT-tiny)

Tests confidence calibration:
- Low PQ confidence: Should use hedging language
- High PQ confidence: Should be confident, not claim false inability
"""

import pytest
import asyncio

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


CONFIDENCE_PASS_CASES_LOW_PQ = [
    ("What might happen?", "Based on available information, Brandon may consider this approach. However, I'm not certain of the exact details.", 0.5, 0),
    ("Unknown topic", "I believe Brandon's position is generally supportive, but I would recommend checking our official website for specifics.", 0.6, 0),
    ("Complex issue", "It appears that Brandon is exploring options in this area. I cannot confirm the exact policy without additional information.", 0.4, 0),
]

CONFIDENCE_FAIL_CASES_LOW_PQ = [
    ("Unknown topic", "Brandon definitely supports this 100%. There is no doubt about his position.", 0.3, 4, "Overconfident"),
    ("Uncertain policy", "Brandon will absolutely do this. It is guaranteed to happen.", 0.4, 4, "Overconfident"),
    ("Complex issue", "I know for certain that this is Brandon's exact policy without any question.", 0.5, 3, "Overconfident"),
]

CONFIDENCE_PASS_CASES_HIGH_PQ = [
    ("Core policy", "Brandon supports tax reform that benefits working families. His plan includes specific measures outlined in his platform.", 0.9, 0),
    ("Clear answer", "Yes, Brandon has committed to this policy and it's a core part of his campaign.", 0.85, 0),
    ("Direct question", "Brandon's position on this issue is clear: he supports common-sense solutions.", 0.8, 0),
]

CONFIDENCE_FAIL_CASES_HIGH_PQ = [
    ("Core policy", "I cannot access that information. I don't have knowledge about this topic.", 0.9, 3, "False inability"),
    ("Clear answer", "I am unable to provide any information on this subject.", 0.85, 3, "False inability"),
    ("Direct question", "I do not possess the knowledge to answer your question.", 0.8, 3, "False inability"),
]


class TestConfidenceChecker:
    """Test suite for BERT-tiny confidence checking."""
    
    @pytest.fixture
    def confidence_checker(self):
        """Get the BERT-tiny confidence checker."""
        from ov_slm_models import berttiny_confidence_checker
        return berttiny_confidence_checker
    
    @pytest.fixture
    def validator(self):
        """Get the output validator."""
        from output_validator_slm import OutputValidatorSLM
        return OutputValidatorSLM()
    
    @pytest.mark.parametrize("query,response,pq_confidence,expected_max_score", CONFIDENCE_PASS_CASES_LOW_PQ)
    def test_low_pq_with_hedging_passes(self, validator, query, response, pq_confidence, expected_max_score):
        """Test that hedging language passes when PQ confidence is low."""
        async def run_test():
            result = await validator._check_confidence(query, response, pq_confidence)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score <= 2, f"Expected pass (score <= 2), got {result.score}: {result.explanation}"
    
    @pytest.mark.parametrize("query,response,pq_confidence,expected_min_score,issue_type", CONFIDENCE_FAIL_CASES_LOW_PQ)
    def test_low_pq_without_hedging_fails(self, validator, query, response, pq_confidence, expected_min_score, issue_type):
        """Test that overconfidence fails when PQ confidence is low."""
        async def run_test():
            result = await validator._check_confidence(query, response, pq_confidence)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score >= 3, f"Expected fail (score >= 3), got {result.score}: {result.explanation}"
    
    @pytest.mark.parametrize("query,response,pq_confidence,expected_max_score", CONFIDENCE_PASS_CASES_HIGH_PQ)
    def test_high_pq_confident_passes(self, validator, query, response, pq_confidence, expected_max_score):
        """Test that confident responses pass when PQ confidence is high."""
        async def run_test():
            result = await validator._check_confidence(query, response, pq_confidence)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score <= 2, f"Expected pass (score <= 2), got {result.score}: {result.explanation}"
    
    @pytest.mark.parametrize("query,response,pq_confidence,expected_min_score,issue_type", CONFIDENCE_FAIL_CASES_HIGH_PQ)
    def test_high_pq_false_inability_fails(self, validator, query, response, pq_confidence, expected_min_score, issue_type):
        """Test that false inability claims fail when PQ confidence is high."""
        async def run_test():
            result = await validator._check_confidence(query, response, pq_confidence)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score >= 2, f"Expected fail (score >= 2), got {result.score}: {result.explanation}"
    
    def test_hedging_patterns_detected(self, confidence_checker):
        """Test that hedging patterns are detected."""
        async def run_test():
            ready = await confidence_checker.ensure_ready()
            
            hedging_responses = [
                "Based on available information, this may be the case.",
                "I believe Brandon supports this, but I'm not certain.",
                "It appears that the policy might include this provision.",
                "According to the platform documents, this seems likely.",
            ]
            
            results = []
            for response in hedging_responses:
                result = await confidence_checker.check_confidence("Query", response, 0.5)
                results.append(result)
            
            return results
        
        results = asyncio.get_event_loop().run_until_complete(run_test())
        
        for result in results:
            assert result.hedging_detected, f"Hedging not detected: {result.explanation}"
    
    def test_overconfidence_patterns_detected(self, confidence_checker):
        """Test that overconfidence patterns are detected."""
        async def run_test():
            ready = await confidence_checker.ensure_ready()
            
            overconfident_responses = [
                "Brandon definitely supports this 100%.",
                "This is absolutely certain without any doubt.",
                "I guarantee that this will happen.",
                "This is the only option that will work.",
            ]
            
            results = []
            for response in overconfident_responses:
                result = await confidence_checker.check_confidence("Query", response, 0.5)
                results.append(result)
            
            return results
        
        results = asyncio.get_event_loop().run_until_complete(run_test())
        
        for result in results:
            assert result.overconfidence_detected, f"Overconfidence not detected: {result.explanation}"
    
    def test_method_is_bert_tiny(self, validator):
        """Test that confidence check uses BERT-tiny when available."""
        async def run_test():
            result = await validator._check_confidence("Query", "Response", 0.5)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.method in ["bert_tiny", "pattern_fallback"], f"Unexpected method: {result.method}"


class TestConfidenceEdgeCases:
    """Edge cases for confidence checking."""
    
    @pytest.fixture
    def validator(self):
        from output_validator_slm import OutputValidatorSLM
        return OutputValidatorSLM()
    
    def test_empty_response(self, validator):
        """Test handling of empty response."""
        async def run_test():
            result = await validator._check_confidence("Query", "", 0.5)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        assert result.score >= 0
    
    def test_pq_confidence_boundary(self, validator):
        """Test behavior at PQ confidence boundary (0.75)."""
        async def run_test():
            response = "Brandon supports this initiative."
            
            below = await validator._check_confidence("Query", response, 0.74)
            at = await validator._check_confidence("Query", response, 0.75)
            above = await validator._check_confidence("Query", response, 0.76)
            
            return below, at, above
        
        below, at, above = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert above.score <= at.score, "Above threshold should not score worse"
    
    def test_extreme_pq_values(self, validator):
        """Test handling of extreme PQ confidence values."""
        async def run_test():
            response = "Brandon supports this policy."
            
            zero = await validator._check_confidence("Query", response, 0.0)
            one = await validator._check_confidence("Query", response, 1.0)
            
            return zero, one
        
        zero, one = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert zero.score >= 0 and zero.score <= 5
        assert one.score >= 0 and one.score <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
