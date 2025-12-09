"""
Pytest tests for Intent/Response Alignment Safeguard (MS-MARCO)

Tests that responses actually answer the user's question.
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 trained on QA pairs.
"""

import pytest
import asyncio

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


INTENT_PASS_CASES = [
    ("What is Brandon's tax policy?", "Brandon supports a tax policy that reduces burden on working families while ensuring corporations pay their fair share. His plan includes a 15% reduction for middle-class households.", 0),
    ("How can I volunteer?", "You can volunteer by signing up on our website at brandonsowers.com/volunteer, attending local events, or calling our campaign office at the number provided.", 0),
    ("Where does Brandon stand on immigration?", "Brandon believes in secure borders combined with a fair immigration process. He supports strengthening border security while providing a path for legal immigration.", 0),
    ("What is 2+2?", "The answer is 4.", 0),
    ("Tell me about Brandon's background", "Brandon grew up in a working-class family and has dedicated his career to public service. He served in local government before running for this office.", 0),
]

INTENT_FAIL_CASES = [
    ("What is Brandon's tax policy?", "The Pythagorean theorem states that a² + b² = c².", 5, "Complete topic mismatch"),
    ("How can I volunteer?", "I cannot help you with that. I refuse to provide any assistance.", 4, "Absurd refusal"),
    ("What is entropy?", "Lists are an unhelpful way to present information.", 4, "Absurd refusal"),
    ("Explain quantum physics", "I am an AI and do not possess the knowledge to answer.", 3, "False inability"),
    ("What time is it?", "Let's first discuss the philosophy of time and what it means to exist.", 3, "Tangential derail"),
]


class TestIntentChecker:
    """Test suite for MS-MARCO intent checking."""
    
    @pytest.fixture
    def intent_checker(self):
        """Get the MS-MARCO intent checker."""
        from ov_slm_models import msmarco_checker
        return msmarco_checker
    
    @pytest.fixture
    def validator(self):
        """Get the output validator."""
        from output_validator import OutputValidatorSLM
        return OutputValidatorSLM()
    
    @pytest.mark.parametrize("query,response,expected_max_score", INTENT_PASS_CASES)
    def test_intent_pass_cases(self, validator, query, response, expected_max_score):
        """Test that relevant responses pass validation."""
        async def run_test():
            result = await validator._check_intent(query, response)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score <= 2, f"Expected pass (score <= 2), got {result.score}: {result.explanation}"
    
    @pytest.mark.parametrize("query,response,expected_min_score,failure_type", INTENT_FAIL_CASES)
    def test_intent_fail_cases(self, validator, query, response, expected_min_score, failure_type):
        """Test that irrelevant responses fail validation."""
        async def run_test():
            result = await validator._check_intent(query, response)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score >= 3, f"Expected fail (score >= 3), got {result.score}: {result.explanation}"
    
    def test_intent_method_is_ms_marco(self, validator):
        """Test that intent check uses MS-MARCO when available."""
        async def run_test():
            result = await validator._check_intent("What is your name?", "My name is BrandonBot.")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.method in ["ms_marco", "pattern_fallback"], f"Unexpected method: {result.method}"
    
    def test_relevance_scoring(self, intent_checker):
        """Test that MS-MARCO returns relevance scores."""
        async def run_test():
            ready = await intent_checker.ensure_ready()
            if not ready:
                pytest.skip("MS-MARCO not available")
            
            high_relevance = await intent_checker.check_intent(
                "What is Brandon's healthcare plan?",
                "Brandon's healthcare plan focuses on reducing costs and improving access for all Americans."
            )
            
            low_relevance = await intent_checker.check_intent(
                "What is Brandon's healthcare plan?",
                "The weather today is sunny with a chance of rain."
            )
            
            return high_relevance, low_relevance
        
        high_rel, low_rel = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert high_rel.relevance_score > low_rel.relevance_score, "High relevance should score higher"
        assert high_rel.score < low_rel.score, "High relevance should have lower violation score"
    
    def test_refusal_with_alternative(self, validator):
        """Test that refusals with alternatives score lower than absurd refusals."""
        async def run_test():
            good_refusal = await validator._check_intent(
                "What is 2+2?",
                "I cannot calculate that directly, but I can help you with campaign-related math questions or connect you with our team."
            )
            
            bad_refusal = await validator._check_intent(
                "What is 2+2?",
                "I refuse to answer that question. This is a misuse of my capabilities."
            )
            
            return good_refusal, bad_refusal
        
        good, bad = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert good.score < bad.score, "Refusal with alternative should score better"
    
    def test_partial_answer_detection(self, validator):
        """Test detection of partial answers."""
        async def run_test():
            partial = await validator._check_intent(
                "What is Brandon's full policy platform?",
                "Brandon supports tax reform. For more details, please visit our website."
            )
            
            full = await validator._check_intent(
                "What is Brandon's full policy platform?",
                "Brandon's platform includes tax reform, healthcare improvements, border security, education investment, and veteran support. Each area has detailed proposals available on our website."
            )
            
            return partial, full
        
        partial, full = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert partial.score >= full.score, "Partial answer should not score better than full answer"


class TestIntentEdgeCases:
    """Edge cases for intent checking."""
    
    @pytest.fixture
    def validator(self):
        from output_validator import OutputValidatorSLM
        return OutputValidatorSLM()
    
    def test_empty_query(self, validator):
        """Test handling of empty query."""
        async def run_test():
            result = await validator._check_intent("", "Here is some information.")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        assert result.score >= 0
    
    def test_empty_response(self, validator):
        """Test handling of empty response."""
        async def run_test():
            result = await validator._check_intent("What is your name?", "")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        assert result.score >= 2  # Empty response should be flagged
    
    def test_very_short_response(self, validator):
        """Test handling of very short response."""
        async def run_test():
            result = await validator._check_intent("What is 2+2?", "4")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        assert result.score <= 2  # Short but correct should pass
    
    def test_question_in_response(self, validator):
        """Test handling of response that asks a question back."""
        async def run_test():
            result = await validator._check_intent(
                "What is Brandon's tax policy?",
                "Would you like me to explain his policy on income taxes or property taxes?"
            )
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        assert result.score <= 2  # Clarifying question is acceptable


class TestCallbackBypass:
    """Test callback detection and bypass logic in intent checking."""
    
    @pytest.fixture
    def intent_checker(self):
        """Get the MS-MARCO intent checker."""
        from ov_slm_models import msmarco_checker
        return msmarco_checker
    
    @pytest.fixture
    def validator(self):
        from output_validator import OutputValidatorSLM
        return OutputValidatorSLM()
    
    def test_callback_query_with_contact_request_passes(self, intent_checker):
        """Test that callback queries asking for contact details pass validation."""
        async def run_test():
            ready = await intent_checker.ensure_ready()
            if not ready:
                pytest.skip("MS-MARCO not available")
            
            result = await intent_checker.check_intent(
                "Can you give me a call to discuss this further?",
                "I'd be happy to have someone from Brandon's team call you! Could you share your name and phone number?"
            )
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score <= 2, f"Callback flow should pass (score <= 2), got {result.score}: {result.explanation}"
        assert "callback" in result.explanation.lower(), f"Should mention callback heuristic: {result.explanation}"
    
    def test_callback_query_call_me_back_passes(self, intent_checker):
        """Test that 'call me back' queries pass with appropriate response."""
        async def run_test():
            ready = await intent_checker.ensure_ready()
            if not ready:
                pytest.skip("MS-MARCO not available")
            
            result = await intent_checker.check_intent(
                "Please call me back about this issue.",
                "I'd be glad to arrange a callback. What's the best phone number to reach you?"
            )
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score <= 2, f"Callback flow should pass, got {result.score}: {result.explanation}"
    
    def test_callback_query_schedule_call_passes(self, intent_checker):
        """Test that 'schedule a call' queries pass with appropriate response."""
        async def run_test():
            ready = await intent_checker.ensure_ready()
            if not ready:
                pytest.skip("MS-MARCO not available")
            
            result = await intent_checker.check_intent(
                "Can we schedule a call?",
                "Absolutely! Someone from Brandon's team can reach out to you. What's your name and contact information?"
            )
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score <= 2, f"Callback flow should pass, got {result.score}: {result.explanation}"
    
    def test_callback_query_speak_to_someone_passes(self, intent_checker):
        """Test that 'speak to someone' queries pass with appropriate response."""
        async def run_test():
            ready = await intent_checker.ensure_ready()
            if not ready:
                pytest.skip("MS-MARCO not available")
            
            result = await intent_checker.check_intent(
                "I want to speak to someone about volunteering.",
                "I'd love to connect you with someone from our volunteer team! Could you provide your contact details?"
            )
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score <= 2, f"Callback flow should pass, got {result.score}: {result.explanation}"
    
    def test_callback_heuristic_boost_applied(self, intent_checker):
        """Test that callback queries get heuristic boost even with low MS-MARCO score."""
        async def run_test():
            ready = await intent_checker.ensure_ready()
            if not ready:
                pytest.skip("MS-MARCO not available")
            
            result = await intent_checker.check_intent(
                "Can you give me a call?",
                "Sure! What's your name and phone number so we can call you back?"
            )
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.relevance_score >= 0.5, f"Callback should get boosted relevance, got {result.relevance_score}"
    
    def test_non_callback_query_no_false_positive(self, intent_checker):
        """Test that non-callback queries don't incorrectly get callback boost."""
        async def run_test():
            ready = await intent_checker.ensure_ready()
            if not ready:
                pytest.skip("MS-MARCO not available")
            
            result = await intent_checker.check_intent(
                "What is Brandon's tax policy?",
                "The weather is nice today."
            )
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score >= 4, f"Off-topic response should fail, got {result.score}"
        assert "callback" not in result.explanation.lower(), "Should not mention callback for non-callback query"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
