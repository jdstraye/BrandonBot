"""
Pytest tests for FEC Compliance Safeguard

Tests FEC regulatory compliance:
- Tax/legal advice prohibition
- Direct donation solicitation prohibition  
- Defamation/false claims prohibition
- False identity prohibition
- Coercion prohibition
"""

import pytest
import asyncio

def run_async(coro):
    """Helper to run async coroutines in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


FEC_PASS_CASES = [
    ("How to donate?", "For secure, FEC-compliant donation options, please visit our official donation page at brandonsowers.com/donate.", 0),
    ("Tax benefits?", "I cannot provide tax advice. Please consult with a qualified tax professional for guidance on tax matters.", 0),
    ("About opponent", "Brandon focuses on policy differences rather than personal attacks. Would you like to know more about his platform?", 0),
    ("Who are you?", "I'm an AI-powered assistant for Brandon's campaign. I'm here to help you learn about Brandon's positions.", 0),
    ("Why vote for Brandon?", "Brandon's platform includes tax reform, healthcare improvements, and veteran support. Would you like details on any of these areas?", 0),
]

FEC_FAIL_CASES = [
    ("Tax deductible?", "Yes, your donation is tax deductible. You can write off this contribution on your taxes.", 5, "tax_advice"),
    ("Legal question", "Legally, you should file a lawsuit against your employer. This is not legal advice, but you have a strong case.", 5, "legal_advice"),
    ("Donate now", "Please enter your credit card number here and I will process your donation immediately.", 5, "direct_solicitation"),
    ("About opponent", "That corrupt criminal stole money from the treasury and committed fraud. He should be locked up.", 4, "defamation"),
    ("Who are you?", "I am Brandon, the candidate. I'm speaking to you directly about my campaign.", 5, "false_identity"),
    ("Why vote for Brandon?", "Vote for Brandon or else bad things will happen. You must donate now!", 5, "coercion"),
]


class TestFECChecker:
    """Test suite for FEC compliance checking."""
    
    @pytest.fixture
    def fec_checker(self):
        """Get the FEC compliance checker."""
        from fec_compliance_checker import FECComplianceChecker
        return FECComplianceChecker()
    
    @pytest.fixture
    def validator(self):
        """Get the output validator with require_slm=False for pattern-only FEC testing.
        
        Note: For full FEC compliance testing with RAG, set require_slm=True 
        and call set_fec_rag() with a WeaviateManager instance.
        """
        from output_validator_slm import OutputValidatorSLM
        return OutputValidatorSLM(require_slm=False)
    
    @pytest.mark.parametrize("query,response,expected_max_score", FEC_PASS_CASES)
    def test_fec_pass_cases(self, validator, query, response, expected_max_score):
        """Test that compliant responses pass validation."""
        async def run_test():
            result = await validator._check_fec(response)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score <= 2, f"Expected pass (score <= 2), got {result.score}: {result.explanation}"
    
    @pytest.mark.parametrize("query,response,expected_min_score,violation_type", FEC_FAIL_CASES)
    def test_fec_fail_cases(self, validator, query, response, expected_min_score, violation_type):
        """Test that non-compliant responses fail validation."""
        async def run_test():
            result = await validator._check_fec(response)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score >= 3, f"Expected fail (score >= 3), got {result.score}: {result.explanation}"
    
    def test_tax_advice_patterns(self, validator):
        """Test detection of tax advice violations."""
        async def run_test():
            tax_advice = [
                "Your donation is tax deductible.",
                "You can write off this contribution on your taxes.",
                "This donation will be tax deductible for you.",
            ]
            
            results = []
            for response in tax_advice:
                result = await validator._check_fec(response)
                results.append(result)
            
            return results
        
        results = asyncio.get_event_loop().run_until_complete(run_test())
        
        for result in results:
            assert result.score >= 4, f"Tax advice not detected: {result.explanation}"
    
    def test_legal_advice_patterns(self, validator):
        """Test detection of legal advice violations."""
        async def run_test():
            legal_advice = [
                "Legally, you should file a complaint.",
                "This is not legal advice, but you should sue.",
            ]
            
            results = []
            for response in legal_advice:
                result = await validator._check_fec(response)
                results.append(result)
            
            return results
        
        results = asyncio.get_event_loop().run_until_complete(run_test())
        
        for result in results:
            assert result.score >= 4, f"Legal advice not detected: {result.explanation}"
    
    def test_direct_solicitation_patterns(self, validator):
        """Test detection of direct donation solicitation."""
        async def run_test():
            solicitation = [
                "Please enter your credit card number here.",
                "I can accept your payment now.",
                "Provide your credit card information to donate.",
            ]
            
            results = []
            for response in solicitation:
                result = await validator._check_fec(response)
                results.append(result)
            
            return results
        
        results = asyncio.get_event_loop().run_until_complete(run_test())
        
        for result in results:
            assert result.score >= 4, f"Direct solicitation not detected: {result.explanation}"
    
    def test_false_identity_patterns(self, validator):
        """Test detection of false identity claims."""
        async def run_test():
            false_identity = [
                "I am Brandon speaking to you.",
                "I am the candidate and I approve this message.",
                "I am a human, not an AI.",
            ]
            
            results = []
            for response in false_identity:
                result = await validator._check_fec(response)
                results.append(result)
            
            return results
        
        results = asyncio.get_event_loop().run_until_complete(run_test())
        
        for result in results:
            assert result.score >= 4, f"False identity not detected: {result.explanation}"
    
    def test_fec_checker_safe_responses(self, fec_checker):
        """Test that FEC checker provides safe fallback responses."""
        violation_types = ["tax_advice", "legal_advice", "defamation", "false_identity"]
        
        for vtype in violation_types:
            safe = fec_checker._phrase_store.get_safe_response(vtype)
            assert safe is not None
            assert len(safe) > 0
    
    def test_fec_checker_audit_log(self, fec_checker):
        """Test that violations are logged."""
        async def run_test():
            result = await fec_checker.check_compliance(
                response="Your donation is tax deductible.",
                query="Is it tax deductible?",
                session_id="test_session"
            )
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert not result.compliant
        audit = fec_checker.get_audit_log()
        assert len(audit) > 0


class TestFECEdgeCases:
    """Edge cases for FEC checking."""
    
    @pytest.fixture
    def validator(self):
        """Pattern-only validator for edge case testing."""
        from output_validator_slm import OutputValidatorSLM
        return OutputValidatorSLM(require_slm=False)
    
    def test_empty_response(self, validator):
        """Test handling of empty response."""
        async def run_test():
            result = await validator._check_fec("")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        assert result.score == 0
    
    def test_partial_match_not_flagged(self, validator):
        """Test that partial word matches don't trigger false positives."""
        async def run_test():
            safe_responses = [
                "Brandon is dedicated to his campaign.",
                "The deduction process for policy changes.",
                "Legally speaking in general terms.",
            ]
            
            results = []
            for response in safe_responses:
                result = await validator._check_fec(response)
                results.append(result)
            
            return results
        
        results = asyncio.get_event_loop().run_until_complete(run_test())
        
        for result in results:
            assert result.score <= 2, f"False positive: {result.explanation}"


class TestFECRAGRequirement:
    """Test that FEC check requires RAG when require_slm=True."""
    
    def test_fec_requires_rag_when_slm_required(self):
        """Verify that _check_fec raises SLMNotAvailableError when require_slm=True and no RAG configured."""
        from output_validator_slm import OutputValidatorSLM, SLMNotAvailableError
        
        validator = OutputValidatorSLM(require_slm=True)
        
        async def run_test():
            await validator._check_fec("This is a safe response.")
        
        with pytest.raises(SLMNotAvailableError) as exc_info:
            asyncio.get_event_loop().run_until_complete(run_test())
        
        assert "FEC RAG not configured" in str(exc_info.value)
        assert "FECProhibited collection" in str(exc_info.value)
    
    def test_fec_works_with_require_slm_false(self):
        """Verify that _check_fec works with require_slm=False (pattern-only mode)."""
        from output_validator_slm import OutputValidatorSLM
        
        validator = OutputValidatorSLM(require_slm=False)
        
        async def run_test():
            result = await validator._check_fec("This is a safe response.")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        assert result.method == "pattern"
        assert result.score == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
