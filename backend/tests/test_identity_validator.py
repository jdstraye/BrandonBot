"""
Tests for Identity Consistency Validator

Ensures BrandonBot never mentions wrong states as Brandon's campaign location.
Brandon Sowers is running for U.S. Congress in ARIZONA - not Pennsylvania or any other state.
"""

import pytest
import asyncio

# Avoid calling `asyncio.get_event_loop()` at import time; tests should use
# `asyncio.run()` or pytest's `event_loop` fixture.


class TestIdentityValidator:
    """Test suite for identity consistency validation."""
    
    @pytest.fixture
    def validator(self):
        """Get the output validator."""
        from output_validator import OutputValidatorSLM
        return OutputValidatorSLM()
    
    def test_correct_arizona_reference_passes(self, validator):
        """Test that Arizona references pass validation."""
        async def run_test():
            result = await validator._check_identity(
                "Brandon is running for Congress in Arizona's 1st Congressional District (AZ-01). "
                "As an Arizona candidate, he understands the issues facing Arizona voters."
            )
            return result
        
        result = asyncio.run(run_test())
        
        assert result.score == 0, f"Arizona reference should pass, got score {result.score}: {result.explanation}"
    
    def test_pennsylvania_campaign_reference_fails(self, validator):
        """Test that Pennsylvania campaign references are caught and rejected."""
        async def run_test():
            result = await validator._check_identity(
                "Brandon's campaign in Pennsylvania focuses on the issues that matter to PA voters."
            )
            return result
        
        result = asyncio.run(run_test())
        
        assert result.score == 5, f"Pennsylvania campaign reference should hard fail, got {result.score}"
        assert "CRITICAL" in result.explanation
        assert "ARIZONA" in result.explanation
    
    def test_running_in_wrong_state_fails(self, validator):
        """Test that 'running in [wrong state]' is caught."""
        async def run_test():
            result = await validator._check_identity(
                "Brandon is running for office in Ohio to represent Ohio constituents."
            )
            return result
        
        result = asyncio.run(run_test())
        
        assert result.score == 5, f"Running in wrong state should fail, got {result.score}"
    
    def test_wrong_state_voters_reference_fails(self, validator):
        """Test that references to wrong state voters/constituents are caught."""
        async def run_test():
            result = await validator._check_identity(
                "Brandon cares deeply about Pennsylvania voters and their concerns."
            )
            return result
        
        result = asyncio.run(run_test())
        
        assert result.score == 5, f"Wrong state voters reference should fail, got {result.score}"
    
    def test_wrong_state_district_reference_fails(self, validator):
        """Test that references to wrong state districts are caught."""
        async def run_test():
            result = await validator._check_identity(
                "Brandon is seeking to represent PA's 7th congressional district."
            )
            return result
        
        result = asyncio.run(run_test())
        
        assert result.score == 5, f"Wrong state district should fail, got {result.score}"
    
    def test_wrong_city_campaign_reference_fails(self, validator):
        """Test that campaign references to wrong state cities are caught."""
        async def run_test():
            result = await validator._check_identity(
                "Brandon's campaign headquarters in Philadelphia is organizing events."
            )
            return result
        
        result = asyncio.run(run_test())
        
        assert result.score == 5, f"Wrong city campaign reference should fail, got {result.score}"
    
    def test_neutral_response_passes(self, validator):
        """Test that responses without state context pass."""
        async def run_test():
            result = await validator._check_identity(
                "Brandon believes in lower taxes and smaller government. "
                "He supports policies that help working families."
            )
            return result
        
        result = asyncio.run(run_test())
        
        assert result.score == 0, f"Neutral response should pass, got {result.score}"
    
    def test_other_state_in_non_campaign_context_passes(self, validator):
        """Test that mentioning other states in non-campaign context is OK."""
        async def run_test():
            result = await validator._check_identity(
                "Texas has similar border issues that Brandon would address. "
                "Like Arizona, Texas faces challenges with immigration policy."
            )
            return result
        
        result = asyncio.run(run_test())
        
        assert result.score == 0, f"Non-campaign state reference should pass, got {result.score}"
    
    def test_representing_wrong_state_fails(self, validator):
        """Test that 'representing [wrong state]' is caught."""
        async def run_test():
            result = await validator._check_identity(
                "Brandon will represent California in Congress."
            )
            return result
        
        result = asyncio.run(run_test())
        
        assert result.score == 5, f"Representing wrong state should fail, got {result.score}"
    
    def test_new_jersey_campaign_fails(self, validator):
        """Test that New Jersey campaign references are caught (coverage for all 50 states)."""
        async def run_test():
            result = await validator._check_identity(
                "Brandon is running for office in New Jersey to help NJ constituents."
            )
            return result
        
        result = asyncio.run(run_test())
        
        assert result.score == 5, f"New Jersey campaign should fail, got {result.score}"
    
    def test_massachusetts_campaign_fails(self, validator):
        """Test that Massachusetts campaign references are caught."""
        async def run_test():
            result = await validator._check_identity(
                "Brandon's campaign in Massachusetts focuses on Boston area issues."
            )
            return result
        
        result = asyncio.run(run_test())
        
        assert result.score == 5, f"Massachusetts campaign should fail, got {result.score}"


class TestIdentityInFullValidation:
    """Test identity check integration with full OV validation."""
    
    @pytest.fixture
    def validator(self):
        from output_validator import OutputValidatorSLM
        return OutputValidatorSLM()
    
    def test_full_validation_catches_wrong_state(self, validator):
        """Test that full validation catches identity violations."""
        async def run_test():
            try:
                result = await validator.validate(
                    query="Where is Brandon running for office?",
                    response="Brandon is running for Congress in Pennsylvania.",
                    pq_confidence=0.9
                )
                return result
            except Exception as e:
                if "FEC RAG" in str(e):
                    pytest.skip("FEC RAG not configured")
                raise
        
        result = asyncio.run(run_test())
        
        from output_validator import OVSafeguard
        identity_result = result.results.get(OVSafeguard.IDENTITY_CONSISTENCY)
        assert identity_result is not None
        assert identity_result.score == 5, f"Should catch Pennsylvania, got {identity_result.score}"
        assert not result.passed, "Full validation should fail on identity violation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
