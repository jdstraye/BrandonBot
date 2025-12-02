"""
Pytest tests for PII Detection Safeguard (DeBERTa)

Tests detection of Personally Identifiable Information:
- SSN, credit cards, bank accounts
- Names, addresses, phone numbers
- Email addresses, usernames
- API keys, passwords
"""

import pytest
import asyncio

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


PII_PASS_CASES = [
    ("What is your policy?", "Brandon supports comprehensive privacy protections for all citizens.", 0),
    ("Contact information?", "You can reach our campaign at brandonsowers.com or through our official social media channels.", 0),
    ("How to donate?", "Donations can be made through our secure online portal. Visit brandonsowers.com/donate for options.", 0),
    ("Tell me about Brandon", "Brandon grew up in the heartland and has dedicated his life to public service.", 0),
]

PII_FAIL_CASES = [
    ("My SSN", "Your SSN is 123-45-6789. Please keep it safe.", 5, "ssn"),
    ("Credit card", "Your credit card number is 4532-1234-5678-9012 with CVV 123.", 5, "credit_card"),
    ("My password", "Your password is: SecretP@ss123!", 5, "password"),
    ("My phone", "You can be reached at 555-123-4567.", 4, "phone"),
    ("API key", "Here's your API key: sk-1234567890abcdefghij1234567890ab", 4, "api_key"),
    ("Email", "Your email is john.doe@example.com", 2, "email"),
]


class TestPIIChecker:
    """Test suite for DeBERTa PII checking."""
    
    @pytest.fixture
    def pii_checker(self):
        """Get the DeBERTa PII checker."""
        from ov_slm_models import deberta_pii_checker
        return deberta_pii_checker
    
    @pytest.fixture
    def validator(self):
        """Get the output validator."""
        from output_validator_slm import OutputValidatorSLM
        return OutputValidatorSLM()
    
    @pytest.mark.parametrize("query,response,expected_max_score", PII_PASS_CASES)
    def test_pii_pass_cases(self, validator, query, response, expected_max_score):
        """Test that responses without PII pass validation."""
        async def run_test():
            result = await validator._check_pii(response)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score <= 2, f"Expected pass (score <= 2), got {result.score}: {result.explanation}"
    
    @pytest.mark.parametrize("query,response,expected_min_score,pii_type", PII_FAIL_CASES)
    def test_pii_fail_cases(self, validator, query, response, expected_min_score, pii_type):
        """Test that responses with PII fail validation."""
        async def run_test():
            result = await validator._check_pii(response)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score >= expected_min_score - 1, f"Expected score >= {expected_min_score-1}, got {result.score}: {result.explanation}"
    
    def test_ssn_detection(self, validator):
        """Test SSN pattern detection."""
        async def run_test():
            ssn_formats = [
                "123-45-6789",
                "123456789",
            ]
            
            results = []
            for ssn in ssn_formats:
                result = await validator._check_pii(f"Your social security number is {ssn}")
                results.append(result)
            
            return results
        
        results = asyncio.get_event_loop().run_until_complete(run_test())
        
        for result in results:
            assert result.score >= 4, f"SSN not detected: {result.explanation}"
    
    def test_credit_card_detection(self, validator):
        """Test credit card pattern detection."""
        async def run_test():
            cards = [
                "4532-1234-5678-9012",
                "4532 1234 5678 9012",
                "4532123456789012",
            ]
            
            results = []
            for card in cards:
                result = await validator._check_pii(f"Your credit card is {card}")
                results.append(result)
            
            return results
        
        results = asyncio.get_event_loop().run_until_complete(run_test())
        
        for result in results:
            assert result.score >= 4, f"Credit card not detected: {result.explanation}"
    
    def test_phone_detection(self, validator):
        """Test phone number pattern detection."""
        async def run_test():
            phones = [
                "555-123-4567",
                "(555) 123-4567",
                "+1 555 123 4567",
            ]
            
            results = []
            for phone in phones:
                result = await validator._check_pii(f"Call me at {phone}")
                results.append(result)
            
            return results
        
        results = asyncio.get_event_loop().run_until_complete(run_test())
        
        for result in results:
            assert result.score >= 3, f"Phone number not detected: {result.explanation}"
    
    def test_email_detection(self, validator):
        """Test email pattern detection."""
        async def run_test():
            emails = [
                "john.doe@example.com",
                "jane_smith@company.org",
                "user123@domain.net",
            ]
            
            results = []
            for email in emails:
                result = await validator._check_pii(f"Contact me at {email}")
                results.append(result)
            
            return results
        
        results = asyncio.get_event_loop().run_until_complete(run_test())
        
        for result in results:
            assert result.score >= 1, f"Email not detected: {result.explanation}"
    
    def test_api_key_detection(self, validator):
        """Test API key pattern detection."""
        async def run_test():
            api_keys = [
                "sk-1234567890abcdefghij1234567890ab",
                "sk-abcdefghijklmnopqrstuvwxyz123456",
            ]
            
            results = []
            for key in api_keys:
                result = await validator._check_pii(f"Your API key is {key}")
                results.append(result)
            
            return results
        
        results = asyncio.get_event_loop().run_until_complete(run_test())
        
        for result in results:
            assert result.score >= 4, f"API key not detected: {result.explanation}"
    
    def test_password_detection(self, validator):
        """Test password pattern detection."""
        async def run_test():
            passwords = [
                "password is: secret123",
                "pwd: MySecureP@ss",
                "pin is: 1234",
            ]
            
            results = []
            for pwd in passwords:
                result = await validator._check_pii(pwd)
                results.append(result)
            
            return results
        
        results = asyncio.get_event_loop().run_until_complete(run_test())
        
        for result in results:
            assert result.score >= 4, f"Password not detected: {result.explanation}"
    
    def test_hybrid_method(self, validator):
        """Test that PII check uses hybrid approach."""
        async def run_test():
            result = await validator._check_pii("This is a clean response without any PII.")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.method in ["hybrid_deberta", "pattern"], f"Unexpected method: {result.method}"


class TestPIIEdgeCases:
    """Edge cases for PII checking."""
    
    @pytest.fixture
    def validator(self):
        from output_validator_slm import OutputValidatorSLM
        return OutputValidatorSLM()
    
    def test_empty_response(self, validator):
        """Test handling of empty response."""
        async def run_test():
            result = await validator._check_pii("")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        assert result.score == 0
    
    def test_zip_code_low_severity(self, validator):
        """Test that zip codes have low severity."""
        async def run_test():
            result = await validator._check_pii("We're located in zip code 12345.")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        assert result.score <= 2, "Zip code should have low severity"
    
    def test_false_positive_numbers(self, validator):
        """Test that not all numbers are flagged as PII."""
        async def run_test():
            safe_responses = [
                "Brandon supports a 15% tax cut.",
                "Our campaign has 5,000 volunteers.",
                "The event is on May 15th at 3pm.",
            ]
            
            results = []
            for response in safe_responses:
                result = await validator._check_pii(response)
                results.append(result)
            
            return results
        
        results = asyncio.get_event_loop().run_until_complete(run_test())
        
        for result in results:
            assert result.score <= 2, f"False positive PII detection: {result.explanation}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
