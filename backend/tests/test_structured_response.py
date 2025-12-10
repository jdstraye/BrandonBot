"""
Tests for structured_response.py - JSON and delimiter-based response parsing.

These tests verify that LLM chain-of-thought reasoning is properly separated
from user-facing responses.
"""

import pytest
from structured_response import (
    parse_structured_response,
    StructuredResponseSchema,
    ParsedResponse,
    get_structured_output_instructions,
    get_ov_regeneration_instructions,
)


class TestParseStructuredResponse:
    """Test parse_structured_response function."""
    
    def test_parses_valid_json(self):
        """JSON with final_response field should be parsed correctly."""
        raw = '{"reasoning": "User asked about taxes", "final_response": "Brandon supports tax reform."}'
        result = parse_structured_response(raw)
        
        assert result.final_response == "Brandon supports tax reform."
        assert result.reasoning == "User asked about taxes"
        assert result.parse_method == "json"
    
    def test_parses_json_with_metadata(self):
        """JSON with metadata field should preserve it."""
        raw = '{"reasoning": "thinking...", "final_response": "Hello!", "metadata": {"confidence": 0.9}}'
        result = parse_structured_response(raw)
        
        assert result.final_response == "Hello!"
        assert result.metadata == {"confidence": 0.9}
        assert result.parse_method == "json"
    
    def test_parses_json_in_code_block(self):
        """JSON wrapped in markdown code block should be extracted."""
        raw = '''Here's my response:
```json
{"reasoning": "Let me think", "final_response": "The answer is 42."}
```'''
        result = parse_structured_response(raw)
        
        assert result.final_response == "The answer is 42."
        assert result.parse_method == "json"
    
    def test_parses_final_response_delimiter(self):
        """<final_response> delimiter should extract content."""
        raw = """I should verify this first.
<final_response>
Brandon supports education funding for public schools.
</final_response>"""
        result = parse_structured_response(raw)
        
        assert result.final_response == "Brandon supports education funding for public schools."
        assert result.reasoning == "I should verify this first."
        assert result.parse_method == "delimiter"
    
    def test_parses_response_delimiter(self):
        """<response> delimiter should also work."""
        raw = """Let me check...<response>Here is the answer.</response>"""
        result = parse_structured_response(raw)
        
        assert result.final_response == "Here is the answer."
        assert result.parse_method == "delimiter"
    
    def test_parses_markdown_final_response(self):
        """### Final Response should be extracted."""
        raw = """## Step 1: Think
I need to verify.

### Final Response
Brandon supports healthcare reform.

### Next Steps
Contact us for more."""
        result = parse_structured_response(raw)
        
        assert "Brandon supports healthcare reform" in result.final_response
        assert result.parse_method == "delimiter"
    
    def test_strips_reasoning_prefixes(self):
        """Reasoning prefixes should be stripped when JSON/delimiters fail."""
        raw = "You're right, I should verify this. Brandon supports tax reform."
        result = parse_structured_response(raw)
        
        assert "You're right" not in result.final_response
        assert "Brandon supports tax reform" in result.final_response
        assert result.parse_method == "chatter_stripped"
    
    def test_strips_step_headings(self):
        """Step headings should be removed."""
        raw = """## Step 1: Verify
I'll check this.

Brandon supports education funding."""
        result = parse_structured_response(raw)
        
        assert "Step 1" not in result.final_response
        assert "Brandon supports education funding" in result.final_response
    
    def test_strips_corrected_response_header(self):
        """'Here's the corrected response' should be stripped."""
        raw = "Here's the corrected response: Brandon supports veterans."
        result = parse_structured_response(raw)
        
        assert "corrected response" not in result.final_response.lower()
        assert "Brandon supports veterans" in result.final_response
    
    def test_preserves_clean_response(self):
        """Clean responses without chatter should be preserved."""
        raw = "Brandon supports affordable housing initiatives to help families find homes."
        result = parse_structured_response(raw)
        
        assert result.final_response == raw
    
    def test_handles_empty_response(self):
        """Empty response should return empty result."""
        result = parse_structured_response("")
        
        assert result.final_response == ""
        assert result.parse_method == "empty"
    
    def test_handles_none_like_empty(self):
        """None-equivalent should be handled gracefully."""
        result = parse_structured_response("   ")
        
        assert result.final_response == ""
    
    def test_full_ov_chatter_sample(self):
        """Real OV chatter sample should be cleaned."""
        raw = """You're right, I should verify this claim before responding. Let me check Brandon's platform.

## Step 1: Verification
I'll search the knowledge base.

## Corrected Response:
Here's the corrected response:

Brandon supports comprehensive education reform including increased funding for public schools, teacher pay raises, and expanded vocational training programs."""
        
        result = parse_structured_response(raw)
        
        assert "You're right" not in result.final_response
        assert "Step 1" not in result.final_response
        assert "Corrected Response" not in result.final_response
        assert "Here's the corrected response" not in result.final_response
        assert "Brandon supports comprehensive education reform" in result.final_response


class TestStructuredResponseSchema:
    """Test Pydantic schema validation."""
    
    def test_valid_schema(self):
        """Valid schema should be created."""
        schema = StructuredResponseSchema(
            reasoning="thinking...",
            final_response="Hello!"
        )
        assert schema.final_response == "Hello!"
        assert schema.reasoning == "thinking..."
    
    def test_minimal_schema(self):
        """Only final_response is required."""
        schema = StructuredResponseSchema(final_response="Hello!")
        assert schema.final_response == "Hello!"
        assert schema.reasoning == ""
        assert schema.metadata is None


class TestInstructionGetters:
    """Test instruction generation functions."""
    
    def test_structured_output_instructions(self):
        """Should return non-empty instructions."""
        instructions = get_structured_output_instructions()
        
        assert len(instructions) > 100
        assert "final_response" in instructions
        assert "reasoning" in instructions
        assert "JSON" in instructions
    
    def test_ov_regeneration_instructions(self):
        """Should return OV-specific instructions."""
        instructions = get_ov_regeneration_instructions()
        
        assert len(instructions) > 50
        assert "final_response" in instructions
        assert "You're right" in instructions or "corrected" in instructions.lower()


class TestParsedResponse:
    """Test ParsedResponse dataclass."""
    
    def test_dataclass_fields(self):
        """ParsedResponse should have expected fields."""
        parsed = ParsedResponse(
            final_response="Hello",
            reasoning="thinking",
            metadata={"key": "value"},
            parse_method="json",
            raw_response="original"
        )
        
        assert parsed.final_response == "Hello"
        assert parsed.reasoning == "thinking"
        assert parsed.metadata == {"key": "value"}
        assert parsed.parse_method == "json"
        assert parsed.raw_response == "original"
    
    def test_default_values(self):
        """Default values should be set."""
        parsed = ParsedResponse(final_response="Hello")
        
        assert parsed.reasoning == ""
        assert parsed.metadata is None
        assert parsed.parse_method == "unknown"
        assert parsed.raw_response == ""
