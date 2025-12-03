"""
Pytest tests for Citation Verification Safeguard

Tests citation anchor system:
- Anchor injection during ingestion
- Anchor resolution against metadata store
- Detection of missing/invalid citations
- Factual claims without citations
"""

import pytest
import asyncio

try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


@pytest.fixture
def citation_store():
    """Create a test citation store with sample anchors."""
    from citation_anchor_system import CitationAnchorStore
    
    store = CitationAnchorStore()
    
    store.generate_anchor(
        source_type="BP",
        document_id="platform_doc_1",
        page_number=3,
        paragraph_id=15,
        content_snippet="Brandon supports a 15% tax reduction for middle-class families.",
        source_file="platform.md"
    )
    
    store.generate_anchor(
        source_type="QA",
        document_id="qa_doc_1",
        page_number=1,
        paragraph_id=42,
        content_snippet="Healthcare should be accessible and affordable.",
        source_file="qa_responses.md"
    )
    
    store.generate_anchor(
        source_type="WEB",
        document_id="web_search_1",
        page_number=1,
        paragraph_id=1,
        content_snippet="Economic growth projections for next year.",
        source_file="web_search.md"
    )
    
    return store


@pytest.fixture
def citation_verifier(citation_store):
    """Create a citation verifier with the test store."""
    from citation_anchor_system import CitationVerifier
    return CitationVerifier(citation_store)


@pytest.fixture
def validator_with_store(citation_store):
    """Create output validator with citation store."""
    from output_validator import OutputValidatorSLM
    validator = OutputValidatorSLM()
    validator.set_citation_store({
        "CITE-BP-001": {"document_id": "platform_doc_1", "page": 3, "line": 15},
        "CITE-QA-001": {"document_id": "qa_doc_1", "page": 1, "line": 42},
        "CITE-WEB-001": {"document_id": "web_search_1", "page": 1, "line": 1},
    })
    return validator


CITATION_PASS_CASES = [
    ("Tax policy?", "Brandon supports a 15% tax reduction [CITE-BP-001].", 0),
    ("Healthcare?", "Healthcare should be accessible [CITE-QA-001] and affordable.", 0),
    ("General question", "Brandon is committed to serving the community.", 0),
    ("Multiple citations", "Tax reform [CITE-BP-001] and healthcare [CITE-QA-001] are priorities.", 0),
]

CITATION_FAIL_CASES = [
    ("Statistics", "Brandon's plan will save $1.2 billion annually.", 5, "Missing citation for statistic"),
    ("Incomplete", "Brandon supports this policy [CITE].", 2, "Incomplete citation"),
    ("Invalid format", "Brandon supports this policy [CITE-123].", 5, "Invalid format"),
    ("Non-standard", "This is supported [CITE: something].", 3, "Non-standard format"),
]


class TestCitationAnchorStore:
    """Test the CitationAnchorStore component."""
    
    def test_generate_anchor(self, citation_store):
        """Test anchor generation."""
        anchor = citation_store.generate_anchor(
            source_type="BP",
            document_id="test_doc",
            page_number=1,
            paragraph_id=1,
            content_snippet="Test content"
        )
        
        assert anchor.startswith("CITE-BP-")
    
    def test_inject_anchor(self, citation_store):
        """Test anchor injection into content."""
        content = "This is the original content."
        anchor = "CITE-BP-001"
        
        injected = citation_store.inject_anchor(content, anchor)
        
        assert f"[{anchor}]" in injected
        assert injected.startswith("This is the original content")
    
    def test_resolve_anchor(self, citation_store):
        """Test anchor resolution."""
        metadata = citation_store.resolve_anchor("CITE-BP-001")
        
        assert metadata is not None
        assert metadata.document_id == "platform_doc_1"
    
    def test_resolve_nonexistent_anchor(self, citation_store):
        """Test resolution of nonexistent anchor."""
        metadata = citation_store.resolve_anchor("CITE-BP-999")
        
        assert metadata is None
    
    def test_get_stats(self, citation_store):
        """Test statistics retrieval."""
        stats = citation_store.get_stats()
        
        assert stats["total"] >= 3
        assert "by_type" in stats


class TestCitationVerifier:
    """Test the CitationVerifier component."""
    
    def test_verify_valid_citations(self, citation_verifier):
        """Test verification of valid citations."""
        response = "Brandon supports tax reform [CITE-BP-001] and healthcare access [CITE-QA-001]."
        
        result = citation_verifier.verify(response)
        
        assert result.valid
        assert len(result.anchors_found) == 2
        assert len(result.anchors_resolved) == 2
    
    def test_verify_unresolved_anchor(self, citation_verifier):
        """Test verification with unresolved anchor."""
        response = "Brandon supports this policy [CITE-BP-999]."
        
        result = citation_verifier.verify(response)
        
        assert not result.valid or result.score > 0
        assert len(result.anchors_unresolved) >= 0
    
    def test_verify_incomplete_citation(self, citation_verifier):
        """Test detection of incomplete citations."""
        response = "Brandon supports this policy [CITE]."
        
        result = citation_verifier.verify(response)
        
        assert len(result.format_issues) > 0
        assert result.score >= 2
    
    def test_verify_invalid_format(self, citation_verifier):
        """Test detection of invalid citation format."""
        response = "Brandon supports this policy [CITE-123]."
        
        result = citation_verifier.verify(response)
        
        assert result.score >= 4
    
    def test_verify_missing_citation_for_statistic(self, citation_verifier):
        """Test detection of missing citation for statistics."""
        response = "Brandon's plan will save approximately $1.2 billion annually."
        
        result = citation_verifier.verify(response)
        
        assert result.score >= 4
        assert len(result.missing_citations) > 0
    
    def test_verify_no_citation_needed(self, citation_verifier):
        """Test that general statements don't require citations."""
        response = "Brandon is committed to serving the community and working for families."
        
        result = citation_verifier.verify(response)
        
        assert result.score == 0


class TestCitationValidation:
    """Test citation validation through the output validator."""
    
    @pytest.mark.parametrize("query,response,expected_max_score", CITATION_PASS_CASES)
    def test_citation_pass_cases(self, validator_with_store, query, response, expected_max_score):
        """Test that properly cited responses pass validation."""
        async def run_test():
            result = await validator_with_store._check_citations(response)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score <= 2, f"Expected pass (score <= 2), got {result.score}: {result.explanation}"
    
    @pytest.mark.parametrize("query,response,expected_min_score,issue_type", CITATION_FAIL_CASES)
    def test_citation_fail_cases(self, validator_with_store, query, response, expected_min_score, issue_type):
        """Test that improperly cited responses fail validation."""
        async def run_test():
            result = await validator_with_store._check_citations(response)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        
        assert result.score >= expected_min_score - 1, f"Expected score >= {expected_min_score-1}, got {result.score}: {result.explanation}"


class TestCitationPromptBuilder:
    """Test the CitationPromptBuilder."""
    
    def test_build_prompt_with_citations(self):
        """Test prompt building with citation requirements."""
        from citation_anchor_system import CitationPromptBuilder
        
        context_chunks = [
            {"content": "Brandon supports tax reform.", "anchor": "CITE-BP-001", "source": "platform.md"},
            {"content": "Healthcare is a priority.", "anchor": "CITE-QA-001", "source": "qa.md"},
        ]
        
        prompt = CitationPromptBuilder.build_prompt(
            query="What is Brandon's policy?",
            context_chunks=context_chunks,
            base_instruction="You are a helpful assistant."
        )
        
        assert "CITE-BP-001" in prompt
        assert "CITE-QA-001" in prompt
        assert "CITATION REQUIREMENTS" in prompt


class TestCitationEdgeCases:
    """Edge cases for citation checking."""
    
    @pytest.fixture
    def validator(self):
        from output_validator import OutputValidatorSLM
        return OutputValidatorSLM()
    
    def test_empty_response(self, validator):
        """Test handling of empty response."""
        async def run_test():
            result = await validator._check_citations("")
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        assert result.score == 0
    
    def test_multiple_citations_same_sentence(self, validator_with_store):
        """Test multiple citations in same sentence."""
        async def run_test():
            response = "Both tax [CITE-BP-001] and healthcare [CITE-QA-001] are important."
            result = await validator_with_store._check_citations(response)
            return result
        
        result = asyncio.get_event_loop().run_until_complete(run_test())
        assert result.score == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
