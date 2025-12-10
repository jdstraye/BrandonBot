"""Tests for bot response sanitization."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation_debug import sanitize_bot_response


class TestSanitizeBotResponse:
    """Test the sanitize_bot_response function."""
    
    def test_strips_corrected_response_header(self):
        """Should remove 'Here's the corrected response' and similar."""
        raw = "Here's the corrected response: Hello! How can I help?"
        cleaned = sanitize_bot_response(raw)
        assert "corrected response" not in cleaned.lower()
        assert "Hello" in cleaned
    
    def test_strips_step_headings(self):
        """Should remove markdown step headings."""
        raw = "## Step 1: Search\nHello there!\n## Step 2: Respond\nHow can I help?"
        cleaned = sanitize_bot_response(raw)
        assert "## Step" not in cleaned
        assert "Step 1" not in cleaned
        assert "Hello there" in cleaned
    
    def test_strips_no_search_needed(self):
        """Should remove 'No search is needed' statements."""
        raw = "No search is needed for greetings. Hi there!"
        cleaned = sanitize_bot_response(raw)
        assert "No search is needed" not in cleaned
        assert "Hi there" in cleaned
    
    def test_strips_verify_statements(self):
        """Should remove 'I'll verify' and similar."""
        raw = "I'll make sure to verify through a search. Welcome to the campaign!"
        cleaned = sanitize_bot_response(raw)
        assert "verify" not in cleaned.lower()
        assert "Welcome" in cleaned
    
    def test_strips_ill_proceed(self):
        """Should remove 'I'll proceed' statements."""
        raw = "So I'll proceed to a friendly response. Nice to meet you!"
        cleaned = sanitize_bot_response(raw)
        assert "proceed" not in cleaned.lower()
        assert "Nice to meet you" in cleaned
    
    def test_full_ov_chatter_sample(self):
        """Test with a full OV chatter sample."""
        raw = '''"You're right, I should verify information through a search. Here's the corrected response:

Hi Jayson! It's great to meet you.

## Step 1: Search for initial information
No search is needed for greetings, so I'll proceed to a friendly response.

The response is: Hi Jayson! How can I help?"'''
        
        cleaned = sanitize_bot_response(raw)
        assert "## Step" not in cleaned
        assert "No search is needed" not in cleaned
        assert "corrected response" not in cleaned.lower()
        assert "Jayson" in cleaned
    
    def test_preserves_clean_response(self):
        """Should preserve already clean responses."""
        raw = "Hi! I'm BrandonBot. Brandon supports fiscal responsibility and limited government."
        cleaned = sanitize_bot_response(raw)
        assert cleaned == raw


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
