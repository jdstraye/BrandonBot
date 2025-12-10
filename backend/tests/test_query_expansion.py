"""
Pytest tests for Query Expansion Module

Tests topic detection, especially callback vs. donation disambiguation.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from query_expansion import get_topic_from_query, expand_query, detect_question_type


class TestTopicDetection:
    """Test suite for get_topic_from_query function."""
    
    def test_callback_phrase_give_me_a_call(self):
        """Test that 'give me a call' is detected as callback, not donate."""
        result = get_topic_from_query("Can you give me a call to discuss this further?")
        assert result == "callback", f"Expected 'callback', got '{result}'"
    
    def test_callback_phrase_call_me_back(self):
        """Test that 'call me back' is detected as callback."""
        result = get_topic_from_query("Please call me back about this issue.")
        assert result == "callback", f"Expected 'callback', got '{result}'"
    
    def test_callback_phrase_call_me(self):
        """Test that 'call me' is detected as callback."""
        result = get_topic_from_query("Call me when you have time.")
        assert result == "callback", f"Expected 'callback', got '{result}'"
    
    def test_callback_phrase_phone_call(self):
        """Test that 'phone call' is detected as callback."""
        result = get_topic_from_query("I'd like a phone call to discuss policies.")
        assert result == "callback", f"Expected 'callback', got '{result}'"
    
    def test_callback_phrase_schedule_a_call(self):
        """Test that 'schedule a call' is detected as callback."""
        result = get_topic_from_query("Can we schedule a call?")
        assert result == "callback", f"Expected 'callback', got '{result}'"
    
    def test_callback_phrase_speak_to_someone(self):
        """Test that 'speak to someone' is detected as callback."""
        result = get_topic_from_query("I want to speak to someone about volunteering.")
        assert result == "callback", f"Expected 'callback', got '{result}'"
    
    def test_callback_phrase_get_in_touch(self):
        """Test that 'get in touch' is detected as callback."""
        result = get_topic_from_query("How can I get in touch with the campaign?")
        assert result == "callback", f"Expected 'callback', got '{result}'"
    
    def test_callback_phrase_reach_out(self):
        """Test that 'reach out to me' is detected as callback."""
        result = get_topic_from_query("Can someone reach out to me?")
        assert result == "callback", f"Expected 'callback', got '{result}'"
    
    def test_callback_phrase_talk_to_someone(self):
        """Test that 'talk to someone' is detected as callback."""
        result = get_topic_from_query("I need to talk to someone about this.")
        assert result == "callback", f"Expected 'callback', got '{result}'"
    
    def test_donate_not_confused_with_callback(self):
        """Test that 'give' in donation context returns donate, not callback."""
        result = get_topic_from_query("I want to give to the campaign.")
        assert result == "donate", f"Expected 'donate', got '{result}'"
    
    def test_donate_phrase_donation(self):
        """Test that donation is detected correctly."""
        result = get_topic_from_query("I want to make a donation.")
        assert result == "donate", f"Expected 'donate', got '{result}'"
    
    def test_donate_phrase_contribute(self):
        """Test that contribute is detected correctly."""
        result = get_topic_from_query("How can I contribute to the campaign?")
        assert result == "donate", f"Expected 'donate', got '{result}'"
    
    def test_healthcare_topic(self):
        """Test that healthcare questions are detected correctly."""
        result = get_topic_from_query("What is Brandon's position on healthcare?")
        assert result == "healthcare", f"Expected 'healthcare', got '{result}'"
    
    def test_immigration_topic(self):
        """Test that immigration questions are detected correctly."""
        result = get_topic_from_query("What is your stance on immigration?")
        assert result == "immigration", f"Expected 'immigration', got '{result}'"
    
    def test_economy_topic(self):
        """Test that economy questions are detected correctly."""
        result = get_topic_from_query("How will you improve the economy and jobs?")
        assert result == "economy", f"Expected 'economy', got '{result}'"
    
    def test_faith_topic(self):
        """Test that faith questions are detected correctly."""
        result = get_topic_from_query("How does your faith inform your policies?")
        assert result == "faith", f"Expected 'faith', got '{result}'"
    
    def test_volunteer_topic(self):
        """Test that volunteer questions are detected correctly."""
        result = get_topic_from_query("How can I volunteer for the campaign?")
        assert result == "volunteer", f"Expected 'volunteer', got '{result}'"
    
    def test_general_fallback(self):
        """Test that unrecognized queries return 'general'."""
        result = get_topic_from_query("Hello, how are you today?")
        assert result == "general", f"Expected 'general', got '{result}'"


class TestCallbackPriority:
    """Test that callback detection takes priority over other topics."""
    
    def test_callback_over_donate_give(self):
        """'Give me a call' should be callback, not donate from 'give'."""
        result = get_topic_from_query("Can you give me a call?")
        assert result == "callback", f"Callback should take priority over donate"
    
    def test_callback_over_healthcare_discuss(self):
        """Callback phrases should take priority even with policy topics mentioned."""
        result = get_topic_from_query("Can you call me back to discuss healthcare?")
        assert result == "callback", f"Callback should take priority when callback phrase present"
    
    def test_callback_over_volunteer_talk(self):
        """Callback phrases should take priority over volunteer mentions."""
        result = get_topic_from_query("I want to talk to someone about volunteering opportunities.")
        assert result == "callback", f"Callback should take priority due to 'talk to someone'"


class TestAdversarialMixedQueries:
    """Test mixed callback+donate and callback+policy queries to ensure proper disambiguation."""
    
    def test_mixed_callback_and_donation_mention(self):
        """Callback phrase with donation mention should still be callback."""
        result = get_topic_from_query("Can you give me a call about donating $50?")
        assert result == "callback", f"Expected 'callback' for mixed query, got '{result}'"
    
    def test_callback_phrase_with_donate_context(self):
        """Callback phrase in donation context should still be callback."""
        result = get_topic_from_query("I want to give to the campaign but call me back first")
        assert result == "callback", f"Expected 'callback', got '{result}'"
    
    def test_donate_then_callback_phrase(self):
        """Donation mention followed by callback phrase should be callback."""
        result = get_topic_from_query("I want to donate. Can you give me a call?")
        assert result == "callback", f"Expected 'callback', got '{result}'"
    
    def test_callback_with_policy_topic(self):
        """Callback phrase with policy topic should be callback."""
        result = get_topic_from_query("Can you call me about immigration policy?")
        assert result == "callback", f"Expected 'callback', got '{result}'"
    
    def test_donate_without_callback_phrase(self):
        """Donation query without callback phrase should be donate."""
        result = get_topic_from_query("I want to give money to the campaign")
        assert result == "donate", f"Expected 'donate', got '{result}'"
    
    def test_contribute_without_callback_phrase(self):
        """Contribution query without callback phrase should be donate."""
        result = get_topic_from_query("How can I contribute financially?")
        assert result == "donate", f"Expected 'donate', got '{result}'"
    
    def test_give_without_callback_context(self):
        """'Give' without callback phrase should be donate."""
        result = get_topic_from_query("I want to give to support Brandon")
        assert result == "donate", f"Expected 'donate', got '{result}'"
    
    def test_callback_with_immigration_topic(self):
        """Callback phrase with immigration topic should be callback."""
        result = get_topic_from_query("Please call me back to discuss immigration")
        assert result == "callback", f"Expected 'callback', got '{result}'"
    
    def test_callback_with_economy_topic(self):
        """Callback phrase with economy topic should be callback."""
        result = get_topic_from_query("I'd like a phone call about the economy")
        assert result == "callback", f"Expected 'callback', got '{result}'"


class TestQueryExpansion:
    """Test suite for expand_query function."""
    
    def test_healthcare_expansion(self):
        """Test that healthcare query gets expanded with medical terms."""
        result = expand_query("healthcare policy")
        assert len(result) > 0, "Should have expanded terms"
        assert any("health" in term.lower() or "medical" in term.lower() for term in result)
    
    def test_callback_expansion(self):
        """Test that callback query gets expanded with related terms."""
        result = expand_query("give me a call")
        assert len(result) > 0, "Should have expanded terms"


class TestQuestionTypeDetection:
    """Test suite for detect_question_type function."""
    
    def test_comparison_question(self):
        """Test detection of comparison questions."""
        result = detect_question_type("How does Brandon compare to other candidates?")
        assert "comparison" in result, f"Expected 'comparison' in {result}"
    
    def test_policy_question(self):
        """Test detection of policy questions."""
        result = detect_question_type("What is Brandon's position on taxes?")
        assert "policy" in result, f"Expected 'policy' in {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
