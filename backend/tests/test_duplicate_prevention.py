"""
Test duplicate prevention and tool context for callbacks, volunteer signups, and donations.

Tests verify:
1. Session tracks callback/volunteer/donation offers
2. get_tool_context() provides useful context to LLM
3. Duplicate response detection works via get_response_hash() and is_response_duplicate()
4. Tool context is provided to prevent repeated offers
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_orchestrator import Session, ConversationRole


class TestSessionTracking:
    """Test that Session properly tracks offers"""
    
    def test_track_callback_offer(self):
        """Test callback offer tracking"""
        session = Session(session_id="test_session")
        
        # Add a callback offer response
        callback_response = "I want to make sure I give you accurate information. Would you like someone from Brandon's team to call you back to discuss this personally?"
        
        # Track it
        result = session.track_callback_offer(callback_response)
        assert result == True
        assert session.callback_offer_count == 1
        assert session.last_callback_offered_turn >= 0
    
    def test_track_volunteer_offer(self):
        """Test volunteer offer tracking"""
        session = Session(session_id="test_session")
        
        volunteer_response = "That's wonderful that you want to volunteer with us! We'd love to have you sign up to get involved."
        
        result = session.track_volunteer_offer(volunteer_response)
        assert result == True
        assert session.volunteer_offer_count == 1
        assert session.last_volunteer_offered_turn >= 0
    
    def test_track_donation_offer(self):
        """Test donation offer tracking"""
        session = Session(session_id="test_session")
        
        donation_response = "If you'd like to support the campaign financially, we'd appreciate your contribution."
        
        result = session.track_donation_offer(donation_response)
        assert result == True
        assert session.donation_offer_count == 1
        assert session.last_donation_offered_turn >= 0
    
    def test_get_tool_context_no_offers(self):
        """Test tool context when no offers have been made"""
        session = Session(session_id="test_session")
        
        context = session.get_tool_context()
        assert context == ""
    
    def test_get_tool_context_with_offers(self):
        """Test tool context reflects what was offered"""
        session = Session(session_id="test_session")
        
        # Add some offers
        session.track_callback_offer("Would you like a callback?")
        session.track_volunteer_offer("Want to volunteer?")
        session.track_donation_offer("Want to donate?")
        
        context = session.get_tool_context()
        assert "Callback has been offered" in context
        assert "Volunteer signup has been offered" in context
        assert "Donation/support signup has been offered" in context
        assert "If the user affirms interest, execute the appropriate tool" in context
    
    def test_get_tool_context_multiple_offers(self):
        """Test tool context with multiple offers of same type"""
        session = Session(session_id="test_session")
        
        # Offer callback with messages that match the pattern
        result1 = session.track_callback_offer("Would you like a callback from our team?")
        assert result1 == True
        assert session.callback_offer_count == 1
        
        result2 = session.track_callback_offer("I can have someone call you back with more details.")
        assert result2 == True
        assert session.callback_offer_count == 2
        
        context = session.get_tool_context()
        assert "Callback has been offered 2 time(s)" in context


class TestResponseDuplication:
    """Test duplicate response detection"""
    
    def test_get_response_hash_consistency(self):
        """Test that same response produces same hash"""
        session = Session(session_id="test_session")
        
        response = "I want to make sure I give you accurate information. Would you like someone from Brandon's team to call you back to discuss this personally?"
        hash1 = session.get_response_hash(response)
        hash2 = session.get_response_hash(response)
        
        assert hash1 == hash2
    
    def test_get_response_hash_case_insensitive(self):
        """Test that hashing is case-insensitive"""
        session = Session(session_id="test_session")
        
        response1 = "I want to make sure I give you accurate information."
        response2 = "I WANT TO MAKE SURE I GIVE YOU ACCURATE INFORMATION."
        
        hash1 = session.get_response_hash(response1)
        hash2 = session.get_response_hash(response2)
        
        assert hash1 == hash2
    
    def test_get_response_hash_different_whitespace(self):
        """Test that hashing is affected by whitespace differences"""
        session = Session(session_id="test_session")
        
        # Different whitespace should produce different hashes
        response1 = "I want to make sure I give you accurate information."
        response2 = "I want to make  sure   I give you accurate information."
        
        hash1 = session.get_response_hash(response1)
        hash2 = session.get_response_hash(response2)
        
        # These should be different because whitespace differs
        assert hash1 != hash2
    
    def test_is_response_duplicate_no_turns(self):
        """Test duplicate detection with no prior turns"""
        session = Session(session_id="test_session")
        
        response = "This is a test response"
        assert session.is_response_duplicate(response) == False
    
    def test_is_response_duplicate_detects_exact_duplicate(self):
        """Test that exact duplicate is detected"""
        session = Session(session_id="test_session")
        
        response = "I want to make sure I give you accurate information. Would you like someone from Brandon's team to call you back to discuss this personally?"
        
        # Add it as an assistant turn
        session.add_turn(ConversationRole.ASSISTANT, response)
        
        # Now check if it would be a duplicate
        assert session.is_response_duplicate(response) == True
    
    def test_is_response_duplicate_respects_window(self):
        """Test that window parameter works"""
        session = Session(session_id="test_session")
        
        response = "This is the response"
        
        # Add multiple turns
        session.add_turn(ConversationRole.ASSISTANT, "Response 1")
        session.add_turn(ConversationRole.USER, "Question 1")
        session.add_turn(ConversationRole.ASSISTANT, "Response 2")
        session.add_turn(ConversationRole.USER, "Question 2")
        session.add_turn(ConversationRole.ASSISTANT, response)
        
        # Check with small window - should find duplicate
        assert session.is_response_duplicate(response, window=3) == True
        
        # With larger window, should still find it
        assert session.is_response_duplicate(response, window=10) == True
    
    def test_is_response_duplicate_not_outside_window(self):
        """Test that duplicates outside window are not detected"""
        session = Session(session_id="test_session")
        
        response = "This is the response"
        
        # Add the response way in the past
        session.add_turn(ConversationRole.ASSISTANT, response)
        session.add_turn(ConversationRole.USER, "Q1")
        session.add_turn(ConversationRole.ASSISTANT, "A1")
        session.add_turn(ConversationRole.USER, "Q2")
        session.add_turn(ConversationRole.ASSISTANT, "A2")
        
        # With window=1, should not find old duplicate
        assert session.is_response_duplicate(response, window=1) == False
        
        # With larger window, should find it
        assert session.is_response_duplicate(response, window=10) == True


class TestCallbackCooldown:
    """Test callback cooldown logic"""
    
    def test_callback_cooldown_tracking(self):
        """Test that callback turns are tracked correctly"""
        session = Session(session_id="test_session")
        
        # Initial state
        assert session.last_callback_offered_turn == -1
        
        # Add some turns
        session.add_turn(ConversationRole.USER, "Question 1")
        session.add_turn(ConversationRole.ASSISTANT, "Answer 1")
        
        # Track callback
        session.track_callback_offer("Would you like a callback?")
        
        # Verify tracking
        assert session.last_callback_offered_turn >= 0
        assert session.callback_offer_count == 1


if __name__ == "__main__":
    # Run tests
    import pytest
    pytest.main([__file__, "-v"])
