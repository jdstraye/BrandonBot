"""
Tests for email notification and verification.

Tests cover:
1. SendGrid email service configuration and sending
2. Guerrilla Mail checker for verification
3. End-to-end email delivery verification
4. Volunteer notification emails
5. Callback request notification emails
6. Donation interest notification emails

NOTE: These tests require SENDGRID_API_KEY and SENDGRID_FROM_EMAIL to be configured.
Tests will skip gracefully if secrets are not available.
"""

import asyncio
import sys
import os
import pytest
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from email_service import EmailService, email_service, TESTING_MODE, BRANDON_EMAIL
from email_checker import (
    GuerrillaMailChecker, 
    generate_test_email, 
    verify_email_delivery,
    verify_email_delivery_async
)

SENDGRID_CONFIGURED = email_service.enabled
SKIP_REASON = "SendGrid not configured (SENDGRID_API_KEY or SENDGRID_FROM_EMAIL missing)"


def run_async(coro):
    """Helper to run async functions in tests"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@pytest.mark.skipif(not SENDGRID_CONFIGURED, reason=SKIP_REASON)
class TestEmailServiceConfiguration:
    """Test email service is properly configured."""
    
    def test_sendgrid_enabled(self):
        """Verify SendGrid is configured with API key and from email."""
        service = EmailService()
        assert service.enabled, "SendGrid should be enabled with SENDGRID_API_KEY and SENDGRID_FROM_EMAIL"
    
    def test_from_email_configured(self):
        """Verify sender email is configured."""
        service = EmailService()
        assert service.from_email, "SENDGRID_FROM_EMAIL should be set"
        assert "@" in service.from_email, "From email should be a valid email address"
    
    def test_testing_mode_configured(self):
        """Verify testing mode is configured (checks env var exists)."""
        import os
        testing_mode = os.environ.get("TESTING_MODE", "").lower()
        assert testing_mode in ("true", "false", ""), "TESTING_MODE should be 'true', 'false', or unset"


class TestGuerrillaMailChecker:
    """Test the Guerrilla Mail disposable email checker."""
    
    def test_generate_email(self):
        """Can generate a disposable email address."""
        checker = GuerrillaMailChecker()
        email = checker.generate_email()
        
        assert email is not None
        assert "@" in email
        assert checker.sid_token is not None
        print(f"Generated: {email}")
    
    def test_check_empty_inbox(self):
        """New inbox should be empty."""
        checker = GuerrillaMailChecker()
        checker.generate_email()
        
        messages = checker.check_inbox()
        assert isinstance(messages, list)
    
    def test_generate_test_email_helper(self):
        """Helper function returns email and checker."""
        email, checker = generate_test_email()
        
        assert email is not None
        assert "@" in email
        assert checker is not None
        assert checker.sid_token is not None


@pytest.mark.skipif(not SENDGRID_CONFIGURED, reason=SKIP_REASON)
class TestEmailSending:
    """Test sending emails via SendGrid."""
    
    def test_send_basic_email(self):
        """Can send a basic email via SendGrid."""
        test_email, checker = generate_test_email()
        
        async def send():
            result = await email_service.send_email(
                to=test_email,
                subject="[TEST] Basic Email",
                text="This is a basic test email.",
                html="<p>This is a basic test email.</p>"
            )
            return result
        
        result = run_async(send())
        
        assert result.success, f"Email should send successfully: {result.error}"
        assert result.message_id is not None
        print(f"Sent to {test_email}, ID: {result.message_id}")
    
    def test_send_volunteer_notification(self):
        """Can send volunteer notification email."""
        test_email, checker = generate_test_email()
        
        async def send():
            service = EmailService()
            service.from_email = email_service.from_email
            service.from_name = email_service.from_name
            service.api_key = email_service.api_key
            service.enabled = email_service.enabled
            
            result = await service.send_email(
                to=test_email,
                subject="[BrandonBot] New Volunteer: Test User",
                text="NEW VOLUNTEER REGISTRATION\n\nName: Test User\nEmail: test@example.com",
                html="<h1>New Volunteer</h1><p>Name: Test User</p>"
            )
            return result
        
        result = run_async(send())
        
        assert result.success, f"Volunteer notification should send: {result.error}"
        print(f"Volunteer notification sent, ID: {result.message_id}")
    
    def test_send_callback_notification(self):
        """Can send callback notification email."""
        test_email, checker = generate_test_email()
        
        async def send():
            service = EmailService()
            service.from_email = email_service.from_email
            service.from_name = email_service.from_name
            service.api_key = email_service.api_key
            service.enabled = email_service.enabled
            
            result = await service.send_email(
                to=test_email,
                subject="[BrandonBot] Callback Request: Test User",
                text="CALLBACK REQUEST\n\nName: Test User\nPhone: 555-123-4567",
                html="<h1>Callback Request</h1><p>Phone: 555-123-4567</p>"
            )
            return result
        
        result = run_async(send())
        
        assert result.success, f"Callback notification should send: {result.error}"
        print(f"Callback notification sent, ID: {result.message_id}")


@pytest.mark.slow
@pytest.mark.skipif(not SENDGRID_CONFIGURED, reason=SKIP_REASON)
class TestEmailDeliveryVerification:
    """Test end-to-end email delivery verification.
    
    These tests send real emails and verify they arrive.
    Marked as slow because they poll for up to 120 seconds.
    """
    
    def test_end_to_end_delivery(self):
        """Send email and verify it arrives in disposable inbox."""
        test_email, checker = generate_test_email()
        unique_id = f"E2E-{int(time.time())}"
        
        async def send():
            result = await email_service.send_email(
                to=test_email,
                subject=f"[TEST] End-to-End Verification {unique_id}",
                text=f"Test ID: {unique_id}\nThis email verifies end-to-end delivery.",
                html=f"<h1>Test {unique_id}</h1><p>End-to-end verification.</p>"
            )
            return result
        
        send_result = run_async(send())
        assert send_result.success, f"Send should succeed: {send_result.error}"
        print(f"Sent to {test_email}, ID: {send_result.message_id}")
        
        received = verify_email_delivery(
            subject_contains=unique_id,
            timeout_seconds=120,
            poll_interval=5,
            checker=checker
        )
        
        assert received is not None, "Email should be received within timeout"
        assert unique_id in received.subject
        print(f"Verified delivery: {received.subject}")
    
    def test_volunteer_email_delivery(self):
        """Verify volunteer notification email is delivered."""
        test_email, checker = generate_test_email()
        unique_id = f"VOL-{int(time.time())}"
        
        async def send():
            result = await email_service.send_email(
                to=test_email,
                subject=f"[BrandonBot] New Volunteer: Test {unique_id}",
                text=f"Volunteer ID: {unique_id}",
                html=f"<h1>Volunteer {unique_id}</h1>"
            )
            return result
        
        send_result = run_async(send())
        assert send_result.success
        
        received = verify_email_delivery(
            subject_contains=unique_id,
            timeout_seconds=120,
            poll_interval=5,
            checker=checker
        )
        
        assert received is not None, "Volunteer email should be delivered"
        assert "Volunteer" in received.subject
        print(f"Volunteer email verified: {received.subject}")
    
    def test_callback_email_delivery(self):
        """Verify callback notification email is delivered."""
        test_email, checker = generate_test_email()
        unique_id = f"CALL-{int(time.time())}"
        
        async def send():
            result = await email_service.send_email(
                to=test_email,
                subject=f"[BrandonBot] Callback Request: Test {unique_id}",
                text=f"Callback ID: {unique_id}",
                html=f"<h1>Callback {unique_id}</h1>"
            )
            return result
        
        send_result = run_async(send())
        assert send_result.success
        
        received = verify_email_delivery(
            subject_contains=unique_id,
            timeout_seconds=120,
            poll_interval=5,
            checker=checker
        )
        
        assert received is not None, "Callback email should be delivered"
        assert "Callback" in received.subject
        print(f"Callback email verified: {received.subject}")


@pytest.mark.skipif(not SENDGRID_CONFIGURED, reason=SKIP_REASON)
class TestEmailServiceMethods:
    """Test the full email service notification methods."""
    
    def test_send_volunteer_notification_method(self):
        """Test the send_volunteer_notification helper method."""
        async def test():
            result = await email_service.send_volunteer_notification(
                name="Integration Test User",
                email="test@example.com",
                phone="555-123-4567",
                zip_code="85001",
                interests=["door_knocking", "phone_banking"],
                availability="weekends"
            )
            return result
        
        result = run_async(test())
        assert result.success, f"Should send volunteer notification: {result.error}"
    
    def test_send_callback_notification_method(self):
        """Test the send_callback_notification helper method."""
        async def test():
            result = await email_service.send_callback_notification(
                name="Integration Test User",
                phone="555-123-4567",
                reason="I have questions about healthcare policy",
                preferred_time="Afternoons",
                session_id="test-session-123"
            )
            return result
        
        result = run_async(test())
        assert result.success, f"Should send callback notification: {result.error}"
    
    def test_send_donation_interest_notification_method(self):
        """Test the send_donation_interest_notification helper method."""
        async def test():
            result = await email_service.send_donation_interest_notification(
                name="Integration Test User",
                email="test@example.com",
                phone="555-123-4567",
                message="I'd like to support the campaign"
            )
            return result
        
        result = run_async(test())
        assert result.success, f"Should send donation interest notification: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
