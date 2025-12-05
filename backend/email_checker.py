"""
Email Checker for BrandonBot Validation
Uses Guerrilla Mail disposable email API for testing email delivery.

Guerrilla Mail API (no auth required):
- Base URL: http://api.guerrillamail.com/ajax.php
- Get email: f=get_email_address
- Check inbox: f=check_email&sid_token=TOKEN
- Read message: f=fetch_email&email_id=ID&sid_token=TOKEN

This module is used by the validation system to verify that
emails sent by BrandonBot (volunteer notifications, callbacks, etc.)
are actually delivered.
"""

import time
import logging
import httpx
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

GUERRILLA_API = "https://www.guerrillamail.com/ajax.php"


@dataclass
class EmailMessage:
    """Represents a received email."""
    id: str
    subject: str
    from_addr: str
    to_addr: str
    date: str
    body_text: str
    body_html: str
    attachments: List[Dict[str, Any]]


class GuerrillaMailChecker:
    """
    Disposable email checker using Guerrilla Mail API.
    No authentication required - perfect for automated testing.
    """
    
    def __init__(self):
        self.email_addr: Optional[str] = None
        self.sid_token: Optional[str] = None
        self.email_timestamp: Optional[int] = None
        self.alias: Optional[str] = None
        self.enabled = True
    
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to Guerrilla Mail API."""
        params["ip"] = "127.0.0.1"
        params["agent"] = "BrandonBot-Validator"
        
        if self.sid_token:
            params["sid_token"] = self.sid_token
        
        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.get(GUERRILLA_API, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "sid_token" in data:
                    self.sid_token = data["sid_token"]
                
                return data
        except Exception as e:
            logger.error(f"Guerrilla API error: {e}")
            raise
    
    def generate_email(self) -> str:
        """
        Generate a new random disposable email address.
        
        Returns:
            Email address string (e.g., "abc123@guerrillamail.com")
        """
        data = self._make_request({"f": "get_email_address"})
        
        self.email_addr = data.get("email_addr")
        self.alias = data.get("alias")
        self.email_timestamp = data.get("email_timestamp")
        
        logger.info(f"Generated test email: {self.email_addr}")
        return self.email_addr
    
    def set_email_user(self, username: str) -> str:
        """
        Set a custom username for the email.
        
        Args:
            username: Desired username (e.g., "testuser")
        
        Returns:
            Full email address
        """
        data = self._make_request({
            "f": "set_email_user",
            "email_user": username
        })
        
        self.email_addr = data.get("email_addr")
        self.alias = data.get("alias")
        
        logger.info(f"Set email to: {self.email_addr}")
        return self.email_addr
    
    def check_inbox(self, seq: int = 0) -> List[Dict[str, Any]]:
        """
        Check inbox for messages.
        
        Args:
            seq: Sequence number for checking new emails (0 for all)
        
        Returns:
            List of message summaries
        """
        if not self.sid_token:
            self.generate_email()
        
        data = self._make_request({
            "f": "check_email",
            "seq": seq
        })
        
        messages = data.get("list", [])
        logger.debug(f"Inbox has {len(messages)} messages")
        return messages
    
    def get_email_list(self, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get list of all emails with offset.
        
        Args:
            offset: Offset for pagination
        
        Returns:
            List of message summaries
        """
        if not self.sid_token:
            self.generate_email()
        
        data = self._make_request({
            "f": "get_email_list",
            "offset": offset
        })
        
        return data.get("list", [])
    
    def read_message(self, email_id: str) -> Optional[EmailMessage]:
        """
        Read full message content.
        
        Args:
            email_id: Message ID from check_inbox()
        
        Returns:
            EmailMessage with full content, or None on error
        """
        if not self.sid_token:
            raise ValueError("No session token. Call generate_email() first.")
        
        try:
            data = self._make_request({
                "f": "fetch_email",
                "email_id": email_id
            })
            
            return EmailMessage(
                id=str(data.get("mail_id", email_id)),
                subject=data.get("mail_subject", ""),
                from_addr=data.get("mail_from", ""),
                to_addr=self.email_addr or "",
                date=data.get("mail_date", ""),
                body_text=data.get("mail_excerpt", ""),
                body_html=data.get("mail_body", ""),
                attachments=[]
            )
        
        except Exception as e:
            logger.error(f"Failed to read message {email_id}: {e}")
            return None
    
    def check_for_email(
        self,
        subject_contains: str,
        from_contains: Optional[str] = None,
        since_minutes: int = 30
    ) -> Optional[EmailMessage]:
        """
        Check inbox for an email matching criteria.
        
        Args:
            subject_contains: Text that must appear in subject
            from_contains: Text that must appear in from address (optional)
            since_minutes: Ignored (Guerrilla only keeps recent emails)
        
        Returns:
            EmailMessage if found, None otherwise
        """
        messages = self.check_inbox()
        
        for msg in messages:
            subject = msg.get("mail_subject", "")
            from_addr = msg.get("mail_from", "")
            
            if subject_contains.lower() not in subject.lower():
                continue
            
            if from_contains and from_contains.lower() not in from_addr.lower():
                continue
            
            full_message = self.read_message(str(msg.get("mail_id")))
            if full_message:
                logger.info(f"Found matching email: {subject}")
                return full_message
        
        return None
    
    def list_recent_emails(self, limit: int = 10) -> List[Dict[str, str]]:
        """
        List recent emails for debugging.
        
        Args:
            limit: Maximum number to return
        
        Returns:
            List of email summaries
        """
        messages = self.check_inbox()
        return [
            {
                "id": str(msg.get("mail_id", "")),
                "subject": msg.get("mail_subject", "")[:100],
                "from": msg.get("mail_from", "")[:50],
                "date": msg.get("mail_date", "")
            }
            for msg in messages[:limit]
        ]
    
    def delete_email(self, email_id: str) -> bool:
        """Delete an email by ID."""
        try:
            self._make_request({
                "f": "del_email",
                "email_ids[]": email_id
            })
            return True
        except:
            return False
    
    def forget_session(self) -> bool:
        """Delete the current session and email address."""
        try:
            self._make_request({"f": "forget_me"})
            self.email_addr = None
            self.sid_token = None
            return True
        except:
            return False


def verify_email_delivery(
    subject_contains: str,
    timeout_seconds: int = 120,
    poll_interval: int = 5,
    checker: Optional[GuerrillaMailChecker] = None
) -> Optional[EmailMessage]:
    """
    Wait for and verify email delivery.
    
    If no checker is provided, generates a new email address.
    
    Args:
        subject_contains: Text that must appear in subject
        timeout_seconds: Maximum time to wait (default 120s)
        poll_interval: Seconds between checks (default 5s)
        checker: Optional pre-configured checker instance
    
    Returns:
        EmailMessage if found within timeout, None otherwise
    """
    if checker is None:
        checker = GuerrillaMailChecker()
        email = checker.generate_email()
        logger.info(f"Waiting for email at: {email}")
    
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        result = checker.check_for_email(subject_contains=subject_contains)
        
        if result:
            elapsed = time.time() - start_time
            logger.info(f"Email verified after {elapsed:.1f}s: {result.subject}")
            return result
        
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed
        logger.info(f"Email not yet received ({elapsed:.0f}s elapsed, {remaining:.0f}s remaining)...")
        time.sleep(poll_interval)
    
    logger.warning(f"Email verification timed out after {timeout_seconds}s")
    return None


async def verify_email_delivery_async(
    subject_contains: str,
    timeout_seconds: int = 120,
    poll_interval: int = 5,
    checker: Optional[GuerrillaMailChecker] = None
) -> Optional[EmailMessage]:
    """
    Async version of verify_email_delivery for use in async contexts.
    """
    import asyncio
    
    if checker is None:
        checker = GuerrillaMailChecker()
        email = checker.generate_email()
        logger.info(f"Waiting for email at: {email}")
    
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        result = checker.check_for_email(subject_contains=subject_contains)
        
        if result:
            elapsed = time.time() - start_time
            logger.info(f"Email verified after {elapsed:.1f}s: {result.subject}")
            return result
        
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed
        logger.info(f"Email not yet received ({elapsed:.0f}s elapsed, {remaining:.0f}s remaining)...")
        await asyncio.sleep(poll_interval)
    
    logger.warning(f"Email verification timed out after {timeout_seconds}s")
    return None


def generate_test_email() -> Tuple[str, GuerrillaMailChecker]:
    """
    Generate a fresh test email and return both the address and checker.
    
    Convenience function for validation tests.
    
    Returns:
        Tuple of (email_address, checker_instance)
    """
    checker = GuerrillaMailChecker()
    email = checker.generate_email()
    return email, checker


email_checker = GuerrillaMailChecker()
