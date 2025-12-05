"""
Email Checker for BrandonBot Validation
Verifies email receipt via NetZero POP3/IMAP for testing.

NetZero Settings:
- POP3: pop.netzero.com:995 (SSL)
- SMTP: smtp.netzero.com:465 (SSL)
- IMAP: imap.netzero.com:993 (SSL)

This module is used by the validation system to verify that
emails sent by BrandonBot (volunteer notifications, callbacks, etc.)
are actually delivered to the test inbox.
"""

import os
import ssl
import email
import email.message
import email.header
import imaplib
import poplib
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

NETZERO_IMAP_HOST = "mail.netzero.com"
NETZERO_IMAP_PORT = 993
NETZERO_POP3_HOST = "pop.netzero.com"
NETZERO_POP3_PORT = 995


@dataclass
class EmailMessage:
    """Represents a received email."""
    subject: str
    from_addr: str
    to_addr: str
    date: str
    body_text: str
    body_html: str
    message_id: str
    raw_headers: Dict[str, str]


class NetZeroEmailChecker:
    """
    Checks NetZero inbox for received emails.
    Used in validation to verify email delivery.
    """
    
    def __init__(self):
        self.username = os.environ.get("NETZERO_USER_NAME", "")
        self.password = os.environ.get("NETZERO_PASSWORD", "")
        self.enabled = bool(self.username and self.password)
        
        if not self.enabled:
            logger.warning("NetZero credentials not configured. Email verification disabled.")
    
    def check_for_email(
        self,
        subject_contains: str,
        from_contains: Optional[str] = None,
        since_minutes: int = 30,
        delete_after_read: bool = False
    ) -> Optional[EmailMessage]:
        """
        Check inbox for an email matching the criteria.
        
        Args:
            subject_contains: Text that must appear in the subject line
            from_contains: Text that must appear in the from address (optional)
            since_minutes: Only check emails from the last N minutes (default 30)
            delete_after_read: Whether to delete the email after reading (default False)
        
        Returns:
            EmailMessage if found, None otherwise
        """
        if not self.enabled:
            logger.warning("Email checker not enabled - credentials missing")
            return None
        
        try:
            return self._check_via_pop3(
                subject_contains=subject_contains,
                from_contains=from_contains,
                since_minutes=since_minutes
            )
        except Exception as e:
            logger.error(f"POP3 check failed: {e}")
            try:
                return self._check_via_imap(
                    subject_contains=subject_contains,
                    from_contains=from_contains,
                    since_minutes=since_minutes,
                    delete_after_read=delete_after_read
                )
            except Exception as e2:
                logger.error(f"IMAP fallback also failed: {e2}")
                return None
    
    def _check_via_imap(
        self,
        subject_contains: str,
        from_contains: Optional[str] = None,
        since_minutes: int = 30,
        delete_after_read: bool = False
    ) -> Optional[EmailMessage]:
        """Check email via IMAP (preferred method)."""
        context = ssl.create_default_context()
        
        with imaplib.IMAP4_SSL(NETZERO_IMAP_HOST, NETZERO_IMAP_PORT, ssl_context=context) as imap:
            imap.login(self.username, self.password)
            imap.select("INBOX")
            
            since_date = datetime.now() - timedelta(minutes=since_minutes)
            date_str = since_date.strftime("%d-%b-%Y")
            
            search_criteria = f'(SINCE "{date_str}")'
            if subject_contains:
                search_criteria = f'(SINCE "{date_str}" SUBJECT "{subject_contains}")'
            
            _, message_numbers = imap.search(None, search_criteria)
            
            if not message_numbers[0]:
                logger.info(f"No emails found matching criteria: {subject_contains}")
                return None
            
            for num in message_numbers[0].split()[::-1]:
                _, msg_data = imap.fetch(num, "(RFC822)")
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        subject = self._decode_header(msg.get("Subject", ""))
                        from_addr = self._decode_header(msg.get("From", ""))
                        
                        if subject_contains.lower() not in subject.lower():
                            continue
                        
                        if from_contains and from_contains.lower() not in from_addr.lower():
                            continue
                        
                        body_text, body_html = self._extract_body(msg)
                        
                        result = EmailMessage(
                            subject=subject,
                            from_addr=from_addr,
                            to_addr=self._decode_header(msg.get("To", "")),
                            date=self._decode_header(msg.get("Date", "")),
                            body_text=body_text,
                            body_html=body_html,
                            message_id=msg.get("Message-ID", ""),
                            raw_headers={k: self._decode_header(v) for k, v in msg.items()}
                        )
                        
                        if delete_after_read:
                            imap.store(num, "+FLAGS", "\\Deleted")
                            imap.expunge()
                            logger.info(f"Deleted email: {subject}")
                        
                        logger.info(f"Found matching email: {subject}")
                        return result
            
            return None
    
    def _check_via_pop3(
        self,
        subject_contains: str,
        from_contains: Optional[str] = None,
        since_minutes: int = 30
    ) -> Optional[EmailMessage]:
        """Check email via POP3 (fallback method)."""
        context = ssl.create_default_context()
        
        with poplib.POP3_SSL(NETZERO_POP3_HOST, NETZERO_POP3_PORT, context=context) as pop:
            pop.user(self.username)
            pop.pass_(self.password)
            
            num_messages = len(pop.list()[1])
            since_date = datetime.now() - timedelta(minutes=since_minutes)
            
            for i in range(num_messages, max(0, num_messages - 50), -1):
                try:
                    _, lines, _ = pop.retr(i)
                    msg_content = b"\r\n".join(lines)
                    msg = email.message_from_bytes(msg_content)
                    
                    subject = self._decode_header(msg.get("Subject", ""))
                    from_addr = self._decode_header(msg.get("From", ""))
                    date_str = msg.get("Date", "")
                    
                    if subject_contains.lower() not in subject.lower():
                        continue
                    
                    if from_contains and from_contains.lower() not in from_addr.lower():
                        continue
                    
                    body_text, body_html = self._extract_body(msg)
                    
                    result = EmailMessage(
                        subject=subject,
                        from_addr=from_addr,
                        to_addr=self._decode_header(msg.get("To", "")),
                        date=date_str,
                        body_text=body_text,
                        body_html=body_html,
                        message_id=msg.get("Message-ID", ""),
                        raw_headers={k: self._decode_header(v) for k, v in msg.items()}
                    )
                    
                    logger.info(f"Found matching email via POP3: {subject}")
                    return result
                    
                except Exception as e:
                    logger.warning(f"Error processing message {i}: {e}")
                    continue
            
            return None
    
    def _decode_header(self, header_value: str) -> str:
        """Decode an email header value."""
        if not header_value:
            return ""
        
        try:
            decoded_parts = email.header.decode_header(header_value)
            result = []
            for part, charset in decoded_parts:
                if isinstance(part, bytes):
                    result.append(part.decode(charset or "utf-8", errors="replace"))
                else:
                    result.append(str(part))
            return " ".join(result)
        except:
            return str(header_value)
    
    def _extract_body(self, msg: email.message.Message) -> tuple:
        """Extract text and HTML body from message."""
        body_text = ""
        body_html = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                
                if "attachment" in content_disposition:
                    continue
                
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        text = payload.decode(charset, errors="replace")
                        
                        if content_type == "text/plain" and not body_text:
                            body_text = text
                        elif content_type == "text/html" and not body_html:
                            body_html = text
                except:
                    pass
        else:
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    text = payload.decode(charset, errors="replace")
                    
                    if msg.get_content_type() == "text/html":
                        body_html = text
                    else:
                        body_text = text
            except:
                pass
        
        return body_text, body_html
    
    def list_recent_emails(self, limit: int = 10) -> List[Dict[str, str]]:
        """
        List recent emails in inbox for debugging.
        
        Args:
            limit: Maximum number of emails to list
        
        Returns:
            List of email summaries (subject, from, date)
        """
        if not self.enabled:
            return []
        
        results = []
        
        try:
            context = ssl.create_default_context()
            
            with imaplib.IMAP4_SSL(NETZERO_IMAP_HOST, NETZERO_IMAP_PORT, ssl_context=context) as imap:
                imap.login(self.username, self.password)
                imap.select("INBOX", readonly=True)
                
                _, message_numbers = imap.search(None, "ALL")
                
                if message_numbers[0]:
                    nums = message_numbers[0].split()[-limit:]
                    
                    for num in reversed(nums):
                        _, msg_data = imap.fetch(num, "(RFC822.HEADER)")
                        
                        for response_part in msg_data:
                            if isinstance(response_part, tuple):
                                msg = email.message_from_bytes(response_part[1])
                                
                                results.append({
                                    "subject": self._decode_header(msg.get("Subject", ""))[:100],
                                    "from": self._decode_header(msg.get("From", ""))[:50],
                                    "date": self._decode_header(msg.get("Date", ""))
                                })
        
        except Exception as e:
            logger.error(f"Error listing emails: {e}")
        
        return results


def verify_email_delivery(
    subject_contains: str,
    timeout_seconds: int = 60,
    poll_interval: int = 5
) -> Optional[EmailMessage]:
    """
    Wait for and verify email delivery.
    
    Polls the inbox at regular intervals until the email is found
    or timeout is reached.
    
    Args:
        subject_contains: Text that must appear in subject
        timeout_seconds: Maximum time to wait (default 60s)
        poll_interval: Seconds between checks (default 5s)
    
    Returns:
        EmailMessage if found within timeout, None otherwise
    """
    import time
    
    checker = NetZeroEmailChecker()
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        result = checker.check_for_email(
            subject_contains=subject_contains,
            since_minutes=5
        )
        
        if result:
            return result
        
        logger.info(f"Email not yet received, waiting {poll_interval}s...")
        time.sleep(poll_interval)
    
    logger.warning(f"Email verification timed out after {timeout_seconds}s")
    return None


email_checker = NetZeroEmailChecker()
