"""
Email Service for BrandonBot
Uses SendGrid for sending notifications.

Email Routing:
- Testing (TESTING_MODE=true): jdstrayer@netzero.net
- Production: campaign@brandonsowers.com
"""

import os
import logging
from typing import Optional, List
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

TESTING_MODE = os.environ.get("TESTING_MODE", "true").lower() == "true"
TEST_EMAIL = "jdstrayer@netzero.net"
PRODUCTION_EMAIL = "campaign@brandonsowers.com"
BRANDON_EMAIL = TEST_EMAIL if TESTING_MODE else PRODUCTION_EMAIL

SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL", "")
SENDGRID_FROM_NAME = os.environ.get("SENDGRID_FROM_NAME", "BrandonBot")

@dataclass
class EmailResult:
    success: bool
    message_id: Optional[str] = None
    accepted: Optional[List[str]] = None
    rejected: Optional[List[str]] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.accepted is None:
            self.accepted = []
        if self.rejected is None:
            self.rejected = []


class EmailService:
    """
    Email service using SendGrid API.
    Falls back to logging if SendGrid is not configured.
    """
    
    def __init__(self):
        self.api_key = SENDGRID_API_KEY
        self.from_email = SENDGRID_FROM_EMAIL
        self.from_name = SENDGRID_FROM_NAME
        self.enabled = bool(self.api_key and self.from_email)
        
        if not self.api_key:
            logger.warning("SENDGRID_API_KEY not set - email service will log instead of sending")
        elif not self.from_email:
            logger.warning("SENDGRID_FROM_EMAIL not set - email service will log instead of sending")
        else:
            logger.info(f"SendGrid email service initialized (from: {self.from_email})")
    
    async def send_email(
        self,
        to: str,
        subject: str,
        text: Optional[str] = None,
        html: Optional[str] = None,
        cc: Optional[str] = None
    ) -> EmailResult:
        """
        Send an email using SendGrid API.
        
        Args:
            to: Recipient email address
            subject: Email subject line
            text: Plain text body
            html: HTML body (optional)
            cc: CC recipient (optional)
        
        Returns:
            EmailResult with success status and details
        """
        if not self.enabled:
            logger.info(f"[EMAIL-LOG] To: {to}, Subject: {subject}")
            logger.info(f"[EMAIL-LOG] Body: {text[:500] if text else 'No text body'}...")
            return EmailResult(
                success=True,
                message_id="logged-only",
                accepted=[to],
                error="SendGrid not configured - email logged instead"
            )
        
        try:
            personalizations = [{
                "to": [{"email": to}],
                "subject": subject
            }]
            
            if cc:
                personalizations[0]["cc"] = [{"email": cc}]
            
            content = []
            if text:
                content.append({"type": "text/plain", "value": text})
            if html:
                content.append({"type": "text/html", "value": html})
            
            if not content:
                content.append({"type": "text/plain", "value": "(No content)"})
            
            payload = {
                "personalizations": personalizations,
                "from": {
                    "email": self.from_email,
                    "name": self.from_name
                },
                "content": content
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.sendgrid.com/v3/mail/send",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                
                if response.status_code in (200, 202):
                    message_id = response.headers.get("X-Message-Id", "sent")
                    logger.info(f"Email sent successfully to {to} via SendGrid (ID: {message_id})")
                    return EmailResult(
                        success=True,
                        message_id=message_id,
                        accepted=[to]
                    )
                else:
                    try:
                        error_data = response.json()
                        errors = error_data.get("errors", [])
                        error_msg = "; ".join([e.get("message", str(e)) for e in errors]) if errors else f"HTTP {response.status_code}"
                    except:
                        error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    
                    logger.error(f"SendGrid email failed: {error_msg}")
                    return EmailResult(
                        success=False,
                        error=error_msg,
                        rejected=[to]
                    )
        
        except Exception as e:
            logger.error(f"Email send exception: {e}")
            return EmailResult(
                success=False,
                error=str(e),
                rejected=[to]
            )
    
    async def send_callback_notification(
        self,
        name: str,
        phone: str,
        reason: str = "",
        preferred_time: str = "",
        session_id: str = ""
    ) -> EmailResult:
        """
        Send callback request notification to Brandon.
        
        Args:
            name: Requester's name
            phone: Their phone number
            reason: Why they want a callback
            preferred_time: When they prefer to be called
            session_id: Chat session ID for context
        
        Returns:
            EmailResult
        """
        subject = f"[BrandonBot] Callback Request: {name}"
        
        text_body = f"""
CALLBACK REQUEST
================

Name: {name}
Phone: {phone}
Preferred Time: {preferred_time or 'Not specified'}

Reason: {reason or 'Not provided'}

Session ID: {session_id or 'Unknown'}

---
This callback was requested through BrandonBot.
The user may have had questions that required personal follow-up.
"""
        
        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background: #c53030; color: white; padding: 20px; text-align: center;">
        <h1 style="margin: 0;">Callback Request</h1>
    </div>
    <div style="padding: 20px; background: #f7fafc;">
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: bold;">Name:</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{name}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: bold;">Phone:</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;"><a href="tel:{phone}">{phone}</a></td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: bold;">Preferred Time:</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{preferred_time or 'Not specified'}</td>
            </tr>
        </table>
        
        {f'<div style="margin-top: 20px; padding: 15px; background: white; border-radius: 8px;"><strong>Reason:</strong><p>{reason}</p></div>' if reason else ''}
        
        <p style="margin-top: 15px; font-size: 12px; color: #666;">Session ID: {session_id or 'Unknown'}</p>
    </div>
    <div style="padding: 15px; background: #e2e8f0; font-size: 12px; color: #666; text-align: center;">
        This callback was requested through BrandonBot.
    </div>
</body>
</html>
"""
        
        return await self.send_email(
            to=BRANDON_EMAIL,
            subject=subject,
            text=text_body,
            html=html_body
        )
    
    async def send_volunteer_notification(
        self,
        name: str,
        email: str,
        phone: str = "",
        zip_code: str = "",
        interests: Optional[List[str]] = None,
        availability: str = "flexible"
    ) -> EmailResult:
        """
        Send volunteer registration notification to Brandon.
        
        Args:
            name: Volunteer's name
            email: Volunteer's email
            phone: Volunteer's phone (optional)
            zip_code: Volunteer's zip code (optional)
            interests: List of volunteer interests
            availability: When they're available
        
        Returns:
            EmailResult
        """
        interests_list = interests or []
        interests_str = ", ".join(interests_list) if interests_list else "Not specified"
        
        subject = f"[BrandonBot] New Volunteer: {name}"
        
        text_body = f"""
NEW VOLUNTEER REGISTRATION
===========================

Name: {name}
Email: {email}
Phone: {phone or 'Not provided'}
Zip Code: {zip_code or 'Not provided'}

Areas of Interest: {interests_str}
Availability: {availability}

---
This notification was sent automatically by BrandonBot.
The volunteer signed up through the campaign website.
"""
        
        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background: #1a365d; color: white; padding: 20px; text-align: center;">
        <h1 style="margin: 0;">New Volunteer Registration</h1>
    </div>
    <div style="padding: 20px; background: #f7fafc;">
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: bold;">Name:</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{name}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: bold;">Email:</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;"><a href="mailto:{email}">{email}</a></td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: bold;">Phone:</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{phone or 'Not provided'}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: bold;">Zip Code:</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{zip_code or 'Not provided'}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: bold;">Interests:</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{interests_str}</td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: bold;">Availability:</td>
                <td style="padding: 10px;">{availability}</td>
            </tr>
        </table>
    </div>
    <div style="padding: 15px; background: #e2e8f0; font-size: 12px; color: #666; text-align: center;">
        This notification was sent automatically by BrandonBot.
    </div>
</body>
</html>
"""
        
        return await self.send_email(
            to=BRANDON_EMAIL,
            subject=subject,
            text=text_body,
            html=html_body
        )
    
    async def send_donation_interest_notification(
        self,
        name: str,
        email: str,
        phone: str = "",
        message: str = ""
    ) -> EmailResult:
        """
        Send donation interest notification to Brandon.
        
        IMPORTANT: This does NOT process donations.
        It only notifies Brandon that someone is interested in donating
        so he can follow up with secure, FEC-compliant donation methods.
        
        Args:
            name: Interested donor's name
            email: Their email
            phone: Their phone (optional)
            message: Any message they included (optional)
        
        Returns:
            EmailResult
        """
        subject = f"[BrandonBot] Donation Interest: {name}"
        
        text_body = f"""
DONATION INTEREST NOTIFICATION
==============================

Someone has expressed interest in donating to the campaign.

Name: {name}
Email: {email}
Phone: {phone or 'Not provided'}

Message: {message or 'No message provided'}

---
IMPORTANT: BrandonBot did NOT collect any financial information.
Please follow up with secure, FEC-compliant donation options.

This notification was sent automatically by BrandonBot.
"""
        
        html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background: #d69e2e; color: white; padding: 20px; text-align: center;">
        <h1 style="margin: 0;">Donation Interest</h1>
    </div>
    <div style="padding: 20px; background: #f7fafc;">
        <p style="font-size: 16px;">Someone has expressed interest in supporting the campaign:</p>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: bold;">Name:</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{name}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: bold;">Email:</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;"><a href="mailto:{email}">{email}</a></td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: bold;">Phone:</td>
                <td style="padding: 10px; border-bottom: 1px solid #e2e8f0;">{phone or 'Not provided'}</td>
            </tr>
        </table>
        
        {f'<div style="margin-top: 20px; padding: 15px; background: white; border-radius: 8px;"><strong>Their Message:</strong><p>{message}</p></div>' if message else ''}
    </div>
    <div style="padding: 15px; background: #fff3cd; border: 1px solid #ffc107; margin: 20px; border-radius: 8px;">
        <strong>Important:</strong> BrandonBot did NOT collect any financial information.
        Please follow up with secure, FEC-compliant donation options.
    </div>
    <div style="padding: 15px; background: #e2e8f0; font-size: 12px; color: #666; text-align: center;">
        This notification was sent automatically by BrandonBot.
    </div>
</body>
</html>
"""
        
        return await self.send_email(
            to=BRANDON_EMAIL,
            subject=subject,
            text=text_body,
            html=html_body
        )


email_service = EmailService()
