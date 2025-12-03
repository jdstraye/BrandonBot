"""
Email Service for BrandonBot
Uses Replit's OpenInt mail service for sending notifications.

Email Routing:
- Testing (TESTING_MODE=true): jdstrayer@netzero.net
- Production: campaign@brandonsowers.com
"""

import os
import logging
import subprocess
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

TESTING_MODE = os.environ.get("TESTING_MODE", "false").lower() == "true"
TEST_EMAIL = "jdstrayer@netzero.net"
PRODUCTION_EMAIL = "campaign@brandonsowers.com"
BRANDON_EMAIL = TEST_EMAIL if TESTING_MODE else PRODUCTION_EMAIL

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
    Email service using Replit's OpenInt mail API.
    Falls back to logging if email service is unavailable.
    """
    
    def __init__(self):
        self.hostname = os.environ.get("REPLIT_CONNECTORS_HOSTNAME", "connectors.replit.com")
        self.enabled = bool(os.environ.get("REPLIT_CONNECTORS_HOSTNAME"))
        
        if not self.enabled:
            logger.warning("REPLIT_CONNECTORS_HOSTNAME not set - email service will log instead of sending")
    
    def _get_auth_token(self) -> Optional[str]:
        """Get Replit identity token for authentication."""
        try:
            result = subprocess.run(
                ["replit", "identity", "create", "--audience", f"https://{self.hostname}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to get identity token: {result.stderr}")
                return None
            
            token = result.stdout.strip()
            if not token:
                logger.error("Empty identity token returned")
                return None
            
            return f"Bearer {token}"
        except FileNotFoundError:
            logger.warning("Replit CLI not found - email service unavailable")
            return None
        except subprocess.TimeoutExpired:
            logger.error("Timeout getting identity token")
            return None
        except Exception as e:
            logger.error(f"Error getting identity token: {e}")
            return None
    
    async def send_email(
        self,
        to: str,
        subject: str,
        text: Optional[str] = None,
        html: Optional[str] = None,
        cc: Optional[str] = None
    ) -> EmailResult:
        """
        Send an email using Replit's mail service.
        
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
                error=None
            )
        
        auth_token = self._get_auth_token()
        if not auth_token:
            logger.warning(f"[EMAIL-FALLBACK] Could not authenticate, logging email instead")
            logger.info(f"[EMAIL-LOG] To: {to}, Subject: {subject}")
            logger.info(f"[EMAIL-LOG] Body: {text[:500] if text else 'No text body'}...")
            return EmailResult(
                success=True,
                message_id="fallback-logged",
                accepted=[to],
                error="Authentication failed, email logged instead"
            )
        
        try:
            payload = {
                "to": to,
                "subject": subject,
            }
            
            if text:
                payload["text"] = text
            if html:
                payload["html"] = html
            if cc:
                payload["cc"] = cc
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"https://{self.hostname}/api/v2/mailer/send",
                    headers={
                        "Content-Type": "application/json",
                        "Replit-Authentication": auth_token
                    },
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Email sent successfully to {to}: {data.get('messageId', 'unknown')}")
                    return EmailResult(
                        success=True,
                        message_id=data.get("messageId"),
                        accepted=data.get("accepted", [to]),
                        rejected=data.get("rejected", [])
                    )
                else:
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get("message", f"HTTP {response.status_code}")
                    logger.error(f"Email send failed: {error_msg}")
                    return EmailResult(
                        success=False,
                        error=error_msg,
                        rejected=[to]
                    )
        
        except Exception as e:
            logger.error(f"Email send exception: {e}")
            logger.info(f"[EMAIL-FALLBACK] Logging email due to error")
            logger.info(f"[EMAIL-LOG] To: {to}, Subject: {subject}")
            return EmailResult(
                success=False,
                error=str(e),
                rejected=[to]
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
