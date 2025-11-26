"""
Security Module for BrandonBot

Provides input sanitization, injection prevention, and rate limiting.
"""

import re
import time
import logging
from typing import Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class SanitizationResult:
    cleaned_text: str
    was_modified: bool
    issues_found: list


class InputSanitizer:
    """Sanitizes user input to prevent injection attacks and malicious content"""
    
    INJECTION_PATTERNS = [
        (r"<script[^>]*>.*?</script>", "script_injection"),
        (r"javascript:", "javascript_protocol"),
        (r"on\w+\s*=", "event_handler"),
        (r"<iframe[^>]*>", "iframe_injection"),
        (r"<object[^>]*>", "object_injection"),
        (r"<embed[^>]*>", "embed_injection"),
        (r"data:\s*text/html", "data_uri_html"),
        (r"vbscript:", "vbscript_protocol"),
        (r"expression\s*\(", "css_expression"),
    ]
    
    PROMPT_INJECTION_PATTERNS = [
        (r"ignore\s+(previous|all|above)\s+(instructions?|prompts?)", "ignore_instruction"),
        (r"forget\s+(everything|all|your)\s+(you|instructions?|learned)", "forget_instruction"),
        (r"you\s+are\s+now\s+(?:a|an|my)", "role_override"),
        (r"pretend\s+(?:to\s+be|you\s+are)", "pretend_command"),
        (r"disregard\s+(?:all|previous|your)", "disregard_instruction"),
        (r"new\s+persona|new\s+identity", "persona_change"),
        (r"system\s*:?\s*\[", "system_prompt_injection"),
        (r"<\|im_start\|>|<\|im_end\|>", "chat_template_injection"),
        (r"\[INST\]|\[/INST\]", "llama_template_injection"),
        (r"<<SYS>>|<</SYS>>", "llama_system_injection"),
    ]
    
    SQL_PATTERNS = [
        (r";\s*(DROP|DELETE|UPDATE|INSERT|TRUNCATE)\s+", "sql_injection"),
        (r"'\s*OR\s+'?1'?\s*=\s*'?1", "sql_always_true"),
        (r"UNION\s+(ALL\s+)?SELECT", "sql_union"),
        (r"--\s*$", "sql_comment"),
    ]
    
    MAX_INPUT_LENGTH = 5000
    MAX_WORD_LENGTH = 100
    
    def sanitize(self, text: str) -> SanitizationResult:
        """
        Sanitize user input text.
        
        Returns:
            SanitizationResult with cleaned text and any issues found
        """
        if not text:
            return SanitizationResult("", False, [])
        
        issues = []
        cleaned = text
        
        if len(cleaned) > self.MAX_INPUT_LENGTH:
            cleaned = cleaned[:self.MAX_INPUT_LENGTH]
            issues.append(("length_truncated", f"Input truncated from {len(text)} to {self.MAX_INPUT_LENGTH}"))
        
        words = cleaned.split()
        truncated_words = []
        for word in words:
            if len(word) > self.MAX_WORD_LENGTH:
                truncated_words.append(word[:self.MAX_WORD_LENGTH])
                issues.append(("word_truncated", word[:20] + "..."))
            else:
                truncated_words.append(word)
        cleaned = " ".join(truncated_words)
        
        for pattern, issue_type in self.INJECTION_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL):
                issues.append((issue_type, "Removed"))
                cleaned = re.sub(pattern, "[removed]", cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        for pattern, issue_type in self.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                issues.append((issue_type, "Detected"))
                logger.warning(f"Prompt injection attempt detected: {issue_type}")
        
        for pattern, issue_type in self.SQL_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                issues.append((issue_type, "Removed"))
                cleaned = re.sub(pattern, "[removed]", cleaned, flags=re.IGNORECASE)
        
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)
        
        was_modified = cleaned != text
        
        if issues:
            logger.info(f"Input sanitized: {len(issues)} issues found")
        
        return SanitizationResult(cleaned, was_modified, issues)
    
    def is_safe(self, text: str) -> Tuple[bool, list]:
        """Check if text is safe without modifying it"""
        result = self.sanitize(text)
        return not result.was_modified, result.issues_found


class RateLimiter:
    """
    Rate limiter for API endpoints and tool calls.
    Uses a sliding window approach.
    """
    
    def __init__(self):
        self.requests: Dict[str, list] = defaultdict(list)
        self.limits = {
            "web_search": (5, 60),
            "query": (30, 60),
            "callback": (3, 300),
            "donation": (5, 60),
            "volunteer": (5, 60),
        }
    
    def check_rate_limit(self, key: str, limit_type: str = "query") -> Tuple[bool, Optional[int]]:
        """
        Check if a request should be rate limited.
        
        Args:
            key: Unique identifier (session_id, IP, etc.)
            limit_type: Type of limit to apply
            
        Returns:
            Tuple of (is_allowed, seconds_until_reset)
        """
        max_requests, window_seconds = self.limits.get(limit_type, (30, 60))
        
        now = time.time()
        window_start = now - window_seconds
        
        full_key = f"{limit_type}:{key}"
        self.requests[full_key] = [t for t in self.requests[full_key] if t > window_start]
        
        if len(self.requests[full_key]) >= max_requests:
            oldest = min(self.requests[full_key])
            seconds_until_reset = int(oldest + window_seconds - now) + 1
            return False, seconds_until_reset
        
        self.requests[full_key].append(now)
        return True, None
    
    def get_remaining(self, key: str, limit_type: str = "query") -> int:
        """Get remaining requests in current window"""
        max_requests, window_seconds = self.limits.get(limit_type, (30, 60))
        window_start = time.time() - window_seconds
        
        full_key = f"{limit_type}:{key}"
        current_count = len([t for t in self.requests[full_key] if t > window_start])
        return max(0, max_requests - current_count)
    
    def cleanup(self, max_age_seconds: int = 600):
        """Clean up old entries to prevent memory bloat"""
        now = time.time()
        cutoff = now - max_age_seconds
        
        for key in list(self.requests.keys()):
            self.requests[key] = [t for t in self.requests[key] if t > cutoff]
            if not self.requests[key]:
                del self.requests[key]


input_sanitizer = InputSanitizer()
rate_limiter = RateLimiter()
