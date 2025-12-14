"""Utility helpers for OV regeneration flows."""
from typing import Any
from backend.output_validator import OVSafeguard


def build_regen_prompt(validation_result: Any, original_query: str) -> str:
    """Build a regeneration prompt for OV-driven retries.

    Adds structured regeneration instructions and, when the failure
    includes an intent checking hard-fail, appends a short reminder
    of the original user query to keep future generations focused.
    """
    base = validation_result.get_feedback_for_retry() or ""
    # Import here to avoid circular imports at module load
    from backend.structured_response import get_ov_regeneration_instructions
    base += get_ov_regeneration_instructions()

    failed_intent = any(
        (s == OVSafeguard.INTENT_CHECKING and r.score > 3)
        for s, r in validation_result.results.items()
    )
    if failed_intent:
        base = base + f"\n\nREMINDER: The user's original question was: '{original_query}'. Please answer that question directly (or ask ONE concise clarifying question if necessary)."
        # Be explicit about callback offers: do not offer a callback unless the
        # user explicitly requested one or the question is escalated.
        base = base + "\n\nNOTE: Unless the user explicitly asked for a callback or provided contact info, do NOT offer to call them back. Focus on answering the question directly."
    return base
