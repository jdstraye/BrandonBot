"""Backend package marker for BrandonBot.

This file ensures `backend` is a proper Python package so absolute
imports like `backend.llm_providers` resolve during test collection
and normal execution.
"""

__all__ = []
