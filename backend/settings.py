"""Global settings accessor for BrandonBot.

This module provides a lightweight Settings wrapper around `BrandonBotConfig`
so callers can import a single `settings` instance instead of repeatedly
calling `load_config()` or reading environment variables directly.
"""
from dataclasses import dataclass
from typing import Optional

from backend.config_loader import load_config, BrandonBotConfig


@dataclass
class Settings:
    cfg: BrandonBotConfig

    @property
    def roles(self):
        return self.cfg.roles

    @property
    def scoring(self):
        return self.cfg.scoring

    @property
    def providers(self):
        return self.cfg.providers


def _load_settings() -> Settings:
    try:
        cfg = load_config()
    except Exception:
        # Fall back to defaults inside load_config
        cfg = load_config()
    return Settings(cfg=cfg)


# Module-level singleton for convenience
settings = _load_settings()

__all__ = ["Settings", "settings"]
