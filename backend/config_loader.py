"""Configuration loader for BrandonBot

Parses `backend/config/BrandonBot.ini` for non-secret runtime controls.
Secrets (API keys) must remain in environment variables.
"""
import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

CONFIG_PATH = Path(__file__).parent / "config" / "BrandonBot.ini"


@dataclass
class RolesConfig:
    judge: List[Tuple[str, str]]  # list of (provider, model)
    user_allow_round_robin: bool


@dataclass
class ScoringConfig:
    require_llama_for_judge: bool
    scoring_whitelist: List[Tuple[str, str]]


@dataclass
class ProvidersConfig:
    ollama_host: str
    default_judge_model: str
    searxng_url: str = ""


@dataclass
class BrandonBotConfig:
    roles: RolesConfig
    scoring: ScoringConfig
    providers: ProvidersConfig
    # validation runtime controls
    validation: Optional[dict] = None
    # judge runtime controls
    judge: Optional[dict] = None


def _parse_provider_model_list(s: Optional[str]) -> List[Tuple[str, str]]:
    if not s:
        return []
    pairs = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            provider, model = item.split(":", 1)
            pairs.append((provider.strip().lower(), model.strip()))
        else:
            # provider only (no model)
            pairs.append((item.strip().lower(), ""))
    return pairs


def load_config() -> BrandonBotConfig:
    cfg = configparser.ConfigParser()
    if not CONFIG_PATH.exists():
        # defaults
        roles = RolesConfig(judge=[("ollama", "llama-4-maverick")], user_allow_round_robin=True)
        scoring = ScoringConfig(require_llama_for_judge=True, scoring_whitelist=[])
        return BrandonBotConfig(roles=roles, scoring=scoring)

    cfg.read(CONFIG_PATH)

    judge_raw = cfg.get("roles", "Judge", fallback="ollama:llama-4-maverick")
    user_rr = cfg.getboolean("roles", "User_allow_round_robin", fallback=True)

    require_llama = cfg.getboolean("scoring", "require_llama_for_judge", fallback=True)
    whitelist_raw = cfg.get("scoring", "scoring_whitelist", fallback="")

    ollama_host = cfg.get("providers", "Ollama_host", fallback="http://localhost:11434")
    default_judge_model = cfg.get("providers", "Default_judge_model", fallback="llama3.2:3b")

    roles = RolesConfig(judge=_parse_provider_model_list(judge_raw), user_allow_round_robin=user_rr)
    scoring = ScoringConfig(require_llama_for_judge=require_llama, scoring_whitelist=_parse_provider_model_list(whitelist_raw))
    searxng_url = cfg.get("providers", "Searxng_url", fallback="").strip()
    providers = ProvidersConfig(ollama_host=ollama_host, default_judge_model=default_judge_model, searxng_url=searxng_url)
    # Validation-specific options
    perf_monitor = cfg.getboolean("validation", "perf_monitor", fallback=False)
    perf_sample_interval_ms = cfg.getint("validation", "perf_sample_interval_ms", fallback=0)
    validation = {"perf_monitor": perf_monitor, "perf_sample_interval_ms": perf_sample_interval_ms}

    # Judge-specific options
    judge_timeout_seconds = cfg.getint("judge", "timeout_seconds", fallback=120)
    judge_retries = cfg.getint("judge", "retries", fallback=1)
    judge_retry_backoff_ms = cfg.getint("judge", "retry_backoff_ms", fallback=500)
    judge = {
        "timeout_seconds": judge_timeout_seconds,
        "retries": judge_retries,
        "retry_backoff_ms": judge_retry_backoff_ms,
    }
    return BrandonBotConfig(roles=roles, scoring=scoring, providers=providers, validation=validation, judge=judge)


__all__ = ["load_config", "BrandonBotConfig", "RolesConfig", "ScoringConfig", "ProvidersConfig"]
