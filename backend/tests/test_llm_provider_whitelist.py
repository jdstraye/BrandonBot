from backend.llm_providers import LLMProviderManager


def test_scout_model_not_in_default_nvidia():
    mgr = LLMProviderManager()
    nvidia = mgr.providers.get("nvidia")
    if not nvidia:
        # If Nvidia provider isn't configured (no API keys), test is inconclusive
        return

    all_models = []
    for slot in nvidia.config.slots:
        all_models.extend(slot.models)

    assert "meta/llama-4-scout-17b-16e-instruct" not in [m.lower() for m in all_models]
