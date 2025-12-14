def test_get_provider_stats_includes_providers():
    from llm_providers import llm_manager
    stats = llm_manager.get_provider_stats()
    assert 'providers' in stats
    # Core providers we expect to be present in the conservative default whitelist
    for expected in ('gemini', 'mistral'):
        assert expected in stats['providers'], f"Expected provider {expected} in stats"
    # HuggingFace may or may not be present depending on operator whitelist; if present, ensure it's well-formed
    if 'huggingface' in stats['providers']:
        assert isinstance(stats['providers']['huggingface']['slots'], list)
    # Ensure slots is a list and models are listed
    gemini_slots = stats['providers']['gemini']['slots']
    assert isinstance(gemini_slots, list)
    assert 'models' in gemini_slots[0]
