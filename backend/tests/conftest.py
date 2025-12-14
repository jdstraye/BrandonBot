"""
Pytest fixtures for BrandonBot test suite.

Provides fixtures for:
- Weaviate manager with FECProhibited collection
- Output validator with FEC RAG configured
- Prequalifier with SLM manager
- SLM manager for frustration/vagueness classification
- Email service and checker for delivery verification
"""

import pytest
import warnings


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    # NOTE: We used to suppress CryptographyDeprecationWarning here, but
    # that hid the underlying problem. Instead of filtering the warning,
    # we prefer to fix code so deprecated features are not used. See
    # TODO.md for dependency-upgrade items related to `pypdf`/`cryptography`.
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def run_async(coro):
    """Helper to run async functions in sync context."""
    # Prefer `asyncio.run()` for running top-level coroutines (Python 3.7+)
    return asyncio.run(coro)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def weaviate_manager():
    """
    Session-scoped Weaviate manager with FECProhibited collection.
    
    Connects to existing Weaviate instance or creates embedded one for tests.
    - FECProhibited collection populated with prohibited phrases
    - Standard collections (BrandonPlatform, PreviousQA, etc.)
    """
    import weaviate
    from weaviate_manager import WeaviateManager
    from weaviate.classes.config import Configure, Property, DataType
    # Use real ingestion pipeline to populate collections including FEC
    from ingest_all import connect_or_start_weaviate, ingest_fec_prohibited
    
    wm = None
    connected_to_existing = False
    
    async def setup():
        nonlocal wm, connected_to_existing
        
        # Try to connect to existing Weaviate instance first
        try:
            client = weaviate.connect_to_local(port=8079, grpc_port=50050)
            if client.is_ready():
                logger.info("Connected to existing Weaviate instance on port 8079")
                wm = WeaviateManager.__new__(WeaviateManager)
                wm.client = client
                wm.initialized = True
                wm.model = None
                connected_to_existing = True
            else:
                client.close()
                raise ConnectionError("Existing Weaviate not ready")
        except Exception as e:
            logger.info(f"No existing Weaviate, creating embedded: {e}")
            wm = WeaviateManager(persist_directory="./weaviate_test_data")
            await wm.initialize()
        
        # Ensure FECProhibited collection exists and is populated using real ingestion helper
        try:
            exists = False
            if getattr(wm, 'client', None):
                try:
                    exists = wm.client.collections.exists("FECProhibited")
                except Exception:
                    exists = False
            elif hasattr(wm, 'rest_url') and wm.rest_url:
                import httpx
                r = httpx.get(f"{wm.rest_url}/v1/schema", timeout=5)
                if r.status_code == 200:
                    classes = r.json().get('classes', [])
                    exists = any(c.get('class') == 'FECProhibited' for c in classes)

            if not exists:
                # Use add_document which will create the class via REST fallback if needed
                try:
                    await wm.add_document("FECProhibited", content="seed doc", source="seed")
                    logger.info("Created/seeded FECProhibited collection via REST fallback")
                except Exception as e:
                    logger.warning(f"Could not create/seed FECProhibited collection: {e}")
        except Exception as e:
            logger.warning(f"Error checking/creating FECProhibited collection: {e}")

        # Use built-in ingestion to populate baseline mandatory FEC data (idempotent)
        await ingest_fec_prohibited(wm)

        # Ensure the module-level output_validator (if imported by tests) is wired
        try:
            from output_validator import output_validator as module_output_validator
            module_output_validator.set_fec_rag(wm)
            module_output_validator.set_weaviate_manager(wm)
            logger.info("Wired module-level output_validator to test WeaviateManager")
        except Exception:
            # Not all test processes import the module-level output_validator; ignore failures
            logger.debug("module-level output_validator not present or could not be wired")

        # Note: Do not attempt to modify the production module-level `fec_checker`
        # at import time. Tests should construct and wire their own instances
        # to preserve fail-closed production semantics.

        return wm
    
    wm = run_async(setup())
    yield wm
    
    async def cleanup():
        nonlocal connected_to_existing
        if not connected_to_existing and wm:
            await wm.close()
    
    run_async(cleanup())


@pytest.fixture(scope="session")
def slm_manager():
    """
    Session-scoped SLM manager for frustration/vagueness classification.
    
    Lazy-loads models on first use to avoid slow test startup.
    """
    from slm_manager import SLMManager
    return SLMManager()


@pytest.fixture
def output_validator_with_rag(weaviate_manager):
    """
    Output validator with FEC RAG configured.
    
    Uses require_slm=True (production mode) with FEC RAG from Weaviate.
    This is the fixture to use for all OV tests.
    """
    from output_validator import OutputValidatorSLM
    
    ov = OutputValidatorSLM(require_slm=True)
    ov.set_fec_rag(weaviate_manager)
    return ov


@pytest.fixture
def prequalifier_with_slm(slm_manager):
    """
    Prequalifier with SLM manager configured.
    
    Uses require_slm=True (production mode) for frustration/vagueness classification.
    This is the fixture to use for all PQ tests.
    """
    from prequalifier import Prequalifier
    
    pq = Prequalifier(require_slm=True, slm_provider=slm_manager)
    return pq


@pytest.fixture
def output_validator_slm_only():
    """
    Output validator with require_slm=True but no FEC RAG.
    
    Use this for tests that don't need FEC checking (ethics, intent, pii, etc.)
    FEC checks will raise SLMNotAvailableError if called.
    """
    from output_validator import OutputValidatorSLM
    return OutputValidatorSLM(require_slm=True)


@pytest.fixture
def prequalifier_slm_only(slm_manager):
    """
    Prequalifier with require_slm=True and SLM manager.
    
    Uses the session-scoped SLM manager for frustration/vagueness classification.
    """
    from prequalifier import Prequalifier
    return Prequalifier(require_slm=True, slm_provider=slm_manager)
