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


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def run_async(coro):
    """Helper to run async functions in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


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
    from fec_compliance_checker import FECProhibitedPhrasesStore
    
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
        
        # Create FECProhibited collection if needed
        if not wm.client.collections.exists("FECProhibited"):
            wm.client.collections.create(
                name="FECProhibited",
                description="FEC prohibited phrases and rules for compliance checking",
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="date", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT),
                    Property(name="confidence_tier", data_type=DataType.INT),
                    Property(name="metadata", data_type=DataType.TEXT),
                    Property(name="violation_type", data_type=DataType.TEXT),
                    Property(name="severity", data_type=DataType.INT),
                ],
                vectorizer_config=Configure.Vectorizer.none()
            )
            logger.info("Created FECProhibited collection")
            
            phrase_store = FECProhibitedPhrasesStore()
            for violation_type, phrases in phrase_store.PROHIBITED_PHRASES.items():
                for phrase, severity in phrases:
                    await wm.add_document(
                        collection_name="FECProhibited",
                        content=phrase,
                        source="FEC Compliance Rules",
                        category=violation_type,
                        metadata={"violation_type": violation_type, "severity": severity}
                    )
            
            count = await wm.get_collection_count("FECProhibited")
            logger.info(f"Populated FECProhibited collection with {count} phrases")
        
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
