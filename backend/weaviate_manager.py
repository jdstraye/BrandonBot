import os
import weaviate
import httpx
from weaviate.classes.config import Configure, Property, DataType
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class WeaviateManager:
    def __init__(self, persist_directory: str = "./weaviate_data"):
        self.persist_directory = persist_directory
        self.client = None
        logger.info("Loading sentence-transformers model (CPU-optimized)...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        logger.info("Embedding model loaded successfully")
        
    async def initialize(self):
        try:
            # Connection strategy (priority):
            # 1. WEAVIATE_URL => connect to remote Weaviate (cloud/hosted)
            # 2. connect_to_local (docker/local) using WEAVIATE_PORT env if present
            # 3. WEAVIATE_EMBEDDED=true => start embedded binary
            weaviate_url = os.environ.get("WEAVIATE_URL")
            weaviate_port = int(os.environ.get("WEAVIATE_PORT", "8079"))
            weaviate_grpc = int(os.environ.get("WEAVIATE_GRPC_PORT", "50050"))
            embedded = os.environ.get("WEAVIATE_EMBEDDED", "false").lower() == "true"

            if weaviate_url:
                logger.info(f"Connecting to remote Weaviate at {weaviate_url}...")
                # Try the standard client constructor (v4 vs v3 compatibility)
                try:
                    if hasattr(weaviate, 'WeaviateClient'):
                        try:
                            self.client = weaviate.WeaviateClient(url=weaviate_url)
                        except TypeError:
                            self.client = weaviate.WeaviateClient(base_url=weaviate_url)
                    else:
                        self.client = weaviate.Client(url=weaviate_url)
                    logger.info("Connected to remote Weaviate successfully")
                except Exception as e:
                    logger.warning(f"Failed to connect to remote Weaviate at {weaviate_url}: {e}")
                    self.client = None
                finally:
                    # set REST base URL for fallback
                    self.rest_url = weaviate_url.rstrip('/')

            if self.client is None:
                # Try a REST-only client to the local Weaviate instance (avoids gRPC init checks)
                try:
                    logger.info("Attempting REST connection to local Weaviate on port %s...", weaviate_port)
                    local_url = f"http://127.0.0.1:{weaviate_port}"
                    # store REST base URL for fallback HTTP requests
                    self.rest_url = local_url
                    # Try to instantiate a client if compatible
                    try:
                        if hasattr(weaviate, 'WeaviateClient'):
                            try:
                                self.client = weaviate.WeaviateClient(url=local_url)
                            except TypeError:
                                self.client = weaviate.WeaviateClient(base_url=local_url)
                        else:
                            self.client = weaviate.Client(url=local_url)
                    except Exception:
                        # client creation failed, but REST may still work
                        self.client = None
                    # check readiness via REST readiness endpoint
                    try:
                        if self.client:
                            if not self.client.is_ready():
                                raise RuntimeError("local Weaviate REST endpoint not ready")
                        else:
                            # probe REST readiness
                            with httpx.Client(timeout=5) as c:
                                r = c.get(f"{self.rest_url}/v1/.well-known/ready")
                                if r.status_code != 200:
                                    raise RuntimeError(f"local Weaviate REST endpoint not ready (status:{r.status_code})")
                        logger.info("Connected to existing local Weaviate instance successfully (REST)")
                    except Exception as ready_err:
                        logger.info(f"Local Weaviate REST client created but not ready: {ready_err}")
                        # discard client and fallthrough to embedded if enabled
                        self.client = None
                except Exception as connect_err:
                    logger.info(f"No existing Weaviate REST instance found on port {weaviate_port}: {connect_err}")
                    if embedded:
                        logger.info("Starting embedded Weaviate binary (WEAVIATE_EMBEDDED=true)")
                        self.client = weaviate.connect_to_embedded(
                            persistence_data_path=self.persist_directory,
                            binary_path=os.environ.get("WEAVIATE_BINARY_PATH", "./weaviate_binary")
                        )
                        logger.info("Weaviate initialized successfully in embedded mode")
                    else:
                        logger.warning("Weaviate not available (no WEAVIATE_URL, no local REST instance, embedded disabled)")
            
            await self._create_collections()
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {str(e)}")
            raise
    
    async def _create_collections(self):
        collections = [
            {
                "name": "BrandonPlatform",
                "description": "Brandon's own statements, speeches, and platform synthesis",
                "confidence_tier": 1
            },
            {
                "name": "PreviousQA",
                "description": "Previously answered questions and responses",
                "confidence_tier": 1
            },
            {
                "name": "PartyPlatform",
                "description": "RNC, Independent, and local Republican platforms",
                "confidence_tier": 2
            },
            {
                "name": "MarketGurus",
                "description": "Marketing and copywriting expert knowledge (Breakthrough Advertising, Boron Letters, etc.)",
                "confidence_tier": 3
            },
            {
                "name": "FECProhibited",
                "description": "FEC prohibited phrases, regulations, and compliance rules",
                "confidence_tier": 1
            }
        ]
        
        for collection_config in collections:
            try:
                if not self.client.collections.exists(collection_config["name"]):
                    self.client.collections.create(
                        name=collection_config["name"],
                        description=collection_config["description"],
                        properties=[
                            Property(name="content", data_type=DataType.TEXT),
                            Property(name="source", data_type=DataType.TEXT),
                            Property(name="date", data_type=DataType.TEXT),
                            Property(name="category", data_type=DataType.TEXT),
                            Property(name="confidence_tier", data_type=DataType.INT),
                            Property(name="metadata", data_type=DataType.TEXT)
                        ],
                        vectorizer_config=Configure.Vectorizer.none()
                    )
                    logger.info(f"Created collection: {collection_config['name']}")
            except Exception as e:
                logger.warning(f"Collection {collection_config['name']} might already exist: {str(e)}")
    
    def encode_text(self, text: str):
        return self.encoder.encode(text).tolist()
    
    async def add_document(self, collection_name: str, content: str, source: str, 
                          date: str = "", category: str = "", metadata=None):
        try:
            # If we have a full client, use its API
            if self.client:
                collection = self.client.collections.get(collection_name)

                confidence_tier_map = {
                    "BrandonPlatform": 1,
                    "PreviousQA": 1,
                    "PartyPlatform": 2,
                    "MarketGurus": 3
                }

                if metadata and isinstance(metadata, dict):
                    import json
                    metadata_str = json.dumps(metadata)
                else:
                    metadata_str = str(metadata) if metadata else ""

                vector = self.encode_text(content)

                collection.data.insert(
                    properties={
                        "content": content,
                        "source": source,
                        "date": date,
                        "category": category,
                        "confidence_tier": confidence_tier_map.get(collection_name, 3),
                        "metadata": metadata_str
                    },
                    vector=vector
                )
                logger.info(f"Added document to {collection_name}")
                return True

            # REST fallback: post to /v1/objects (Weaviate will auto-create class)
            if hasattr(self, 'rest_url') and self.rest_url:
                vector = self.encode_text(content)
                payload = {
                    "class": collection_name,
                    "properties": {
                        "content": content,
                        "source": source,
                        "date": date,
                        "category": category,
                        "confidence_tier": 3,
                        "metadata": metadata if metadata else ""
                    },
                    "vector": vector
                }
                try:
                    resp = httpx.post(f"{self.rest_url}/v1/objects", json=payload, timeout=20)
                    resp.raise_for_status()
                    logger.info(f"Added document to {collection_name} via REST")
                    return True
                except Exception as e:
                    logger.error(f"Failed to add document via REST: {e}")
                    return False
            else:
                raise RuntimeError("No Weaviate client or REST URL available to add document")
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            return False
    
    async def search(self, collection_name: str, query: str, limit: int = 10):
        try:
            # Prefer using the full client if available
            if self.client:
                collection = self.client.collections.get(collection_name)
                query_vector = self.encode_text(query)

                response = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=limit,
                    return_metadata=['distance']
                )

                results = []
                for obj in response.objects:
                    distance = obj.metadata.distance if hasattr(obj.metadata, 'distance') else 1.0
                    similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))

                    results.append({
                        "content": obj.properties.get("content", ""),
                        "source": obj.properties.get("source", ""),
                        "date": obj.properties.get("date", ""),
                        "category": obj.properties.get("category", ""),
                        "confidence": similarity,
                        "confidence_tier": obj.properties.get("confidence_tier", 3),
                        "metadata": obj.properties.get("metadata", "")
                    })

                return results

            # REST fallback: use GraphQL nearVector query against the REST endpoint
            if hasattr(self, 'rest_url') and self.rest_url:
                query_vector = self.encode_text(query)
                # Build a GraphQL query string for nearVector search
                gql_query = {
                    "query": f"query {{ Get {{ {collection_name}(nearVector: {{vector: [{', '.join(map(str, query_vector))}] }}, limit: {limit}) {{ content source date category confidence_tier _additional {{distance}} }} }} }}"
                }
                try:
                    resp = httpx.post(f"{self.rest_url}/v1/graphql", json=gql_query, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    hits = data.get('data', {}).get('Get', {}).get(collection_name, [])

                    results = []
                    for h in hits:
                        add = h.get('_additional', {})
                        distance = add.get('distance', 1.0)
                        similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
                        results.append({
                            "content": h.get('content', ''),
                            "source": h.get('source', ''),
                            "date": h.get('date', ''),
                            "category": h.get('category', ''),
                            "confidence": similarity,
                            "confidence_tier": h.get('confidence_tier', 3),
                            "metadata": h.get('metadata', '')
                        })

                    return results
                except Exception as e:
                    logger.error(f"REST GraphQL search failed for {collection_name}: {e}")
                    return []

            # No client or REST URL available
            raise RuntimeError("No Weaviate client or REST URL available for search")
        except Exception as e:
            logger.error(f"Search failed in {collection_name}: {str(e)}")
            return []
    
    async def get_collection_count(self, collection_name: str) -> int:
        try:
            if self.client:
                collection = self.client.collections.get(collection_name)
                result = collection.aggregate.over_all(total_count=True)
                return result.total_count if hasattr(result, 'total_count') else 0

            # REST fallback: query objects and count returned items (small dataset)
            if hasattr(self, 'rest_url') and self.rest_url:
                try:
                    resp = httpx.get(f"{self.rest_url}/v1/objects", params={"class": collection_name, "limit": 1000}, timeout=20)
                    resp.raise_for_status()
                    data = resp.json()
                    objs = data.get('objects') if isinstance(data, dict) else None
                    if objs is None:
                        # older API might return 'objects' under 'result'
                        objs = data.get('result', {}).get('objects', []) if isinstance(data, dict) else []
                    return len(objs or [])
                except Exception as e:
                    logger.error(f"REST get_collection_count failed: {e}")
                    return 0

            return 0
        except Exception as e:
            logger.error(f"Failed to get collection count for {collection_name}: {str(e)}")
            return 0
    
    async def health_check(self):
        try:
            return self.client.is_ready()
        except:
            return False
    
    async def close(self):
        if self.client:
            self.client.close()
