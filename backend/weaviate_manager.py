import weaviate
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
            logger.info("Starting Weaviate in embedded mode (no Docker required)...")
            self.client = weaviate.connect_to_embedded(
                persistence_data_path=self.persist_directory,
                binary_path="./weaviate_binary"
            )
            await self._create_collections()
            logger.info("Weaviate initialized successfully in embedded mode")
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
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            return False
    
    async def search(self, collection_name: str, query: str, limit: int = 10):
        try:
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
                # Weaviate cosine distance ranges from 0 (identical) to 2 (opposite)
                # Convert to similarity: similarity = 1 - (distance / 2) to get [0, 1] range
                # This gives us: distance 0 → similarity 1.0, distance 1 → similarity 0.5, distance 2 → similarity 0.0
                similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
                
                results.append({
                    "content": obj.properties.get("content", ""),
                    "source": obj.properties.get("source", ""),
                    "date": obj.properties.get("date", ""),
                    "category": obj.properties.get("category", ""),
                    "confidence": similarity,  # This is raw similarity before trust multiplier
                    "confidence_tier": obj.properties.get("confidence_tier", 3),
                    "metadata": obj.properties.get("metadata", "")
                })
            
            return results
        except Exception as e:
            logger.error(f"Search failed in {collection_name}: {str(e)}")
            return []
    
    async def get_collection_count(self, collection_name: str) -> int:
        try:
            collection = self.client.collections.get(collection_name)
            result = collection.aggregate.over_all(total_count=True)
            return result.total_count if hasattr(result, 'total_count') else 0
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
