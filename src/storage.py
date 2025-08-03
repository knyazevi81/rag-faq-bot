from qdrant_client import qdrant_client
from langchain_community.vectorstores import Qdrant

from src.config import settings
from src.embedding import embedding_model

qd_client = qdrant_client.QdrantClient(
    # api_key=settings.QDRANT_API_KEY,
    url=settings.QDRANT_URL
)

vectorstore = Qdrant(
    client=qd_client,
    collection_name=settings.QDRANT_COLLECTION_NAME,
    embeddings=embedding_model
)