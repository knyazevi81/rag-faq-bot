from langchain_community.embeddings import OllamaEmbeddings

from dataclasses import dataclass

from src.config import settings

embedding_model = OllamaEmbeddings(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.EMBEDDING_MODEL
)

    
@dataclass
class EmbedingModelConfig:

    def validate(self) -> None:
        ...


class EmbedingModelFactory:

    @staticmethod
    async def create_embedding_model(config: EmbedingModelConfig):
        ...