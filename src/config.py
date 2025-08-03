from pydantic_settings import BaseSettings

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    APP_TITLE: str = "🤖 FAQ Assistant API"
    LOGGING_FORMAT: str = "%(asctime)s | %(levename)s | %(name)s %(message)s"
    LOGGING_DATEFMT: str = "%Y-%m-d %H:%M:S"


class Settings(BaseSettings):
    QDRANT_API_KEY: str
    QDRANT_URL: str
    QDRANT_COLLECTION_NAME: str
    OLLAMA_BASE_URL: str
    AI_MODEL: str
    EMBEDDING_MODEL: str 

    class Config:
        env_file = ".env"
        extra = "allow"
        
settings = Settings() # type: ignore