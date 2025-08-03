from fastapi import FastAPI
from qdrant_client import models

from contextlib import asynccontextmanager
from typing import AsyncIterable

from src.config import AppConfig, settings
from src.storage import qd_client
# from src.logging import loggers
from src.storages.router import router as storage_router
from src.chat.router import router as chat_router

# 🔍 Что можно улучшить потом:
# Использовать ConversationalRetrievalChain для чата с историей
# Добавить source_documents=True чтобы возвращать, откуда ответ
# Добавить логирование или кеширование
# Хочешь пример с ConversationalRetrievalChain и историей чата — могу дополнить.


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterable[None]:
    # logger.info("Проверка наличия коллекция в qdrant")
    collections = qd_client.get_collections().collections
    exisisting_collections = [collection.name for collection in collections]
    if settings.QDRANT_COLLECTION_NAME not in exisisting_collections:
        # logger.info(f"Коллекция {settings.QDRANT_COLLECTION_NAME} не создана")
        qd_client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE
            )
        )
        qd_client.create_payload_index(
            collection_name=f"{settings.QDRANT_COLLECTION_NAME}",
            field_name="name_of_the_field_to_index",
            field_schema="keyword",
        )
        # logger.info(f"Коллекция {settings.QDRANT_COLLECTION_NAME} успешно создана")
    else:
        ...
        # logger.info(f"Коллекция {settings.QDRANT_COLLECTION_NAME} уже создана")
    yield
    qd_client.close()


def create_app() -> FastAPI:
    _app = FastAPI(
        title=AppConfig.APP_TITLE,
        lifespan=lifespan
    )
    _app.include_router(storage_router)
    _app.include_router(chat_router)
    return _app

app = create_app()