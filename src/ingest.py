import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

from src.config import settings

# Функция для предварительной обработки текста
def preprocess_text(text: str):
    # Сшивка переносов слов
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Удаление лишних пробелов
    text = re.sub(r"\s+", " ", text)
    # Удаление технических артефактов
    text = re.sub(r"Adobe PDF Library|Layout by [A-Z\.]+", "", text)
    return text

# Функция для извлечения именованных сущностей
def extract_entities(text: str):
    try:
        ner = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner")
        entities = ner(text)
        return list(set([e['word'] for e in entities if e['score'] > 0.8]))
    except:
        return []

def ingest_documents():
    # Инициализация эмбеддингов
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Адаптивный чанкинг с увеличенным размером и перекрытием
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", "\. ", "! ", "? ", " "],
        length_function=len,
        is_separator_regex=False,
    )
    
    documents = []
    for filename in os.listdir("docs"):
        if filename.endswith(".pdf"):
            filepath = os.path.join("docs", filename)
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            
            for page in pages:
                # Предварительная обработка текста
                clean_text = preprocess_text(page.page_content)
                page.page_content = clean_text
                
                # Извлечение сущностей для метаданных
                entities = extract_entities(clean_text)
                
                # Обновление метаданных
                page.metadata["key_entities"] = entities
                page.metadata["source_file"] = filename
                
                # Разделение на чанки
                chunks = text_splitter.split_documents([page])
                documents.extend(chunks)
    
    # Создание векторного хранилища
    Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        location=":memory:",
        collection_name="insurance_base",
        distance_func="Cosine",
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )

if __name__ == "__main__":
    ingest_documents()