from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from src.config import settings
from src.storage import qd_client


# Функция для реранжирования результатов
def rerank_results(query: str, documents: list):
    try:
        # Используем Jina reranker для русского языка
        model_name = "jinaai/jina-reranker-v1-base-ru"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Подготавливаем пары запрос-документ
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Получаем оценки релевантности
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            scores = model(**inputs).logits
            
        # Сортируем документы по убыванию релевантности
        sorted_indices = torch.argsort(scores, descending=True)
        return [documents[i] for i in sorted_indices]
    except Exception as e:
        print(f"Reranking error: {e}")
        return documents

def get_qa_chain():
    # Инициализация моделей
    embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
    llm = Ollama(model=settings.AI_MODEL)
    
    # Подключаемся к Qdrant
    vector_store = Qdrant(
        client=qd_client,
        collection_name="insurance_base",
        embeddings=embeddings,
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )
    
    # Создаем ретривер
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 10,
            "score_threshold": 0.7,
        }
    )
    
    # Кастомная функция поиска с реранжированием
    def custom_retriever(query: str):
        # Получаем документы
        docs = retriever.get_relevant_documents(query)
        
        # Реранжируем результаты
        ranked_docs = rerank_results(query, docs)
        
        # Возвращаем топ-3 наиболее релевантных
        return ranked_docs[:3]
    
    # Улучшенный промпт для ответов
    prompt_template = """
    Ты ассистент, отвечающий на вопросы на основе предоставленного контекста.
    Отвечай точно и подробно, используя только информацию из контекста.
    Если в контексте нет полного ответа, скажи "В предоставленных материалах нет полной информации".
    
    Контекст:
    {context}
    
    Вопрос: {question}
    Подробный ответ:
    """
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=custom_retriever,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        },
        return_source_documents=True,
    )