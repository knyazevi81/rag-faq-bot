from langchain.chains import StuffDocumentsChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from qdrant_client import QdrantClient
import torch

from src.config import settings


# Функция реранжирования результатов
def rerank_results(query: str, documents: list):
    try:
        model_name = "jinaai/jina-reranker-v1-base-ru"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        pairs = [(query, doc.page_content) for doc in documents]
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)

        with torch.no_grad():
            scores = model(**inputs).logits.squeeze()

        sorted_indices = torch.argsort(scores, descending=True)
        return [documents[i] for i in sorted_indices]
    except Exception as e:
        print(f"Reranking error: {e}")
        return documents


def get_qa_chain():
    # LLM и эмбеддинги
    llm = Ollama(
        base_url=settings.AI_MODEL,
        model=settings.AI_MODEL
    )
    embeddings = OllamaEmbeddings(
        base_url=settings.AI_MODEL,
        model=settings.EMBEDDING_MODEL
    )

    # Qdrant клиент
    qd_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None
    )

    # Векторное хранилище
    vector_store = Qdrant(
        client=qd_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embeddings=embeddings,
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 15, "score_threshold": 0.6}
    )

    # Промпт
    prompt_template = PromptTemplate(
        template="""
Ты ассистент, отвечающий на вопросы на основе предоставленного контекста.
Отвечай точно и подробно, используя только информацию из контекста.
Если в контексте нет полного ответа, скажи "В предоставленных материалах нет полной информации".

Контекст:
{context}

Вопрос: {question}
Подробный ответ:
""",
        input_variables=["context", "question"]
    )

    # Сборка ответа из документов
    combine_chain = StuffDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=prompt_template),
        document_variable_name="context"
    )

    # Полный Runnable-пайплайн
    chain = (
        RunnableMap({
            "docs": lambda x: retriever.get_relevant_documents(x["query"]),
            "query": lambda x: x["query"]
        })
        | RunnableLambda(lambda inputs: {
            "query": inputs["query"],
            "docs": rerank_results(inputs["query"], inputs["docs"])[:3]
        })
        | RunnableLambda(lambda inputs: {
            "result": combine_chain.run({
                "input_documents": inputs["docs"],
                "question": inputs["query"]
            }),
            "source_documents": inputs["docs"]
        })
    )

    return chain
