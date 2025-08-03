from fastapi import APIRouter, Query
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from src.storage import qd_client
from src.embedding import embedding_model
from src.config import settings

router = APIRouter(prefix="/chat")

# LLM-модель через Ollama
llm = Ollama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.AI_MODEL,
    temperature=0.3
)

# Кастомный промпт
prompt_template = PromptTemplate.from_template("""
Ты — литературный ассистент. Ответь на вопрос, используя только контекст из романа "Мастер и Маргарита".

Контекст:
{context}

Вопрос: {question}

Если ответа в контексте нет — скажи честно, что информации недостаточно.
""")

@router.get("/ask")
async def ask(query: str = Query(...)):
    vectorstore = Qdrant(
        client=qd_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embeddings=embedding_model
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

    response = qa_chain.run(query)
    return {"answer": response}


@router.get("/debug")
async def debug(query: str = Query(...)):
    vectorstore = Qdrant(
        client=qd_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embeddings=embedding_model
    )

    docs = vectorstore.similarity_search(query, k=15)
    return {
        "query": query,
        "chunks": [doc.page_content for doc in docs]
    }
