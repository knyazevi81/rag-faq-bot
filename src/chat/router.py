from fastapi import APIRouter, Query
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant

from src.storage import qd_client
from src.embedding import embedding_model
from src.config import settings


router = APIRouter(
    prefix="/chat"
)

llm = Ollama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.AI_MODEL,
    temperature=0
)

@router.get("/ask")
async def ask(query: str = Query(...)):
    vectorstore = Qdrant(
        client=qd_client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embeddings=embedding_model
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff"
    )

    response = qa_chain.run(query)
    return {"answer": response}
