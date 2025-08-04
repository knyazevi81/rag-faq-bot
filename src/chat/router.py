from fastapi import APIRouter, Query
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from fastapi import HTTPException

from src.storage import qd_client
from src.embedding import embedding_model
from src.config import settings
from src.chain import get_qa_chain


router = APIRouter(prefix="/chat")

# LLM-модель через Ollama
llm = Ollama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.AI_MODEL,
    temperature=0.3
)

# torch.set_num_threads(4)

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        qa_chain = get_qa_chain()
        result = qa_chain({"query": request.question})
        
        # Формирование ответа с источниками
        answer = result["result"]
        sources = []
        for doc in result.get("source_documents", []):
            source_info = {
                "source": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "entities": doc.metadata.get("key_entities", [])
            }
            sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


# @router.post("/debug")
# async def debug(query: QuestionRequest = Query(...)):
#     qa_chain = get_qa_chain()
#     result = qa_chain({"query": request.question})
    
#     return {
#         "query": query,
#         "chunks": [doc.page_content for doc in docs]
#     }
