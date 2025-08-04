from fastapi import APIRouter, Query, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant

from datetime import datetime
import tempfile
import shutil
from pathlib import Path
import os

from src.storage import qd_client
from src.embedding import embedding_model
from src.config import settings
from src.utils import preprocess_text, extract_entities

router = APIRouter(
    prefix="/storages",
    tags=["Эндпоинт для работы с векторным хранилищем"]
)


@router.get('/collections')
async def get_collections():
    # надо чекнуть что возвращает и сделать пайдентик модель
    # Сделать асинхронность
    result = qd_client.get_collections().collections
    return result


@router.get("/search")
async def search(query: str = Query(...)):
    vectorstore = Qdrant(
        client=qd_client, 
        collection_name=settings.QDRANT_COLLECTION_NAME, 
        embeddings=embedding_model
    )
    results_with_scores = vectorstore.similarity_search_with_score(query, k=5)

    return [{
        "text": doc.page_content,
        "metadata": doc.metadata,
        "score": score
    } for doc, score in results_with_scores]


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    media_type = file.filename.split(".")[-1].lower()

    # Сохраняем файл во временное место
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{media_type}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Загружаем документ
        if media_type == "pdf":
            loader = PyPDFLoader(str(tmp_path))
        else:
            loader = TextLoader(str(tmp_path))

        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", "\. ", "! ", "? ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )


        documents = []
        for page in pages:
            clean_text = preprocess_text(page)
            page.page_content = clean_text

            entities = extract_entities(clean_text)

            page.metadata["key_entities"] = entities
            page.metadata["source_file"] = file.filename

            chunks = text_splitter.split_documents([page])
            documents.extend(chunks)
        
        vectorstore = Qdrant(
            client=qd_client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embeddings=embedding_model
        )
            
        vectorstore.add_documents(chunks)

        # chunks = text_splitter.split_documents(documents)
# 
        # for chunk in chunks:
            # chunk.metadata.update({
                # "filename": file.filename,
                # "uploaded_at": datetime.utcnow().isoformat(),
                # "source": f"/docs/{file.filename}"
            # })
# 
# 
        return {"message": "Document uploaded and processed successfully."}

    finally:
        # Удаляем временный файл в любом случае
        if tmp_path.exists():
            os.remove(tmp_path)



# @router.post("/upload")
# async def add_new_data(
#     media_type: MediaType = Query(...),
#     file: UploadFile = File(...)
# ):
#     if media_type == MediaType.pdf:
#         try:
#             reader = PdfFileReader(file.file)
#             text = '\n'.join([page.extract_text() or "" for page in reader.pages])
#             text = text.strip()
#             if not text:
#                 raise TextNotFoundException()
            
#             embedding = embedding_model.encode(text).tolist()
#             return embedding
#             # point = PointStruct(
#                 # id=str(uuid.uuid4()),
#                 # vector=embedding,
#                 # payload={"filename": file.filename}
#             # )
#             # await qd_client.upsert(
#                 # collection_name=settings.QDRANT_COLLECTION_NAME,
#                 # points=[point]
#             # )
#             return {"status":"ok", "message": f"The {file.filename} file has been successfully processed "}
#         except Exception as e:
#             pass
    
#     raise WrongFormatDocException()
    
