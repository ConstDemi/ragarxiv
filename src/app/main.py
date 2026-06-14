# main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

import config
from rag_pipeline import ScienceRAG

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Тяжёлые объекты грузим один раз при старте; настройки берём из config.
    ml_models["rag"] = ScienceRAG(
        qdrant_host=config.QDRANT_HOST,
        qdrant_port=config.QDRANT_PORT,
        collection_name=config.COLLECTION_NAME,
        embed_model=config.EMBED_MODEL,
        llm_model=config.LLM_MODEL,
        system_prompt=config.SYSTEM_PROMPT,
        max_new_tokens=config.MAX_NEW_TOKENS,
        max_input_tokens=config.MAX_INPUT_TOKENS,
        retrieve_k=config.RETRIEVE_K,
        context_k=config.CONTEXT_K,
        max_papers=config.MAX_PAPERS,
    )
    yield
    # Освобождаем ресурсы (VRAM) при остановке.
    ml_models["rag"].cleanup()
    ml_models.clear()


app = FastAPI(title="arXiv RAG API", lifespan=lifespan)


# --- Контракт ---
class AskRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class Chunk(BaseModel):
    text: str
    score: float
    chunk_index: int = 0
    header_1: str = ""
    header_2: str = ""
    header_3: str = ""


class PaperSource(BaseModel):
    doc_id: str
    title: str
    authors: str = ""
    published: str = ""
    url: str
    score: float              # лучший score среди чанков статьи
    chunks: list[Chunk]


class AskResponse(BaseModel):
    answer: str
    sources: list[PaperSource]


class SearchResponse(BaseModel):
    sources: list[PaperSource]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    rag = ml_models["rag"]
    return rag.answer(query=request.text)


@app.post("/search", response_model=SearchResponse)
def search(request: AskRequest):
    rag = ml_models["rag"]
    return {"sources": rag.search(query=request.text)}
