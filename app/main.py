import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from sqlalchemy import text

from app.api import chats, documents, evaluation, messages
from app.config import get_settings
from app.db.session import SessionLocal
from app.rag.ingestion import ingest_document, processing_document_ids
from app.storage import ensure_bucket

logger = logging.getLogger(__name__)


def _operation_id(route: APIRoute) -> str:
    tag = route.tags[0] if route.tags else "default"
    return f"{tag}-{route.name}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ensure_bucket()
    except Exception:
        logger.exception("Object storage is not available")
    try:
        for document_id in processing_document_ids():
            threading.Thread(
                target=ingest_document,
                args=(document_id,),
                daemon=True,
            ).start()
    except Exception:
        logger.exception("Could not requeue documents that were still processing")
    yield


app = FastAPI(
    title="RAGMAD",
    version="0.1.0",
    lifespan=lifespan,
    generate_unique_id_function=_operation_id,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chats.router)
app.include_router(documents.router)
app.include_router(messages.router)
app.include_router(evaluation.router)


@app.get("/healthz", tags=["health"], name="check")
def healthz() -> dict[str, str]:
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    return {"status": "ok"}
