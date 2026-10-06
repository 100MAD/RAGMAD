import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.chats import _get_chat
from app.db.models import Document
from app.db.session import get_db
from app.rag.ingestion import delete_vectors, ingest_document
from app.schemas import DocumentOut
from app.storage import delete_key, open_stream, put_bytes

router = APIRouter(tags=["documents"])


def _get_document(document_id: uuid.UUID, db: Session) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.post("/chats/{chat_id}/documents", status_code=202, name="upload")
def upload_documents(
    chat_id: uuid.UUID,
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> list[DocumentOut]:
    _get_chat(chat_id, db)
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one file")

    created: list[Document] = []
    for upload in files:
        filename = Path(upload.filename or "document").name or "document"
        data = upload.file.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"{filename} is empty")
        document = Document(
            chat_id=chat_id,
            filename=filename,
            content_type=upload.content_type or "application/octet-stream",
            size=len(data),
            storage_key="",
            status="processing",
        )
        db.add(document)
        db.flush()
        document.storage_key = f"chats/{chat_id}/{document.id}/{filename}"
        put_bytes(document.storage_key, data, document.content_type)
        created.append(document)

    db.commit()
    for document in created:
        background.add_task(ingest_document, document.id)
        db.refresh(document)
    return [DocumentOut.model_validate(document) for document in created]


@router.get("/chats/{chat_id}/documents", name="list")
def list_documents(chat_id: uuid.UUID, db: Session = Depends(get_db)) -> list[DocumentOut]:
    _get_chat(chat_id, db)
    documents = db.scalars(
        select(Document).where(Document.chat_id == chat_id).order_by(Document.created_at.asc())
    ).all()
    return [DocumentOut.model_validate(document) for document in documents]


@router.get("/documents/{document_id}/download", name="download")
def download_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    document = _get_document(document_id, db)
    stream = open_stream(document.storage_key)
    filename = quote(document.filename)
    return StreamingResponse(
        stream,
        media_type=document.content_type or "application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


@router.delete("/documents/{document_id}", status_code=204, name="delete")
def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    document = _get_document(document_id, db)
    storage_key = document.storage_key
    document_key = str(document.id)
    db.delete(document)
    db.flush()
    delete_key(storage_key)
    delete_vectors(document_id=document_key)
