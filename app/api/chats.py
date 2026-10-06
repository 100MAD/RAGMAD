import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chat
from app.db.session import get_db
from app.rag.ingestion import delete_vectors
from app.schemas import ChatCreate, ChatOut, ChatUpdate
from app.storage import delete_prefix

router = APIRouter(tags=["chats"])


def _get_chat(chat_id: uuid.UUID, db: Session) -> Chat:
    chat = db.get(Chat, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.post("/chats", status_code=201, name="create")
def create_chat(body: ChatCreate | None = None, db: Session = Depends(get_db)) -> ChatOut:
    title = (body.title.strip() if body and body.title else "") or "New chat"
    chat = Chat(title=title)
    db.add(chat)
    db.flush()
    db.refresh(chat)
    return ChatOut.model_validate(chat)


@router.get("/chats", name="list")
def list_chats(db: Session = Depends(get_db)) -> list[ChatOut]:
    chats = db.scalars(
        select(Chat).where(Chat.hidden.is_(False)).order_by(Chat.updated_at.desc())
    ).all()
    return [ChatOut.model_validate(chat) for chat in chats]


@router.get("/chats/{chat_id}", name="get")
def get_chat(chat_id: uuid.UUID, db: Session = Depends(get_db)) -> ChatOut:
    return ChatOut.model_validate(_get_chat(chat_id, db))


@router.patch("/chats/{chat_id}", name="update")
def update_chat(chat_id: uuid.UUID, body: ChatUpdate, db: Session = Depends(get_db)) -> ChatOut:
    chat = _get_chat(chat_id, db)
    chat.title = body.title.strip()
    chat.updated_at = datetime.now(timezone.utc)
    db.flush()
    db.refresh(chat)
    return ChatOut.model_validate(chat)


@router.delete("/chats/{chat_id}", status_code=204, name="delete")
def delete_chat(chat_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    chat = _get_chat(chat_id, db)
    delete_prefix(f"chats/{chat.id}/")
    delete_vectors(chat_id=str(chat.id))
    db.delete(chat)
