import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.chats import _get_chat
from app.db.models import Chat, Document, Message
from app.db.session import get_db
from app.rag.chat import answer_question
from app.schemas import MessageCreate, MessageOut, SourceOut

router = APIRouter(tags=["messages"])


def _to_out(message: Message) -> MessageOut:
    sources = None
    if message.sources is not None:
        sources = [SourceOut.model_validate(source) for source in message.sources]
    return MessageOut(
        id=message.id,
        chat_id=message.chat_id,
        role=message.role,
        content=message.content,
        sources=sources,
        created_at=message.created_at,
    )


@router.get("/chats/{chat_id}/messages", name="list")
def list_messages(chat_id: uuid.UUID, db: Session = Depends(get_db)) -> list[MessageOut]:
    _get_chat(chat_id, db)
    messages = db.scalars(
        select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at.asc())
    ).all()
    return [_to_out(message) for message in messages]


@router.post("/chats/{chat_id}/messages", name="create")
def create_message(
    chat_id: uuid.UUID,
    body: MessageCreate,
    db: Session = Depends(get_db),
) -> MessageOut:
    chat = _get_chat(chat_id, db)
    ready = db.scalar(
        select(func.count())
        .select_from(Document)
        .where(Document.chat_id == chat_id, Document.status == "ready")
    )
    if not ready:
        raise HTTPException(
            status_code=409,
            detail="Upload a document and wait until it is ready before asking questions.",
        )

    history = [
        {"role": message.role, "content": message.content}
        for message in db.scalars(
            select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at.asc())
        ).all()
    ]
    first_message = len(history) == 0
    try:
        answer, sources = answer_question(chat_id, body.content, history)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"The language model request failed: {exc}",
        ) from exc

    user_message = Message(chat_id=chat_id, role="user", content=body.content, sources=None)
    assistant_message = Message(
        chat_id=chat_id,
        role="assistant",
        content=answer,
        sources=sources,
    )
    db.add(user_message)
    db.add(assistant_message)
    if first_message:
        chat.title = body.content.strip()[:80] or chat.title
    chat.updated_at = datetime.now(timezone.utc)
    db.flush()
    db.refresh(assistant_message)
    return _to_out(assistant_message)
