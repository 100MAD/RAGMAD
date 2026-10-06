"""Answer a question from the passages stored for one chat."""

from __future__ import annotations

import uuid
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.schema import MetadataMode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from app.config import get_settings
from app.rag.components import get_index, get_llm

SYSTEM_PROMPT = (
    "You answer questions using the passages retrieved from the user's uploaded documents. "
    "If the passages do not contain the answer, say you cannot find it in the uploaded documents. "
    "Mention the filename when you use a passage."
)

CONTEXT_PROMPT = (
    "Passages from the user's documents:\n"
    "{context_str}\n"
    "Answer using only these passages. "
    "If they do not contain the answer, say you cannot find it in the uploaded documents."
)


def _history_messages(history: list[dict[str, Any]]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for item in history:
        role = item.get("role")
        content = item.get("content") or ""
        if role == "user":
            messages.append(ChatMessage(role=MessageRole.USER, content=content))
        elif role == "assistant":
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content))
    return messages


def _source_payload(source_node) -> dict[str, Any]:
    node = source_node.node
    metadata = node.metadata or {}
    page = metadata.get("page")
    score = source_node.score
    return {
        "document_id": metadata.get("document_id"),
        "filename": metadata.get("filename"),
        "page": int(page) if page is not None and str(page).isdigit() else page if isinstance(page, int) else None,
        "text": node.get_content(metadata_mode=MetadataMode.NONE),
        "score": float(score) if score is not None else None,
    }


def answer_question(
    chat_id: uuid.UUID,
    question: str,
    history: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    settings = get_settings()
    chat_history = _history_messages(history)
    retriever = get_index().as_retriever(
        similarity_top_k=settings.similarity_top_k,
        filters=MetadataFilters(
            filters=[MetadataFilter(key="chat_id", value=str(chat_id))]
        ),
    )
    engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=get_llm(),
        system_prompt=SYSTEM_PROMPT,
        context_prompt=CONTEXT_PROMPT,
        chat_history=chat_history,
    )
    response = engine.chat(question, chat_history=chat_history)
    sources = [_source_payload(source) for source in response.source_nodes]
    return str(response.response).strip(), sources
