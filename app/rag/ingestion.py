"""Parse, chunk, embed, and index one uploaded document."""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.db.models import Document
from app.db.session import SessionLocal
from app.rag.components import get_converter, get_index, get_node_parser, get_vector_store
from app.storage import get_bytes

logger = logging.getLogger(__name__)

_docling_semaphore = threading.Semaphore(1)

_EXCLUDED_METADATA = (
    "doc_items",
    "origin",
    "schema_name",
    "version",
    "chat_id",
    "document_id",
)


def page_from_doc_items(doc_items: Any) -> int | None:
    if isinstance(doc_items, str):
        try:
            doc_items = json.loads(doc_items)
        except json.JSONDecodeError:
            return None
    if not isinstance(doc_items, list):
        return None
    pages: list[int] = []
    for item in doc_items:
        if not isinstance(item, dict):
            continue
        provenances = item.get("prov") or []
        if isinstance(provenances, dict):
            provenances = [provenances]
        for provenance in provenances:
            if isinstance(provenance, dict) and provenance.get("page_no") is not None:
                pages.append(int(provenance["page_no"]))
    return min(pages) if pages else None


def _primitive(value: Any) -> Any:
    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, list) and all(
        isinstance(item, (str, int, float)) or item is None for item in value
    ):
        return value
    return json.dumps(value, default=str)


def _exclude(node, key: str) -> None:
    if key not in node.excluded_embed_metadata_keys:
        node.excluded_embed_metadata_keys.append(key)
    if key not in node.excluded_llm_metadata_keys:
        node.excluded_llm_metadata_keys.append(key)


def _annotate_nodes(nodes, *, chat_id: uuid.UUID, document_id: uuid.UUID, filename: str):
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

    prepared = []
    source = RelatedNodeInfo(node_id=str(document_id))
    for node in nodes:
        text = (node.get_content() or "").strip()
        if not text:
            continue
        metadata = dict(node.metadata or {})
        page = page_from_doc_items(metadata.get("doc_items"))
        metadata["chat_id"] = str(chat_id)
        metadata["document_id"] = str(document_id)
        metadata["filename"] = filename
        if page is not None:
            metadata["page"] = page
        node.metadata = {key: _primitive(value) for key, value in metadata.items()}
        for key in _EXCLUDED_METADATA:
            _exclude(node, key)
        node.relationships[NodeRelationship.SOURCE] = source
        prepared.append(node)
    return prepared


def _set_status(
    document_id: uuid.UUID,
    *,
    status: str,
    error: str | None = None,
    chunk_count: int | None = None,
) -> None:
    with SessionLocal() as session:
        document = session.get(Document, document_id)
        if document is None:
            return
        document.status = status
        document.error = error
        if chunk_count is not None:
            document.chunk_count = chunk_count
        session.commit()


def _document_exists(document_id: uuid.UUID) -> bool:
    with SessionLocal() as session:
        return session.get(Document, document_id) is not None


def delete_vectors(*, chat_id: str | None = None, document_id: str | None = None) -> None:
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

    store = get_vector_store()
    if not store.client.collection_exists(store.collection_name):
        return
    filters = []
    if chat_id is not None:
        filters.append(MetadataFilter(key="chat_id", value=chat_id))
    if document_id is not None:
        filters.append(MetadataFilter(key="document_id", value=document_id))
    if not filters:
        return
    store.delete_nodes(filters=MetadataFilters(filters=filters))


def ingest_document(document_id: uuid.UUID) -> None:
    with SessionLocal() as session:
        document = session.get(Document, document_id)
        if document is None or document.status == "ready":
            return
        chat_id = document.chat_id
        filename = document.filename
        storage_key = document.storage_key

    try:
        data = get_bytes(storage_key)
        with _docling_semaphore:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / filename
                path.write_bytes(data)
                from llama_index.readers.docling import DoclingReader

                reader = DoclingReader(
                    export_type=DoclingReader.ExportType.JSON,
                    doc_converter=get_converter(),
                )
                documents = reader.load_data(path)
                nodes = _annotate_nodes(
                    get_node_parser().get_nodes_from_documents(documents),
                    chat_id=chat_id,
                    document_id=document_id,
                    filename=filename,
                )
        if not nodes:
            delete_vectors(document_id=str(document_id))
            _set_status(document_id, status="failed", error="no extractable text", chunk_count=0)
            return
        if not _document_exists(document_id):
            return
        delete_vectors(document_id=str(document_id))
        get_index().insert_nodes(nodes)
        if not _document_exists(document_id):
            delete_vectors(document_id=str(document_id))
            return
        _set_status(document_id, status="ready", error=None, chunk_count=len(nodes))
    except Exception as exc:
        logger.exception("Ingestion failed for document %s", document_id)
        try:
            delete_vectors(document_id=str(document_id))
        except Exception:
            logger.exception("Could not remove vectors for document %s", document_id)
        _set_status(document_id, status="failed", error=str(exc)[:2000], chunk_count=0)


def processing_document_ids() -> list[uuid.UUID]:
    with SessionLocal() as session:
        return list(
            session.scalars(select(Document.id).where(Document.status == "processing")).all()
        )
