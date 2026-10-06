"""Shared RAG objects. Heavy libraries are imported on first use."""

from __future__ import annotations

import threading
from typing import Any

from llama_index.core import Settings as LlamaSettings
from llama_index.core import VectorStoreIndex

from app.config import get_settings

_lock = threading.Lock()
_embed_model: Any = None
_llm: Any = None
_vector_store: Any = None
_converter: Any = None
_node_parser: Any = None
_parser_signature: tuple[Any, ...] | None = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        with _lock:
            if _embed_model is None:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding

                settings = get_settings()
                _embed_model = HuggingFaceEmbedding(
                    model_name=settings.embed_model,
                    query_instruction=settings.embed_query_instruction,
                    trust_remote_code=False,
                )
                LlamaSettings.embed_model = _embed_model
    return _embed_model


def get_llm():
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                from llama_index.llms.openai_like import OpenAILike

                settings = get_settings()
                _llm = OpenAILike(
                    model=settings.llm_model,
                    api_base=settings.llm_base_url,
                    api_key=settings.llm_api_key,
                    is_chat_model=True,
                    is_function_calling_model=False,
                    context_window=settings.llm_context_window,
                    timeout=120,
                )
                LlamaSettings.llm = _llm
    return _llm


def get_eval_llm():
    settings = get_settings()
    if not settings.eval_llm_model and not settings.eval_llm_base_url:
        return get_llm()
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=settings.eval_llm_model or settings.llm_model,
        api_base=settings.eval_llm_base_url or settings.llm_base_url,
        api_key=settings.eval_llm_api_key or settings.llm_api_key,
        is_chat_model=True,
        is_function_calling_model=False,
        context_window=settings.llm_context_window,
        timeout=120,
    )


def get_qdrant_client():
    from qdrant_client import QdrantClient

    return QdrantClient(url=get_settings().qdrant_url)


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        with _lock:
            if _vector_store is None:
                from llama_index.vector_stores.qdrant import QdrantVectorStore

                settings = get_settings()
                store = QdrantVectorStore(
                    client=get_qdrant_client(),
                    collection_name=settings.qdrant_collection,
                    payload_indexes=[
                        {"field_name": "chat_id", "field_schema": "keyword"},
                        {"field_name": "document_id", "field_schema": "keyword"},
                    ],
                )
                # Custom metadata is stored on the payload root. Leave flat_metadata
                # off so list values such as headings are allowed.
                _vector_store = store
    return _vector_store


def get_index() -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(
        vector_store=get_vector_store(),
        embed_model=get_embed_model(),
    )


def get_converter():
    global _converter
    if _converter is None:
        with _lock:
            if _converter is None:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption

                pipeline_options = PdfPipelineOptions(do_ocr=False)
                _converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
    return _converter


def get_node_parser():
    global _node_parser, _parser_signature
    settings = get_settings()
    signature = (settings.embed_model, settings.chunk_max_tokens)
    if _node_parser is None or _parser_signature != signature:
        with _lock:
            if _node_parser is None or _parser_signature != signature:
                from docling.chunking import HybridChunker
                from docling_core.transforms.chunker.tokenizer.huggingface import (
                    HuggingFaceTokenizer,
                )
                from llama_index.node_parser.docling import DoclingNodeParser
                from transformers import AutoTokenizer

                tokenizer = HuggingFaceTokenizer(
                    tokenizer=AutoTokenizer.from_pretrained(settings.embed_model),
                    max_tokens=settings.chunk_max_tokens,
                )
                _node_parser = DoclingNodeParser(chunker=HybridChunker(tokenizer=tokenizer))
                _parser_signature = signature
    return _node_parser


def reset_components() -> None:
    """Drop cached models and stores. Used by the evaluation script and tests."""
    global _embed_model, _llm, _vector_store, _converter, _node_parser, _parser_signature
    with _lock:
        _embed_model = None
        _llm = None
        _vector_store = None
        _converter = None
        _node_parser = None
        _parser_signature = None
