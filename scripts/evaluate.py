"""Score the RAG pipeline with ragas.

Install the extra dependencies first:

    uv sync --group eval

Then, from the RAGMAD directory, with Postgres, Qdrant and MinIO running:

    uv run --group eval python scripts/evaluate.py
    uv run --group eval python scripts/evaluate.py --top-k 3 --chunk-tokens 256

Changing the chunk size or the embedding model writes vectors to a separate
Qdrant collection so those runs do not mix with the application index.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()[:40] or "model"


def _apply_overrides(args: argparse.Namespace) -> str:
    index_changed = args.chunk_tokens is not None or args.embed_model is not None
    if args.top_k is not None:
        os.environ["SIMILARITY_TOP_K"] = str(args.top_k)
    if args.chunk_tokens is not None:
        os.environ["CHUNK_MAX_TOKENS"] = str(args.chunk_tokens)
    if args.embed_model:
        os.environ["EMBED_MODEL"] = args.embed_model
    if args.collection:
        os.environ["QDRANT_COLLECTION"] = args.collection
    elif index_changed:
        tokens = args.chunk_tokens if args.chunk_tokens is not None else "default"
        model = _slug(args.embed_model or "default")
        os.environ["QDRANT_COLLECTION"] = f"ragmad_eval_t{tokens}_{model}"
    parts = [
        f"k{args.top_k or 'default'}",
        f"t{args.chunk_tokens or 'default'}",
        _slug(args.embed_model or "default"),
    ]
    return "-".join(parts)


def _load_dataset(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("question", "reference", "documents"):
            if key not in row:
                raise SystemExit(f"{path}:{line_number} is missing {key}")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path} has no questions")
    return rows


def _prepare_chat(session, title: str, files: list[Path]):
    from app.db.models import Chat, Document
    from app.rag.ingestion import ingest_document
    from app.storage import put_bytes

    chat = Chat(title=title, hidden=True)
    session.add(chat)
    session.flush()
    for path in files:
        data = path.read_bytes()
        document = Document(
            chat_id=chat.id,
            filename=path.name,
            content_type="text/markdown",
            size=len(data),
            storage_key="",
            status="processing",
        )
        session.add(document)
        session.flush()
        document.storage_key = f"chats/{chat.id}/{document.id}/{path.name}"
        put_bytes(document.storage_key, data, document.content_type)
        document_id = document.id
        session.commit()
        ingest_document(document_id)
        session.expire_all()
        stored = session.get(Document, document_id)
        if stored is None or stored.status != "ready":
            error = stored.error if stored is not None else "document disappeared"
            raise SystemExit(f"Failed to ingest {path.name}: {error}")
    return chat.id


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAGMAD with ragas")
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval" / "dataset.jsonl")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection. Required implicitly when chunking or the embedding model changes.",
    )
    args = parser.parse_args()
    config_name = _apply_overrides(args)

    from alembic import command
    from alembic.config import Config

    from app.config import get_settings
    from app.db.session import SessionLocal
    from app.rag.chat import answer_question
    from app.rag.components import get_embed_model, get_eval_llm, reset_components
    from app.storage import ensure_bucket

    get_settings.cache_clear()
    reset_components()
    settings = get_settings()

    eval_llm = get_eval_llm()
    try:
        probe = eval_llm.complete("Reply with only the word ok.")
        if not str(getattr(probe, "text", "")).strip():
            raise SystemExit("EVAL LLM returned an empty response")
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            "EVAL LLM is not reachable. Check EVAL_LLM_BASE_URL / "
            f"EVAL_LLM_MODEL ({settings.eval_llm_model or settings.llm_model}): {exc}"
        ) from exc
    print(
        f"eval llm: {settings.eval_llm_model or settings.llm_model} "
        f"@ {settings.eval_llm_base_url or settings.llm_base_url}"
    )

    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LlamaIndexEmbeddingsWrapper
    from ragas.llms import LlamaIndexLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    ensure_bucket()

    docs_dir = args.dataset.parent / "docs"
    rows = _load_dataset(args.dataset)
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row["documents"])].append(row)

    samples = []
    with SessionLocal() as session:
        for documents, questions in grouped.items():
            files = []
            for name in documents:
                path = docs_dir / name
                if not path.is_file():
                    raise SystemExit(f"Missing evaluation document: {path}")
                files.append(path)
            chat_id = _prepare_chat(session, "Evaluation", files)
            for row in questions:
                answer, sources = answer_question(chat_id, row["question"], [])
                samples.append(
                    {
                        "user_input": row["question"],
                        "response": answer,
                        "retrieved_contexts": [source["text"] for source in sources],
                        "reference": row["reference"],
                    }
                )

    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=LlamaIndexLLMWrapper(eval_llm),
        embeddings=LlamaIndexEmbeddingsWrapper(get_embed_model()),
    )
    frame = result.to_pandas()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    created_at = datetime.now(timezone.utc)
    run_id = f"{stamp}-{config_name}"
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.csv"
    frame.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

    from app.evaluation import TEXT_COLUMNS, build_run_payload

    questions = []
    for _, row in frame.iterrows():
        scores = {}
        for column in frame.columns:
            if column in TEXT_COLUMNS:
                continue
            value = row[column]
            if value is None or (isinstance(value, float) and value != value):
                continue
            try:
                scores[column] = float(value)
            except (TypeError, ValueError):
                continue
        contexts = row.get("retrieved_contexts")
        if hasattr(contexts, "tolist"):
            contexts = contexts.tolist()
        questions.append(
            {
                "question": str(row.get("user_input") or ""),
                "response": str(row.get("response") or ""),
                "reference": str(row.get("reference") or ""),
                "retrieved_contexts": contexts,
                "scores": scores,
            }
        )

    payload = build_run_payload(
        run_id=run_id,
        questions=questions,
        config={
            "name": config_name,
            "top_k": args.top_k or settings.similarity_top_k,
            "chunk_tokens": args.chunk_tokens or settings.chunk_max_tokens,
            "embed_model": args.embed_model or settings.embed_model,
            "collection": settings.qdrant_collection,
            "answer_model": settings.llm_model,
            "eval_model": settings.eval_llm_model or settings.llm_model,
        },
        created_at=created_at,
    )
    serializable = {
        **payload,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
    }
    json_path = out_dir / f"{run_id}.json"
    latest_path = out_dir / "latest.json"
    encoded = json.dumps(serializable, indent=2, ensure_ascii=False) + "\n"
    json_path.write_text(encoded, encoding="utf-8")
    latest_path.write_text(encoded, encoding="utf-8")

    print(f"collection: {settings.qdrant_collection}")
    print(f"questions: {len(samples)}")
    print(f"wrote {out_path}")
    print(f"wrote {json_path}")
    summary = payload["summary"]
    if summary["overall_score"] is not None:
        print(f"overall: {summary['overall_score']:.3f}")
    for key, stats in summary["metrics"].items():
        print(f"{key}: {stats['mean']:.3f}")


if __name__ == "__main__":
    main()
