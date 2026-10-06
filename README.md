# RAGMAD

Backend for a document chat. Uploaded files are parsed with Docling, chunked with Docling's hybrid chunker, embedded with `BAAI/bge-small-en-v1.5`, stored in Qdrant, and sent to an OpenAI-compatible chat model as retrieved context.

The React client lives in the `RAGMAD-UI` repository.

## Local services

```bash
docker compose up -d
cp .env.example .env
```

Postgres is published on host port 5433 so it does not collide with a local Postgres on 5432. Qdrant uses 6333 and MinIO uses 9000.

Edit `.env` and set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`. Any OpenAI-compatible API works, including OpenAI, OpenRouter, vLLM, and Ollama.

## Docker images

From this directory, with `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` set in `.env`:

```bash
docker compose up -d --build
```

That builds `ragmad-api` and `ragmad-ui` and starts them with Postgres, Qdrant, and MinIO. The chat screen is at http://localhost:8080. The API is at http://localhost:8000.

## API

```bash
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000
```

On an Intel Mac, PyTorch is pinned to 2.2.2 because newer wheels are not published for that platform. NumPy stays below 2, and SciPy and scikit-learn stay on the releases that work with that combination.

Documents are processed in the API process. A scanned PDF is marked `failed` with `no extractable text`, because OCR is turned off.

## Evaluation

```bash
uv sync --group eval
uv run --group eval python scripts/evaluate.py
uv run --group eval python scripts/evaluate.py --top-k 8 --chunk-tokens 256
```

Questions and reference answers are in `eval/dataset.jsonl`. Sample documents are in `eval/docs/`. Each run writes `eval/results/<timestamp>-<config>.csv` plus a JSON summary used by `GET /evaluation`.

`--chunk-tokens` and `--embed-model` use a separate Qdrant collection. Optional `EVAL_LLM_BASE_URL`, `EVAL_LLM_API_KEY`, and `EVAL_LLM_MODEL` select the model that judges the answers. Use an OpenAI-compatible base URL (no `/chat/completions` suffix).
