import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.evaluation import get_run, list_runs
from app.main import app


def _write_sample(directory: Path) -> None:
    payload = {
        "id": "20261006T120000Z-kdefault-tdefault-default",
        "created_at": "2026-10-06T12:00:00Z",
        "config": {
            "name": "kdefault-tdefault-default",
            "top_k": 5,
            "chunk_tokens": 512,
            "embed_model": "BAAI/bge-small-en-v1.5",
            "answer_model": "gpt-5-nano",
            "eval_model": "deepseek-v4-pro",
        },
        "questions": [
            {
                "question": "What time does the library close on Friday?",
                "response": "It closes at 18:00.",
                "reference": "The main library closes at 18:00 on Friday.",
                "retrieved_contexts": ["On Friday it closes at 18:00."],
                "scores": {
                    "faithfulness": 1.0,
                    "answer_relevancy": 0.9,
                    "llm_context_precision_with_reference": 0.8,
                    "context_recall": 1.0,
                },
            },
            {
                "question": "Is the library open on Sunday?",
                "response": "No, it is closed on Sunday.",
                "reference": "The library is closed on Sunday.",
                "retrieved_contexts": ["The library is closed on Sunday."],
                "scores": {
                    "faithfulness": 1.0,
                    "answer_relevancy": 1.0,
                    "llm_context_precision_with_reference": 1.0,
                    "context_recall": 1.0,
                },
            },
        ],
    }
    encoded = json.dumps(payload, indent=2) + "\n"
    (directory / f"{payload['id']}.json").write_text(encoded, encoding="utf-8")
    (directory / "latest.json").write_text(encoded, encoding="utf-8")


def test_list_and_get_runs(tmp_path: Path, monkeypatch):
    _write_sample(tmp_path)
    monkeypatch.setattr("app.evaluation.RESULTS_DIR", tmp_path)

    runs = list_runs()
    assert len(runs) == 1
    assert runs[0]["summary"]["overall_score"] == 0.9625
    assert get_run()["id"] == runs[0]["id"]

    client = TestClient(app)
    latest = client.get("/evaluation")
    assert latest.status_code == 200
    body = latest.json()
    assert body["id"].startswith("20261006T120000Z")
    assert len(body["questions"]) == 2
    assert body["summary"]["metrics"]["faithfulness"]["mean"] == 1.0

    listed = client.get("/evaluation/runs")
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == body["id"]

    single = client.get(f"/evaluation/runs/{body['id']}")
    assert single.status_code == 200
    assert single.json()["questions"][0]["question"].startswith("What time")
