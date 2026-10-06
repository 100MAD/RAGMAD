"""Load processed evaluation runs written by scripts/evaluate.py."""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "eval" / "results"

TEXT_COLUMNS = {"user_input", "response", "retrieved_contexts", "reference"}
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "response_relevancy": "Answer relevancy",
    "llm_context_precision_with_reference": "Context precision",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "llm_context_recall": "Context recall",
}


def results_dir() -> Path:
    return RESULTS_DIR


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _parse_stamp(run_id: str) -> datetime | None:
    match = re.match(r"^(\d{8}T\d{6}Z)", run_id)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _metric_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(mean(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _summarize_questions(questions: list[dict[str, Any]]) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {}
    for question in questions:
        for key, value in (question.get("scores") or {}).items():
            number = _finite(value)
            if number is None:
                continue
            metric_values.setdefault(key, []).append(number)

    metrics = {
        key: {
            "label": METRIC_LABELS.get(key, key.replace("_", " ").title()),
            **_metric_stats(values),
        }
        for key, values in metric_values.items()
        if values
    }
    overall = mean([item["mean"] for item in metrics.values()]) if metrics else None
    scored = sum(1 for question in questions if any(_finite(v) is not None for v in (question.get("scores") or {}).values()))
    return {
        "question_count": len(questions),
        "scored_count": scored,
        "overall_score": round(overall, 4) if overall is not None else None,
        "metrics": metrics,
    }


def build_run_payload(
    *,
    run_id: str,
    questions: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    stamp = created_at or _parse_stamp(run_id) or datetime.now(timezone.utc)
    summary = _summarize_questions(questions)
    return {
        "id": run_id,
        "created_at": stamp,
        "config": config or {},
        "summary": summary,
        "questions": questions,
    }


def run_from_csv(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    questions: list[dict[str, Any]] = []
    for row in rows:
        scores = {
            key: value
            for key, raw in row.items()
            if key not in TEXT_COLUMNS and (value := _finite(raw)) is not None
        }
        questions.append(
            {
                "question": row.get("user_input") or "",
                "response": row.get("response") or "",
                "reference": row.get("reference") or "",
                "retrieved_contexts": row.get("retrieved_contexts") or "",
                "scores": scores,
            }
        )

    config_name = path.stem.split("-", 1)[1] if "-" in path.stem else path.stem
    return build_run_payload(
        run_id=path.stem,
        questions=questions,
        config={"name": config_name},
        created_at=_parse_stamp(path.stem),
    )


def run_from_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "questions" not in payload:
        raise ValueError(f"{path} is missing questions")
    run_id = payload.get("id") or path.stem
    created_at = payload.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    else:
        created_at = _parse_stamp(run_id)
    return build_run_payload(
        run_id=run_id,
        questions=payload["questions"],
        config=payload.get("config") or {},
        created_at=created_at,
    )


def _candidate_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    json_files = [
        path
        for path in directory.glob("*.json")
        if path.name != "latest.json"
    ]
    if json_files:
        return sorted(json_files, key=lambda path: path.name, reverse=True)
    return sorted(directory.glob("*.csv"), key=lambda path: path.name, reverse=True)


def list_runs(directory: Path | None = None) -> list[dict[str, Any]]:
    directory = directory or results_dir()
    runs = []
    for path in _candidate_paths(directory):
        try:
            run = run_from_json(path) if path.suffix == ".json" else run_from_csv(path)
        except (OSError, ValueError, json.JSONDecodeError, csv.Error):
            continue
        runs.append(
            {
                "id": run["id"],
                "created_at": run["created_at"],
                "config": run["config"],
                "summary": run["summary"],
            }
        )
    return runs


def get_run(run_id: str | None = None, directory: Path | None = None) -> dict[str, Any] | None:
    directory = directory or results_dir()
    if run_id is None:
        latest = directory / "latest.json"
        if latest.is_file():
            try:
                return run_from_json(latest)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        candidates = _candidate_paths(directory)
        if not candidates:
            return None
        path = candidates[0]
        return run_from_json(path) if path.suffix == ".json" else run_from_csv(path)

    json_path = directory / f"{run_id}.json"
    if json_path.is_file():
        return run_from_json(json_path)
    csv_path = directory / f"{run_id}.csv"
    if csv_path.is_file():
        return run_from_csv(csv_path)
    return None
