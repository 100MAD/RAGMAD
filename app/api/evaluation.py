from fastapi import APIRouter, HTTPException

from app.evaluation import get_run, list_runs
from app.schemas import EvaluationRunOut, EvaluationRunSummaryOut

router = APIRouter(tags=["evaluation"])


@router.get("/evaluation", name="latest")
def latest_evaluation() -> EvaluationRunOut:
    run = get_run()
    if run is None:
        raise HTTPException(
            status_code=404,
            detail="No evaluation results yet. Run scripts/evaluate.py first.",
        )
    return EvaluationRunOut.model_validate(run)


@router.get("/evaluation/runs", name="list")
def list_evaluations() -> list[EvaluationRunSummaryOut]:
    return [EvaluationRunSummaryOut.model_validate(run) for run in list_runs()]


@router.get("/evaluation/runs/{run_id}", name="get")
def get_evaluation(run_id: str) -> EvaluationRunOut:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return EvaluationRunOut.model_validate(run)
