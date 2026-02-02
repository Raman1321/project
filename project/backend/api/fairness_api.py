from fastapi import APIRouter
from fairness.metrics import compute_fairness

router = APIRouter(prefix="/fairness", tags=["Fairness"])

@router.post("/metrics")
def fairness_api(y_true: list, y_pred: list, sensitive: list):
    return compute_fairness(y_true, y_pred, sensitive)
