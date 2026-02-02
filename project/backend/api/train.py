from fastapi import APIRouter
from core.utils import load_data
from core.config import DATA_PATH, TARGET_COL
from preprocessing.encoder import encode_and_scale
from preprocessing.split import split_data
from models.baseline import train_baseline
from fairness.metrics import compute_fairness

router = APIRouter(prefix="/train", tags=["Training"])

@router.get("/baseline")
def train_baseline_api():
    df = load_data(DATA_PATH)

    y = df[TARGET_COL]
    sensitive = df["gender"]
    X = df.drop(TARGET_COL, axis=1)

    X_encoded, _, _ = encode_and_scale(X)

    X_train, X_test, y_train, y_test, s_train, s_test = split_data(
        X_encoded, y, sensitive
    )

    model, acc, preds = train_baseline(
        X_train, y_train, X_test, y_test
    )

    fairness = compute_fairness(y_test, preds, s_test)

    return {
        "accuracy": acc,
        "fairness": fairness
    }
