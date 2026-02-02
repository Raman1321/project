from fastapi import FastAPI, UploadFile, File
import pandas as pd
from ml.model import train_model
from ml.fairness import compute_fairness
from ml.explainability import compute_shap_values

app = FastAPI(title="Fairness + Explainability API")

DATA_PATH = "data/uploaded.csv"


@app.post("/upload-dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df.to_csv(DATA_PATH, index=False)
    return {"rows": df.shape[0], "columns": list(df.columns)}


@app.post("/train-model/")
def train(model_type: str = "xgboost"):
    return train_model(DATA_PATH, model_type)


@app.get("/fairness/")
def fairness(sensitive_attribute: str):
    return compute_fairness(DATA_PATH, sensitive_attribute)


@app.get("/explain/shap/")
def shap_explain():
    return compute_shap_values()
