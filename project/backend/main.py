from fastapi import FastAPI
from api.train import router as train_router
from api.fairness_api import router as fairness_router
from api.explain_api import router as explain_router

app = FastAPI(title="ML Fairness Monitoring System")

app.include_router(train_router)
app.include_router(fairness_router)
app.include_router(explain_router)

@app.get("/")
def root():
    return {"status": "System Running"}
