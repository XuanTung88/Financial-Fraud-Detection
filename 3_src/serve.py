"""ML service (FastAPI). Node.js gọi service này, không tự chạy model.

Chạy (trong thư mục 3_src):  uvicorn serve:app --port 8000
Tài liệu tự sinh:            http://localhost:8000/docs
"""
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from inference import Predictor

app = FastAPI(title="Fraud Detection ML Service", version="1.0")
predictor = Predictor()


class Transaction(BaseModel):
    TransactionID: Optional[int] = None          # chỉ trả lại nguyên vẹn để Node ghép với DB
    TransactionDT: float = Field(..., ge=0, description="Số giây tương đối (như dữ liệu IEEE-CIS)")
    TransactionAmt: float = Field(..., gt=0)
    ProductCD: str
    card1: int
    card4: Optional[str] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    C1: Optional[float] = None
    C2: Optional[float] = None
    C6: Optional[float] = None
    D1: Optional[float] = None
    D4: Optional[float] = None
    D10: Optional[float] = None
    M4: Optional[str] = None
    M6: Optional[str] = None


class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1, max_length=1000)


def _score(items: List[Transaction]):
    import pandas as pd

    df = pd.DataFrame([t.model_dump() for t in items])
    pred = predictor.predict_df(df.drop(columns=["TransactionID"]))
    out = []
    for t, row in zip(items, pred.to_dict("records")):
        out.append({
            "transaction_id": t.TransactionID,
            "fraud_probability": round(row["fraud_probability"], 6),
            "is_fraud": bool(row["is_fraud"]),
            "risk_level": row["risk_level"],
            "nn_probability": round(row["nn_probability"], 6),
            "rf_probability": round(row["rf_probability"], 6),
            "model_version": predictor.version,
        })
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model_version": predictor.version}


@app.get("/schema")
def schema():
    return predictor.pipe.schema()


@app.post("/predict")
def predict(tx: Transaction):
    return _score([tx])[0]


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    return {"results": _score(req.transactions)}
