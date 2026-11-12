"""Predictor: nạp pipeline + NN + RF và chấm điểm giao dịch thô.

Chạy thử nhanh (smoke test):  python inference.py
"""
import json

import joblib
import pandas as pd

from config import ARTIFACT_DIR, MODEL_DIR
from pipeline import FraudPipeline


class Predictor:
    def __init__(self):
        from tensorflow import keras  # import muộn vì nặng

        self.pipe = FraudPipeline.load(ARTIFACT_DIR / "pipeline.joblib")
        with open(ARTIFACT_DIR / "model_config.json", encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.nn = keras.models.load_model(MODEL_DIR / "nn_final.keras")
        self.rf = joblib.load(MODEL_DIR / "random_forest_fair.pkl")

    @property
    def version(self):
        return self.cfg["model_version"]

    def risk_level(self, p):
        """high: vượt ngưỡng quyết định; medium: vượt một nửa ngưỡng (quy ước, không có cơ sở thống kê)."""
        thr = self.cfg["ensemble_threshold"]
        if p >= thr:
            return "high"
        return "medium" if p >= 0.5 * thr else "low"

    def predict_df(self, raw_df):
        tree, nn_x = self.pipe.transform_raw(raw_df)
        p_nn = self.nn.predict(nn_x, batch_size=4096, verbose=0).ravel()
        p_rf = self.rf.predict_proba(tree)[:, 1]

        w = self.cfg["ensemble_weight_nn"]
        p_en = w * p_nn + (1 - w) * p_rf
        thr = self.cfg["ensemble_threshold"]

        return pd.DataFrame({
            "fraud_probability": p_en,
            "is_fraud": (p_en >= thr).astype(int),
            "risk_level": [self.risk_level(p) for p in p_en],
            "nn_probability": p_nn,
            "rf_probability": p_rf,
        })


if __name__ == "__main__":
    from data_loader import load_raw_sample

    raw = load_raw_sample().sample(5, random_state=1)
    truth = raw["isFraud"].to_numpy()
    pred = Predictor().predict_df(raw.drop(columns=["isFraud"]))
    pred["thuc_te_isFraud"] = truth
    print(pred.round(3).to_string())
