"""Dựng và kiểm chứng các artifact cần cho API. Chạy 1 lần:  python build_artifacts.py

4 bước:
  1. Fit pipeline (median, bảng mã hóa, bảng tần suất, scaler) và lưu lại.
  2. KIỂM TRA KHỚP: pipeline dựng lại từ dữ liệu thô phải ra đúng bảng đặc trưng của notebook.
  3. KIỂM TRA MODEL: nn_final.keras + random_forest_fair.pkl chấm lại phải ra số cũ.
  4. Chọn ngưỡng/trọng số trên VALIDATION và ghi model_config.json.
"""
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import (ARTIFACT_DIR, CAT_COLS, CLEANED, DROP_MULTICOLLINEAR, LABEL,
                    MODEL_DIR)
from data_loader import load_raw_sample, load_splits
from evaluator import best_threshold_on_val, evaluate_model
from pipeline import FraudPipeline

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
t_start = time.time()

# ---------------------------------------------------------------- 1. fit
print("[1/4] Fit pipeline ...")
raw = load_raw_sample()
sp = load_splits()
pipe = FraudPipeline().fit(raw, sp["X_train"])
pipe.save(ARTIFACT_DIR / "pipeline.joblib")
with open(ARTIFACT_DIR / "schema.json", "w", encoding="utf-8") as f:
    json.dump(pipe.schema(), f, ensure_ascii=False, indent=2)
print(f"      {len(pipe.base_columns)} cột base -> {len(pipe.tree_columns)} cột model")
print(f"      median điền thiếu: { {k: round(v, 1) for k, v in pipe.medians.items()} }")

# -------------------------------------------------- 2. khớp với notebook
print("\n[2/4] Kiểm tra pipeline khớp notebook (200.000 dòng) ...")
base_new = pipe.raw_to_base(raw)

ref = pd.read_csv(CLEANED).drop(columns=[LABEL] + DROP_MULTICOLLINEAR)
ref = pd.get_dummies(ref, columns=[f"{c}_enc" for c in CAT_COLS], drop_first=True,
                     prefix=[f"{c}_enc" for c in CAT_COLS])
bool_cols = ref.select_dtypes(include="bool").columns
ref[bool_cols] = ref[bool_cols].astype(int)
ref = ref[pipe.base_columns]

diff = np.abs(base_new.to_numpy(dtype=float) - ref.to_numpy(dtype=float)).max(axis=0)
bad = [(c, float(d)) for c, d in zip(pipe.base_columns, diff) if d > 1e-6]
if bad:
    print("      LỆCH ở các cột:", bad[:10])
    sys.exit("DỪNG: pipeline không khớp notebook. Gửi nội dung dòng trên để kiểm tra.")
print("      OK: mọi cột khớp tuyệt đối với notebook")

# ------------------------------------------------------ 3. kiểm tra model
print("\n[3/4] Nạp model và chấm lại trên validation ...")
from tensorflow import keras  # import muộn vì nặng

nn = keras.models.load_model(MODEL_DIR / "nn_final.keras")
rf = joblib.load(MODEL_DIR / "random_forest_fair.pkl")

tree_va, nn_va = pipe.transform_base(sp["X_val"])
tree_te, nn_te = pipe.transform_base(sp["X_test"])
y_va, y_te = sp["y_val"].to_numpy(), sp["y_test"].to_numpy()

p_nn_va = nn.predict(nn_va, batch_size=4096, verbose=0).ravel()
p_nn_te = nn.predict(nn_te, batch_size=4096, verbose=0).ravel()
p_rf_va = rf.predict_proba(tree_va)[:, 1]
p_rf_te = rf.predict_proba(tree_te)[:, 1]

ap_nn = average_precision_score(y_va, p_nn_va)
ap_rf = average_precision_score(y_va, p_rf_va)
print(f"      NN val PR-AUC = {ap_nn:.4f}  (notebook: ~0.72)")
print(f"      RF val PR-AUC = {ap_rf:.4f}  (notebook: ~0.71)")
if ap_nn < 0.68 or ap_rf < 0.66:
    sys.exit("DỪNG: điểm thấp bất thường, pipeline hoặc file model không khớp nhau.")

# -------------------------------------------- 4. ngưỡng + trọng số + cấu hình
print("\n[4/4] Chọn ngưỡng và trọng số trên validation, chấm test 1 lần ...")
thr_nn, _ = best_threshold_on_val(y_va, p_nn_va)
thr_rf, _ = best_threshold_on_val(y_va, p_rf_va)

ws = np.linspace(0, 1, 11)
w_best = float(ws[int(np.argmax([average_precision_score(y_va, w * p_nn_va + (1 - w) * p_rf_va)
                                 for w in ws]))])
p_en_va = w_best * p_nn_va + (1 - w_best) * p_rf_va
p_en_te = w_best * p_nn_te + (1 - w_best) * p_rf_te
thr_en, _ = best_threshold_on_val(y_va, p_en_va)

res = {}
evaluate_model("NN_final_thr", y_te, (p_nn_te >= thr_nn).astype(int), p_nn_te, res, verbose=False)
evaluate_model("RF_fair_thr", y_te, (p_rf_te >= thr_rf).astype(int), p_rf_te, res, verbose=False)
evaluate_model("Ensemble_NN_RF_fair", y_te, (p_en_te >= thr_en).astype(int), p_en_te, res, verbose=False)

print(f"\n      {'Mô hình':<22}{'Prec':>8}{'Recall':>8}{'F1':>8}{'PR-AUC':>8}")
for k, r in res.items():
    print(f"      {k:<22}{r['precision']:>8.4f}{r['recall']:>8.4f}{r['f1']:>8.4f}{r['pr_auc']:>8.4f}")

cfg = {
    "model_version": "v1-" + time.strftime("%Y%m%d"),
    "serving_model": "ensemble",
    "nn_threshold": thr_nn,
    "rf_threshold": thr_rf,
    "ensemble_weight_nn": w_best,
    "ensemble_threshold": thr_en,
    "test_metrics": {k: {m: float(r[m]) for m in ["precision", "recall", "f1", "roc_auc", "pr_auc"]}
                     for k, r in res.items()},
    "note": "Ngưỡng và trọng số chọn trên validation. Tỷ lệ fraud 10.3% là do lấy mẫu, "
            "không phải tỷ lệ thật nên precision ngoài thực tế sẽ thấp hơn.",
}
with open(ARTIFACT_DIR / "model_config.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print(f"\nXONG sau {time.time() - t_start:.0f}s. Đã ghi vào {ARTIFACT_DIR}:")
for p in sorted(ARTIFACT_DIR.iterdir()):
    print(f"   - {p.name}")
