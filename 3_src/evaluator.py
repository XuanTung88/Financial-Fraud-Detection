"""Đánh giá mô hình: cùng một bộ metric cho mọi mô hình để so sánh công bằng."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)


def evaluate_model(model_name, y_true, y_pred, y_proba, results_dict,
                   train_time=None, verbose=True):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)

    # PR-AUC quan trọng hơn ROC-AUC khi lớp mất cân bằng
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(rec_curve, prec_curve)

    results_dict[model_name] = {
        "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba,
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc, "train_time": train_time,
    }

    if verbose:
        print("=" * 55)
        print(f"  KẾT QUẢ: {model_name}")
        print("=" * 55)
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1-score  : {f1:.4f}")
        print(f"  ROC-AUC   : {roc_auc:.4f}")
        print(f"  PR-AUC    : {pr_auc:.4f}")
        if train_time:
            print(f"  Thời gian train: {train_time:.2f}s")
        cm = confusion_matrix(y_true, y_pred)
        print("\n  Confusion Matrix:")
        print("              Dự đoán Không-fraud  Dự đoán Fraud")
        print(f"  Thực Không-fraud    {cm[0, 0]:>10,}      {cm[0, 1]:>10,}")
        print(f"  Thực Fraud          {cm[1, 0]:>10,}      {cm[1, 1]:>10,}")


def best_threshold_on_val(y_val_true, y_val_proba):
    """Chọn ngưỡng cho F1 cao nhất trên tập VALIDATION (không bao giờ chọn trên test)."""
    prec, rec, thr = precision_recall_curve(y_val_true, y_val_proba)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    i = int(np.argmax(f1s))
    return float(thr[i]), float(f1s[i])


def summary_table(results_dict, keys=None):
    keys = keys or list(results_dict)
    cols = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    return (pd.DataFrame({k: {m: results_dict[k][m] for m in cols} for k in keys})
            .T.round(4).sort_values("pr_auc", ascending=False))
