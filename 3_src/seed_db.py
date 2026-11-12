"""Đổ TẬP TEST (30.000 giao dịch chưa từng dùng để huấn luyện) vào PostgreSQL.

Chạy:  python seed_db.py            (thêm --reset để xóa dữ liệu replay cũ rồi nạp lại)
       python seed_db.py --dsn postgresql://user:pass@localhost:5432/fraud_db

Vì sao chỉ dùng tập test: nếu chấm điểm cả dữ liệu train thì dashboard sẽ đẹp giả tạo
(model đã "thấy" các giao dịch đó). Tập test cho số liệu trung thực.

Các bước:
  1. Khôi phục đúng 30.000 dòng test từ dữ liệu thô (tái hiện phép chia seed=42 của notebook,
     rồi đối chiếu với X_test.csv để chắc chắn khớp).
  2. Chấm điểm bằng Predictor (NN + RF), đối chiếu với model_config.json.
  3. Ghi vào PostgreSQL: model_versions, transactions, predictions, labels.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from config import (ARTIFACT_DIR, CAT_COLS, CLEANED, DROP_MULTICOLLINEAR, LABEL,
                    RAW_CATEGORICAL, RAW_NUMERIC, SEED)

DEFAULT_DSN = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5432/fraud_db")
BASE_TIME = pd.Timestamp("2017-12-01", tz="UTC")   # mốc giả định, chỉ để hiển thị ngày giờ


def _py(v):
    """NaN -> None, numpy -> kiểu Python thuần (để psycopg2 và JSON xử lý được)."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v.item() if hasattr(v, "item") else v


# --------------------------------------------------------------------------
def recover_test_rows(base_columns, X_test_csv, y_test_csv):
    """Tái hiện phép chia của notebook 04 để biết dòng nào của 02_sampled.csv thuộc tập test."""
    from sklearn.model_selection import train_test_split

    ref = pd.read_csv(CLEANED)
    y = ref[LABEL]
    X = ref.drop(columns=[LABEL] + DROP_MULTICOLLINEAR)
    enc_cols = [f"{c}_enc" for c in CAT_COLS]
    X = pd.get_dummies(X, columns=enc_cols, drop_first=True, prefix=enc_cols)
    bools = X.select_dtypes(include="bool").columns
    X[bools] = X[bools].astype(int)
    X = X[base_columns]

    # Phải giống hệt notebook: test 15%, seed 42, stratify theo nhãn
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)

    same_x = np.allclose(X_test.to_numpy(float), X_test_csv[base_columns].to_numpy(float))
    same_y = np.array_equal(y_test.to_numpy(), y_test_csv.to_numpy())
    if not (same_x and same_y):
        sys.exit("DỪNG: không tái hiện được đúng tập test của notebook "
                 "(có thể khác phiên bản scikit-learn). Gửi lại dòng này để xử lý.")
    return X_test.index.to_numpy(), y_test.to_numpy()


# --------------------------------------------------------------------------
def write_to_db(dsn, raw_test, pred, y_true, cfg, reset=False):
    """Ghi vào PostgreSQL. raw_test: dữ liệu thô; pred: kết quả Predictor.predict_df."""
    import psycopg2
    from psycopg2.extras import Json, execute_values

    payload_cols = RAW_NUMERIC + RAW_CATEGORICAL
    recs = raw_test.reset_index(drop=True).to_dict("records")

    tx_rows, ext_ids = [], []
    for d in recs:
        dt = int(d["TransactionDT"])
        ext = _py(d.get("TransactionID"))
        ext_ids.append(ext)
        payload = {c: _py(d.get(c)) for c in payload_cols}
        tx_rows.append((
            ext, "replay", dt, (BASE_TIME + pd.Timedelta(seconds=dt)).to_pydatetime(),
            (dt // 3600) % 24, (dt // 86400) % 7, round(float(d["TransactionAmt"]), 2),
            _py(d.get("ProductCD")), _py(d.get("card1")), _py(d.get("card4")), _py(d.get("card6")),
            _py(d.get("P_emaildomain")), _py(d.get("R_emaildomain")), Json(payload),
        ))

    conn = psycopg2.connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            if reset:
                cur.execute("DELETE FROM transactions WHERE source = 'replay'")  # cascade xóa predictions, labels
                print(f"      đã xóa {cur.rowcount:,} giao dịch replay cũ")

            cur.execute("""
                INSERT INTO model_versions
                    (version, nn_threshold, rf_threshold, ensemble_weight_nn, ensemble_threshold, test_metrics, note)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (version) DO UPDATE SET
                    nn_threshold = EXCLUDED.nn_threshold, rf_threshold = EXCLUDED.rf_threshold,
                    ensemble_weight_nn = EXCLUDED.ensemble_weight_nn,
                    ensemble_threshold = EXCLUDED.ensemble_threshold,
                    test_metrics = EXCLUDED.test_metrics, note = EXCLUDED.note""",
                        (cfg["model_version"], cfg["nn_threshold"], cfg["rf_threshold"],
                         cfg["ensemble_weight_nn"], cfg["ensemble_threshold"],
                         Json(cfg["test_metrics"]), cfg.get("note")))

            res = execute_values(cur, """
                INSERT INTO transactions
                    (external_id, source, tx_dt, tx_time, hour_of_day, day_of_week, amount,
                     product_cd, card1, card4, card6, p_emaildomain, r_emaildomain, payload)
                VALUES %s
                ON CONFLICT (external_id) DO UPDATE SET source = EXCLUDED.source
                RETURNING id, external_id""", tx_rows, page_size=2000, fetch=True)
            id_by_ext = {ext: tid for tid, ext in res}
            ids = [id_by_ext[e] for e in ext_ids]

            version = cfg["model_version"]
            pred_rows = [
                (tid, version, float(p), float(n), float(r), bool(f), lvl)
                for tid, p, n, r, f, lvl in zip(
                    ids, pred["fraud_probability"], pred["nn_probability"],
                    pred["rf_probability"], pred["is_fraud"], pred["risk_level"])
            ]
            execute_values(cur, """
                INSERT INTO predictions
                    (transaction_id, model_version, fraud_probability, nn_probability,
                     rf_probability, is_fraud, risk_level)
                VALUES %s
                ON CONFLICT (transaction_id, model_version) DO UPDATE SET
                    fraud_probability = EXCLUDED.fraud_probability,
                    nn_probability = EXCLUDED.nn_probability, rf_probability = EXCLUDED.rf_probability,
                    is_fraud = EXCLUDED.is_fraud, risk_level = EXCLUDED.risk_level""",
                           pred_rows, page_size=2000)

            label_rows = [(tid, bool(y)) for tid, y in zip(ids, y_true)]
            execute_values(cur, """
                INSERT INTO labels (transaction_id, is_fraud) VALUES %s
                ON CONFLICT (transaction_id) DO UPDATE SET is_fraud = EXCLUDED.is_fraud""",
                           label_rows, page_size=2000)
    finally:
        conn.close()
    return len(ids)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--reset", action="store_true", help="xóa dữ liệu replay cũ trước khi nạp")
    args = ap.parse_args()

    from sklearn.metrics import auc, precision_recall_curve

    from data_loader import load_raw_sample, load_splits
    from inference import Predictor

    print("[1/3] Khôi phục tập test từ dữ liệu thô ...")
    raw = load_raw_sample()
    sp = load_splits()
    predictor = Predictor()
    idx, y_test = recover_test_rows(predictor.pipe.base_columns, sp["X_test"], sp["y_test"])
    raw_test = raw.loc[idx].reset_index(drop=True)
    print(f"      {len(raw_test):,} giao dịch test, fraud = {int(y_test.sum()):,} ({y_test.mean():.1%})")

    print("[2/3] Chấm điểm và đối chiếu với model_config.json ...")
    pred = predictor.predict_df(raw_test)
    cfg = predictor.cfg
    exp = cfg["test_metrics"]["Ensemble_NN_RF_fair"]

    flagged = pred["is_fraud"].to_numpy().astype(bool)
    tp = int((flagged & (y_test == 1)).sum())
    precision, recall = tp / max(flagged.sum(), 1), tp / y_test.sum()
    p_c, r_c, _ = precision_recall_curve(y_test, pred["fraud_probability"])
    pr_auc = auc(r_c, p_c)
    print(f"      precision {precision:.4f} (config {exp['precision']:.4f}) | "
          f"recall {recall:.4f} (config {exp['recall']:.4f}) | PR-AUC {pr_auc:.4f} (config {exp['pr_auc']:.4f})")
    if abs(precision - exp["precision"]) > 1e-3 or abs(recall - exp["recall"]) > 1e-3:
        sys.exit("DỪNG: kết quả không khớp model_config.json.")

    print("[3/3] Ghi vào PostgreSQL ...")
    n = write_to_db(args.dsn, raw_test, pred, y_test, cfg, reset=args.reset)
    print(f"XONG: đã ghi {n:,} giao dịch + dự đoán + nhãn thật (model {cfg['model_version']}).")


if __name__ == "__main__":
    main()
