-- Hệ thống phát hiện giao dịch bất thường: schema PostgreSQL
-- Chạy an toàn nhiều lần (IF NOT EXISTS).

-- Mỗi lần huấn luyện lại model = 1 phiên bản; mọi dự đoán gắn với phiên bản đã tạo ra nó.
CREATE TABLE IF NOT EXISTS model_versions (
    version             TEXT PRIMARY KEY,
    nn_threshold        DOUBLE PRECISION NOT NULL,
    rf_threshold        DOUBLE PRECISION NOT NULL,
    ensemble_weight_nn  DOUBLE PRECISION NOT NULL,
    ensemble_threshold  DOUBLE PRECISION NOT NULL,
    test_metrics        JSONB NOT NULL,
    note                TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Giao dịch. source = 'replay' (tập test đổ vào để demo) hoặc 'api' (chấm điểm qua API).
CREATE TABLE IF NOT EXISTS transactions (
    id             BIGSERIAL PRIMARY KEY,
    external_id    BIGINT UNIQUE,                 -- TransactionID gốc (có thể NULL)
    source         TEXT NOT NULL DEFAULT 'api' CHECK (source IN ('replay', 'api')),
    tx_dt          BIGINT NOT NULL,               -- TransactionDT gốc (giây, mốc tương đối)
    tx_time        TIMESTAMPTZ NOT NULL,          -- mốc giả định 2017-12-01 + tx_dt (chỉ để hiển thị)
    hour_of_day    SMALLINT NOT NULL CHECK (hour_of_day BETWEEN 0 AND 23),
    day_of_week    SMALLINT NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
    amount         NUMERIC(14, 2) NOT NULL CHECK (amount > 0),
    product_cd     TEXT,
    card1          INTEGER,
    card4          TEXT,
    card6          TEXT,
    p_emaildomain  TEXT,
    r_emaildomain  TEXT,
    payload        JSONB NOT NULL,                -- toàn bộ trường thô đã gửi vào model
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Kết quả dự đoán. Lưu xác suất (không chỉ nhãn) để dashboard đổi ngưỡng được.
CREATE TABLE IF NOT EXISTS predictions (
    id                 BIGSERIAL PRIMARY KEY,
    transaction_id     BIGINT NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    model_version      TEXT NOT NULL REFERENCES model_versions(version),
    fraud_probability  DOUBLE PRECISION NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    nn_probability     DOUBLE PRECISION,
    rf_probability     DOUBLE PRECISION,
    is_fraud           BOOLEAN NOT NULL,          -- theo ngưỡng của model lúc dự đoán
    risk_level         TEXT NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
    predicted_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (transaction_id, model_version)
);

-- Nhãn thật (ground truth). Tách riêng vì ngoài đời nhãn đến muộn, sau khi giao dịch xảy ra.
CREATE TABLE IF NOT EXISTS labels (
    transaction_id  BIGINT PRIMARY KEY REFERENCES transactions(id) ON DELETE CASCADE,
    is_fraud        BOOLEAN NOT NULL,
    source          TEXT NOT NULL DEFAULT 'ground_truth',
    labeled_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pred_model_prob ON predictions (model_version, fraud_probability DESC);
CREATE INDEX IF NOT EXISTS idx_tx_hour         ON transactions (hour_of_day);
CREATE INDEX IF NOT EXISTS idx_tx_dt           ON transactions (tx_dt);
CREATE INDEX IF NOT EXISTS idx_tx_card1        ON transactions (card1);
CREATE INDEX IF NOT EXISTS idx_tx_source       ON transactions (source);
