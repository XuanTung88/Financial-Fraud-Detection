const express = require('express');
const { pool, query, getActiveModel } = require('../db');
const { BASE_TIME_MS } = require('../config');
const { scoreTransaction } = require('../ml');
const { HttpError, wrap, getContext } = require('../utils');

const router = express.Router();

const SORTS = { probability: 'p.fraud_probability', amount: 't.amount', time: 't.tx_dt', id: 't.id' };

// Danh sách giao dịch bị cảnh báo (xác suất >= ngưỡng), có lọc, sắp xếp, phân trang
router.get('/alerts', wrap(async (req, res) => {
  const { model, threshold, source } = await getContext(req, 'all');
  const q = req.query;

  const mediumUp = q.level === 'medium_up';           // lấy thêm nhóm "trung bình" (>= 50% ngưỡng)
  const cutoff = mediumUp ? threshold * 0.5 : threshold;
  const page = Math.max(1, parseInt(q.page, 10) || 1);
  const pageSize = Math.min(100, Math.max(1, parseInt(q.pageSize, 10) || 15));
  const sortCol = SORTS[q.sort] || SORTS.probability;
  const dir = q.dir === 'asc' ? 'ASC' : 'DESC';

  const params = [model.version, threshold, cutoff];
  const where = ['p.model_version = $1', 'p.fraud_probability >= $3'];
  if (source !== 'all') { params.push(source); where.push(`t.source = $${params.length}`); }
  if (q.product) { params.push(String(q.product)); where.push(`t.product_cd = $${params.length}`); }
  if (q.minAmount && Number.isFinite(Number(q.minAmount)) && Number(q.minAmount) > 0) {
    params.push(Number(q.minAmount)); where.push(`t.amount >= $${params.length}`);
  }
  if (q.hour !== undefined && q.hour !== '' && Number.isInteger(Number(q.hour))) {
    params.push(Number(q.hour)); where.push(`t.hour_of_day = $${params.length}`);
  }
  params.push(pageSize, (page - 1) * pageSize);

  const { rows } = await query(`
    SELECT t.id, t.external_id, t.source, t.tx_dt::float8 AS tx_dt, t.hour_of_day::int AS hour,
      t.amount::float8 AS amount, t.product_cd, t.card1, t.card4, t.card6,
      p.fraud_probability, p.nn_probability, p.rf_probability,
      CASE WHEN p.fraud_probability >= $2 THEN 'high' ELSE 'medium' END AS risk_level,
      l.is_fraud AS actual_fraud,
      COUNT(*) OVER()::int AS total
    FROM predictions p
    JOIN transactions t ON t.id = p.transaction_id
    LEFT JOIN labels l ON l.transaction_id = t.id
    WHERE ${where.join(' AND ')}
    ORDER BY ${sortCol} ${dir}, t.id
    LIMIT $${params.length - 1} OFFSET $${params.length}`, params);

  const total = rows.length ? rows[0].total : 0;
  res.json({
    threshold, page, pageSize, total, pages: Math.max(1, Math.ceil(total / pageSize)),
    items: rows.map(({ total: _t, ...r }) => r),
  });
}));

// Chi tiết một giao dịch: trường thô, điểm số từng model, nhãn thật (nếu có)
router.get('/transactions/:id', wrap(async (req, res) => {
  const id = Number(req.params.id);
  if (!Number.isInteger(id) || id <= 0) throw new HttpError(400, 'id không hợp lệ');
  const model = await getActiveModel();

  const { rows: [r] } = await query(`
    SELECT t.id, t.external_id, t.source, t.tx_dt::float8 AS tx_dt, t.tx_time, t.hour_of_day::int AS hour,
      t.amount::float8 AS amount, t.payload,
      p.model_version, p.fraud_probability, p.nn_probability, p.rf_probability, p.risk_level, p.predicted_at,
      l.is_fraud AS actual_fraud
    FROM transactions t
    LEFT JOIN predictions p ON p.transaction_id = t.id AND p.model_version = $2
    LEFT JOIN labels l ON l.transaction_id = t.id
    WHERE t.id = $1`, [id, model ? model.version : null]);
  if (!r) throw new HttpError(404, 'Không tìm thấy giao dịch');
  res.json({ ...r, ensemble_threshold: model ? model.ensemble_threshold : null });
}));

// ----------------------------------------------------------------------------
// Chấm điểm một giao dịch mới: kiểm tra -> gọi ML service -> lưu DB -> trả kết quả
// ----------------------------------------------------------------------------
const ALLOWED_FIELDS = [
  'TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card4', 'card6',
  'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
  'C1', 'C2', 'C6', 'D1', 'D4', 'D10', 'M4', 'M6',
];

function validateTransaction(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    throw new HttpError(400, 'Body phải là một object JSON');
  }
  const tx = {};
  for (const k of ALLOWED_FIELDS) if (body[k] !== undefined && body[k] !== null) tx[k] = body[k];

  const need = ['TransactionDT', 'TransactionAmt', 'ProductCD', 'card1'];
  const missing = need.filter((k) => tx[k] === undefined || tx[k] === '');
  if (missing.length) throw new HttpError(400, `Thiếu trường bắt buộc: ${missing.join(', ')}`);

  if (!Number.isFinite(Number(tx.TransactionDT)) || Number(tx.TransactionDT) < 0) {
    throw new HttpError(400, 'TransactionDT phải là số >= 0 (giây)');
  }
  if (!Number.isFinite(Number(tx.TransactionAmt)) || Number(tx.TransactionAmt) <= 0) {
    throw new HttpError(400, 'TransactionAmt phải là số > 0');
  }
  if (!Number.isInteger(Number(tx.card1))) throw new HttpError(400, 'card1 phải là số nguyên');
  if (typeof tx.ProductCD !== 'string') throw new HttpError(400, 'ProductCD phải là chuỗi');
  return tx;
}

router.post('/transactions/score', wrap(async (req, res) => {
  const tx = validateTransaction(req.body);
  const model = await getActiveModel();
  if (!model) throw new HttpError(503, 'Chưa có model nào trong DB. Hãy chạy seed_db.py trước.');

  const result = await scoreTransaction(tx);
  if (result.model_version !== model.version) {
    throw new HttpError(409,
      `ML service đang dùng model ${result.model_version} nhưng DB mới nhất là ${model.version}. Hãy chạy lại seed_db.py.`);
  }

  const dt = Math.floor(Number(tx.TransactionDT));
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    const ins = await client.query(`
      INSERT INTO transactions
        (external_id, source, tx_dt, tx_time, hour_of_day, day_of_week, amount,
         product_cd, card1, card4, card6, p_emaildomain, r_emaildomain, payload)
      VALUES ($1, 'api', $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
      ON CONFLICT (external_id) DO NOTHING
      RETURNING id`,
    [tx.TransactionID ?? null, dt, new Date(BASE_TIME_MS + dt * 1000),
      Math.floor(dt / 3600) % 24, Math.floor(dt / 86400) % 7, Number(tx.TransactionAmt),
      tx.ProductCD, tx.card1, tx.card4 ?? null, tx.card6 ?? null,
      tx.P_emaildomain ?? null, tx.R_emaildomain ?? null, JSON.stringify(tx)]);
    if (!ins.rows.length) throw new HttpError(409, `TransactionID ${tx.TransactionID} đã tồn tại`);

    const id = ins.rows[0].id;
    await client.query(`
      INSERT INTO predictions
        (transaction_id, model_version, fraud_probability, nn_probability, rf_probability, is_fraud, risk_level)
      VALUES ($1, $2, $3, $4, $5, $6, $7)`,
    [id, result.model_version, result.fraud_probability, result.nn_probability,
      result.rf_probability, result.is_fraud, result.risk_level]);
    await client.query('COMMIT');
    res.status(201).json({ id, ...result });
  } catch (e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
  }
}));

module.exports = router;
