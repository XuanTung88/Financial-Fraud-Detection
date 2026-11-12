const fs = require('fs');
const path = require('path');
const express = require('express');
const { query } = require('../db');
const { BASE_PREVALENCE } = require('../config');
const { wrap, getContext, metricsFrom } = require('../utils');

const router = express.Router();

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const headers = lines[0].split(',').map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const values = [];
    let cur = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (c === '"') { inQuotes = !inQuotes; }
      else if (c === ',' && !inQuotes) { values.push(cur.trim()); cur = ''; }
      else { cur += c; }
    }
    values.push(cur.trim());
    const obj = {};
    headers.forEach((h, idx) => {
      const val = values[idx] ?? '';
      const num = Number(val);
      obj[h || 'model_name'] = val !== '' && !isNaN(num) ? num : val;
    });
    return obj;
  });
}

// Phần FROM/WHERE dùng chung: $1 = phiên bản model, $2 = nguồn dữ liệu ('all' = tất cả)
const SCOPE = `
  FROM predictions p
  JOIN transactions t ON t.id = p.transaction_id
  LEFT JOIN labels l ON l.transaction_id = t.id
  WHERE p.model_version = $1 AND ($2::text = 'all' OR t.source = $2::text)`;

// Thông tin model + số liệu lúc huấn luyện (để đối chiếu với số tính lại từ DB)
router.get('/model', wrap(async (req, res) => {
  const { model } = await getContext(req);
  res.json({
    version: model.version,
    thresholds: {
      ensemble: model.ensemble_threshold, nn: model.nn_threshold, rf: model.rf_threshold,
      weight_nn: model.ensemble_weight_nn,
    },
    test_metrics: model.test_metrics,
    note: model.note,
    base_prevalence: BASE_PREVALENCE,
    created_at: model.created_at,
  });
}));

// KPI tổng quan + ma trận nhầm lẫn tại ngưỡng đang chọn
router.get('/overview', wrap(async (req, res) => {
  const { model, threshold, source } = await getContext(req);
  const { rows: [r] } = await query(`
    SELECT COUNT(*)::int AS total,
      COUNT(*) FILTER (WHERE p.fraud_probability >= $3)::int AS flagged,
      COALESCE(SUM(t.amount) FILTER (WHERE p.fraud_probability >= $3), 0)::float8 AS flagged_amount,
      COUNT(*) FILTER (WHERE l.transaction_id IS NOT NULL)::int AS labeled,
      COUNT(*) FILTER (WHERE l.is_fraud = true  AND p.fraud_probability >= $3)::int AS tp,
      COUNT(*) FILTER (WHERE l.is_fraud = false AND p.fraud_probability >= $3)::int AS fp,
      COUNT(*) FILTER (WHERE l.is_fraud = true  AND p.fraud_probability <  $3)::int AS fn,
      COUNT(*) FILTER (WHERE l.is_fraud = false AND p.fraud_probability <  $3)::int AS tn,
      COALESCE(SUM(t.amount) FILTER (WHERE l.is_fraud = true  AND p.fraud_probability >= $3), 0)::float8 AS amount_caught,
      COALESCE(SUM(t.amount) FILTER (WHERE l.is_fraud = true  AND p.fraud_probability <  $3), 0)::float8 AS amount_missed
    ${SCOPE}`, [model.version, source, threshold]);

  res.json({
    model_version: model.version, threshold, source,
    total: r.total, flagged: r.flagged, flagged_rate: r.total ? r.flagged / r.total : null,
    flagged_amount: r.flagged_amount, labeled: r.labeled,
    amount_caught: r.amount_caught, amount_missed: r.amount_missed,
    actual_fraud_rate: r.labeled ? (r.tp + r.fn) / r.labeled : null,
    metrics: metricsFrom(r),
    base_prevalence: BASE_PREVALENCE,
  });
}));

// Theo giờ trong ngày: lượng giao dịch, tỷ lệ cảnh báo, tỷ lệ fraud thực tế
router.get('/stats/hourly', wrap(async (req, res) => {
  const { model, threshold, source } = await getContext(req);
  const { rows } = await query(`
    SELECT t.hour_of_day::int AS hour, COUNT(*)::int AS total,
      COUNT(*) FILTER (WHERE p.fraud_probability >= $3)::int AS flagged,
      COUNT(*) FILTER (WHERE l.is_fraud)::int AS actual_fraud,
      COUNT(*) FILTER (WHERE l.is_fraud AND p.fraud_probability >= $3)::int AS caught
    ${SCOPE}
    GROUP BY 1 ORDER BY 1`, [model.version, source, threshold]);
  res.json(rows.map((x) => ({
    ...x,
    flagged_rate: x.total ? x.flagged / x.total : null,
    actual_fraud_rate: x.total ? x.actual_fraud / x.total : null,
  })));
}));

// Theo tuần tương đối (TransactionDT / 7 ngày)
router.get('/stats/weekly', wrap(async (req, res) => {
  const { model, threshold, source } = await getContext(req);
  const { rows } = await query(`
    SELECT FLOOR(t.tx_dt / 604800.0)::int AS week, COUNT(*)::int AS total,
      COUNT(*) FILTER (WHERE p.fraud_probability >= $3)::int AS flagged,
      COUNT(*) FILTER (WHERE l.is_fraud)::int AS actual_fraud,
      COUNT(*) FILTER (WHERE l.is_fraud AND p.fraud_probability >= $3)::int AS caught
    ${SCOPE}
    GROUP BY 1 ORDER BY 1`, [model.version, source, threshold]);
  res.json(rows);
}));

// Theo loại sản phẩm
router.get('/stats/by-product', wrap(async (req, res) => {
  const { model, threshold, source } = await getContext(req);
  const { rows } = await query(`
    SELECT COALESCE(t.product_cd, '?') AS product, COUNT(*)::int AS total,
      COUNT(*) FILTER (WHERE p.fraud_probability >= $3)::int AS flagged,
      COUNT(*) FILTER (WHERE l.is_fraud)::int AS actual_fraud,
      COUNT(*) FILTER (WHERE l.is_fraud AND p.fraud_probability >= $3)::int AS caught
    ${SCOPE}
    GROUP BY 1 ORDER BY 2 DESC`, [model.version, source, threshold]);
  res.json(rows.map((x) => ({
    ...x,
    flagged_rate: x.total ? x.flagged / x.total : null,
    actual_fraud_rate: x.total ? x.actual_fraud / x.total : null,
  })));
}));

// Phân phối điểm số theo nhãn thật (20 khoảng): cho thấy model tách 2 lớp tốt đến đâu
router.get('/stats/score-distribution', wrap(async (req, res) => {
  const { model, source } = await getContext(req);
  const { rows } = await query(`
    SELECT LEAST(FLOOR(p.fraud_probability * 20), 19)::int AS bin, l.is_fraud, COUNT(*)::int AS n
    ${SCOPE} AND l.transaction_id IS NOT NULL
    GROUP BY 1, 2`, [model.version, source]);

  const fraud = Array(20).fill(0);
  const normal = Array(20).fill(0);
  rows.forEach((r) => { (r.is_fraud ? fraud : normal)[r.bin] = r.n; });
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  res.json({ bins: 20, fraud, normal, fraud_total: sum(fraud), normal_total: sum(normal) });
}));

// Đường Precision-Recall quét qua 49 ngưỡng (0,02 đến 0,98).
// Cách làm nhanh: đếm số giao dịch theo 50 khoảng điểm (1 lần quét bảng) rồi cộng dồn từ cao xuống thấp.
router.get('/stats/pr-curve', wrap(async (req, res) => {
  const { model, source } = await getContext(req);
  const { rows } = await query(`
    SELECT LEAST(FLOOR(p.fraud_probability * 50), 49)::int AS bin, l.is_fraud, COUNT(*)::int AS n
    ${SCOPE} AND l.transaction_id IS NOT NULL
    GROUP BY 1, 2`, [model.version, source]);

  const pos = Array(50).fill(0);
  const neg = Array(50).fill(0);
  rows.forEach((r) => { (r.is_fraud ? pos : neg)[r.bin] = r.n; });
  const posTotal = pos.reduce((a, b) => a + b, 0);
  const negTotal = neg.reduce((a, b) => a + b, 0);

  const out = [];
  for (let g = 1; g <= 49; g++) {              // cảnh báo khi điểm >= g/50, tức khoảng điểm >= g
    let tp = 0; let fp = 0;
    for (let b = g; b < 50; b++) { tp += pos[b]; fp += neg[b]; }
    out.push({ threshold: g / 50, ...metricsFrom({ tp, fp, fn: posTotal - tp, tn: negTotal - fp }) });
  }
  res.json(out);
}));

// Lấy danh sách kết quả thử nghiệm mô hình và NN tuning từ CSV
router.get('/stats/experiments', wrap(async (req, res) => {
  const compPath = path.join(__dirname, '..', '..', '..', '1_data', 'exports', 'model_comparison.csv');
  const nnPath = path.join(__dirname, '..', '..', '..', '1_data', 'exports', 'nn_experiments.csv');
  let models = [];
  let nn_experiments = [];
  try {
    if (fs.existsSync(compPath)) {
      models = parseCSV(fs.readFileSync(compPath, 'utf8'));
    }
  } catch (e) { console.error('Error reading model_comparison.csv', e); }
  try {
    if (fs.existsSync(nnPath)) {
      nn_experiments = parseCSV(fs.readFileSync(nnPath, 'utf8'));
    }
  } catch (e) { console.error('Error reading nn_experiments.csv', e); }
  res.json({ models, nn_experiments });
}));

module.exports = router;
