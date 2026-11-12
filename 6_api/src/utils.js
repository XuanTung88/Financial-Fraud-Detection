const { getActiveModel } = require('./db');
const { BASE_PREVALENCE } = require('./config');

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

// Bọc handler async để lỗi tự chuyển tới error middleware
const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

const SOURCES = ['replay', 'api', 'all'];

// Đọc ngữ cảnh chung của mọi API thống kê: model, ngưỡng, nguồn dữ liệu
async function getContext(req, defaultSource = 'replay') {
  const model = await getActiveModel();
  if (!model) throw new HttpError(503, 'Chưa có model nào trong DB. Hãy chạy seed_db.py trước.');

  let threshold = model.ensemble_threshold;
  if (req.query.threshold !== undefined) {
    threshold = Number(req.query.threshold);
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
      throw new HttpError(400, 'threshold phải là số trong khoảng 0 đến 1');
    }
  }
  const source = req.query.source || defaultSource;
  if (!SOURCES.includes(source)) throw new HttpError(400, `source phải là một trong: ${SOURCES.join(', ')}`);
  return { model, threshold, source };
}

// Từ ma trận nhầm lẫn -> các chỉ số. tp/fp/fn/tn là số nguyên.
function metricsFrom({ tp, fp, fn, tn }, prevalence = BASE_PREVALENCE) {
  const precision = tp + fp > 0 ? tp / (tp + fp) : null;
  const recall = tp + fn > 0 ? tp / (tp + fn) : null;
  const fpr = fp + tn > 0 ? fp / (fp + tn) : null;
  const f1 = precision != null && recall != null && precision + recall > 0
    ? (2 * precision * recall) / (precision + recall) : null;

  // Precision hiệu chỉnh theo tỷ lệ fraud thật (giả định: chỉ tỷ lệ lớp đổi, phân phối điểm số giữ nguyên)
  //   precision' = TPR*pi / (TPR*pi + FPR*(1-pi))
  let precisionAdjusted = null;
  if (recall != null && fpr != null) {
    const num = recall * prevalence;
    const den = num + fpr * (1 - prevalence);
    precisionAdjusted = den > 0 ? num / den : null;
  }
  return { tp, fp, fn, tn, precision, recall, f1, fpr, precision_adjusted: precisionAdjusted };
}

module.exports = { HttpError, wrap, getContext, metricsFrom };
