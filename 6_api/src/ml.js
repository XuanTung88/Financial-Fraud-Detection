const { ML_SERVICE_URL } = require('./config');
const { HttpError } = require('./utils');

async function callMl(pathname, options = {}) {
  let res;
  try {
    res = await fetch(`${ML_SERVICE_URL}${pathname}`, {
      ...options,
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(15000),
    });
  } catch (e) {
    throw new HttpError(502, `Không kết nối được ML service tại ${ML_SERVICE_URL} (đã chạy uvicorn chưa?)`);
  }
  const body = await res.json().catch(() => ({}));
  if (res.status === 422) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ')
      : 'dữ liệu không hợp lệ';
    throw new HttpError(400, `ML service từ chối dữ liệu: ${detail}`);
  }
  if (!res.ok) throw new HttpError(502, `ML service lỗi ${res.status}`);
  return body;
}

const scoreTransaction = (tx) => callMl('/predict', { method: 'POST', body: JSON.stringify(tx) });
const mlHealth = () => callMl('/health');

module.exports = { scoreTransaction, mlHealth };
