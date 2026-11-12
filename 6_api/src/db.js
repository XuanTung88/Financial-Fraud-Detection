const { Pool, types } = require('pg');
const { DATABASE_URL } = require('./config');

// pg mặc định trả BIGINT (id, external_id) dạng chuỗi; id của đồ án này nhỏ hơn 2^53 nên đọc thành số an toàn
types.setTypeParser(types.builtins.INT8, (v) => parseInt(v, 10));

const pool = new Pool({ connectionString: DATABASE_URL });

// Lấy phiên bản model mới nhất (nguồn của ngưỡng mặc định)
async function getActiveModel() {
  const { rows } = await pool.query(
    'SELECT * FROM model_versions ORDER BY created_at DESC LIMIT 1');
  return rows[0] || null;
}

module.exports = { pool, query: (text, params) => pool.query(text, params), getActiveModel };
