// Tạo bảng:  npm run db:init      Xóa sạch rồi tạo lại:  npm run db:reset
const fs = require('fs');
const path = require('path');
const { pool } = require('../src/db');

(async () => {
  try {
    if (process.argv.includes('--reset')) {
      await pool.query('DROP TABLE IF EXISTS labels, predictions, transactions, model_versions CASCADE');
      console.log('Đã xóa các bảng cũ.');
    }
    await pool.query(fs.readFileSync(path.join(__dirname, 'schema.sql'), 'utf8'));
    const { rows } = await pool.query(
      "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY 1");
    console.log('Các bảng hiện có:', rows.map((r) => r.table_name).join(', '));
  } catch (e) {
    console.error('Lỗi:', e.message);
    process.exitCode = 1;
  } finally {
    await pool.end();
  }
})();
