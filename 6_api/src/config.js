const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env'), quiet: true });

module.exports = {
  PORT: Number(process.env.PORT) || 3000,
  DATABASE_URL: process.env.DATABASE_URL || 'postgresql://postgres:password@localhost:5432/fraud_db',
  ML_SERVICE_URL: (process.env.ML_SERVICE_URL || 'http://localhost:8000').replace(/\/$/, ''),
  // Tỷ lệ fraud thật của IEEE-CIS (20.663 / 590.540 ≈ 3,5%)
  BASE_PREVALENCE: Number(process.env.BASE_PREVALENCE) || 0.035,
  // Mốc giả định để đổi TransactionDT (giây tương đối) thành ngày giờ hiển thị
  BASE_TIME_MS: Date.UTC(2017, 11, 1),
};
