const path = require('path');
const express = require('express');
const { PORT } = require('./config');
const { query } = require('./db');
const { mlHealth } = require('./ml');
const { wrap } = require('./utils');

const app = express();
app.use(express.json({ limit: '100kb' }));
app.use(express.static(path.join(__dirname, '..', 'public')));
app.use('/reports', express.static(path.join(__dirname, '..', '..', '5_reports', 'figures')));

// Kiểm tra sống: DB và ML service
app.get('/api/health', wrap(async (req, res) => {
  const out = { api: 'ok', database: 'error', ml_service: 'error' };
  try { await query('SELECT 1'); out.database = 'ok'; } catch (e) { out.database_error = e.message; }
  try { const h = await mlHealth(); out.ml_service = 'ok'; out.ml_model_version = h.model_version; }
  catch (e) { out.ml_error = e.message; }
  res.status(out.database === 'ok' ? 200 : 503).json(out);
}));

app.use('/api', require('./routes/stats'));
app.use('/api', require('./routes/transactions'));

app.use('/api', (req, res) => res.status(404).json({ error: 'Không có endpoint này' }));

// Gom mọi lỗi về JSON { error }
// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  const status = err.status || 500;
  if (status === 500) console.error(err);
  res.status(status).json({ error: status === 500 ? 'Lỗi máy chủ nội bộ' : err.message });
});

if (require.main === module) {
  app.listen(PORT, () => console.log(`Dashboard: http://localhost:${PORT}`));
}
module.exports = app;
