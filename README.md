# Hệ thống phân tích hành vi và phát hiện giao dịch tài chính bất thường

Đồ án học tập kết hợp 3 môn: **Học máy và Khai phá dữ liệu**, **Phân tích và Trực quan hóa dữ liệu**, **Ngôn ngữ kịch bản (Node.js)**.

Bài toán: dự đoán một giao dịch thẻ có phải gian lận hay không (phân loại có nhãn), dùng bộ dữ liệu **IEEE-CIS Fraud Detection** (Kaggle), so sánh 3 thuật toán **Logistic Regression**, **Random Forest**, **Neural Network**, rồi đưa kết quả lên dashboard web (Node.js + PostgreSQL).

---

## 1. Kết quả chính

Đánh giá trên **tập test** (30.000 giao dịch, 3.099 ca fraud). Ngưỡng quyết định và trọng số ensemble được chọn trên tập validation, tập test chỉ chấm một lần.

| Mô hình | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.225 | 0.697 | 0.340 | 0.330 | 0.786 |
| Random Forest (có đặc trưng tần suất) | 0.667 | 0.626 | 0.646 | 0.710 | 0.924 |
| Neural Network (đã tối ưu) | 0.663 | 0.650 | 0.656 | 0.716 | 0.919 |
| **Ensemble NN + RF** (đang phục vụ) | 0.710 | 0.645 | 0.676 | 0.738 | 0.927 |

Lưu ý khi so sánh: Logistic Regression được chấm ở ngưỡng 0,5 (chưa tối ưu ngưỡng), ba mô hình còn lại dùng ngưỡng chọn trên validation. PR-AUC và ROC-AUC không phụ thuộc ngưỡng nên vẫn so sánh được giữa các dòng. Precision, Recall, F1 của Logistic Regression thì không công bằng bằng các dòng còn lại.

Các kết luận có số liệu đối chiếu:

1. Mô hình phi tuyến vượt hẳn Logistic Regression (PR-AUC gấp hơn 2 lần).
2. Tối ưu Neural Network nâng PR-AUC từ 0.590 (bản cơ bản) lên 0.716.
3. **Đặc trưng quan trọng hơn thuật toán**: đặc trưng tần suất `card1_freq`, `addr1_freq` giúp Random Forest tăng 0.677 lên 0.710; log transform cho các cột lệch giúp Neural Network (cây quyết định không nhạy với phân phối lệch).
4. Neural Network và Random Forest ngang nhau (chênh 0.006 PR-AUC, nằm trong mức nhiễu của tập test). Ensemble nhỉnh hơn một chút.

### Hạn chế cần nêu khi báo cáo

- **Tỷ lệ fraud 10,3% là do lấy mẫu**, tỷ lệ thật khoảng 3,5% (20.663 / 590.540). ROC-AUC không đổi nhưng precision ngoài thực tế thấp hơn. Dashboard có thẻ "Precision hiệu chỉnh" ước tính điều này (giả định chỉ tỷ lệ lớp đổi, phân phối điểm số giữ nguyên).
- Chia train/val/test **ngẫu nhiên, không theo thời gian**, nên kết quả có thể lạc quan hơn so với dự đoán tương lai thật.
- `TransactionDT` là số giây **tương đối**, không chắc là giờ thật. Kết luận "fraud tập trung 5h-9h" cần nói thận trọng. Ngày giờ trên dashboard là mốc giả định.
- Median dùng để điền thiếu được tính trên toàn bộ 200.000 dòng, trước khi chia tập (rò rỉ rất nhẹ).
- Mỗi thí nghiệm chạy một lần với một seed. Random Forest chưa được tune (200 cây, độ sâu 15 cố định) trong khi Neural Network đã qua nhiều vòng thử.
- Chỉ dùng một nguồn dữ liệu.

---

## 2. Cấu trúc dự án

```
Detai/
├── 1_data/
│   ├── raw/                       Dữ liệu gốc tải từ Kaggle (không bao giờ sửa)
│   │   ├── train_transaction.csv
│   │   └── train_identity.csv
│   ├── processed/                 Dữ liệu qua từng bước xử lý
│   │   ├── 02_sampled.csv            200.000 dòng, 28 cột thô (còn chữ)
│   │   ├── 03_cleaned.csv            Đã làm sạch + mã hóa nhãn (25 cột)
│   │   ├── X_train/val/test.csv      47 cột sau one-hot (chưa scale)
│   │   ├── y_train/val/test.csv
│   │   └── X_*_scaled.csv, *_smote.csv
│   └── exports/                   Bảng so sánh mô hình, log thí nghiệm
│
├── 2_notebooks/                   Khám phá và huấn luyện (chạy theo thứ tự số)
│   ├── 01_merge_and_sample.ipynb     Ghép 2 file, lấy mẫu 200.000 dòng
│   ├── 02_EDA.ipynb                  Phân tích và trực quan hóa
│   ├── 03_cleaning_v2.ipynb          Làm sạch, biến đổi
│   ├── 04_preprocessing_for_models.ipynb   One-hot, chia train/val/test, scale, SMOTE
│   └── 05_modeling.ipynb             Huấn luyện, tối ưu, so sánh 3 thuật toán
│
├── 3_src/                         Code Python dùng lại (huấn luyện + suy luận + nạp DB)
│   ├── config.py                     Đường dẫn, hằng số, danh sách cột
│   ├── data_loader.py                Đọc dữ liệu đã chuẩn bị
│   ├── cleaner.py                    Làm sạch một giao dịch thô
│   ├── feature_engineering.py        Gom nhóm text, đặc trưng thời gian, log, tần suất
│   ├── evaluator.py                  evaluate_model, best_threshold_on_val
│   ├── pipeline.py                   FraudPipeline: thô -> đầu vào model (dùng chung train và API)
│   ├── build_artifacts.py            Lưu scaler, bảng mã hóa, bảng tần suất, ngưỡng (chạy 1 lần)
│   ├── inference.py                  Predictor: nạp model, chấm điểm
│   ├── serve.py                      ML service (FastAPI), Node.js gọi vào đây
│   └── seed_db.py                    Đổ tập test vào PostgreSQL
│
├── 4_models/
│   ├── nn_final.keras                Neural Network đã tối ưu
│   ├── random_forest_fair.pkl        Random Forest cùng bộ đặc trưng với NN
│   └── artifacts/                    Tạo bởi build_artifacts.py
│       ├── pipeline.joblib              median, bảng mã hóa, bảng tần suất, scaler
│       ├── model_config.json            ngưỡng, trọng số ensemble, điểm test
│       └── schema.json                  trường đầu vào và giá trị hợp lệ
│
├── 5_reports/figures/             Biểu đồ xuất ra từ notebook
│
├── 6_api/                         Backend Node.js + dashboard
│   ├── db/
│   │   ├── schema.sql                Cấu trúc 4 bảng PostgreSQL
│   │   └── init.js                   Tạo bảng (npm run db:init)
│   ├── src/
│   │   ├── app.js                    Điểm vào Express
│   │   ├── config.js, db.js, utils.js, ml.js
│   │   └── routes/
│   │       ├── stats.js              API thống kê cho biểu đồ
│   │       └── transactions.js       Danh sách cảnh báo, chi tiết, chấm điểm
│   ├── public/index.html             Dashboard (HTML + Chart.js)
│   ├── package.json
│   └── .env                          Cấu hình (tự tạo, xem mục 4)
│
├── requirements.txt
└── README.md
```


### Luồng dữ liệu

```
 Trình duyệt ──► Node.js (Express, :3000) ──► PostgreSQL
 (dashboard)          │                        transactions / predictions / labels
                      │ POST /predict (chỉ khi chấm điểm giao dịch mới)
                      ▼
               FastAPI (:8000) ──► FraudPipeline ──► Neural Network + Random Forest
```

Vì sao tách ML service bằng Python: mô hình Keras và scikit-learn chạy trong Python, nên Node.js không tự chạy được. Node lo kiểm tra đầu vào, ghi cơ sở dữ liệu, phục vụ dashboard. Python chỉ lo chấm điểm.

---

## 3. Cài đặt một lần

### 3.1 Yêu cầu

- Python 3.10+ (đã dùng 3.12), Node.js 18+ (cần `fetch` có sẵn), PostgreSQL 14+.
- Đã chạy xong các notebook 01 đến 05 và có các file trong `4_models/`.

### 3.2 Thư viện Python

```bash
pip install fastapi uvicorn joblib psycopg2-binary
```

(Các thư viện pandas, scikit-learn, tensorflow đã cài từ giai đoạn huấn luyện. Nên thêm 4 thư viện trên vào `requirements.txt`.)

### 3.3 PostgreSQL

Cài PostgreSQL từ postgresql.org (Windows: ghi nhớ mật khẩu của user `postgres`, giữ cổng 5432). Mở **SQL Shell (psql)** từ menu Start, đăng nhập bằng user `postgres`, chạy:

```sql
CREATE USER tung WITH PASSWORD 'fraud123';
CREATE DATABASE fraud_db OWNER tung;
```

Đổi mật khẩu `fraud123` theo ý bạn, nhưng nhớ sửa cho khớp ở bước 3.5 và 5.

### 3.4 Thư viện Node.js

```bash
cd "/d/hocki7/Data Annalyst/Detai/6_api"
npm install express pg dotenv
```

Nếu `package.json` đã có phần `dependencies` thì chỉ cần `npm install`.

### 3.5 File cấu hình `6_api/.env`

Tạo file tên `.env` (không có phần tên trước dấu chấm) trong `6_api/`:

```
PORT=3000
DATABASE_URL=postgresql://fraud:fraud123@localhost:5432/fraud_db
ML_SERVICE_URL=http://localhost:8000
BASE_PREVALENCE=0.035
```

Không đưa file `.env` lên Git (tạo `6_api/.gitignore` chứa 2 dòng `node_modules/` và `.env`).

### 3.6 Tạo bảng

```bash
cd 6_api
npm run db:init
```

Kết quả đúng: `Các bảng hiện có: labels, model_versions, predictions, transactions`. Chạy lại nhiều lần không sao. Muốn xóa sạch tạo lại: `npm run db:reset`.

---

## 4. Chuẩn bị model và dữ liệu (chạy một lần, Python)

Chạy trong thư mục `3_src/`:

```bash
cd "/d/hocki7/Data Annalyst/Detai/3_src"

python build_artifacts.py      # lưu scaler, bảng mã hóa, ngưỡng; kiểm tra khớp notebook
python seed_db.py              # đổ 30.000 giao dịch test vào PostgreSQL
```

`build_artifacts.py` dừng ngay nếu pipeline lệch notebook hoặc điểm model thấp bất thường. Kết quả đúng có dòng `OK: mọi cột khớp tuyệt đối với notebook`.

`seed_db.py` làm 3 việc và tự đối chiếu số liệu với `model_config.json`:

1. Khôi phục đúng 30.000 dòng **thuộc tập test** (không dùng dữ liệu train để dashboard không đẹp giả tạo).
2. Chấm điểm bằng Neural Network + Random Forest.
3. Ghi `model_versions`, `transactions`, `predictions`, `labels`.

Nếu mật khẩu PostgreSQL khác mặc định:

```bash
python seed_db.py --dsn postgresql://fraud:MATKHAU@localhost:5432/fraud_db
```

Nạp lại từ đầu (xóa dữ liệu replay cũ): `python seed_db.py --reset`.

---

## 5. Chạy hệ thống hằng ngày

Cần **hai terminal** chạy song song (và PostgreSQL đang chạy).

**Terminal 1: ML service** (phải đứng trong `3_src`, mất 10 đến 20 giây để nạp TensorFlow):

```bash
cd "/d/hocki7/Data Annalyst/Detai/3_src"
uvicorn serve:app --port 8000
```

**Terminal 2: Node.js + dashboard:**

```bash
cd "/d/hocki7/Data Annalyst/Detai/6_api"
npm start
```

Mở trình duyệt: **http://localhost:3000**

Kiểm tra nhanh: `http://localhost:3000/api/health` phải trả `database: ok` và `ml_service: ok`. Tài liệu tự sinh của ML service: `http://localhost:8000/docs`.

---

## 6. Dùng dashboard

| Thành phần | Cách đọc |
|---|---|
| **Thanh ngưỡng** (góc phải trên) | Kéo để đổi ngưỡng cảnh báo, mọi số liệu và biểu đồ tính lại ngay. Nút "Ngưỡng mặc định" đặt về ngưỡng chọn trên validation. |
| **Thẻ KPI** | Số liệu **tính lại từ cơ sở dữ liệu**. Ở ngưỡng mặc định phải khớp số lúc huấn luyện (ghi nhỏ bên dưới thẻ). |
| **Precision hiệu chỉnh** | Ước tính precision nếu tỷ lệ fraud thật là 3,5% thay vì 10,3% của mẫu. Số này là ước tính, không phải đo được. |
| **Ma trận nhầm lẫn** | Hàng = thực tế, cột = dự đoán. Báo động giả (cam) và bỏ sót (đỏ) là hai loại sai. |
| **Đường Precision-Recall** | Chấm đỏ là ngưỡng hiện tại; đường đứt là phiên bản hiệu chỉnh. |
| **Theo giờ** | Cột xám là số giao dịch: giờ nào cột thấp thì tỷ lệ dao động mạnh, đừng kết luận vội. |
| **Phân phối điểm số** | Hai lớp càng tách xa nhau thì model càng đáng tin. |
| **Danh sách cảnh báo** | Có lọc, sắp xếp, phân trang. Cột "Thực tế" là nhãn thật để tự kiểm tra cảnh báo đúng hay sai. "Chi tiết" hiện điểm từng model. |
| **Thử chấm điểm** | Gửi một giao dịch JSON, kết quả được lưu vào DB (nguồn "Chấm qua API"). Chọn nguồn đó ở bộ lọc để xem lại. |

---

## 7. API Node.js

Mọi endpoint trả JSON; lỗi có dạng `{ "error": "..." }`. Tham số chung: `threshold` (0 đến 1, mặc định là ngưỡng của model) và `source` (`replay`, `api`, `all`).

| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| GET | `/api/health` | Kiểm tra API, database, ML service |
| GET | `/api/model` | Phiên bản model, ngưỡng, điểm lúc huấn luyện |
| GET | `/api/overview` | KPI, ma trận nhầm lẫn, precision/recall/F1 tại ngưỡng |
| GET | `/api/stats/hourly` | Theo giờ trong ngày |
| GET | `/api/stats/weekly` | Theo tuần tương đối |
| GET | `/api/stats/by-product` | Theo loại sản phẩm |
| GET | `/api/stats/score-distribution` | Phân phối điểm theo nhãn thật |
| GET | `/api/stats/pr-curve` | Precision/recall tại 49 ngưỡng |
| GET | `/api/alerts` | Danh sách cảnh báo. Lọc: `level` (`high`/`medium_up`), `product`, `minAmount`, `hour`, `source`. Sắp xếp: `sort` (`probability`/`amount`/`time`), `dir`. Phân trang: `page`, `pageSize` (tối đa 100) |
| GET | `/api/transactions/:id` | Chi tiết một giao dịch |
| POST | `/api/transactions/score` | Chấm điểm giao dịch mới và lưu vào DB |

### Ví dụ chấm điểm

Bắt buộc 4 trường: `TransactionDT` (giây), `TransactionAmt` (> 0), `ProductCD`, `card1`. Trường thiếu được điền giá trị trung vị lúc huấn luyện; category lạ được xếp vào nhóm "Unknown".

```bash
curl -X POST http://localhost:3000/api/transactions/score \
  -H "Content-Type: application/json" \
  -d '{"TransactionDT": 15000000, "TransactionAmt": 120.5, "ProductCD": "W",
       "card1": 9633, "card4": "visa", "card6": "debit", "P_emaildomain": "gmail.com"}'
```

Phản hồi:

```json
{ "id": 30001, "transaction_id": null, "fraud_probability": 0.1234, "is_fraud": false,
  "risk_level": "low", "nn_probability": 0.11, "rf_probability": 0.14,
  "model_version": "v1-20261105" }
```

Mức rủi ro: `high` khi điểm ≥ ngưỡng quyết định; `medium` khi điểm ≥ một nửa ngưỡng (quy ước của đồ án, không có cơ sở thống kê); còn lại là `low`.

Mã lỗi: `400` dữ liệu không hợp lệ, `404` không tìm thấy, `409` trùng `TransactionID` hoặc lệch phiên bản model, `502` không kết nối được ML service.

---

## 8. Cơ sở dữ liệu

```
model_versions (version PK, ngưỡng NN/RF/ensemble, trọng số, test_metrics JSONB)
      ▲
      │ model_version
predictions (id, transaction_id FK, model_version FK, fraud_probability,
             nn_probability, rf_probability, is_fraud, risk_level)   UNIQUE(transaction_id, model_version)
      │ transaction_id
      ▼
transactions (id, external_id UNIQUE, source 'replay'|'api', tx_dt, tx_time,
              hour_of_day, day_of_week, amount, product_cd, card1, card4, card6,
              p_emaildomain, r_emaildomain, payload JSONB)
      ▲
      │ transaction_id
labels (transaction_id PK/FK, is_fraud, source, labeled_at)
```

Các quyết định thiết kế:

- **Lưu xác suất, không chỉ nhãn**: để dashboard đổi ngưỡng mà không phải chấm điểm lại.
- **Bảng `labels` tách riêng**: ngoài thực tế nhãn thật đến muộn sau giao dịch. Giao dịch chấm qua API chưa có nhãn nên không tính vào precision/recall.
- **`model_versions`**: mỗi lần huấn luyện lại là một phiên bản, mọi dự đoán gắn với phiên bản tạo ra nó, nên so sánh được các lần huấn luyện.
- Câu lệnh SQL đều dùng tham số (`$1`, `$2`, ...) nên không bị SQL injection.

---

## 9. Xử lý lỗi thường gặp

| Triệu chứng | Nguyên nhân và cách xử lý |
|---|---|
| `ECONNREFUSED 127.0.0.1:5432` | PostgreSQL chưa chạy. Windows: mở Services, tìm `postgresql-x64-xx`, Start. |
| `password authentication failed for user "fraud"` | Sai mật khẩu. Sửa `DATABASE_URL` trong `.env`, hoặc dùng `--dsn` với `seed_db.py`. |
| `relation "model_versions" does not exist` | Chưa chạy `npm run db:init`. |
| Dashboard báo "Chưa có model trong DB" | Chưa chạy `python seed_db.py`. |
| `502 Không kết nối được ML service` | Chưa chạy `uvicorn serve:app --port 8000`, hoặc chạy sai thư mục (phải `cd 3_src`). |
| `ModuleNotFoundError: No module named 'config'` | Đang chạy `uvicorn`/`python` ngoài thư mục `3_src`. |
| `409 ML service đang dùng model ... nhưng DB mới nhất là ...` | Đã huấn luyện lại model. Chạy lại `build_artifacts.py` rồi `seed_db.py`. |
| `seed_db.py`: "không tái hiện được đúng tập test" | Khác phiên bản scikit-learn so với lúc chia dữ liệu. Gửi lại thông báo này để xử lý, đừng bỏ qua cảnh báo. |
| `seed_db.py`: "kết quả không khớp model_config.json" | Model trong `4_models/` không khớp `artifacts/`. Chạy lại `build_artifacts.py`. |
| Biểu đồ trống nhưng số liệu có | Trình duyệt không tải được Chart.js từ cdnjs (mất Internet). Bảng và thẻ KPI vẫn dùng được. |
| `EADDRINUSE` (cổng đã dùng) | Đổi `PORT` trong `.env`, hoặc `uvicorn serve:app --port 8001` (nhớ sửa `ML_SERVICE_URL`). |

---

## 10. Tái lập toàn bộ từ đầu

Thứ tự chạy khi làm lại từ dữ liệu thô:

```
1. Tải IEEE-CIS từ Kaggle vào 1_data/raw/
2. 01_merge_and_sample.ipynb         -> 02_sampled.csv
3. 02_EDA.ipynb                      -> biểu đồ, insight
4. 03_cleaning_v2.ipynb              -> 03_cleaned.csv
5. 04_preprocessing_for_models.ipynb -> X_*/y_*.csv
6. 05_modeling.ipynb                 -> nn_final.keras, random_forest_fair.pkl
7. python build_artifacts.py         -> 4_models/artifacts/
8. npm run db:init  +  python seed_db.py
9. uvicorn serve:app  +  npm start
```

Mọi bước ngẫu nhiên đều dùng `random_state = 42` / seed 42.
