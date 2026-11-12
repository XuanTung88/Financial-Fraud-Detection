"""Cấu hình dùng chung: đường dẫn, hằng số, danh sách cột.

Mọi file khác import từ đây, không viết cứng đường dẫn ở nơi khác.
Vị trí file: <ROOT>/3_src/config.py  ->  parents[1] chính là thư mục gốc dự án (Detai).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "1_data" / "processed"
EXPORT_DIR = ROOT / "1_data" / "exports"
MODEL_DIR = ROOT / "4_models"
ARTIFACT_DIR = MODEL_DIR / "artifacts"
FIG_DIR = ROOT / "5_reports" / "figures"

RAW_SAMPLED = DATA_DIR / "02_sampled.csv"   # dữ liệu thô sau khi lấy mẫu
CLEANED = DATA_DIR / "03_cleaned.csv"       # dữ liệu đã làm sạch + label encode

SEED = 42
LABEL = "isFraud"

# ---- Các trường THÔ mà API nhận vào -------------------------------------
RAW_NUMERIC = [
    "TransactionDT", "TransactionAmt", "card1", "addr1", "addr2",
    "dist1", "dist2", "C1", "C2", "C6", "D1", "D4", "D10",
]
RAW_CATEGORICAL = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "M4", "M6",
]

# Cột số được điền median khi thiếu (median tính trên dữ liệu huấn luyện)
MEDIAN_COLS = ["card1", "addr1", "addr2", "dist1", "dist2",
               "C1", "C2", "C6", "D1", "D4", "D10"]

# Chuẩn hóa chữ hoa/thường cho cột categorical (khớp với dữ liệu gốc IEEE-CIS)
CASE_RULES = {"ProductCD": "upper", "card4": "lower", "card6": "lower",
              "M4": "upper", "M6": "upper"}

# Giá trị mã hóa dự phòng khi gặp category chưa từng thấy lúc huấn luyện
CAT_FALLBACK = {"ProductCD": "Unknown", "card4": "Unknown", "card6": "Unknown",
                "M4": "Unknown", "M6": "Unknown",
                "P_emaildomain": "unknown", "R_emaildomain": "unknown"}

# ---- Cột đưa vào one-hot (theo đúng notebook 04) --------------------------
CAT_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "M4", "M6"]

# Cột bị loại do đa cộng tuyến (theo kết quả EDA)
DROP_MULTICOLLINEAR = ["C1", "C2", "C6", "D1", "D4"]

# Cột có phân phối lệch -> log có dấu (chỉ cần cho Neural Network, cây không nhạy)
SKEW_COLS = ["dist1", "dist2", "D10", "D_total", "C_total"]

# Cột tạo đặc trưng tần suất (đếm trên tập TRAIN)
FREQ_COLS = ["card1", "addr1"]
