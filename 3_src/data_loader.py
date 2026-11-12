"""Đọc dữ liệu đã chuẩn bị ở các notebook trước."""
import pandas as pd

from config import DATA_DIR, RAW_SAMPLED


def load_raw_sample():
    """Dữ liệu thô sau khi lấy mẫu (02_sampled.csv), còn nguyên chuỗi chữ."""
    return pd.read_csv(RAW_SAMPLED)


def load_splits():
    """Các tập đã chia (chưa scale, đã one-hot, 47 cột). y là Series."""
    out = {}
    for name in ["train", "val", "test"]:
        out[f"X_{name}"] = pd.read_csv(DATA_DIR / f"X_{name}.csv")
        out[f"y_{name}"] = pd.read_csv(DATA_DIR / f"y_{name}.csv").squeeze("columns")
    return out
