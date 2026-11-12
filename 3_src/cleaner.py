"""Làm sạch giao dịch THÔ (một dòng hoặc nhiều dòng) theo đúng quy tắc lúc huấn luyện."""
import numpy as np
import pandas as pd

from config import RAW_NUMERIC, RAW_CATEGORICAL, MEDIAN_COLS, CASE_RULES
from feature_engineering import group_email


def ensure_raw_columns(df):
    """Đảm bảo đủ cột thô; cột nào API không gửi thì coi như thiếu (NaN)."""
    df = df.copy()
    for c in RAW_NUMERIC + RAW_CATEGORICAL:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _norm_cat(s, mode):
    """Cắt khoảng trắng, đưa về chữ hoa/thường chuẩn; chuỗi rỗng -> NaN."""
    s = s.astype("object")
    mask = s.notna()
    fn = {"lower": str.lower, "upper": str.upper}[mode]
    s[mask] = s[mask].map(lambda x: fn(str(x).strip()))
    return s.replace("", np.nan)


def compute_medians(df):
    """Median từng cột số (bỏ qua giá trị thiếu), dùng để điền thiếu lúc suy luận."""
    return {c: float(pd.to_numeric(df[c], errors="coerce").median()) for c in MEDIAN_COLS}


def clean_raw(df, medians):
    df = ensure_raw_columns(df)

    for c in RAW_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c, m in medians.items():
        df[c] = df[c].fillna(m)

    for c, mode in CASE_RULES.items():
        df[c] = _norm_cat(df[c], mode).fillna("Unknown")

    for c in ["P_emaildomain", "R_emaildomain"]:
        df[c] = df[c].astype("object").fillna("unknown").map(group_email)
    return df
