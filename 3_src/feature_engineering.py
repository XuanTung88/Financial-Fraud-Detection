"""Các hàm tạo đặc trưng. Dùng chung cho notebook (huấn luyện) và API (suy luận)."""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Gom nhóm text bẩn (giữ NGUYÊN logic của notebook 03_cleaning_v2)
# ---------------------------------------------------------------------------
def group_email(val):
    d = str(val).lower()
    if d in ["nan", "unknown", "none"]:
        return "unknown"
    for name in ["gmail", "yahoo", "hotmail", "outlook", "icloud", "live", "msn", "anonymous"]:
        if name in d:
            return name
    return "other"


def group_device(val):
    d = str(val).lower()
    if d in ["nan", "unknown", "none"]:
        return "unknown"
    if "windows" in d:
        return "windows"
    if any(x in d for x in ["ios", "iphone", "ipad"]):
        return "ios"
    if any(x in d for x in ["android", "samsung", "sm-", "huawei"]):
        return "android"
    if "mac" in d:
        return "mac"
    if "linux" in d:
        return "linux"
    return "other"


def group_browser(val):
    d = str(val).lower()
    if d in ["nan", "unknown", "none"]:
        return "unknown"
    for name in ["chrome", "firefox", "safari", "edge", "ie"]:
        if name in d:
            return name
    return "other"


# ---------------------------------------------------------------------------
# Đặc trưng thời gian và tương tác
# ---------------------------------------------------------------------------
def add_time_features(df, dt_col="TransactionDT"):
    """TransactionDT (giây) -> giờ trong ngày, thứ trong tuần, cờ ban đêm."""
    df = df.copy()
    df["hour_of_day"] = (df[dt_col] // 3600) % 24
    df["day_of_week"] = (df[dt_col] // 86400) % 7
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)
    return df


def add_interactions(df):
    """C_total = C1+C2+C6, D_total = D1+D4+D10 (cần đủ 6 cột thô)."""
    df = df.copy()
    df["C_total"] = df["C1"] + df["C2"] + df["C6"]
    df["D_total"] = df["D1"] + df["D4"] + df["D10"]
    return df


# ---------------------------------------------------------------------------
# Log có dấu (cho cột lệch, D_total có thể âm)
# ---------------------------------------------------------------------------
def slog(df, cols):
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = np.sign(d[c]) * np.log1p(d[c].abs())
    return d


# ---------------------------------------------------------------------------
# Đặc trưng tần suất: đếm trên TRAIN, áp dụng cho val/test/API (tránh rò rỉ)
# ---------------------------------------------------------------------------
def fit_freq_maps(df_train, cols):
    return {c: df_train[c].value_counts().to_dict() for c in cols}


def apply_freq_maps(df, freq_maps):
    d = df.copy()
    for c, mapping in freq_maps.items():
        d[c + "_freq"] = np.log1p(d[c].map(mapping).fillna(0).astype(float))
    return d


def add_freq(dtr, dva, dte, cols):
    """Giữ nguyên chữ ký cũ của notebook: trả về [train, val, test] đã thêm *_freq."""
    maps = fit_freq_maps(dtr, cols)
    return [apply_freq_maps(d, maps) for d in (dtr, dva, dte)]
