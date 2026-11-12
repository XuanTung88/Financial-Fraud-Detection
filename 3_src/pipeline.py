"""FraudPipeline: biến một giao dịch THÔ thành đầu vào đúng chuẩn của model.

Thứ tự biến đổi (phải giống hệt lúc huấn luyện):
  thô -> làm sạch -> đặc trưng thời gian/tương tác -> log(Amt) -> mã hóa category
      -> one-hot (47 cột "base")
      -> log có dấu cho cột lệch + đặc trưng tần suất ("tree features", 49 cột) -> RF dùng
      -> StandardScaler                                                       -> NN dùng
"""
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cleaner import clean_raw, compute_medians, ensure_raw_columns
from config import CAT_COLS, CAT_FALLBACK, FREQ_COLS, SKEW_COLS
from feature_engineering import (
    add_interactions, add_time_features, apply_freq_maps, fit_freq_maps, slog,
)

_DUMMY_RE = re.compile(r"^(?P<col>.+)_enc_(?P<k>\d+)$")


class FraudPipeline:
    def __init__(self):
        self.medians = {}        # median từng cột số (điền thiếu)
        self.cat_codes = {}      # {cột: {giá trị chữ: mã số}} (thay cho LabelEncoder)
        self.base_columns = []   # 47 cột sau one-hot, đúng thứ tự lúc train
        self.tree_columns = []   # base + *_freq
        self.freq_maps = {}      # đếm card1/addr1 trên TRAIN
        self.scaler = None       # StandardScaler fit trên tree features của TRAIN

    # ------------------------------------------------------------------ fit
    def fit(self, raw_df, X_train):
        """raw_df: 02_sampled.csv (còn chữ). X_train: X_train.csv (47 cột base)."""
        raw_df = ensure_raw_columns(raw_df)
        self.medians = compute_medians(raw_df)

        cleaned = clean_raw(raw_df, self.medians)
        # sorted() cho cùng thứ tự với LabelEncoder (xếp theo bảng chữ cái)
        self.cat_codes = {
            c: {v: i for i, v in enumerate(sorted(cleaned[c].unique()))} for c in CAT_COLS
        }

        self.base_columns = list(X_train.columns)
        self.freq_maps = fit_freq_maps(X_train, FREQ_COLS)

        tree_train = self.base_to_tree(X_train)
        self.tree_columns = list(tree_train.columns)
        self.scaler = StandardScaler().fit(tree_train)
        return self

    # ----------------------------------------------------------- transforms
    def raw_to_base(self, raw_df):
        """Giao dịch thô -> 47 cột base (chưa log, chưa scale)."""
        df = clean_raw(raw_df, self.medians)
        df = add_time_features(df)
        df = add_interactions(df)
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

        for col in CAT_COLS:
            codes = self.cat_codes[col]
            fallback = codes.get(CAT_FALLBACK[col], -1)
            df[col + "_enc"] = df[col].map(codes).fillna(fallback).astype(int)

        out = {}
        for name in self.base_columns:
            if name in df.columns:                       # cột số thường
                out[name] = df[name].to_numpy()
            else:                                        # cột one-hot: ProductCD_enc_3 ...
                m = _DUMMY_RE.match(name)
                if m is None:
                    raise KeyError(f"Không dựng được cột '{name}' từ dữ liệu thô")
                out[name] = (df[m["col"] + "_enc"] == int(m["k"])).astype(int).to_numpy()
        return pd.DataFrame(out, index=df.index)

    def base_to_tree(self, base_df):
        """47 cột base -> 49 cột (log cột lệch + tần suất). Random Forest dùng bản này."""
        d = slog(base_df, SKEW_COLS)
        return apply_freq_maps(d, self.freq_maps)

    def tree_to_nn(self, tree_df):
        """Chuẩn hóa để đưa vào Neural Network."""
        return self.scaler.transform(tree_df[self.tree_columns]).astype("float32")

    def transform_base(self, base_df):
        tree = self.base_to_tree(base_df)
        return tree, self.tree_to_nn(tree)

    def transform_raw(self, raw_df):
        return self.transform_base(self.raw_to_base(raw_df))

    # ---------------------------------------------------------------- schema
    def schema(self):
        """Mô tả trường đầu vào + giá trị hợp lệ, để Node.js kiểm tra dữ liệu."""
        return {
            "required": ["TransactionDT", "TransactionAmt", "ProductCD", "card1"],
            "categorical_values": {c: list(self.cat_codes[c]) for c in CAT_COLS},
            "median_fill": self.medians,
            "n_base_features": len(self.base_columns),
            "n_model_features": len(self.tree_columns),
        }

    # ------------------------------------------------------------- save/load
    def save(self, path):
        joblib.dump(self.__dict__, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.__dict__.update(joblib.load(path))
        return obj
