"""
Feature engineering dùng chung giữa train (01) và inference (02, 04).

Quyết định kỹ thuật đã chốt:
  - Features: short_enc, name_enc, detector_enc, alert_cat_enc,
    ip_log_freq, host_log_freq, ip_oct1, ip_oct2
  - KHÔNG dùng hour/day_of_week (data leakage)
  - KHÔNG dùng StandardScaler (RF/XGBoost là cây quyết định)
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    """Tạo 8 feature cho model từ alert thô; lưu/đọc LabelEncoder để train và inference dùng chung mã hoá."""

    # Thứ tự cột đưa vào model — phải giống nhau giữa lúc train và inference
    FEATURES = [
        "short_enc", "name_enc", "detector_enc", "alert_cat_enc",
        "ip_log_freq", "host_log_freq", "ip_oct1", "ip_oct2",
    ]
    # Các cột dạng chuỗi cần LabelEncoder để chuyển thành số
    ENCODED_COLUMNS = ["short", "name", "detector", "alert_cat"]

    def __init__(self):
        self.encoders = {}   # col -> fitted LabelEncoder, điền bởi fit_transform/load_encoders

    @staticmethod
    def _ip_octet(series, n):
        """Lấy octet thứ n của IP (vd n=2: 192.168.1.5 → 168); IP lỗi → 0."""
        return (series.astype(str).str.split(".").str[n - 1]
                .apply(pd.to_numeric, errors="coerce").fillna(0).astype(int))

    def _add_raw_features(self, df):
        """Feature không cần encoder: frequency + IP octet + tách detector/alert_cat từ short."""
        # Đếm số alert của mỗi IP / host TRONG CÙNG scenario.
        # IP tấn công (vd chạy dirb) sinh hàng trăm nghìn alert → tần suất rất cao.
        df["ip_count"]   = df.groupby(["scenario", "ip"])["ip"].transform("count")
        df["host_count"] = df.groupby(["scenario", "host"])["host"].transform("count")
        # log1p nén khoảng giá trị (10 → 2.4, 400.000 → 12.9) để giá trị cực lớn không lấn át
        df["ip_log_freq"]   = np.log1p(df["ip_count"])
        df["host_log_freq"] = np.log1p(df["host_count"])
        # 2 octet đầu của IP ~ subnet nguồn
        df["ip_oct1"] = self._ip_octet(df["ip"], 1)
        df["ip_oct2"] = self._ip_octet(df["ip"], 2)
        # short có dạng "W-Acc-400": ký tự đầu = IDS (W/S/A), phần giữa = loại alert
        df["detector"]  = df["short"].astype(str).str[0]
        df["alert_cat"] = df["short"].astype(str).str.split("-").str[1]
        return df

    def fit_transform(self, df):
        """TRAIN: fit LabelEncoder mới cho mỗi cột categorical trên toàn bộ df truyền vào."""
        df = self._add_raw_features(df)
        for col in self.ENCODED_COLUMNS:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
        return df

    def transform(self, df):
        """INFERENCE: áp encoder đã load; nhãn chưa từng thấy → fallback về class đầu tiên."""
        if not self.encoders:
            raise RuntimeError("Chưa có encoder — gọi load_encoders() hoặc fit_transform() trước.")
        df = self._add_raw_features(df)
        for col in self.ENCODED_COLUMNS:
            le = self.encoders[col]
            known = set(le.classes_)
            safe = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            df[f"{col}_enc"] = le.transform(safe)
        return df

    def save_encoders(self, output_dir, suffix="v2"):
        """Lưu encoder ra output/le_<cột>_v2.pkl để các bước sau mã hoá giống hệt lúc train."""
        for col, le in self.encoders.items():
            joblib.dump(le, f"{output_dir}/le_{col}_{suffix}.pkl")

    def load_encoders(self, output_dir, suffix="v2"):
        """Đọc lại encoder đã lưu bởi 01_eda_train.py; trả về self để gọi nối tiếp."""
        for col in self.ENCODED_COLUMNS:
            self.encoders[col] = joblib.load(f"{output_dir}/le_{col}_{suffix}.pkl")
        return self
