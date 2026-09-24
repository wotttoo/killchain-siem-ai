"""
Model training (Random Forest + XGBoost) và inference có confidence threshold.

ThresholdedPredictor/ThresholdTuner đóng gói quyết định đã chốt ở
02_correlation_engine.py: predict_proba() + threshold=0.75 thay vì
predict() mặc định, để giảm false positive ở cấp alert (xem README.md).
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


class ModelTrainer:
    """Train + lưu Random Forest và XGBoost trên cùng 1 bộ feature (01_eda_train.py)."""

    def __init__(self, features):
        self.features = features
        self.rf = None
        self.xgb = None
        self.label_encoder = None   # chỉ XGBoost cần (yêu cầu label dạng số)

    def train_random_forest(self, X_train, y_train, **overrides):
        """Train RF; class_weight="balanced" tăng trọng số lớp hiếm để bù mất cân bằng dữ liệu."""
        params = dict(
            n_estimators=300, max_depth=20, min_samples_leaf=2,
            n_jobs=-1, random_state=42, class_weight="balanced",
        )
        params.update(overrides)
        self.rf = RandomForestClassifier(**params)
        self.rf.fit(X_train, y_train)
        return self.rf

    def train_xgboost(self, X_train, y_train, X_test=None, y_test=None, **overrides):
        """Train XGBoost; không có class_weight nên cân bằng lớp bằng sample_weight."""
        # XGBoost chỉ nhận nhãn dạng số 0..N-1 → mã hoá tên phase trước khi train
        self.label_encoder = LabelEncoder()
        y_train_enc = self.label_encoder.fit_transform(y_train)
        sample_weight = compute_sample_weight("balanced", y_train)

        params = dict(
            n_estimators=400, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            n_jobs=-1, random_state=42, verbosity=0,
        )
        params.update(overrides)
        self.xgb = XGBClassifier(**params)

        # eval_set chỉ để theo dõi loss trên tập test, không ảnh hưởng tới việc train
        eval_set = None
        if X_test is not None and y_test is not None:
            eval_set = [(X_test, self.label_encoder.transform(y_test))]
        self.xgb.fit(X_train, y_train_enc, sample_weight=sample_weight,
                      eval_set=eval_set, verbose=False)
        return self.xgb

    def predict_xgb(self, X):
        """Dự đoán bằng XGBoost rồi đổi nhãn số về lại tên phase."""
        return self.label_encoder.inverse_transform(self.xgb.predict(X))

    def save(self, output_dir, suffix="v2"):
        """Lưu 2 model + label encoder của XGBoost vào output/."""
        joblib.dump(self.rf, f"{output_dir}/model_rf_{suffix}.pkl")
        joblib.dump(self.xgb, f"{output_dir}/model_xgb_{suffix}.pkl")
        joblib.dump(self.label_encoder, f"{output_dir}/le_label_{suffix}.pkl")


class ThresholdedPredictor:
    """
    Bọc quanh 1 classifier đã train: chỉ chấp nhận nhãn attack (khác Benign)
    khi predict_proba đủ tự tin (>= threshold), ngược lại fallback về Benign.
    """

    def __init__(self, model, features, benign_label="Benign"):
        self.model = model
        self.features = features
        self.benign_label = benign_label

    @classmethod
    def from_path(cls, model_path, features, benign_label="Benign"):
        return cls(joblib.load(model_path), features, benign_label)

    def predict_proba(self, df):
        """Trả về (proba matrix, model.classes_) cho toàn bộ df[self.features]."""
        return self.model.predict_proba(df[self.features]), self.model.classes_

    def apply_threshold(self, proba, classes, threshold):
        """Chọn lớp có xác suất cao nhất, nhưng hạ về Benign nếu là lớp tấn công mà độ tin cậy < threshold."""
        argmax_idx = proba.argmax(axis=1)
        max_proba = proba.max(axis=1)
        pred = classes[argmax_idx].copy()
        low_confidence_attack = (pred != self.benign_label) & (max_proba < threshold)
        pred[low_confidence_attack] = self.benign_label
        return pred

    def predict(self, df, threshold):
        """predict_proba + apply_threshold trong 1 bước."""
        proba, classes = self.predict_proba(df)
        return self.apply_threshold(proba, classes, threshold)


class ThresholdTuner:
    """Quét 1 grid threshold trên (proba, y_true) đã biết nhãn, chọn giá trị tối ưu F1."""

    def __init__(self, predictor, grid):
        self.predictor = predictor
        self.grid = grid
        self.results = []
        self.best = None

    def tune(self, proba, classes, y_true, verbose=True):
        """Thử từng threshold, tính precision/recall/F1 nhị phân (tấn công vs Benign), trả về dòng F1 cao nhất."""
        benign = self.predictor.benign_label
        self.results = []
        for t in self.grid:
            pred = self.predictor.apply_threshold(proba, classes, t)
            tp = int(((pred != benign) & (y_true != benign)).sum())
            fp = int(((pred != benign) & (y_true == benign)).sum())
            fn = int(((pred == benign) & (y_true != benign)).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            row = dict(threshold=t, precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)
            self.results.append(row)
            if verbose:
                print(f"  threshold={t:.2f}  precision={precision:.3f}  recall={recall:.3f}  "
                      f"f1={f1:.3f}  (TP={tp:,} FP={fp:,} FN={fn:,})")
        self.best = max(self.results, key=lambda r: r["f1"])
        return self.best
