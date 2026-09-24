"""
SIEM AI - Kill Chain Detection (01)
EDA + Feature Engineering + Train Random Forest / XGBoost

Toàn bộ logic nghiệp vụ (load data, feature engineering, train model) sống
trong package `siem/`; script này chỉ orchestrate theo đúng thứ tự và in kết
quả — xem `siem/data.py`, `siem/features.py`, `siem/model.py`.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from siem import AlertDataLoader, FeatureEngineer, ModelTrainer, KillChain

warnings.filterwarnings("ignore")


class EDATrainPipeline:
    """Orchestrator: load → feature engineer → split → train RF/XGBoost → plot → save."""

    def __init__(self, data_dir, output_dir, test_scenarios):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.test_scenarios = test_scenarios

        self.loader = AlertDataLoader(data_dir)
        self.feature_engineer = FeatureEngineer()
        self.trainer = ModelTrainer(FeatureEngineer.FEATURES)

        self.data = None
        self.train = None
        self.test = None
        self.y_pred_rf = None
        self.y_pred_xgb = None

    # ── 1-2. LOAD + FEATURE ENGINEERING ─────────────────────────────
    def load_and_engineer(self):
        print("=" * 60)
        print("1. LOAD & CLEAN")
        print("=" * 60)
        data = self.loader.load(verbose=True)
        print(f"\nTotal rows: {len(data):,}")

        print("\n" + "=" * 60)
        print("2. FEATURE ENGINEERING")
        print("=" * 60)
        data = self.feature_engineer.fit_transform(data)
        print(f"Features: {FeatureEngineer.FEATURES}")
        print("\n[Kill Chain phase distribution]")
        print(data["gt_phase"].value_counts().to_string())

        self.data = data
        return data

    # ── 3. TRAIN / TEST SPLIT ───────────────────────────────────────
    def split(self):
        print("\n" + "=" * 60)
        print("3. TRAIN / TEST SPLIT")
        print("=" * 60)
        # Chia theo scenario, KHÔNG chia ngẫu nhiên: IP nội bộ lặp lại trong cùng scenario,
        # chia ngẫu nhiên sẽ cho accuracy ảo ~99%. Test = fox + harrison (chưa từng thấy lúc train).
        self.train = self.data[~self.data["scenario"].isin(self.test_scenarios)]
        self.test = self.data[self.data["scenario"].isin(self.test_scenarios)]
        print(f"Train: {len(self.train):,}  |  Test: {len(self.test):,}")
        print(f"\nClass distribution (train):\n"
              f"{self.train['gt_phase'].value_counts().to_string()}")
        return self.train, self.test

    def _xy(self, split_df):
        """Tách feature (X) và nhãn Kill Chain (y)."""
        return split_df[FeatureEngineer.FEATURES], split_df["gt_phase"]

    # ── 4. RANDOM FOREST ─────────────────────────────────────────────
    def train_random_forest(self):
        print("\n" + "=" * 60)
        print("4. RANDOM FOREST")
        print("=" * 60)
        X_train, y_train = self._xy(self.train)
        X_test, y_test = self._xy(self.test)

        # Đây là dự đoán argmax thường (chưa áp threshold) — threshold được tune ở bước 02
        rf = self.trainer.train_random_forest(X_train, y_train)
        self.y_pred_rf = rf.predict(X_test)

        print("\n[Random Forest — Classification Report]")
        print(classification_report(y_test, self.y_pred_rf, zero_division=0))

        self.feat_importance = pd.Series(
            rf.feature_importances_, index=FeatureEngineer.FEATURES
        ).sort_values(ascending=False)
        print("[Feature Importance]")
        print(self.feat_importance.to_string())
        return rf

    # ── 5. XGBOOST ────────────────────────────────────────────────────
    def train_xgboost(self):
        print("\n" + "=" * 60)
        print("5. XGBOOST")
        print("=" * 60)
        X_train, y_train = self._xy(self.train)
        X_test, y_test = self._xy(self.test)

        self.trainer.train_xgboost(X_train, y_train, X_test, y_test)
        self.y_pred_xgb = self.trainer.predict_xgb(X_test)

        print("\n[XGBoost — Classification Report]")
        print(classification_report(y_test, self.y_pred_xgb, zero_division=0))
        return self.trainer.xgb

    # ── 6. PLOTS ──────────────────────────────────────────────────────
    def plot_results(self):
        print("\n" + "=" * 60)
        print("6. PLOTS")
        print("=" * 60)
        _, y_test = self._xy(self.test)
        classes = sorted(self.data["gt_phase"].unique())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, y_pred, title in [
            (axes[0], self.y_pred_rf, "Random Forest v2"),
            (axes[1], self.y_pred_xgb, "XGBoost v2"),
        ]:
            # normalize="true": mỗi hàng chia cho tổng hàng → đọc được recall của từng lớp
            cm = confusion_matrix(y_test, y_pred, labels=classes, normalize="true")
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/v2_confusion_matrices.png", dpi=150)
        plt.close()
        print(f"[Saved] {self.output_dir}/v2_confusion_matrices.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        self.feat_importance.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title("Random Forest v2 — Feature Importance")
        ax.set_ylabel("Importance")
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/v2_feature_importance.png", dpi=150)
        plt.close()
        print(f"[Saved] {self.output_dir}/v2_feature_importance.png")

    # ── 7. SAVE ───────────────────────────────────────────────────────
    def save(self):
        print("\n" + "=" * 60)
        print("7. SAVING")
        print("=" * 60)
        self.trainer.save(self.output_dir)
        self.feature_engineer.save_encoders(self.output_dir)
        print(f"Models saved to '{self.output_dir}/'")

    def run(self):
        """Chạy toàn bộ bước 1 theo thứ tự."""
        self.load_and_engineer()
        self.split()
        self.train_random_forest()
        self.train_xgboost()
        self.plot_results()
        self.save()
        print("\nDone.")


if __name__ == "__main__":
    pipeline = EDATrainPipeline(
        data_dir="alert-data-set-main/alerts_csv/alerts_csv",
        output_dir="output",
        test_scenarios=["fox", "harrison"],
    )
    pipeline.run()
