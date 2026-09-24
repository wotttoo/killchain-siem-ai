"""
SIEM AI - Đánh giá End-to-End Pipeline (04)
Alert (raw) → Model (RF v2 + threshold) → Correlation Engine (session) → Kill Chain

Đánh giá toàn bộ pipeline ở 3 mức:
  1. Alert-level      : classification_report cho model (đã áp threshold)
  2. Session-level    : precision/recall/F1 phát hiện attack, phase agreement
  3. Operational value: alert reduction, detection latency

Dùng lại các quyết định threshold đã CHỐT ở 02_correlation_engine.py
(alert threshold=0.75, MIN_PHASE_ALERTS=3) — không tune lại ở đây.
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

from siem import AlertDataLoader, FeatureEngineer, ThresholdedPredictor, SessionEvaluator, KillChain

warnings.filterwarnings("ignore")


class PipelineEvaluator:
    """Orchestrator đánh giá end-to-end: cấp alert → cấp session → theo scenario → giá trị vận hành → latency."""

    def __init__(self, data_dir, output_dir, test_scenarios,
                 alert_threshold=0.75, min_phase_alerts=3):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.test_scenarios = test_scenarios
        self.alert_threshold = alert_threshold
        self.min_phase_alerts = min_phase_alerts

        self.loader = AlertDataLoader(data_dir)
        self.feature_engineer = FeatureEngineer().load_encoders(output_dir)
        self.predictor = ThresholdedPredictor.from_path(
            f"{output_dir}/model_rf_v2.pkl", FeatureEngineer.FEATURES
        )

        self.data = None
        self.sessions = None
        self.df_sess = None
        self.alert_report = None
        self.alert_accuracy = None
        self.session_metrics_all = None
        self.session_metrics_test = None
        self.multi_phase_recall = None
        self.phase_agreement = None
        self.df_scenario = None
        self.latencies = []
        self.escalations = []

    # ── 1-2. LOAD + PREDICT (threshold cố định, không tune lại) ─────
    def load_and_predict(self):
        print("=" * 60)
        print("1. LOAD RAW ALERTS")
        print("=" * 60)
        self.data = self.loader.load()
        print(f"Tổng alert (8 scenario): {len(self.data):,}")

        self.data = self.feature_engineer.transform(self.data)

        print("\n" + "=" * 60)
        print("2. MODEL PREDICTION")
        print("=" * 60)
        proba, classes = self.predictor.predict_proba(self.data)
        pred = self.predictor.apply_threshold(proba, classes, self.alert_threshold)
        # Đếm số alert mà argmax nói "tấn công" nhưng bị threshold hạ về Benign
        argmax_pred = classes[proba.argmax(axis=1)]
        low_conf_count = int(((argmax_pred != "Benign") & (pred == "Benign")).sum())
        self.data["pred_phase"] = pred
        print(f"Áp threshold={self.alert_threshold} → {low_conf_count:,} alert bị fallback về Benign")
        return self.data

    # ── 3. ALERT-LEVEL EVALUATION ────────────────────────────────────
    def evaluate_alert_level(self):
        print("\n" + "=" * 60)
        print("3. ALERT-LEVEL EVALUATION (fox + harrison)")
        print("=" * 60)
        test_mask = self.data["scenario"].isin(self.test_scenarios)
        y_true = self.data.loc[test_mask, "gt_phase"]
        y_pred = self.data.loc[test_mask, "pred_phase"]

        self.alert_accuracy = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy: {self.alert_accuracy:.4f}")
        print("\n[Classification Report]")
        print(classification_report(y_true, y_pred, labels=KillChain.CLASS_ORDER, zero_division=0))
        self.alert_report = classification_report(
            y_true, y_pred, labels=KillChain.CLASS_ORDER, zero_division=0, output_dict=True
        )

        cm = confusion_matrix(y_true, y_pred, labels=KillChain.CLASS_ORDER, normalize="true")
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=KillChain.CLASS_ORDER)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
        ax.set_title(f"Alert-level Confusion Matrix (threshold={self.alert_threshold}, test scenarios)")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/04_alert_confusion_matrix.png", dpi=150)
        plt.close()
        print(f"[Saved] {self.output_dir}/04_alert_confusion_matrix.png")

    # ── 4. LOAD SESSIONS ĐÃ BUILD BỞI 02 ─────────────────────────────
    def load_sessions(self):
        print("\n" + "=" * 60)
        print("4. LOAD SESSIONS")
        print("=" * 60)
        sessions_path = f"{self.output_dir}/attack_sessions.json"
        if not os.path.exists(sessions_path):
            raise SystemExit(
                f"Không tìm thấy {sessions_path}. Chạy `python3 02_correlation_engine.py` trước."
            )
        with open(sessions_path, encoding="utf-8") as f:
            self.sessions = json.load(f)

        self.df_sess = pd.DataFrame(self.sessions)
        self.df_sess["has_pred_attack"] = self.df_sess["phases_pred"].apply(len) > 0
        self.df_sess["has_gt_attack"] = self.df_sess["phases_gt"].apply(len) > 0
        print(f"Loaded {len(self.df_sess):,} sessions")

    # ── 5. SESSION-LEVEL EVALUATION ──────────────────────────────────
    def evaluate_session_level(self):
        print("\n" + "=" * 60)
        print("5. SESSION-LEVEL EVALUATION")
        print("=" * 60)
        df = self.df_sess

        self.session_metrics_all = SessionEvaluator.from_dataframe(df)
        is_test = df["scenario"].isin(self.test_scenarios)
        self.session_metrics_test = SessionEvaluator.from_dataframe(df[is_test])

        SessionEvaluator.print_metrics("Tất cả 8 scenario", self.session_metrics_all)
        SessionEvaluator.print_metrics("Test scenarios (fox, harrison)", self.session_metrics_test)

        gt_multi = df[df["phases_gt"].apply(lambda x: len(x) >= 2)]
        self.multi_phase_recall = (
            gt_multi["is_multi_phase"].sum() / len(gt_multi) if len(gt_multi) > 0 else 0
        )
        print(f"\n[Multi-phase detection] GT multi-phase={len(gt_multi)}  "
              f"Pred đúng multi-phase={int(gt_multi['is_multi_phase'].sum())}  "
              f"Recall={self.multi_phase_recall:.3f}")

        # Phase agreement: trong các session phát hiện đúng, phase cao nhất dự đoán có khớp thực tế không
        tp_sessions = df[df["has_pred_attack"] & df["has_gt_attack"]]
        self.phase_agreement = (
            (tp_sessions["max_phase_pred"] == tp_sessions["max_phase_gt"]).mean()
            if len(tp_sessions) > 0 else 0
        )
        print(f"\n[Phase agreement — trong {len(tp_sessions)} session TP] "
              f"max_phase_pred == max_phase_gt: {self.phase_agreement:.3f}")

        if len(tp_sessions) > 0:
            phase_cm = confusion_matrix(
                tp_sessions["max_phase_gt"], tp_sessions["max_phase_pred"],
                labels=KillChain.PHASE_ORDER,
            )
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=phase_cm, display_labels=KillChain.PHASE_ORDER)
            disp.plot(ax=ax, colorbar=False, cmap="Oranges", values_format="d")
            ax.set_title("Session-level max_phase: GT vs Predicted (TP sessions)")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/04_session_phase_confusion.png", dpi=150)
            plt.close()
            print(f"[Saved] {self.output_dir}/04_session_phase_confusion.png")

    # ── 6. PER-SCENARIO BREAKDOWN ─────────────────────────────────────
    def evaluate_per_scenario(self):
        print("\n" + "=" * 60)
        print("6. PER-SCENARIO BREAKDOWN")
        print("=" * 60)
        rows = []
        for scenario in sorted(self.df_sess["scenario"].unique()):
            sub = self.df_sess[self.df_sess["scenario"] == scenario]
            m = SessionEvaluator.from_dataframe(sub)
            rows.append({
                "scenario": scenario,
                "split": "test" if scenario in self.test_scenarios else "train",
                "sessions": len(sub),
                "gt_attack_sessions": int(sub["has_gt_attack"].sum()),
                "precision": round(m["precision"], 3),
                "recall": round(m["recall"], 3),
                "f1": round(m["f1"], 3),
            })
        self.df_scenario = pd.DataFrame(rows)
        print(self.df_scenario.to_string(index=False))
        path = f"{self.output_dir}/pipeline_eval_per_scenario.csv"
        self.df_scenario.to_csv(path, index=False)
        print(f"\n[Saved] {path}")

    # ── 7. OPERATIONAL VALUE ─────────────────────────────────────────
    def evaluate_operational_value(self):
        print("\n" + "=" * 60)
        print("7. OPERATIONAL VALUE — ALERT REDUCTION")
        print("=" * 60)
        # Hệ thống giảm tải cho SOC bao nhiêu lần so với việc xem từng alert thô
        n_alerts = len(self.data)
        n_sessions = len(self.df_sess)
        n_high_crit = int(self.df_sess["risk_level"].isin(["CRITICAL", "HIGH"]).sum())
        self.reduction_to_sessions = n_alerts / n_sessions if n_sessions else 0
        self.reduction_to_triage = n_alerts / n_high_crit if n_high_crit else 0
        self.n_high_crit = n_high_crit

        print(f"  Tổng alert thô           : {n_alerts:,}")
        print(f"  → Attack sessions        : {n_sessions:,}  (giảm {self.reduction_to_sessions:,.0f} lần)")
        print(f"  → Session HIGH/CRITICAL  : {n_high_crit:,}  "
              f"(giảm {self.reduction_to_triage:,.0f} lần so với alert thô)")
        print(f"  Trung bình alert/session : {n_alerts / n_sessions:,.1f}")

    # ── 8. DETECTION LATENCY ─────────────────────────────────────────
    def evaluate_detection_latency(self):
        print("\n" + "=" * 60)
        print("8. DETECTION LATENCY (session TP)")
        print("=" * 60)
        for s in self.sessions:
            has_pred = len(s["phases_pred"]) > 0
            has_gt = len(s["phases_gt"]) > 0
            if not (has_pred and has_gt) or not s["kill_chain_progress"]:
                continue
            # Latency = từ lúc session bắt đầu tới lúc alert tấn công đầu tiên được model nhận ra
            first_seen_vals = [p["first_seen"] for p in s["kill_chain_progress"]]
            self.latencies.append((min(first_seen_vals) - s["start_time"]) / 60)

            # Với session multi-phase: bao lâu thì tấn công leo tới phase cao nhất
            if s["is_multi_phase"]:
                max_phase_entry = next(
                    (p for p in s["kill_chain_progress"] if p["phase"] == s["max_phase_pred"]),
                    None,
                )
                if max_phase_entry:
                    self.escalations.append((max_phase_entry["first_seen"] - s["start_time"]) / 60)

        if self.latencies:
            lat = np.array(self.latencies)
            print(f"  Time-to-first-detection (n={len(lat)}): "
                  f"median={np.median(lat):.1f} phút, mean={lat.mean():.1f} phút, "
                  f"p90={np.percentile(lat, 90):.1f} phút")
        if self.escalations:
            esc = np.array(self.escalations)
            print(f"  Time-to-max-phase multi-phase (n={len(esc)}): "
                  f"median={np.median(esc):.1f} phút, mean={esc.mean():.1f} phút")

    # ── 9. LƯU METRICS TỔNG HỢP ────────────────────────────────────────
    def save_summary(self):
        print("\n" + "=" * 60)
        print("9. LƯU METRICS TỔNG HỢP")
        print("=" * 60)
        summary = {
            "config": {
                "alert_threshold": self.alert_threshold,
                "min_phase_alerts": self.min_phase_alerts,
                "test_scenarios": self.test_scenarios,
            },
            "alert_level": {
                "accuracy_test": round(self.alert_accuracy, 4),
                "classification_report_test": self.alert_report,
            },
            "session_level": {
                "all_scenarios": self.session_metrics_all,
                "test_scenarios": self.session_metrics_test,
                "multi_phase_recall": round(self.multi_phase_recall, 3),
                "phase_agreement_tp": round(self.phase_agreement, 3),
            },
            "operational": {
                "total_alerts": int(len(self.data)),
                "total_sessions": int(len(self.df_sess)),
                "high_critical_sessions": self.n_high_crit,
                "reduction_alerts_to_sessions": round(self.reduction_to_sessions, 1),
                "reduction_alerts_to_triage": round(self.reduction_to_triage, 1),
            },
            "detection_latency_min": {
                "median_time_to_first_detection": (
                    round(float(np.median(self.latencies)), 1) if self.latencies else None
                ),
                "mean_time_to_first_detection": (
                    round(float(np.mean(self.latencies)), 1) if self.latencies else None
                ),
                "median_time_to_max_phase_multi": (
                    round(float(np.median(self.escalations)), 1) if self.escalations else None
                ),
            },
        }
        path = f"{self.output_dir}/pipeline_eval_metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {path}")

    def run(self):
        """Chạy toàn bộ bước 4 theo thứ tự."""
        self.load_and_predict()
        self.evaluate_alert_level()
        self.load_sessions()
        self.evaluate_session_level()
        self.evaluate_per_scenario()
        self.evaluate_operational_value()
        self.evaluate_detection_latency()
        self.save_summary()
        print("\nDone.")


if __name__ == "__main__":
    evaluator = PipelineEvaluator(
        data_dir="alert-data-set-main/alerts_csv/alerts_csv",
        output_dir="output",
        test_scenarios=["fox", "harrison"],
        alert_threshold=0.75,
        min_phase_alerts=3,
    )
    evaluator.run()
