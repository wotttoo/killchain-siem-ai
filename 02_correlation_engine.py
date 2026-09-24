"""
SIEM AI - Correlation Engine (02)
Nhóm alert theo IP + time window → attack session → Kill Chain timeline

Toàn bộ logic nghiệp vụ sống trong package `siem/`; script này chỉ
orchestrate theo đúng thứ tự và in kết quả. Xem `siem/model.py` (threshold
tuning cấp alert) và `siem/session.py` (session building + threshold cấp
session) cho chi tiết 2 quyết định threshold đã chốt (README.md).
"""

import json
import warnings

import pandas as pd

from siem import (
    AlertDataLoader, FeatureEngineer, ThresholdedPredictor, ThresholdTuner,
    RiskScorer, SessionBuilder, SessionThresholdTuner, SessionEvaluator, KillChain,
)

warnings.filterwarnings("ignore")


class CorrelationEnginePipeline:
    """Orchestrator: dự đoán alert → tune threshold → gom session → chấm risk → đánh giá → lưu JSON/CSV."""

    def __init__(self, data_dir, output_dir, test_scenarios,
                 session_gap=1800, min_alerts=2,      # 1800 giây = 30 phút im lặng thì cắt session
                 threshold_grid=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
                 min_phase_alerts_grid=(1, 2, 3, 5, 10, 20)):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.test_scenarios = test_scenarios
        self.threshold_grid = list(threshold_grid)
        self.min_phase_alerts_grid = list(min_phase_alerts_grid)

        self.loader = AlertDataLoader(data_dir)
        # Dùng lại encoder + model RF v2 đã train ở bước 01
        self.feature_engineer = FeatureEngineer().load_encoders(output_dir)
        self.predictor = ThresholdedPredictor.from_path(
            f"{output_dir}/model_rf_v2.pkl", FeatureEngineer.FEATURES
        )
        self.session_builder = SessionBuilder(
            session_gap=session_gap, min_alerts=min_alerts,
            risk_scorer=RiskScorer(KillChain),
        )

        self.data = None
        self.best_alert_threshold = None
        self.best_min_phase_alerts = None
        self.sessions = None
        self.df_sess = None

    # ── 1-2. LOAD + FEATURE ENGINEERING ─────────────────────────────
    def load_and_engineer(self):
        print("=" * 60)
        print("1. LOAD DATA")
        print("=" * 60)
        data = self.loader.load()
        print(f"Loaded {len(data):,} alerts từ {self.loader.n_scenarios} scenario")

        print("\n" + "=" * 60)
        print("2. FEATURE ENGINEERING")
        print("=" * 60)
        data = self.feature_engineer.transform(data)
        print(f"Features ready: {FeatureEngineer.FEATURES}")

        self.data = data
        return data

    # ── 3 + 3b. MODEL PREDICTION + ALERT-LEVEL THRESHOLD TUNING ─────
    def tune_alert_threshold(self):
        print("\n" + "=" * 60)
        print("3. MODEL PREDICTION")
        print("=" * 60)
        proba, classes = self.predictor.predict_proba(self.data)
        print(f"Predicted {len(self.data):,} alerts (predict_proba, {len(classes)} classes)")

        print("\n[Default argmax phase distribution — predict() thường]")
        default_pred = classes[proba.argmax(axis=1)]
        print(pd.Series(default_pred).value_counts().to_string())

        print("\n" + "=" * 60)
        print("3b. THRESHOLD TUNING (trên test scenarios: fox, harrison)")
        print("=" * 60)
        # Chỉ tune trên test scenarios (fox, harrison)
        is_test = self.data["scenario"].isin(self.test_scenarios).values
        proba_test = proba[is_test]
        gt_test = self.data.loc[is_test, "gt_phase"].values

        tuner = ThresholdTuner(self.predictor, self.threshold_grid)
        best = tuner.tune(proba_test, classes, gt_test)
        self.best_alert_threshold = best["threshold"]
        print(f"\n[Chosen] threshold={self.best_alert_threshold:.2f} "
              f"(best F1={best['f1']:.3f} on alert-level, test scenarios)")

        # Áp threshold tốt nhất cho toàn bộ alert của 8 scenario
        self.data["pred_phase"] = self.predictor.apply_threshold(
            proba, classes, self.best_alert_threshold
        )
        print("\n[Final predicted phase distribution — sau threshold tuning]")
        print(self.data["pred_phase"].value_counts().to_string())
        return self.best_alert_threshold

    # ── 4 + 4b. BUILD SESSIONS + SESSION-LEVEL THRESHOLD TUNING ─────
    def build_sessions(self):
        print("\n" + "=" * 60)
        print("4. BUILD ATTACK SESSIONS")
        print("=" * 60)
        sessions = self.session_builder.build_all(self.data)
        print(f"\nTổng: {len(sessions):,} sessions")

        print("\n" + "=" * 60)
        print("4b. SESSION-LEVEL THRESHOLD TUNING (MIN_PHASE_ALERTS)")
        print("=" * 60)
        test_sessions_raw = [s for s in sessions if s["scenario"] in self.test_scenarios]
        tuner = SessionThresholdTuner(self.session_builder, self.min_phase_alerts_grid)
        best = tuner.tune(test_sessions_raw)
        self.best_min_phase_alerts = best["min_phase_alerts"]
        print(f"\n[Chosen] min_phase_alerts={self.best_min_phase_alerts} "
              f"(best F1={best['f1']:.3f} on session-level, test scenarios)")

        # Áp MIN_PHASE_ALERTS đã chọn → điền phases_pred, risk_score, risk_level cho mọi session
        self.session_builder.finalize_all(sessions, self.best_min_phase_alerts)
        self.sessions = sessions
        self.df_sess = pd.DataFrame(sessions)
        return sessions

    # ── 5. THỐNG KÊ ───────────────────────────────────────────────────
    def print_statistics(self):
        print("\n" + "=" * 60)
        print("5. THỐNG KÊ SESSIONS")
        print("=" * 60)
        df = self.df_sess
        print("\n[Risk level distribution]")
        print(df["risk_level"].value_counts().to_string())
        print("\n[Max phase reached (predicted)]")
        print(df["max_phase_pred"].value_counts().to_string())
        print("\n[Multi-phase sessions]")
        print(f"  Tổng multi-phase: {int(df['is_multi_phase'].sum())}")
        print(f"  CRITICAL + HIGH:  {len(df[df['risk_level'].isin(['CRITICAL', 'HIGH'])])}")

    # ── 6. TOP 10 ─────────────────────────────────────────────────────
    def print_top_sessions(self, n=10):
        print("\n" + "=" * 60)
        print(f"6. TOP {n} SESSION NGUY HIỂM NHẤT")
        print("=" * 60)
        top = self.df_sess.sort_values("risk_score", ascending=False).head(n)
        for _, row in top.iterrows():
            phases_str = " → ".join(row["phases_pred"]) if row["phases_pred"] else "Benign only"
            print(f"\n  [{row['risk_level']:8s}] score={row['risk_score']:3d}  "
                  f"IP={row['source_ip']:16s}  scenario={row['scenario']}")
            print(f"           Kill Chain: {phases_str}")
            print(f"           Alerts: {row['alert_count']:,}  Duration: {row['duration_min']:.0f} min")
            print(f"           Action: {row['recommended_action']}")

    # ── 7. VÍ DỤ CHI TIẾT MULTI-PHASE ────────────────────────────────
    def print_multi_phase_example(self):
        print("\n" + "=" * 60)
        print("7. VÍ DỤ CHI TIẾT — ATTACK SESSION MULTI-PHASE")
        print("=" * 60)
        multi = self.df_sess[self.df_sess["is_multi_phase"]].sort_values(
            "risk_score", ascending=False
        )
        if len(multi) > 0:
            example = self.sessions[multi.index[0]]
            print(json.dumps(example, indent=2, ensure_ascii=False))
        else:
            print("Không tìm thấy multi-phase session (kiểm tra MIN_ALERTS threshold)")

    # ── 8. ĐÁNH GIÁ ───────────────────────────────────────────────────
    def evaluate(self):
        print("\n" + "=" * 60)
        print("8. ĐÁNH GIÁ CORRELATION ENGINE")
        print("=" * 60)
        df = self.df_sess
        # Session được coi là "tấn công" nếu có ít nhất 1 phase tấn công
        has_pred = df["phases_pred"].apply(len) > 0
        has_gt = df["phases_gt"].apply(len) > 0

        metrics_all = SessionEvaluator.binary_metrics(has_pred, has_gt)
        SessionEvaluator.print_metrics("Session-level detection — TẤT CẢ 8 scenario", metrics_all)

        is_test = df["scenario"].isin(self.test_scenarios)
        metrics_test = SessionEvaluator.binary_metrics(has_pred[is_test], has_gt[is_test])
        SessionEvaluator.print_metrics(
            f"Session-level detection — CHỈ test scenarios: fox, harrison "
            f"(alert_threshold={self.best_alert_threshold:.2f}, "
            f"min_phase_alerts={self.best_min_phase_alerts})",
            metrics_test,
        )

        # Trong các session thật sự đi qua ≥ 2 phase, bao nhiêu session được dự đoán multi-phase
        gt_multi = df[df["phases_gt"].apply(lambda x: len(x) >= 2)]
        if len(gt_multi) > 0:
            recall = gt_multi["is_multi_phase"].sum() / len(gt_multi)
            print("\n[Multi-phase detection]")
            print(f"  GT multi-phase sessions   : {len(gt_multi)}")
            print(f"  Pred multi-phase (đúng)   : {int(gt_multi['is_multi_phase'].sum())}")
            print(f"  Multi-phase recall        : {recall:.3f}")

        return metrics_all, metrics_test

    # ── 9. LƯU KẾT QUẢ ─────────────────────────────────────────────────
    def save(self):
        print("\n" + "=" * 60)
        print("9. LƯU KẾT QUẢ")
        print("=" * 60)

        out_path = f"{self.output_dir}/attack_sessions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.sessions, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {out_path}  ({len(self.sessions):,} sessions)")

        danger = [s for s in self.sessions if s["risk_level"] in ("CRITICAL", "HIGH")]
        danger_path = f"{self.output_dir}/attack_sessions_high_risk.json"
        with open(danger_path, "w", encoding="utf-8") as f:
            json.dump(danger, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {danger_path}  ({len(danger)} HIGH/CRITICAL sessions)")

        # Bản CSV bỏ các cột dạng list để mở được bằng Excel
        summary_path = f"{self.output_dir}/sessions_summary.csv"
        self.df_sess.drop(columns=["kill_chain_progress", "phases_pred", "phases_gt"],
                           errors="ignore").to_csv(summary_path, index=False)
        print(f"[Saved] {summary_path}")

    def run(self):
        """Chạy toàn bộ bước 2 theo thứ tự."""
        self.load_and_engineer()
        self.tune_alert_threshold()
        self.build_sessions()
        self.print_statistics()
        self.print_top_sessions()
        self.print_multi_phase_example()
        self.evaluate()
        self.save()
        print("\nDone.")


if __name__ == "__main__":
    pipeline = CorrelationEnginePipeline(
        data_dir="alert-data-set-main/alerts_csv/alerts_csv",
        output_dir="output",
        test_scenarios=["fox", "harrison"],
    )
    pipeline.run()
