"""
Gom alert theo IP + cửa sổ thời gian thành attack session, và tune
MIN_PHASE_ALERTS (số alert tối thiểu/phase để tính là "phát hiện thật").

SessionBuilder tách làm 2 bước có chủ đích:
  - build()    : dựng kill_chain_progress RAW (đầy đủ, chưa lọc nhiễu) — chạy 1 lần.
  - finalize() : áp MIN_PHASE_ALERTS lên progress raw → phases_pred/risk_score/...
                 rẻ vì không cần nhóm lại alert, nên có thể gọi nhiều lần khi sweep
                 threshold (SessionThresholdTuner) mà không phải build lại session.
"""

from .killchain import KillChain
from .risk import RiskScorer


class SessionBuilder:
    def __init__(self, session_gap=1800, min_alerts=2, kill_chain=KillChain, risk_scorer=None):
        self.session_gap = session_gap    # giây — tạo session mới nếu IP im lặng lâu hơn
        self.min_alerts = min_alerts      # bỏ qua session có ít hơn N alert (noise)
        self.kc = kill_chain
        self.risk_scorer = risk_scorer or RiskScorer(kill_chain)

    def build(self, df, scenario_name, pred_col="pred_phase", gt_col="gt_phase"):
        """Trả về list session RAW cho 1 scenario (phases_pred chưa được điền)."""
        sessions = []
        for ip, grp in df.groupby("ip"):
            grp = grp.sort_values("time").reset_index(drop=True)
            t = grp["time"].values
            n = len(grp)

            cuts = [0]
            for i in range(1, n):
                if t[i] - t[i - 1] > self.session_gap:
                    cuts.append(i)
            cuts.append(n)

            for k in range(len(cuts) - 1):
                seg = grp.iloc[cuts[k]:cuts[k + 1]]
                if len(seg) < self.min_alerts:
                    continue
                sessions.append(self._build_one(seg, ip, scenario_name, pred_col, gt_col))
        return sessions

    def build_all(self, df, scenario_col="scenario", pred_col="pred_phase", gt_col="gt_phase",
                  verbose=True):
        """Dựng session cho toàn bộ scenario có trong df, in tiến độ nếu verbose."""
        sessions = []
        for scenario in df[scenario_col].unique():
            sub = df[df[scenario_col] == scenario]
            sess = self.build(sub, scenario, pred_col=pred_col, gt_col=gt_col)
            sessions.extend(sess)
            if verbose:
                print(f"  {scenario:20s}: {len(sess):4d} sessions")
        return sessions

    def _build_one(self, seg, ip, scenario_name, pred_col, gt_col):
        gt_phases = seg[gt_col].unique().tolist()
        attack_phases_gt = set(gt_phases) - {self.kc.BENIGN}

        kc_progress = []
        seen = set()
        for _, row in seg.iterrows():
            phase = row[pred_col]
            if phase != self.kc.BENIGN and phase not in seen:
                seen.add(phase)
                phase_rows = seg[seg[pred_col] == phase]
                kc_progress.append({
                    "phase": phase,
                    "first_seen": int(row["time"]),
                    "alert_count": int(len(phase_rows)),
                    "top_alert": (str(phase_rows["short"].value_counts().index[0])
                                  if len(phase_rows) > 0 else ""),
                })
        kc_progress.sort(key=lambda p: self.kc.phase_rank(p["phase"]))

        t_start = int(seg["time"].min())
        t_end = int(seg["time"].max())

        return {
            "session_id": f"{scenario_name}-{ip}-{t_start}",
            "scenario": scenario_name,
            "source_ip": str(ip),
            "start_time": t_start,
            "end_time": t_end,
            "duration_min": round((t_end - t_start) / 60, 1),
            "alert_count": len(seg),
            "kill_chain_progress": kc_progress,
            "phases_gt": self.kc.sort_phases(attack_phases_gt),
            "max_phase_gt": self.kc.max_phase(attack_phases_gt),
        }

    def finalize(self, session, min_phase_alerts):
        """Áp MIN_PHASE_ALERTS lên kill_chain_progress raw, ghi đè phases_pred/risk/... vào session."""
        kept = [p for p in session["kill_chain_progress"] if p["alert_count"] >= min_phase_alerts]
        attack_phases_pred = {p["phase"] for p in kept}

        session["phases_pred"] = self.kc.sort_phases(attack_phases_pred)
        session["max_phase_pred"] = self.kc.max_phase(attack_phases_pred)
        session["is_multi_phase"] = len(attack_phases_pred) >= 2
        self.risk_scorer.annotate(session, attack_phases_pred)
        return session

    def finalize_all(self, sessions, min_phase_alerts):
        for session in sessions:
            self.finalize(session, min_phase_alerts)
        return sessions


class SessionThresholdTuner:
    """Quét MIN_PHASE_ALERTS trên 1 tập session RAW, chọn giá trị tối ưu F1 (binary attack/benign)."""

    def __init__(self, session_builder, grid):
        self.session_builder = session_builder
        self.grid = grid
        self.results = []
        self.best = None

    @staticmethod
    def _binary_metrics(sessions):
        tp = fp = fn = 0
        for s in sessions:
            has_pred = len(s["phases_pred"]) > 0
            has_gt = len(s["phases_gt"]) > 0
            if has_pred and has_gt:
                tp += 1
            elif has_pred and not has_gt:
                fp += 1
            elif has_gt:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)

    def tune(self, sessions, verbose=True):
        self.results = []
        for m in self.grid:
            # shallow copy: không mutate kill_chain_progress gốc giữa các vòng lặp threshold
            trial = [self.session_builder.finalize(dict(s), m) for s in sessions]
            metrics = self._binary_metrics(trial)
            metrics["min_phase_alerts"] = m
            self.results.append(metrics)
            if verbose:
                print(f"  min_phase_alerts={m:3d}  precision={metrics['precision']:.3f}  "
                      f"recall={metrics['recall']:.3f}  f1={metrics['f1']:.3f}  "
                      f"(TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})")
        self.best = max(self.results, key=lambda r: r["f1"])
        return self.best
