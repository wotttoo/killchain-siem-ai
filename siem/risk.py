"""Chấm điểm rủi ro cho 1 attack session dựa trên các phase Kill Chain đạt được."""

from .killchain import KillChain


class RiskScorer:
    """
    risk_score (0-100) = phase nguy hiểm nhất (base) + bonus multi-phase
    + bonus tiến triển đúng thứ tự Kill Chain + bonus volume alert.
    """

    # (ngưỡng risk_score tối thiểu, tên mức) — duyệt theo thứ tự giảm dần
    LEVELS = [(80, "CRITICAL"), (60, "HIGH"), (30, "MEDIUM"), (0, "LOW")]

    def __init__(self, kill_chain=KillChain):
        self.kc = kill_chain

    def score(self, attack_phases, alert_count):
        if not attack_phases:
            return 0

        base = max(self.kc.PHASE_RISK.get(p, 0) for p in attack_phases)
        multi_bonus = (len(attack_phases) - 1) * 10

        ordered = [p for p in self.kc.PHASE_ORDER if p in attack_phases]
        progression_bonus = 0
        if len(ordered) >= 2:
            progression_bonus = 15
        if len(ordered) >= 3:
            progression_bonus = 25
        if len(ordered) >= 4:
            progression_bonus = 35

        volume_bonus = min(5, alert_count // 10000)

        return min(100, base + multi_bonus + progression_bonus + volume_bonus)

    def level(self, score):
        if score <= 0:
            return "BENIGN"
        for threshold, label in self.LEVELS:
            if score >= threshold:
                return label
        return "BENIGN"

    @staticmethod
    def status(score):
        return "ACTIVE" if score >= 50 else "MONITORED"

    @staticmethod
    def recommended_action(score, ip):
        if score >= 80:
            return f"ISOLATE host + BLOCK {ip}"
        if score >= 60:
            return f"BLOCK source IP {ip}"
        if score >= 30:
            return f"MONITOR IP {ip}"
        return "No action required"

    def annotate(self, session, attack_phases):
        """Điền risk_score/risk_level/status/recommended_action trực tiếp vào session dict."""
        score = self.score(attack_phases, session["alert_count"])
        session["risk_score"] = score
        session["risk_level"] = self.level(score)
        session["status"] = self.status(score)
        session["recommended_action"] = self.recommended_action(score, session["source_ip"])
        return session
