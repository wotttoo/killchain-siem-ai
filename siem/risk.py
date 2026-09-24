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
        """Tính risk_score 0-100 từ tập phase tấn công của session."""
        if not attack_phases:
            return 0

        # Điểm nền = phase nguy hiểm nhất (Recon 15, Exploit 50, Install 75, Actions 90)
        base = max(self.kc.PHASE_RISK.get(p, 0) for p in attack_phases)
        # +10 cho mỗi phase thêm vào
        multi_bonus = (len(attack_phases) - 1) * 10

        # Thưởng khi tấn công tiến triển qua nhiều giai đoạn Kill Chain
        ordered = [p for p in self.kc.PHASE_ORDER if p in attack_phases]
        progression_bonus = 0
        if len(ordered) >= 2:
            progression_bonus = 15
        if len(ordered) >= 3:
            progression_bonus = 25
        if len(ordered) >= 4:
            progression_bonus = 35

        # +1 mỗi 10.000 alert, tối đa +5
        volume_bonus = min(5, alert_count // 10000)

        return min(100, base + multi_bonus + progression_bonus + volume_bonus)

    def level(self, score):
        """Đổi điểm số sang mức: CRITICAL ≥ 80, HIGH ≥ 60, MEDIUM ≥ 30, LOW > 0, BENIGN = 0."""
        if score <= 0:
            return "BENIGN"
        for threshold, label in self.LEVELS:
            if score >= threshold:
                return label
        return "BENIGN"

    @staticmethod
    def status(score):
        """ACTIVE = cần xử lý (điểm ≥ 50), MONITORED = chỉ theo dõi."""
        return "ACTIVE" if score >= 50 else "MONITORED"

    @staticmethod
    def recommended_action(score, ip):
        """Hành động đề xuất cho SOC tương ứng với mức rủi ro."""
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
