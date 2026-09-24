"""
Response module (mục 4.6 trong spec) — sinh playbook phản ứng tự động cho
attack session đã tiến tới giai đoạn Exploitation trở lên: block IP, tạo
case TheHive, cảnh báo Telegram/Email.

Phạm vi đồ án không có hạ tầng thật (firewall gateway, TheHive server,
Telegram bot) — dataset là log lịch sử (AIT-ADS), không phải traffic sống,
và các IP trong dataset là IP nội bộ lab không tồn tại thật. Vì vậy mọi
hành động ở đây chỉ SIMULATE (sinh payload đúng format API thật + log lại),
không có tác động thật lên hệ thống/mạng. Payload khớp format API thật
(TheHive Alert object, Telegram sendMessage text) nên có thể nối thẳng vào
hạ tầng thật sau này chỉ bằng cách thay thân hàm trong `ResponseExecutor`.
"""

import datetime
from collections import Counter

from .killchain import KillChain


class ResponsePlaybook:
    """Sinh 3 hành động phản ứng cho 1 attack session đã trigger."""

    # Thang severity 1-4 của TheHive
    SEVERITY_MAP = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

    def __init__(self, kill_chain=KillChain):
        self.kc = kill_chain

    def should_trigger(self, session):
        """Theo spec 4.6: trigger khi session đã đi từ Exploitation trở lên."""
        phase = session.get("max_phase_pred")
        if not phase:
            return False
        return self.kc.phase_rank(phase) >= self.kc.phase_rank(self.kc.EXPLOITATION)

    def iptables_command(self, session):
        """Lệnh chặn toàn bộ traffic từ IP nguồn của session (chỉ sinh chuỗi, không chạy)."""
        return f"iptables -A INPUT -s {session['source_ip']} -j DROP"

    def thehive_case(self, session):
        """Payload tạo case/alert trên TheHive, đúng các trường của API thật."""
        phases = (" → ".join(session["phases_pred"]) if session["phases_pred"]
                  else session["max_phase_pred"])
        return {
            "title": f"Attack Session Detected - Kill Chain Escalation ({session['max_phase_pred']})",
            "description": (
                f"Session {session['session_id']} (scenario={session['scenario']}) "
                f"progressed through: {phases}. Duration {session['duration_min']} min, "
                f"{session['alert_count']} alerts."
            ),
            "severity": self.SEVERITY_MAP.get(session["risk_level"], 1),
            "source": "AI-SIEM-Correlation-Engine",
            "sourceRef": session["session_id"],
            "artifacts": [{"dataType": "ip", "data": session["source_ip"]}],
        }

    def notification_text(self, session):
        """Nội dung tin nhắn cảnh báo gửi qua Telegram/Email."""
        phases = " → ".join(session["phases_pred"]) if session["phases_pred"] else "N/A"
        start = datetime.datetime.utcfromtimestamp(session["start_time"]).strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"[{session['risk_level']}] Attack session {session['session_id']}\n"
            f"Scenario: {session['scenario']}  |  Source IP: {session['source_ip']}\n"
            f"Kill Chain: {phases}\n"
            f"Start: {start} UTC  |  Duration: {session['duration_min']} min  |  "
            f"Alerts: {session['alert_count']}\n"
            f"Risk score: {session['risk_score']}/100\n"
            f"Recommended action: {session['recommended_action']}"
        )

    def build(self, session):
        """Gộp 3 hành động phản ứng của 1 session thành 1 bản ghi."""
        return {
            "session_id": session["session_id"],
            "scenario": session["scenario"],
            "risk_level": session["risk_level"],
            "risk_score": session["risk_score"],
            "iptables_command": self.iptables_command(session),
            "thehive_case": self.thehive_case(session),
            "notification_text": self.notification_text(session),
        }


class ResponseExecutor:
    """
    Chạy playbook cho toàn bộ session trigger. Luôn SIMULATE (xem docstring
    module) — không có đường dẫn nào thực thi iptables/gọi API thật trong
    lớp này, tránh việc vô tình sửa firewall thật hoặc gửi cảnh báo thật
    dựa trên IP lab không có ý nghĩa ngoài đời.
    """

    def __init__(self, playbook=None):
        self.playbook = playbook or ResponsePlaybook()
        self.actions = []

    def run(self, sessions, verbose=True):
        """Sinh playbook cho các session đủ điều kiện và đánh dấu simulated=True; không thực thi gì thật."""
        actions = []
        for s in sessions:
            if not self.playbook.should_trigger(s):
                continue
            action = self.playbook.build(s)
            action["simulated"] = True
            action["simulated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            actions.append(action)
            if verbose:
                print(f"  [SIMULATED] {action['risk_level']:8s} {action['session_id']}")
                print(f"      iptables : {action['iptables_command']}")
                print(f"      TheHive  : {action['thehive_case']['title']}")
        self.actions = actions
        return actions

    def summary(self):
        """Đếm số response action theo risk_level."""
        return dict(Counter(a["risk_level"] for a in self.actions))
