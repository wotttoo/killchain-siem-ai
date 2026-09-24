"""Domain constants: ánh xạ nhãn tấn công thô sang giai đoạn Cyber Kill Chain."""


class KillChain:
    """
    Namespace các hằng số Kill Chain — không giữ state, chỉ tập hợp mapping
    và các phép tra cứu liên quan để mọi module khác dùng chung một nguồn
    sự thật duy nhất (không định nghĩa lại mapping ở nhiều nơi).
    """

    BENIGN = "Benign"
    RECONNAISSANCE = "Reconnaissance"
    EXPLOITATION = "Exploitation"
    INSTALLATION = "Installation"
    ACTIONS_ON_OBJECTIVES = "Actions_on_Objectives"

    # Thứ tự tiến triển Kill Chain (không gồm Benign)
    PHASE_ORDER = [RECONNAISSANCE, EXPLOITATION, INSTALLATION, ACTIONS_ON_OBJECTIVES]
    CLASS_ORDER = [BENIGN] + PHASE_ORDER

    # time_label (dataset gốc) → Kill Chain phase — QUYẾT ĐỊNH ĐÃ CHỐT, không đổi
    LABEL_MAP = {
        "network_scans":        RECONNAISSANCE,
        "service_scans":        RECONNAISSANCE,
        "wpscan":                RECONNAISSANCE,
        "dirb":                  RECONNAISSANCE,
        "cracking":               EXPLOITATION,
        "privilege_escalation":   EXPLOITATION,
        "webshell":               INSTALLATION,
        "reverse_shell":          INSTALLATION,
        "dnsteal":                ACTIONS_ON_OBJECTIVES,
        "service_stop":           ACTIONS_ON_OBJECTIVES,
        "false_positive":         BENIGN,
    }

    # Điểm rủi ro nền theo phase — dùng bởi RiskScorer
    PHASE_RISK = {
        BENIGN: 0,
        RECONNAISSANCE: 15,
        EXPLOITATION: 50,
        INSTALLATION: 75,
        ACTIONS_ON_OBJECTIVES: 90,
    }

    @classmethod
    def phase_rank(cls, phase):
        """Vị trí của phase trong Kill Chain (số nhỏ = sớm hơn); phase lạ xếp cuối."""
        return cls.PHASE_ORDER.index(phase) if phase in cls.PHASE_ORDER else 99

    @classmethod
    def sort_phases(cls, phases):
        """Sắp xếp 1 tập phase theo đúng thứ tự Kill Chain, trả về list."""
        return sorted(phases, key=cls.phase_rank)

    @classmethod
    def max_phase(cls, phases):
        """Phase cao nhất (nguy hiểm nhất) đạt được trong 1 tập phase, None nếu rỗng."""
        for p in reversed(cls.PHASE_ORDER):
            if p in phases:
                return p
        return None
