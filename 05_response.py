"""
SIEM AI - Response Module (05)
Đọc attack_sessions.json → sinh playbook phản ứng (block IP, case TheHive,
cảnh báo Telegram/Email) cho session đã tiến tới Exploitation trở lên.

Toàn bộ logic sống trong `siem/response.py` (ResponsePlaybook, ResponseExecutor);
script này chỉ orchestrate + in kết quả. Đây LÀ SIMULATION (xem docstring
`siem/response.py`) — không có hạ tầng thật (firewall/TheHive/Telegram) trong
phạm vi đồ án, dataset là log lịch sử nên không có tác động thật lên hệ thống.
"""

import json

from siem import KillChain, ResponsePlaybook, ResponseExecutor


class ResponsePipeline:
    """Orchestrator: đọc session → chạy playbook (mô phỏng) → thống kê → lưu response_actions.json."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.sessions = None
        self.executor = ResponseExecutor(ResponsePlaybook(KillChain))

    def load_sessions(self):
        print("=" * 60)
        print("1. LOAD ATTACK SESSIONS")
        print("=" * 60)
        path = f"{self.output_dir}/attack_sessions.json"
        with open(path, encoding="utf-8") as f:
            self.sessions = json.load(f)
        print(f"Loaded {len(self.sessions):,} sessions từ {path}")
        return self.sessions

    def run_playbooks(self):
        print("\n" + "=" * 60)
        print("2. CHẠY PLAYBOOK (SIMULATED — xem siem/response.py)")
        print("=" * 60)
        eligible = [s for s in self.sessions if s.get("max_phase_pred")]
        print(f"Session có phase tấn công: {len(eligible):,} / {len(self.sessions):,}")
        # Executor tự lọc: chỉ session có phase cao nhất từ Exploitation trở lên mới sinh playbook
        actions = self.executor.run(self.sessions)
        print(f"\nTổng số session trigger response (>= Exploitation): {len(actions):,}")
        return actions

    def print_summary(self):
        print("\n" + "=" * 60)
        print("3. THỐNG KÊ RESPONSE ACTIONS")
        print("=" * 60)
        summary = self.executor.summary()
        for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            if level in summary:
                print(f"  {level:8s}: {summary[level]}")
        if not summary:
            print("  Không có session nào trigger response.")

    def print_example(self):
        if not self.executor.actions:
            return
        print("\n" + "=" * 60)
        print("4. VÍ DỤ 1 RESPONSE ACTION")
        print("=" * 60)
        example = max(self.executor.actions, key=lambda a: a["risk_score"])
        print(json.dumps(example, indent=2, ensure_ascii=False))

    def save(self):
        print("\n" + "=" * 60)
        print("5. LƯU KẾT QUẢ")
        print("=" * 60)
        out_path = f"{self.output_dir}/response_actions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.executor.actions, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {out_path}  ({len(self.executor.actions):,} actions)")

    def run(self):
        """Chạy toàn bộ bước 5 theo thứ tự."""
        self.load_sessions()
        self.run_playbooks()
        self.print_summary()
        self.print_example()
        self.save()
        print("\nDone.")


if __name__ == "__main__":
    pipeline = ResponsePipeline(output_dir="output")
    pipeline.run()
