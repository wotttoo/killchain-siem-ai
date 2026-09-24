"""Load + làm sạch alert thô từ các file CSV theo scenario."""

import glob
import os

import pandas as pd

from .killchain import KillChain


class AlertDataLoader:
    """
    Đọc toàn bộ file `*_alerts.txt` trong `data_dir` (mỗi file = 1 scenario),
    gắn cột `scenario`, lọc các dòng có time_label hợp lệ, và tạo nhãn
    ground-truth `gt_phase` theo Kill Chain mapping.
    """

    def __init__(self, data_dir, kill_chain=KillChain):
        self.data_dir = data_dir
        self.kill_chain = kill_chain
        self.n_scenarios = 0

    def _scenario_files(self):
        # Mỗi scenario là 1 file <tên>_alerts.txt (nội dung dạng CSV)
        return sorted(glob.glob(os.path.join(self.data_dir, "*.txt")))

    def load(self, verbose=False):
        """Gộp alert của mọi scenario thành 1 DataFrame đã làm sạch, có cột gt_phase."""
        files = self._scenario_files()
        dfs = []
        for f in files:
            scenario = os.path.basename(f).replace("_alerts.txt", "")
            df = pd.read_csv(f)
            df["scenario"] = scenario
            dfs.append(df)
            if verbose:
                print(f"  {scenario:20s}: {len(df):>8,} rows")

        data = pd.concat(dfs, ignore_index=True)
        # Bỏ dòng header bị lặp lại bên trong file
        data = data[data["time_label"] != "time_label"]
        # Chỉ giữ alert có nhãn nằm trong mapping Kill Chain
        data = data[data["time_label"].isin(self.kill_chain.LABEL_MAP)].copy()
        # Nhãn đúng (ground truth) ở cấp Kill Chain phase — dùng để train và đánh giá
        data["gt_phase"] = data["time_label"].map(self.kill_chain.LABEL_MAP)
        # time là Unix timestamp (giây); dòng thiếu time/ip không gom session được nên bỏ
        data["time"] = pd.to_numeric(data["time"], errors="coerce")
        data = data.dropna(subset=["time", "ip"]).copy()

        self.n_scenarios = len(files)
        return data
