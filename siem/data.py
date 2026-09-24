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
        return sorted(glob.glob(os.path.join(self.data_dir, "*.txt")))

    def load(self, verbose=False):
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
        data = data[data["time_label"] != "time_label"]
        data = data[data["time_label"].isin(self.kill_chain.LABEL_MAP)].copy()
        data["gt_phase"] = data["time_label"].map(self.kill_chain.LABEL_MAP)
        data["time"] = pd.to_numeric(data["time"], errors="coerce")
        data = data.dropna(subset=["time", "ip"]).copy()

        self.n_scenarios = len(files)
        return data
