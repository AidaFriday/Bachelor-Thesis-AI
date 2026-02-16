import psutil
import os
from collections import deque


class LiveMemoryUsage:
    """
    Tracks process RAM usage and model-specific delta.
    Values are in MB.
    """
    def __init__(self, window: int = 30):
        self.process = psutil.Process(os.getpid())
        self.rss_hist = deque(maxlen=window)
        self.vms_hist = deque(maxlen=window)

        self._baseline_rss = None
        self._baseline_vms = None

    def snapshot_baseline(self):
        """Call BEFORE loading a model"""
        mem = self.process.memory_info()
        self._baseline_rss = mem.rss
        self._baseline_vms = mem.vms

    def tick(self):
        mem = self.process.memory_info()
        self.rss_hist.append(mem.rss)
        self.vms_hist.append(mem.vms)

    def mean_rss_mb(self) -> float:
        if not self.rss_hist:
            return 0.0
        return sum(self.rss_hist) / len(self.rss_hist) / (1024 ** 2)

    def mean_vms_mb(self) -> float:
        if not self.vms_hist:
            return 0.0
        return sum(self.vms_hist) / len(self.vms_hist) / (1024 ** 2)

    def model_rss_delta_mb(self) -> float:
        if self._baseline_rss is None:
            return 0.0
        current = self.process.memory_info().rss
        return (current - self._baseline_rss) / (1024 ** 2)

    def model_vms_delta_mb(self) -> float:
        if self._baseline_vms is None:
            return 0.0
        current = self.process.memory_info().vms
        return (current - self._baseline_vms) / (1024 ** 2)
