# src/components/utilities/live_fps_latency.py

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Any, Optional, Tuple


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _find_value_any_schema(obj: Any, target_key: str) -> Optional[float]:
    """
    Tries to find a numeric value for `target_key` inside unknown JSON schemas.
    Supports:
      - {"arcface": 123, "facenet_camera": 456}
      - {"models": {"arcface": {"fps": 12.3}}}
      - [{"model": "arcface", "fps": 12.3}, ...]
      - {"results": [{"name":"arcface","value": 12.3}, ...]}
    """
    if obj is None:
        return None

    # direct mapping
    if isinstance(obj, dict):
        if target_key in obj and isinstance(obj[target_key], (int, float, str)):
            return _safe_float(obj[target_key])

        # common nested: models -> model_name -> metric
        for k in ("models", "results", "data", "report"):
            if k in obj:
                v = _find_value_any_schema(obj[k], target_key)
                if v is not None:
                    return v

        # brute force: walk dict
        for _, v in obj.items():
            val = _find_value_any_schema(v, target_key)
            if val is not None:
                return val

    # list of entries
    if isinstance(obj, list):
        for item in obj:
            val = _find_value_any_schema(item, target_key)
            if val is not None:
                return val

        # also try list of dicts like [{"model":..., "fps":...}]
        for item in obj:
            if isinstance(item, dict):
                model = item.get("model") or item.get("name") or item.get("model_name")
                if model == target_key:
                    # try common fields
                    for mkey in ("fps", "latency", "latency_ms", "avg_latency_ms", "mean_ms", "value"):
                        if mkey in item:
                            return _safe_float(item[mkey])

    return None


def _read_json(path: Path) -> Optional[Any]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@dataclass
class BenchMetrics:
    fps: Optional[float] = None
    latency_ms: Optional[float] = None


class LiveFpsLatency:
    """
    Rolling LIVE metrics.
    - FPS from frame-to-frame delta.
    - Latency from processing duration for update_frame().
    """
    def __init__(self, fps_window: int = 30, latency_window: int = 30):
        self.fps_hist = deque(maxlen=fps_window)
        self.lat_hist = deque(maxlen=latency_window)
        self._last_t = None  # type: Optional[float]

    def tick_frame(self) -> float:
        now = time.perf_counter()
        if self._last_t is None:
            self._last_t = now
            return 0.0
        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            fps = 1.0 / dt
            self.fps_hist.append(fps)
            return fps
        return 0.0

    def tick_latency(self, latency_ms: float) -> None:
        if latency_ms > 0:
            self.lat_hist.append(latency_ms)

    def mean_fps(self) -> float:
        return float(sum(self.fps_hist) / len(self.fps_hist)) if self.fps_hist else 0.0

    def mean_latency_ms(self) -> float:
        return float(sum(self.lat_hist) / len(self.lat_hist)) if self.lat_hist else 0.0

    def reset(self) -> None:
        self.fps_hist.clear()
        self.lat_hist.clear()
        self._last_t = None


class BenchmarkMetricsProvider:
    """
    Reads benchmark metrics from:
      src/fps_report.json
      src/latency_cpu_report.json
    and returns the numbers for the given model key.
    """
    def __init__(self, src_root: Optional[Path] = None):
        # utilities/ -> components/ -> src/
        self.src_root = src_root or Path(__file__).resolve().parents[2]
        self.fps_path = self.src_root / "fps_report.json"
        self.lat_path = self.src_root / "latency_cpu_report.json"
        self._fps_json = _read_json(self.fps_path)
        self._lat_json = _read_json(self.lat_path)

    def reload(self) -> None:
        self._fps_json = _read_json(self.fps_path)
        self._lat_json = _read_json(self.lat_path)

    def get(self, model_name: str) -> BenchMetrics:
        fps_val = None
        lat_val = None

        # FPS
        if self._fps_json is not None:
            # Try direct: fps_json[model_name]
            fps_val = _find_value_any_schema(self._fps_json, model_name)
            # Some schemas might store actual fps under a nested metric key:
            if fps_val is None and isinstance(self._fps_json, dict):
                # try: {"arcface": {"fps": 12.3}}
                m = self._fps_json.get(model_name)
                if isinstance(m, dict):
                    for k in ("fps", "mean_fps", "avg_fps", "value"):
                        if k in m:
                            fps_val = _safe_float(m[k])
                            break

        # Latency (ms)
        if self._lat_json is not None:
            lat_val = _find_value_any_schema(self._lat_json, model_name)
            if lat_val is None and isinstance(self._lat_json, dict):
                m = self._lat_json.get(model_name)
                if isinstance(m, dict):
                    for k in ("latency_ms", "avg_latency_ms", "mean_ms", "latency", "value"):
                        if k in m:
                            lat_val = _safe_float(m[k])
                            break

        return BenchMetrics(fps=fps_val, latency_ms=lat_val)


def format_metric(v: Optional[float], unit: str = "", nd: int = 1) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{nd}f}{unit}"
