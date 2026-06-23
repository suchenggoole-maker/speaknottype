"""性能计时工具 - 精确测量流水线各阶段耗时"""

import time
import logging

log = logging.getLogger("SpeakNotType.Timer")


class PerfTimer:
    """轻量级分段计时器，自动累积和汇总"""

    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self._marks: list[tuple[str, float, float]] = []  # (label, start, end)

    def mark(self, label: str) -> float:
        """记录一个时间点（秒），返回自上次标记的间隔"""
        now = time.perf_counter()
        if self._marks:
            _, prev_start, _ = self._marks[-1]
            delta = now - prev_start
        else:
            delta = 0
        self._marks.append((label, now, delta))
        return delta

    def log_summary(self, threshold_ms: float = 1.0, label: str = ""):
        """打印耗时汇总，跳过低于 threshold_ms 的项"""
        if not self._marks:
            return

        total = self._marks[-1][1] - self._marks[0][1]
        header = f"[PERF] {'=' * 40}"
        log.info(header)
        log.info(f"[PERF] {label or self.name} 总耗时: {total:.3f}s ({total*1000:.0f}ms)")

        for i, (label_, _, delta) in enumerate(self._marks):
            pct = (delta / total * 100) if total > 0 else 0
            ms = delta * 1000
            if ms >= threshold_ms:
                log.info(f"[PERF]   ├─ {label_}: {ms:.0f}ms ({pct:.0f}%)")

        log.info(f"[PERF] {'=' * 40}")
        return total