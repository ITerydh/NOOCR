"""机器性能基线自检。

**为什么需要这个工具**：开发本项目时，同一段OCR 代码在几分钟内测出
551ms / 982ms / 11500ms / 15430ms 四种结果。逐层排查后发现，
同一段与 OCR 完全无关的 numpy 矩阵乘也从 1484ms 涨到 3395ms——
是**机器负载波动**，不是代码问题。

因此在分析任何性能数字之前，先跑本工具确认环境是否稳定。

用法::

    python tests/diag_loadcheck.py          # 短检查
    python tests/diag_loadcheck.py --long   # 持续监测（默认60s）
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


class CpuBaseline:
    """用纯 numpy 矩阵乘探测当前算力。不涉及任何 OCR 代码。"""

    def __init__(self, size: int = 64, reps: int = 120_000):
        self.size = size
        self.reps = reps
        self._a = np.random.rand(size, size).astype(np.float32)

    def run(self) -> float:
        a = self._a
        t = time.perf_counter()
        for _ in range(reps):
            a = a @ a.T / 64.0
        self._a = a
        return (time.perf_counter() - t) * 1000


def _sysinfo() -> str:
    import os

    lines = [f"逻辑核数: {os.cpu_count()}"]
    try:
        import ctypes

        class M(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        m = M()
        m.dwLength = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        lines.append(
            f"内存占用: {m.dwMemoryLoad}%  可用 {m.ullAvailPhys/1e9:.1f}GB / "
            f"{m.ullTotalPhys/1e9:.1f}GB"
        )
    except Exception:
        pass
    try:
        import onnxruntime as ort

        lines.append(f"onnxruntime: {ort.__version__}")
        lines.append(f"可用 EP: {', '.join(ort.get_available_providers())}")
    except Exception:
        pass
    return "\n  ".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="检查机器性能是否稳定")
    ap.add_argument("--long", action="store_true", help="持续监测 60 秒")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    print("机器信息:")
    print("  " + _sysinfo())

    print("\n预热...")
    for _ in range(3):
        CpuBaseline(reps=50_000).run()

    samples: List[float] = []
    rounds = args.rounds if args.long else 5
    print(f"\n采样中（{rounds} 轮，间隔 {args.interval}s）...")
    for i in range(rounds):
        v = min(CpuBaseline().run(), CpuBaseline().run())
        samples.append(v)
        bar = "#" * int(v / max(samples) * 40)
        print(f"  第{i+1:2d}轮 {v:7.0f}ms {bar}")
        if i < rounds - 1:
            time.sleep(args.interval)

    lo, hi = min(samples), max(samples)
    ratio = hi / lo
    cv = statistics.stdev(samples) / statistics.mean(samples) if len(samples) > 1 else 0.0

    print(f"\n最快 {lo:.0f}ms  最慢 {hi:.0f}ms  波动 {ratio:.2f}x  变异系数 {cv:.1%}")
    if ratio <= 1.15:
        print("结论: 机器性能稳定，可以可靠地做性能对比。")
        return 0
    if ratio <= 1.5:
        print("结论: 有轻微波动。性能对比建议用多次取最小值，不要用单次读数。")
        return 0
    print("结论: **机器性能不稳定**。此时任何性能数字都不可信——")
    print("      请用 tests/bench.py 的归一化模式（按同期 CPU 基线折算），")
    print("      或待机器空闲后重测。")
    return 1


if __name__ == "__main__":
    sys.exit(main())