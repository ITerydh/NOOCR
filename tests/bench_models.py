"""跨模型/跨档位性能与精度对比（归一化）。

本脚本回答两个问题：

1. **PP-OCRv6 真的比 v5 快吗？**——必须用归一化基线换算，否则本机
   的剧烈负载波动（同一张图先后测出 551ms / 982ms / 11500ms）会让任何
   结论失效。
2. **提速的代价是什么？**——行数、平均置信度是可对比的硬指标。

用法::

    python tests/bench_models.py                 # 全部档位
    python tests/bench_models.py --backends ppocrv5 ppocrv6
    python tests/bench_models.py --images receipt_bank_statement.jpg
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATIC = ROOT / "noocr" / "web" / "static"

DEFAULT_IMAGES = [
    "receipt_bank_statement.jpg",
    "medical_lab_report.jpg",
    "scene_bank_branch.jpg",
    "product_spec_sheet.jpg",
]


class CpuBaseline:
    """测量本机当前算力，作为所有耗时测量的归一化基准。"""

    def __init__(self, size: int = 64, reps: int = 120_000) -> None:
        self.size = size
        self.reps = reps
        self._a = np.random.rand(size, size).astype(np.float32)

    def run(self) -> float:
        a = self._a
        t = time.perf_counter()
        for _ in range(self.reps):
            a = a @ a.T / 64.0
        self._a = a
        return (time.perf_counter() - t) * 1000.0


def build(name: str):
    """按名构造后端实例。"""
    from noocr.backends import get_backend

    return get_backend(name, rec_batch_size=6)


def bench(name: str, image_paths: List[Path], baseline_ms: float, repeat: int):
    """对单个后端跑一组图片，返回聚合结果。"""
    import cv2

    backend = build(name)
    backend.ensure_loaded()

    per_image: List[Dict[str, object]] = []
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        best = None
        lines: List = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            page = backend.recognize_image(img)
            dt = (time.perf_counter() - t0) * 1000.0
            if best is None or dt < best:
                best = dt
                lines = page.lines
        if best is None:
            continue
        confs = [ln.confidence for ln in lines] or [0.0]
        per_image.append(
            {
                "image": p.name,
                "ms": best,
                "norm": best * baseline_ms / 1000.0,
                "lines": len(lines),
                "conf": float(np.mean(confs)),
                "chars": sum(len(ln.text) for ln in lines),
            }
        )
    backend.unload()
    return per_image


def main() -> int:
    ap = argparse.ArgumentParser(description="NOOCR 跨模型对比基准")
    ap.add_argument("--backends", nargs="*", default=None)
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    names = args.backends or ["ppocrv5", "ppocrv6-tiny", "ppocrv6-small"]
    img_names = args.images or DEFAULT_IMAGES
    paths = [STATIC / n for n in img_names]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        print("没有可用测试图")
        return 1

    print("测量本机算力基线...", flush=True)
    bl = CpuBaseline()
    base = min(bl.run() for _ in range(2))
    print(f"  基线 {base:.0f}ms\n")

    all_rows: Dict[str, List[Dict[str, object]]] = {}
    errors: Dict[str, str] = {}
    for name in names:
        try:
            all_rows[name] = bench(name, paths, base, args.repeat)
        except Exception as e:
            errors[name] = str(e).split("\n")[0][:90]

    header = f"{'后端':<18}{'图片':<32}{'原始ms':>10}{'归一ms':>10}{'行数':>7}{'均置信':>9}{'字符':>7}"
    print("=" * 94)
    print(header)
    print("=" * 94)
    for name, rows in all_rows.items():
        for i, r in enumerate(rows):
            label = f"{name:<18}{r['image'] if i == 0 else '':<32}"
            print(
                f"{label}{r['ms']:10.0f}{r['norm']:10.1f}"
                f"{r['lines']:7d}{r['conf']:9.3f}{r['chars']:7d}"
            )
        print("-" * 94)

    for name, msg in errors.items():
        print(f"  [跳过] {name}: {msg}")

    # 汇总
    print("\n" + "=" * 94)
    print(f"{'后端':<18}{'归一合计ms':>12}{'相对v5':>10}{'总行数':>8}{'总字符':>8}{'加权置信':>10}")
    print("=" * 94)
    ref = all_rows.get("ppocrv5")
    ref_norm = sum(r["norm"] for r in ref) if ref else None
    for name, rows in all_rows.items():
        if not rows:
            continue
        tot = sum(r["norm"] for r in rows)
        nlines = sum(r["lines"] for r in rows)
        nchars = sum(r["chars"] for r in rows)
        wconf = (
            sum(r["conf"] * r["lines"] for r in rows) / nlines if nlines else 0.0
        )
        rel = f"{tot / ref_norm:.2f}x" if ref_norm else "-"
        print(f"{name:<18}{tot:12.0f}{rel:>10}{nlines:8d}{nchars:8d}{wconf:10.3f}")
    print("\n注: 归一ms 已除以本机算力基线，可横向比较。")
    print("    「相对v5」<1 表示更快；行数/字符数减少过多说明漏识别。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
