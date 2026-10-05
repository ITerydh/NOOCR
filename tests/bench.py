"""归一化基准测试：本机负载波动很大（实测纯 numpy 负载可慢 2.3 倍），
因此所有 OCR 耗时都必须除以同期测得的 CPU 基线速度，才能横向比较。

用法::

    python bench.py                 # 默认测试图
    python bench.py --repeat 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ CPU 基线


class CpuBaseline:
    """测量本机当前算力，作为其他测量的归一化基准。

    为什么需要：开发本项目时曾把「同一进程内第 5 次调用突然慢 14倍」
    误判为代码退化，逐层排查后才发现是**机器负载波动**——同一段
    与OCR 完全无关的 numpy 矩阵乘也从 1484ms 涨到 3395ms。
    有了这个基线，任何 OCR 耗时都能换算成「机器满速时的等效耗时」。
    """

    def __init__(self, size: int = 64, reps: int = 120_000):
        self.size = size
        self.reps = reps
        self._a = np.random.rand(size, size).astype(np.float32)

    def run(self) -> float:
        a = self._a
        t = time.perf_counter()
        for _ in range(self.reps):
            a = a @ a.T / 64.0
        self._a = a
        return (time.perf_counter() - t) * 1000


def measure(
    fn: Callable[[], object],
    baseline: CpuBaseline,
    repeat: int = 3,
    warmup: int = 1,
) -> Dict[str, float]:
    """测量 ``fn``，并用同期 CPU 基线归一化。

    Returns:
        ``{"raw_ms": 最优原始耗时, "norm_ms": 归一化耗时, "speed": 基线倍率}``
    """
    for _ in range(warmup):
        fn()
    base = min(baseline.run() for _ in range(2))
    times: List[float] = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    raw = min(times)
    return {
        "raw_ms": round(raw, 1),
        "norm_ms": round(raw * base / 1000.0, 1),
        "speed": round(base / 1000.0, 2),
        "all": [round(x, 1) for x in times],
    }


# ------------------------------------------------------------------ OCR 基准


def find_test_image() -> Path:
    for pat in ("*.jpg", "*.png", "*.jpeg"):
        cands = sorted((ROOT / "noocr" / "web" / "static").glob(pat))
        if cands:
            return cands[0]
    raise SystemExit("找不到测试图片")


def run_bench(repeat: int = 3) -> int:
    from noocr.backends.ppocr import PPOCRBackend
    from noocr.engine.imageops import imread

    img_path = find_test_image()
    img = imread(img_path)
    if img is None:
        raise SystemExit(f"无法读取 {img_path}")
    print(f"测试图: {img_path.name}  {img.shape[1]}x{img.shape[0]}")
    print(f"CPU 基线将全程测量并归一化（repeat={repeat}）\n")

    baseline = CpuBaseline()
    print("预热 CPU 基线...")
    baseline.run()

    # ---- 端到端 ----
    print("\n=== 端到端 recognize_image ===")
    print("  bs   原始ms   归一ms   相对bs=6   行数")
    results = {}
    base_norm: Optional[float] = None
    for bs in (1, 2, 4, 6, 8, 12, 24, 32):
        be = PPOCRBackend(device="cpu", rec_batch_size=bs, use_cls=False)
        be.load()
        r = measure(lambda be=be: be.recognize_image(img), baseline, repeat)
        n_lines = len(be.recognize_image(img).lines)
        if bs == 6:
            base_norm = r["norm_ms"]
        results[bs] = r
        rel = f"x{r['norm_ms']/base_norm:.2f}" if base_norm else "-"
        print(f"{bs:4d}  {r['raw_ms']:8.1f}  {r['norm_ms']:8.1f}  {rel:>9}  {n_lines:5d}")

    # ---- 分段 ----
    print("\n=== 分段耗时（bs=6，归一化后）===")
    import noocr.engine.imageops as iops

    be = PPOCRBackend(device="cpu", rec_batch_size=6, use_cls=False)
    be.load()
    resized, shape = iops.resize_keep_ratio(img, be.det_limit, be.det_limit)
    blob = iops.normalize_db(resized)[None, ...]
    prob = be._det_session.run(be._det_out, {be._det_in: blob})[0]
    boxes = be.db(prob, shape)
    order = iops.sort_reading_order(list(boxes))
    crops = [iops.crop_quad(img, boxes[i]) for i in order]

    stages = [
        ("检测预处理", lambda: iops.resize_keep_ratio(img, be.det_limit, be.det_limit)),
        ("DB 归一化", lambda: iops.normalize_db(resized)),
        ("检测推理", lambda: be._det_session.run(be._det_out, {be._det_in: blob})),
        ("DB 后处理", lambda: be.db(prob, shape)),
        ("框排序", lambda: iops.sort_reading_order(list(boxes))),
        ("裁剪", lambda: [iops.crop_quad(img, boxes[i]) for i in order]),
        ("识别(全流程)", lambda: be._recognize(crops)),
    ]
    total = 0.0
    print("  阶段          原始ms   归一ms   占比")
    for name, fn in stages:
        r = measure(fn, baseline, repeat)
        total += r["norm_ms"]
        print(f"  {name:12s}  {r['raw_ms']:8.1f}  {r['norm_ms']:8.1f}  {r['norm_ms']/total*100:5.1f}%")
    print(f"  {'合计':12s}  {'':8s}  {total:8.1f}")

    # ---- 一致性 ----
    print("\n=== 识别一致性（不同 bs 应得到相同文本与顺序）===")
    ref_text: Optional[List[str]] = None
    ref_xy: Optional[List[Tuple[float, float]]] = None
    ok = True
    for bs in (1, 6, 32):
        b2 = PPOCRBackend(device="cpu", rec_batch_size=bs, use_cls=False)
        b2.load()
        page = b2.recognize_image(img)
        texts = [ln.text for ln in page.lines]
        xy = [(round(ln.box.points[0][0]), round(ln.box.points[0][1])) for ln in page.lines]
        if ref_text is None:
            ref_text, ref_xy = texts, xy
            print(f"  bs={bs:3d}: 基准 ({len(texts)} 行)")
        else:
            same_t = texts == ref_text
            same_x = xy == ref_xy
            print(f"  bs={bs:3d}: 文本{'一致' if same_t else '不一致'} / "
                  f"坐标顺序{'一致' if same_x else '不一致'}")
            if not (same_t and same_x):
                ok = False
                for i, (a, b_) in enumerate(zip(ref_text, texts)):
                    if a != b_:
                        print(f"      首个差异 @{i}: bs6={a!r} bs{bs}={b_!r}")
                        break

    print("\n" + "=" * 60)
    print(f"一致性: {'通过 ✓' if ok else '失败 ✗'}")
    print("提示: 归一ms 已按同期 CPU 基线折算，可跨时段比较。")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    sys.exit(run_bench(args.repeat))