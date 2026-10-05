"""分析识别阶段的分桶与串行调用次数。

分桶机制的目的是让 ``rec_batch_size`` 不影响输出（CTC 时间步数
= 输入宽度 / 4，必须固定）。代价是 16 个档位会把一批裁剪打散成
十几次串行 ``session.run``——这在 CPU 上划算（算力主导），
在 GPU 上则可能与每次调用的固定开销构成权衡。

配合 ``ab_tiers.py`` 一起看：先看这里统计出的「每档平均张数」，
再决定是否值得动档位。

用法::

    python scripts/perf/bench_buckets.py cuda
"""

from __future__ import annotations

import collections
import sys
import time

import cv2
from _boot import IMAGES, SAMPLES  # noqa: E402  须在 sys.path 调整之后

from noocr.backends import get_backend  # noqa: E402

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"

be = get_backend("ppocrv6-small", device=device)
be.load()

calls: list = []
orig_rec = be._recognize


def spy(crops):
    """复用后端自身的预处理与分桶逻辑，只统计不改动行为。"""
    prepared = [be._prepare_crop(c) for c in crops]
    buckets = be._assign_width_buckets([p.shape[2] for p in prepared])
    dist = collections.Counter(buckets)
    t0 = time.perf_counter()
    r = orig_rec(crops)
    calls.append(
        {
            "n": len(crops),
            "tiers": len(dist),
            "dist": dict(dist),
            "total": (time.perf_counter() - t0) * 1000,
        }
    )
    return r


be._recognize = spy

print(f"{'图片':<28}{'裁剪':>5}{'档位':>5}{'run数':>7}{'rec总ms':>10}{'单run均':>9}")
print("-" * 68)
tot_runs = tot_crops = 0
for name in IMAGES:
    img = cv2.imread(str(SAMPLES / name), cv2.IMREAD_COLOR)
    if img is None:
        continue
    be.recognize_image(img)
    calls.clear()
    be.recognize_image(img)
    if not calls:
        continue
    info = calls[0]
    runs = sum(
        (v + be.rec_batch_size - 1) // be.rec_batch_size for v in info["dist"].values()
    )
    tot_runs += runs
    tot_crops += info["n"]
    print(
        f"{name:<28}{info['n']:>5}{info['tiers']:>5}{runs:>7}"
        f"{info['total']:>10.0f}{info['total'] / max(runs, 1):>9.1f}"
    )

print(f"\n合计: {tot_crops} 张裁剪 → {tot_runs} 次串行 session.run")
print(f"平均每档 {(tot_crops / tot_runs) if tot_runs else 0:.2f} 张"
      f"（rec_batch_size={be.rec_batch_size}）")

print("\n各图档位分布:")
for name in IMAGES:
    img = cv2.imread(str(SAMPLES / name), cv2.IMREAD_COLOR)
    if img is None:
        continue
    be.recognize_image(img)
    calls.clear()
    be.recognize_image(img)
    if calls:
        d = calls[0]["dist"]
        top = ", ".join(f"{k}x{v}" for k, v in sorted(d.items()))
        print(f"  {name:<28}{len(d):>3} 种: {top}")

be.unload()
