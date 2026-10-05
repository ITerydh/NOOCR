"""确定宽度档位数的最优取值。

背景：CTC 时间步数 = 输入宽度 / 4，因此**同一张裁剪必须落在固定宽度**
才能保证输出与 batch 大小无关。这正是 ``_WIDTH_TIERS`` 存在的理由。

但档位越密，合批率越高、padding 越少，代价是串行 ``run`` 次数暴涨。
GPU 上每次 run 有固定开销，**档位密度就成了速度与 padding 的权衡**。

结论（本机实测，RTX 4070 Ti SUPER + ppocrv6-small）：

- 稀疏档位总体更快（最多省 35%），但**输出文本与 dense 不一致**，
  说明 padding 宽度确实影响 CTC 输出。因此不能为了速度改档位。
- 保持 dense 档位，改从别处省时间（见 ``bench_device.py``）。

用法::

    python scripts/perf/ab_tiers.py cuda
"""

from __future__ import annotations

import statistics
import sys
import time

import cv2
from _boot import IMAGES, SAMPLES  # noqa: E402  须在 sys.path 调整之后

from noocr.backends import get_backend  # noqa: E402
from noocr.backends.ppocrv6 import PPOCRv6Backend  # noqa: E402

CANDIDATES = {
    "dense16": (32, 48, 64, 96, 128, 160, 200, 256, 320, 416, 544, 704, 896,
                1152, 1472, 1920),
    "medium10": (32, 64, 96, 128, 192, 256, 384, 576, 896, 1472),
    "sparse7": (32, 64, 128, 256, 448, 768, 1472),
    "sparse5": (48, 96, 192, 384, 960),
}


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    original = PPOCRv6Backend._WIDTH_TIERS

    imgs = {}
    for n in IMAGES:
        im = cv2.imread(str(SAMPLES / n), cv2.IMREAD_COLOR)
        if im is not None:
            imgs[n] = im

    print(f"device={device}  图片 {len(imgs)} 张，每张跑 {reps} 次取中位数\n")
    print(f"{'档位集':<12}{'总ms':>9}{'相对':>8}{'行数':>6}{'文本一致':>12}")
    print("-" * 52)

    reference = None
    base = 0.0
    for tag, tiers in CANDIDATES.items():
        PPOCRv6Backend._WIDTH_TIERS = tiers
        be = get_backend("ppocrv6-small", device=device)
        be.load()
        total = 0.0
        texts = {}
        lines = 0
        for name, im in imgs.items():
            be.recognize_image(im)
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                pr = be.recognize_image(im)
                ts.append((time.perf_counter() - t0) * 1000)
            total += statistics.median(ts)
            texts[name] = [ln.text for ln in pr.lines]
            lines += len(pr.lines)
        be.unload()

        if reference is None:
            reference, base = texts, total
            same = "基准"
        else:
            same = "是" if texts == reference else "否（已变化）"
        print(f"{tag:<12}{total:>9.0f}{total / base * 100:>7.0f}%{lines:>6}{same:>12}")

    PPOCRv6Backend._WIDTH_TIERS = original


if __name__ == "__main__":
    main()
