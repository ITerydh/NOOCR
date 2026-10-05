"""CPU / GPU 端到端对比基准。

同一批图、同一后端、同一权重，只切 ``device``，其余条件完全对齐。
先用一把固定的 CPU 参照物测出本轮基准，再把所有耗时除以它，
得到「相对于本机 CPU 参照」的比值——这样跨时段、跨机器才有可比性。

用法::

    python scripts/perf/bench_device.py cpu
    python scripts/perf/bench_device.py cuda
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import cv2
from _boot import IMAGES, SAMPLES  # noqa: E402  须在 sys.path 调整之后

from noocr.backends import get_backend  # noqa: E402


def measure_baseline(backend_name: str) -> float:
    """纯 CPU 跑一张中等负载的图，取中位数作为本轮参照。"""
    probe = cv2.imread(str(SAMPLES / IMAGES[1]), cv2.IMREAD_COLOR)
    be = get_backend(backend_name, device="cpu")
    be.load()
    for _ in range(2):
        be.recognize_image(probe)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        be.recognize_image(probe)
        ts.append((time.perf_counter() - t0) * 1000)
    be.unload()
    return statistics.median(ts)


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    backend_name = "ppocrv6-small"

    base = measure_baseline(backend_name)
    print(f"CPU 参照（{IMAGES[1]}，3 次中位数）: {base:.1f} ms\n")

    be = get_backend(backend_name, device=device)
    t0 = time.perf_counter()
    be.load()
    print(f"device={device}  模型加载 {(time.perf_counter() - t0) * 1000:.0f} ms\n")
    print(f"{'图片':<26}{'总ms':>9}{'归一ms':>9}{'det':>8}{'cls':>8}{'rec':>9}{'行':>5}")
    print("-" * 78)

    rows = []
    for name in IMAGES:
        img = cv2.imread(str(SAMPLES / name), cv2.IMREAD_COLOR)
        if img is None:
            print(f"跳过 {name}（未找到）")
            continue
        be.recognize_image(img)  # 预热，剔除首次 kernel 编译开销
        totals, stages, nlines = [], [], []
        for _ in range(reps):
            pr = be.recognize_image(img)
            totals.append(pr.processing_time * 1000)
            stages.append(pr.debug.get("stage_ms", {}))
            nlines.append(len(pr.lines))

        def med(key: str, data: list = stages) -> float:
            """取某阶段耗时的中位数；缺该阶段返回 0。"""
            vals = [s[key] for s in data if key in s]
            return statistics.median(vals) if vals else 0.0

        tot = statistics.median(totals)
        det = med("det_preprocess") + med("det_infer") + med("det_postprocess")
        cls, rec = med("cls"), med("rec")
        rows.append(
            {
                "image": name,
                "total": tot,
                "norm": tot / base * 1000,
                "det": det,
                "cls": cls,
                "rec": rec,
                "lines": nlines[0],
            }
        )
        print(f"{name:<26}{tot:>9.0f}{tot / base * 1000:>9.0f}"
              f"{det:>8.0f}{cls:>8.0f}{rec:>9.0f}{nlines[0]:>5}")

    be.unload()
    out = f"_bench_{device}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"device": device, "baseline": base, "rows": rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
