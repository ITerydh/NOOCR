"""CPU 与 GPU 的端到端耗时对比，供 README「性能」章节使用。

与 :mod:`bench_noocr` 同一批图、同一口径（含图片解码），只切 ``device``：

- CPU 侧与 GPU 侧各跑**独立进程**——同一进程里先跑 CPU 再跑 GPU 会让
  CPU 线程数配置互相干扰，且长时间连续负载会触发 CPU 降频。
- 每档1 轮预热，测量轮数由命令行给出。

用法::

    python scripts/perf/bench_device.py cpu 3 > device_cpu.json
    python scripts/perf/bench_device.py cuda 3 > device_cuda.json
    python scripts/perf/bench_device_merge.py     # 合成表格
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

IMAGES = _HERE / "images"

WARMUP = 1
REPS = 3
BACKEND = "ppocrv6-small"


def main() -> int:
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else REPS
    backend_name = sys.argv[3] if len(sys.argv) > 3 else BACKEND

    from noocr.backends import get_backend
    from noocr.inputs.loader import load_document

    imgs = sorted(IMAGES.glob("*.jpg"))
    if not imgs:
        print(f"没有找到测试图: {IMAGES}", file=sys.stderr)
        return 1

    be = get_backend(backend_name, device=device)
    be.load()
    per_img = {}
    for p in imgs:
        ts = []
        lines = 0
        for i in range(WARMUP + reps):
            t0 = time.perf_counter()
            doc = load_document(str(p))
            img = next(iter(doc.images))
            pr = be.recognize_image(img, page_index=0)
            dt = (time.perf_counter() - t0) * 1000
            if i >= WARMUP:
                ts.append(dt)
            lines = len(pr.lines)
        per_img[p.stem] = {"avg_ms": round(statistics.mean(ts), 1),
                           "lines": lines}
        print(f"{p.stem:44s} {statistics.mean(ts):9.1f} ms  {lines:4d} 行",
              file=sys.stderr)
    be.unload()

    print(json.dumps({
        "device": device,
        "backend": backend_name,
        "warmup": WARMUP,
        "reps": reps,
        "per_image": per_img,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
