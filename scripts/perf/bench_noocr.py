"""NOOCR 侧基准：与 compare_onnxocr_side.py 完全同口径。

同图集、同权重目录、同预热轮数、同统计量、同 GPU。唯一差别是引擎本身，
这样表格里的差值才只反映实现差异。

环境变量：

``NOOCR_MODELS_DIR``
    权重根目录。对比测试要指向 ``weights_shared/``，让两侧读同一批文件。
``BENCH_SUFFIX``
    追加到输出文件名，便于区分不同权重来源的多次测量。

用法::

    NOOCR_MODELS_DIR=.../weights_shared python bench_noocr.py ppocrv6-medium cuda
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

IMAGES = _HERE / "images"

#: 预热只留 1 轮，测量 10 轮取均值——与 OnnxOCR 侧完全一致。
#: 单次抖动可到 ±40%，多轮预热会抹平机器状态波动，均值才反映真实期望。
WARMUP = 1
ROUNDS = 10
BACKEND = sys.argv[1] if len(sys.argv) > 1 else "ppocrv6-small"
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cuda"


def main() -> int:
    # 与 CLI 一致的调用路径：load_document 负责读图与解码，
    # recognize_image 是单图识别入口。不用高层封装，
    # 这样测到的是引擎本身而非接口开销。
    from noocr.backends import get_backend
    from noocr.inputs.loader import load_document
    from noocr.models import MODELS_ROOT

    imgs = sorted(IMAGES.glob("*.jpg"))
    if not imgs:
        print(f"没有找到测试图: {IMAGES}")
        return 1

    t0 = time.perf_counter()
    backend = get_backend(BACKEND, device=DEVICE)
    boot = time.perf_counter() - t0
    print(f"加载耗时 {boot:.2f}s  档位 {BACKEND}  设备 {DEVICE}")
    print(f"权重根目录 {MODELS_ROOT}")

    per_img = {}
    total_lines = 0
    for p in imgs:
        timings = []
        lines = 0
        for i in range(WARMUP + ROUNDS):
            t = time.perf_counter()
            doc = load_document(str(p))
            # images 是生成器，只能消费一次；每轮重新 load_document。
            # 解码刻意留在计时内：OnnxOCR 侧同样走 cv2.imread 解码，
            # 两边都含解码才是同口径。只算纯推理反而不真实。
            img = next(iter(doc.images))
            page = backend.recognize_image(img, page_index=0)
            dt = time.perf_counter() - t
            if i >= WARMUP:
                timings.append(dt)
            lines = len(page.lines)
        avg = statistics.mean(timings)
        per_img[p.name] = {"avg_ms": round(avg * 1000, 1),
                           "lines": lines,
                           "min_ms": round(min(timings) * 1000, 1),
                           "max_ms": round(max(timings) * 1000, 1)}
        total_lines += lines
        print(f"{p.name:44s} {avg * 1000:8.1f} ms  {lines:4d} 行"
              f"  [{min(timings) * 1000:.0f}-{max(timings) * 1000:.0f}]")

    avg_all = statistics.mean([v["avg_ms"] for v in per_img.values()])
    out = {
        "engine": "NOOCR",
        "backend": BACKEND,
        "device": DEVICE,
        "boot_ms": round(boot * 1000, 1),
        "avg_ms": round(avg_all, 1),
        "total_lines": total_lines,
        "images": len(imgs),
        # 权重根目录写进结果：换权重来源时必须能从产物里看出来
        # 这份数字是哪套权重跑出来的。
        "weights": str(MODELS_ROOT),
        "per_image": per_img,
    }
    suffix = os.environ.get("BENCH_SUFFIX", "")
    dst = _HERE / f"bench_noocr_{BACKEND.replace('-', '_')}{suffix}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n均值 {avg_all:.1f} ms/张  共 {total_lines} 行  -> {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
