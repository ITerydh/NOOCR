"""测各档位的相对速度与加权置信度，供 README「后端对比」表使用。

口径：

- **相对速度**：同一批示例图、同一设备，各档位总耗时之比，
  以 ``ppocrv5`` 为 1.00x 基准。
- **加权置信度**：所有图所有文本行的置信度按字数加权平均。
  不按行数平均——一行 50 个字和一行 1 个字对结果的贡献不同。

**每个档位必须独立进程跑。** 同一进程里连续建 12 个 session 会让显存
吃紧、叠加十几分钟的连续负载还会触发 GPU 热降频，测出来的数字与
真实单档使用场景对不上（本机实测偏差可达 40%）。所以这里只测一个
档位就退出，由外层循环调四次：

用法::

    for t in ppocrv5 ppocrv6-tiny ppocrv6-small ppocrv6-medium; do \\
        python scripts/perf/bench_tiers.py $t cuda; done

环境变量 ``NOOCR_MODELS_DIR`` 可切换权重根目录。
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

#: 相对速度的基准档位
BASE = "ppocrv5"

#: 预热 1 轮、测量 5 轮取中位数——每档独立进程后抖动很小，
#: 中位数足够稳；档位表看的是量级关系，不追求小数点后一位。
WARMUP = 1
REPS = 5


def measure(tier: str, device: str) -> dict:
    """测单个档位，返回总耗时 / 加权置信度 / 文本行数。"""
    from noocr.backends import get_backend
    from noocr.inputs.loader import load_document
    from noocr.models import MODELS_ROOT

    imgs = sorted(IMAGES.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"没有找到测试图: {IMAGES}")

    be = get_backend(tier, device=device)
    be.load()
    total = 0.0
    w_sum = w_conf = 0.0
    lines = 0
    for p in imgs:
        # 每轮重新 load_document：解码计入耗时，与主对比表同口径。
        doc = load_document(str(p))
        img = next(iter(doc.images))
        for _ in range(WARMUP):
            be.recognize_image(img, page_index=0)
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            pr = be.recognize_image(img, page_index=0)
            ts.append((time.perf_counter() - t0) * 1000)
        total += statistics.median(ts)
        for ln in pr.lines:
            n = max(1, len(ln.text))
            w_sum += n
            w_conf += n * ln.confidence
        lines += len(pr.lines)
    be.unload()
    return {
        "tier": tier,
        "device": device,
        "total_ms": round(total, 1),
        "avg_ms": round(total / len(imgs), 1),
        "weighted_confidence": round(w_conf / w_sum, 4) if w_sum else 0.0,
        "lines": lines,
        "weights": str(MODELS_ROOT),
    }


def main() -> int:
    tier = sys.argv[1] if len(sys.argv) > 1 else BASE
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    r = measure(tier, device)
    print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
