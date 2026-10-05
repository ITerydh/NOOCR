"""分阶段性能剖析：用归一化基线定位真正的耗时靶点。

为什么需要它
------------
优化前必须知道**时间花在哪**。本机CPU 性能波动极大（同一张图先后测出
551ms / 982ms / 11500ms / 15430ms），直接比较两次测量的绝对值毫无意义。

本脚本的做法：
1. 先跑一段纯 numpy 负载测出「本机当前算力」``base``
2. 各阶段耗时除以 ``base`` 得到**归一化毫秒**，此后才可横向比较

用法::

    python tests/profile_stages.py                     # 默认三张图
    python tests/profile_stages.py --images a.jpg b.jpg
    python tests/profile_stages.py --repeat 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATIC = ROOT / "noocr" / "web" / "static"

DEFAULT_IMAGES = [
    "receipt_bank_statement.jpg",
    "exam_math_primary.jpg",
    "doc_english_single_column.jpg",
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


class Stopwatch:
    """累计各阶段耗时（原始毫秒）。"""

    def __init__(self) -> None:
        self.stages: Dict[str, float] = {}
        self._t0 = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.stages[name] = self.stages.get(name, 0.0) + (now - self._t0) * 1000.0
        self._t0 = now

    def raw(self) -> float:
        return sum(self.stages.values())


def profile_one(backend, image_path: Path, baseline_ms: float, repeat: int) -> Dict[str, object]:
    """剖析单张图，返回各阶段的原始/归一化耗时。"""
    from noocr.engine.imageops import crop_quad, normalize_db, resize_keep_ratio, sort_reading_order

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"无法读取图片: {image_path}")

    agg: Dict[str, List[float]] = {}

    def push(stage: str, ms: float) -> None:
        agg.setdefault(stage, []).append(ms)

    backend.ensure_loaded()

    for _ in range(repeat):
        sw = Stopwatch()

        # 1) 预处理：等比resize + DB 归一化
        resized, shape = resize_keep_ratio(img, backend.det_limit, backend.det_limit)
        blob = normalize_db(resized)[np.newaxis, ...]
        sw.lap("preprocess")

        # 2) 检测推理
        prob = backend._det_session.run(backend._det_out, {backend._det_in: blob})[0]
        sw.lap("det_infer")

        # 3) DB 后处理（阈值/轮廓/最小外接矩形/多边形外扩）
        det_boxes = backend.db(prob, shape)
        sw.lap("det_post")

        if not det_boxes:
            sw.lap("crop")
            sw.lap("cls")
            sw.lap("rec_prep")
            sw.lap("rec_infer")
            sw.lap("ctc_decode")
            for k, v in sw.stages.items():
                push(k, v)
            continue

        # 4) 排序 + 裁剪
        order = sort_reading_order([b for b in det_boxes])
        det_boxes = [det_boxes[i] for i in order]
        crops = [crop_quad(img, b) for b in det_boxes]
        sw.lap("crop")

# 5) 方向分类
        backend._classify_angles(crops)
        sw.lap("cls")

        # 6) 识别预处理（逐条 resize + 归一化）
        prepared = [backend._prepare_crop(c) for c in crops]
        widths = [p.shape[2] for p in prepared]
        target = backend._assign_width_buckets(widths)
        sw.lap("rec_prep")

        # 7) 识别推理 + 8) CTC 解码
        groups: Dict[int, List[int]] = {}
        for i, w in enumerate(target):
            groups.setdefault(w, []).append(i)

        rec_infer = 0.0
        decode = 0.0
        for max_w in sorted(groups):
            indices = groups[max_w]
            for start in range(0, len(indices), backend.rec_batch_size):
                chunk = indices[start : start + backend.rec_batch_size]
                b = np.full((len(chunk), 3, backend.rec_img_h, max_w), 1.0, np.float32)
                for slot, i in enumerate(chunk):
                    arr = prepared[i]
                    b[slot, :, :, : arr.shape[2]] = arr
                t0 = time.perf_counter()
                preds = backend._rec_session.run(backend._rec_out, {backend._rec_in: b})[0]
                rec_infer += (time.perf_counter() - t0) * 1000.0
                t0 = time.perf_counter()
                backend.decoder(preds)
                decode += (time.perf_counter() - t0) * 1000.0
        push("rec_infer", rec_infer)
        push("ctc_decode", decode)

        for k, v in sw.stages.items():
            push(k, v)

    rows = []
    for stage, vals in agg.items():
        raw = min(vals)
        rows.append(
            {
                "stage": stage,
                "raw": raw,
                "norm": raw * baseline_ms / 1000.0,
            }
        )
    order = {s: i for i, s in enumerate(
        ["preprocess", "det_infer", "det_post", "crop", "cls",
         "rec_prep", "rec_infer", "ctc_decode"]
    )}
    rows.sort(key=lambda r: order.get(r["stage"], 99))
    return {
        "image": image_path.name,
        "shape": f"{img.shape[1]}x{img.shape[0]}",
        "rows": rows,
        "total_raw": sum(r["raw"] for r in rows) / repeat,
        "total_norm": sum(r["norm"] for r in rows) / repeat,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="NOOCR 分阶段性能剖析")
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--det-limit", type=int, default=960)
    ap.add_argument("--batch", type=int, default=6)
    args = ap.parse_args()

    names = args.images or DEFAULT_IMAGES
    paths = [STATIC / n if not Path(n).is_absolute() else Path(n) for n in names]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print("缺少图片: " + ", ".join(str(p) for p in missing))
        return 1

    print("测量本机算力基线...", flush=True)
    bl = CpuBaseline()
    base = min(bl.run() for _ in range(2))
    print(f"  基线: {base:.0f}ms（作为 1.0 归一化单位）\n")

    from noocr.backends.ppocr import PPOCRBackend

    backend = PPOCRBackend(det_limit=args.det_limit, rec_batch_size=args.batch)
    backend.load()

    grand: Dict[str, List[float]] = {}
    results = []
    for p in paths:
        r = profile_one(backend, p, base, args.repeat)
        results.append(r)
        for row in r["rows"]:
            grand.setdefault(row["stage"], []).append(row["norm"])
        print(f"■ {r['image']}  {r['shape']}")
        for row in r["rows"]:
            share = 100.0 * row["norm"] / max(r["total_norm"], 1e-9)
            print(f"    {row['stage']:<12} 原始 {row['raw']:8.1f}ms   归一 {row['norm']:8.1f}ms   {share:5.1f}%")
        print(f"    {'合计':<12} 原始 {r['total_raw']:8.1f}ms   归一 {r['total_norm']:8.1f}ms\n")

    print("=" * 72)
    print("全部图像汇总（归一化 ms，取各图最优）")
    print("=" * 72)
    order = ["preprocess", "det_infer", "det_post", "crop", "cls",
             "rec_prep", "rec_infer", "ctc_decode"]
    tot = 0.0
    for stage in order:
        if stage not in grand:
            continue
        v = min(grand[stage])
        tot += v
        print(f"    {stage:<12} {v:8.1f}ms")
    print(f"    {'——':<12} {'——':>8}")
    print(f"    {'总计':<12} {tot:8.1f}ms")

    print("\n占比排序（优化靶点）:")
    ranked = sorted(
        ((s, min(grand[s])) for s in grand),
        key=lambda kv: -kv[1],
    )
    for stage, v in ranked:
        if tot > 0:
            print(f"    {stage:<12} {v:8.1f}ms  {100.0*v/tot:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
