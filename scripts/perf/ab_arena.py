"""det 的 ORT 内存策略 A/B：``dynamic_shape`` 开/ 关的真实代价。

背景
----
:func:`noocr.engine.session._make_session_options` 里有一条硬规则：
``dynamic_shape=True`` 会强制关闭 ``enable_cpu_mem_arena`` 与
``enable_mem_pattern``。这条规则是**为rec 立的**——rec 的输入宽度逐行
变化，arena 在持续变化的形状下要反复重规划，实测会退化到秒级。

但 det 的情况完全不同：短边恒为 736，长边只在 32 的倍数上变动且下界固定，
形状集合是**有限且离散**的。理论上 arena 正擅长这种场景。

PP-OCRv6 的 ``load()`` 目前给det 传了 ``dynamic_shape=True``，等于白扔
arena。这个脚本用真实图片验证「白扔」到底值多少，而不是只靠推理。

方法
----
交替执行 A/B 两组（同一批图、同一session 对、同一线程数），多轮交错
以抵消本机性能漂移，最后报中位数与相对变化。

.. warning::
   A/B 两组必须建**两个独立 session**。ORT 的 arena 是 session 私有的，
   在同一 session 内切换形状再测，测到的是 arena 已预热后的第二次运行，
   会严重高估开启组的收益。

用法::

    python scripts/perf/ab_arena.py                 # CPU，small档
    python scripts/perf/ab_arena.py --tier tiny
    python scripts/perf/ab_arena.py --rounds 7 --device cuda
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from _boot import IMAGES, SAMPLES  # noqa: E402  脚本内导入，需先补 sys.path


def _load_images(names: List[str]) -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []
    for name in names:
        path = SAMPLES / name
        if not path.is_file():
            print(f"  跳过（不存在）: {name}", file=sys.stderr)
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            out.append((name, img))
    return out


def _resize_for_det(image: np.ndarray, short_side: int, align: int, max_long: int):
    """复刻 PP-OCRv6 的检测预处理，保证两组输入形状完全一致。"""
    src_h, src_w = image.shape[:2]
    short = min(src_h, src_w)
    long_side = max(src_h, src_w)
    scale = min(short_side / max(short, 1), 2.0)
    if long_side * scale > max_long:
        scale = max_long / long_side
    new_h = max(align, int(math.ceil(src_h * scale / align)) * align)
    new_w = max(align, int(math.ceil(src_w * scale / align)) * align)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def _build_session(model_path: Path, device: str, dynamic_shape: bool, threads: int):
    """建一个 session，显式指定 dynamic_shape。"""
    from noocr.engine.session import create_session, detect_device

    dev = detect_device(device)
    return create_session(
        model_path,
        dev,
        threads=threads,
        dynamic_shape=dynamic_shape,
        warmup_shape=None,
    )


def _time_group(session, blobs, out_names, in_name) -> float:
    """跑完一组 blob，返回单张平均耗时（毫秒）。"""
    for blob in blobs:  # 预热，不计入
        session.run(out_names, {in_name: blob})
    t0 = time.perf_counter()
    for blob in blobs:
        session.run(out_names, {in_name: blob})
    return (time.perf_counter() - t0) / len(blobs) * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description="det 内存策略 A/B")
    ap.add_argument("--tier", default="small", choices=("tiny", "small"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=5, help="交错轮数，越大越稳")
    args = ap.parse_args()

    from noocr.models import model_path

    mp = Path(model_path(f"ppocrv6-{args.tier}", f"PP-OCRv6_det_{args.tier}.onnx"))
    if not mp.is_file():
        print(f"缺少模型 {mp}\n先执行: python -m noocr models --get ppocrv6-{args.tier}")
        return 1

    images = _load_images(IMAGES)
    if not images:
        print("没有可用示例图", file=sys.stderr)
        return 1

    from noocr.engine.imageops import normalize_db

    blobs = [
        normalize_db(_resize_for_det(img, 736, 32, 4000))[np.newaxis, ...]
        for _, img in images
    ]
    shapes = " ".join(f"{b.shape[2]}x{b.shape[3]}" for b in blobs)
    print(f"模型: {mp.name}")
    print(f"样本: {len(blobs)} 张，输入形状: {shapes}")

    print("建 A(session: dynamic_shape=True → arena 关) ...")
    a_sess = _build_session(mp, args.device, True, args.threads)
    print("建 B(session: dynamic_shape=False → arena 开) ...")
    b_sess = _build_session(mp, args.device, False, args.threads)

    a_in = a_sess.get_inputs()[0].name
    b_in = b_sess.get_inputs()[0].name
    a_out = [o.name for o in a_sess.get_outputs()]
    b_out = [o.name for o in b_sess.get_outputs()]

    a_times: List[float] = []
    b_times: List[float] = []
    for r in range(args.rounds):
        # 每轮都重新按A-B-B-A 顺序跑，抵消单向漂移
        a_times.append(_time_group(a_sess, blobs, a_out, a_in))
        b_times.append(_time_group(b_sess, blobs, b_out, b_in))
        b_times.append(_time_group(b_sess, blobs, b_out, b_in))
        a_times.append(_time_group(a_sess, blobs, a_out, a_in))
        print(f"  轮 {r + 1}: A={a_times[-1]:7.1f}ms  B={b_times[-1]:7.1f}ms")

    a_med = statistics.median(a_times)
    b_med = statistics.median(b_times)
    delta = (b_med - a_med) / a_med * 100

    print("\n" + "=" * 60)
    print(f"A (dynamic_shape=True,arena off) 中位数: {a_med:8.1f} ms")
    print(f"B (dynamic_shape=False,arena on ) 中位数: {b_med:8.1f} ms")
    print(f"B 相对 A: {delta:+.1f}%   （负数表示 B 更快）")
    print(f"A 波动范围: {min(a_times):.1f}~{max(a_times):.1f} ms")
    print(f"B 波动范围: {min(b_times):.1f}~{max(b_times):.1f} ms")

    # 两组波动区间重叠时，任何差异都不可信，必须如实说明而不是硬下结论
    overlap = min(a_times) < max(b_times) and min(b_times) < max(a_times)
    print("\n结论:", end=" ")
    if overlap:
        print("波动区间重叠，差异不显著，建议加 --rounds 重测再定论")
    elif delta < -5:
        print(f"B 快 {abs(delta):.1f}%，建议把 det 的 dynamic_shape 改为 False")
    elif delta > 5:
        print(f"A 反而快 {abs(delta):.1f}%，保持现状")
    else:
        print(f"差异仅 {delta:+.1f}%，收益不足以支撑改动")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
