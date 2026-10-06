"""OnnxOCR 侧基准：在同一批示例图上测端到端耗时与文本行数。

必须与 NOOCR 用**同一批图、同一个GPU、同一统计口径**，
否则表格里的数字没有可比性。

输出 JSON 供 noocr 侧的对照脚本读取。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # .../bench
sys.path.insert(0, str(_HERE / "OnnxOCR"))

IMAGES = _HERE / "images"


def _register_dll_dirs() -> None:
    """把 CUDA / cuDNN 的 dll 目录登记进进程。

    Windows 上**只改 PATH 不够**：ONNX Runtime 用 LoadLibrary 加载
    provider 桥接dll 时，走的是进程级搜索路径，而 Python 3.8+
    的 PATH 变更不会自动同步到 DLL 搜索路径，必须显式
    ``os.add_dll_directory``。少了这一步 CUDA provider 会静默
    初始化失败并回退 CPU——OnnxOCR 自身没有这层处理，
    所以它在这台机器上默认只能跑 CPU。
    """
    import os

    here = _HERE / "onnxocr-env" / "Lib" / "site-packages"
    roots = [here / "cudnn", here / "nvidia" / "cudnn" / "bin"]
    roots += sorted((here / "nvidia").glob("*/bin"))
    roots.append(Path(r"C:\Users\iterhui\Desktop\ocr\noocr-gpu\Lib\site-packages\cudnn"))
    seen = set()
    for r in roots:
        if not r.is_dir() or str(r) in seen:
            continue
        seen.add(str(r))
        if os.name == "nt":
            os.add_dll_directory(str(r))
        os.environ["PATH"] = str(r) + os.pathsep + os.environ.get("PATH", "")


_register_dll_dirs()
WARMUP = 2
ROUNDS = 5
#第一个样本单独计为预热，不计入统计——它含CUDA kernel 编译
# 与首次内存分配，把它算进中位数会系统性抬高耗时。

import cv2
from onnxocr.onnx_paddleocr import ONNXPaddleOcr  # noqa: E402


def main() -> int:
    imgs = sorted(IMAGES.glob("*.jpg"))
    if not imgs:
        print(f"没有找到测试图: {IMAGES}")
        return 1

    t0 = time.perf_counter()
    # use_angle_cls=True 才加载方向分类器。OnnxOCR 默认 False，
    # 而 NOOCR 的 180° 纠正默认开启——不开等于让它少做一步，
    # 那样的对比不诚实。
    ocr = ONNXPaddleOcr(use_angle_cls=True)
    boot = time.perf_counter() - t0
    print(f"加载耗时 {boot:.2f}s")

    per_img = {}
    total_lines = 0
    for p in imgs:
        timings = []
        lines = 0
        for i in range(WARMUP + ROUNDS):
            t = time.perf_counter()
            # 解码放在计时内，与 NOOCR 侧一致。
            # 必须是 numpy 数组：pipeline 直接取 img.shape。
            img = cv2.imread(str(p))
            res = ocr.ocr(img, det=True, rec=True, cls=True)
            dt = time.perf_counter() - t
            if i >= WARMUP:
                timings.append(dt)
            if res and res[0]:
                lines = len(res[0])
        med = statistics.median(timings)
        per_img[p.name] = {"median_ms": round(med * 1000, 1),
                           "lines": lines,
                           "min_ms": round(min(timings) * 1000, 1),
                           "max_ms": round(max(timings) * 1000, 1)}
        total_lines += lines
        print(f"{p.name:44s} {med*1000:8.1f} ms  {lines:4d} 行"
              f"  [{min(timings)*1000:.0f}-{max(timings)*1000:.0f}]")

    med_all = statistics.median([v["median_ms"] for v in per_img.values()])
    out = {
        "engine": "OnnxOCR",
        "backend": "ppocrv5",
        "device": "GPU",
        "boot_ms": round(boot * 1000, 1),
        "median_ms": round(med_all, 1),
        "mean_ms": round(statistics.mean(
            [v["median_ms"] for v in per_img.values()]), 1),
        "total_lines": total_lines,
        "images": len(imgs),
        "per_image": per_img,
    }
    dst = Path(__file__).resolve().parent / "bench_onnxocr.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n中位 {med_all:.1f} ms/张共 {total_lines} 行-> {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
