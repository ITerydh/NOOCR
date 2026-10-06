"""OnnxOCR 侧基准：在同一批示例图上测端到端耗时与文本行数。

必须与 NOOCR 用**同一批图、同一个 GPU、同一统计口径**，
否则表格里的数字没有可比性。

两侧都从 ``weights_shared/`` 读权重——那个目录里的每个文件都从
OnnxOCR ``ppocrv6`` 分支自带的 ``models/`` 复制而来，复制时校验过
md5。共用一份是必须的：本项目与 OnnxOCR 各自导出的 v6 det 图不同源
（我们的带 ``p2o.pd_op.*`` 图优化痕迹），det 那一半无法证明逐字节相同，
共用一个目录就把这个问题彻底消掉了。

用法::

    python compare_onnxocr_side.py ppocrv5
    python compare_onnxocr_side.py ppocrv6-tiny
    python compare_onnxocr_side.py ppocrv6-small
    python compare_onnxocr_side.py ppocrv6-medium

输出 ``bench_onnxocr_<档位>.json`` 供 :mod:`compare_make_table` 读取。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # .../scripts/perf
sys.path.insert(0, str(_HERE / "OnnxOCR"))

IMAGES = _HERE / "images"

#: **两侧共用的权重根目录**
SHARED = _HERE / "weights_shared"

#: 预热轮数。只留 1 轮——含 CUDA kernel 编译与首次内存分配，
#: 多轮预热会把机器状态波动一并抹平，反而看不出真实抖动。
WARMUP = 1

#: 测量轮数。10 轮取**均值**：单次抖动可到 ±40%（本机实测 v5 首样本
#: 4951ms、末样本 10135ms），中位数会把这个方差藏起来，均值才反映
#: 真实期望耗时。轮数够多，均值自身也稳。
ROUNDS = 10


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

import cv2  # noqa: E402
from onnxocr.onnx_paddleocr import ONNXPaddleOcr  # noqa: E402

#: 档位 -> 共用目录里的 det / rec / 字典
WEIGHTS = {
    "ppocrv5": {
        "det_model_dir": SHARED / "ppocrv5/det/det.onnx",
        "rec_model_dir": SHARED / "ppocrv5/rec/rec.onnx",
        "rec_char_dict_path": SHARED / "ppocrv5/ppocrv5_dict.txt",
    },
    "ppocrv6-tiny": {
        "det_model_dir": SHARED / "ppocrv6/det/PP-OCRv6_det_tiny.onnx",
        "rec_model_dir": SHARED / "ppocrv6/rec/PP-OCRv6_rec_tiny.onnx",
        "rec_char_dict_path": SHARED / "ppocrv6/ppocrv6_tiny_dict.txt",
    },
    "ppocrv6-small": {
        "det_model_dir": SHARED / "ppocrv6/det/PP-OCRv6_det_small.onnx",
        "rec_model_dir": SHARED / "ppocrv6/rec/PP-OCRv6_rec_small.onnx",
        "rec_char_dict_path": SHARED / "ppocrv6/ppocrv6_dict.txt",
    },
    "ppocrv6-medium": {
        "det_model_dir": SHARED / "ppocrv6/det/PP-OCRv6_det_medium.onnx",
        "rec_model_dir": SHARED / "ppocrv6/rec/PP-OCRv6_rec_medium.onnx",
        "rec_char_dict_path": SHARED / "ppocrv6/ppocrv6_dict.txt",
    },
}


def build(backend: str):
    """按后端名构造 OnnxOCR 实例。

    use_angle_cls=True 才加载方向分类器。OnnxOCR 默认 False，
    而 NOOCR 的 180° 纠正默认开启——不开等于让它少做一步，
    那样的对比不诚实。

    OnnxOCR 的 v6 默认档是 medium（``utils.py:275``），要测tiny/small
    必须显式传 ``det_model_dir`` / ``rec_model_dir``。
    """
    kwargs = {"use_angle_cls": True}
    for k, v in WEIGHTS[backend].items():
        if not v.is_file():
            raise SystemExit(
                f"缺少共用权重: {v}\n"
                "见 compare_onnxocr.md 的「准备共用权重目录」一节。"
            )
        kwargs[k] = str(v)
    return ONNXPaddleOcr(**kwargs)


def main() -> int:
    backend = sys.argv[1] if len(sys.argv) > 1 else "ppocrv5"
    if backend not in WEIGHTS:
        raise SystemExit(f"档位须为 {sorted(WEIGHTS)} 之一，收到 {backend!r}")

    imgs = sorted(IMAGES.glob("*.jpg"))
    if not imgs:
        print(f"没有找到测试图: {IMAGES}")
        return 1

    t0 = time.perf_counter()
    ocr = build(backend)
    boot = time.perf_counter() - t0
    print(f"加载耗时 {boot:.2f}s  档位 {backend}  权重 {SHARED}")

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
        "engine": "OnnxOCR",
        "backend": backend,
        "device": "GPU",
        "boot_ms": round(boot * 1000, 1),
        "avg_ms": round(avg_all, 1),
        "total_lines": total_lines,
        "images": len(imgs),
        "weights": str(SHARED),
        "per_image": per_img,
    }
    dst = _HERE / f"bench_onnxocr_{backend.replace('-', '_')}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n均值 {avg_all:.1f} ms/张  共 {total_lines} 行  -> {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
