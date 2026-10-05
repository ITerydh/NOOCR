"""验证 cuDNN Conv 调优开关到底有没有生效，并对比开关差异。

Fallback 的成因可能不是"没开穷举"，而是 cuDNN 9 与该模型图里的
卷积参数组合（depthwise / 分组 / 特定 stride）没有可用实现。
本脚本用同一个模型做A/B，对比 Conv 算子的 device 与耗时。
"""

from __future__ import annotations

import collections
import json
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL = str(Path(__file__).resolve().parents[1] / "models/ppocrv6/rec/PP-OCRv6_rec_small.onnx")


def run(tag: str, entries: dict, bs: int = 6, width: int = 320) -> dict:
    out_dir = Path(tempfile.gettempdir()) / "noocr_prof"
    out_dir.mkdir(exist_ok=True)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = str(out_dir / tag)
    for k, v in entries.items():
        opts.add_session_config_entry(k, v)

    import noocr.engine.session as S

    S.ensure_gpu_runtime()
    sess = ort.InferenceSession(
        MODEL, opts, providers=[("CUDAExecutionProvider", {"device_id": 0})]
    )
    blob = np.random.rand(bs, 3, 48, width).astype(np.float32)
    feed = {sess.get_inputs()[0].name: blob}
    outs = [o.name for o in sess.get_outputs()]
    for _ in range(3):
        sess.run(outs, feed)
    path = sess.end_profiling()

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    evts = [e for e in data if e.get("cat") == "Node"]
    by_dev = collections.Counter(e.get("args", {}).get("device", "?") for e in evts)
    dur = collections.Counter()
    for e in evts:
        dur[e.get("args", {}).get("op_name", "?")] += e.get("dur", 0)
    total = sum(dur.values())
    conv_ms = dur.get("Conv", 0) / 1000
    print(f"[{tag}] 算子数={len(evts)} device={dict(by_dev)} "
          f"总耗时={total/1000:.1f}ms Conv={conv_ms:.1f}ms")
    del sess
    return {"total_ms": total / 1000, "conv_ms": conv_ms, "devices": dict(by_dev)}


if __name__ == "__main__":
    base = run("ab_base", {})
    exh = run("ab_exh", {
        "ep.cuda.cudnn_conv_algo_search": "EXHAUSTIVE",
        "ep.cuda.cudnn_conv_use_max_workspace": "1",
    })
    heu = run("ab_heuristic", {"ep.cuda.cudnn_conv_algo_search": "HEURISTIC"})
    print(f"\nConv 对比:默认 {base['conv_ms']:.0f}ms | "
          f"EXHAUSTIVE {exh['conv_ms']:.0f}ms | HEURISTIC {heu['conv_ms']:.0f}ms")
