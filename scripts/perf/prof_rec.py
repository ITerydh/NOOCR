"""用 ONNX Runtime 自带 profiler 拆解 rec 模型的算子耗时分布。

回答的问题是「哪些算子吃掉了时间」。本机实测发现PP-OCRv6-rec 的
228 个 Conv 在 ``cudnn_conv_algo_search=DEFAULT`` 下会集体退回 CPU
实现（ORT 日志打印 ``running in Fallback mode``），Conv 占了 92% 的
算子耗时。定位这类问题靠猜是没用的，得看 profiler 的逐算子数据。

注意 profiler 必须在**建session 之前**打开，而 ``InferenceSession``
不暴露 ``sess_options``，所以这里临时 patch 构造函数。

用法::

    python scripts/perf/prof_rec.py cuda
"""

from __future__ import annotations

import collections
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from _boot import ROOT  # noqa: E402  须在 sys.path 调整之后

from noocr.engine import session as S  # noqa: E402

MODEL = ROOT / "models" / "ppocrv6" / "rec" / "PP-OCRv6_rec_small.onnx"

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"

out_dir = Path(tempfile.gettempdir()) / "noocr_prof"
out_dir.mkdir(exist_ok=True)
prefix = str(out_dir / f"rec_{device}")

_orig = S._make_session_options


def patched(*a, **kw):
    opts = _orig(*a, **kw)
    opts.enable_profiling = True
    opts.profile_file_prefix = prefix
    return opts


S._make_session_options = patched

import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(
    str(MODEL),
    patched(dynamic_shape=True, on_gpu=True),
    providers=[("CUDAExecutionProvider", {"device_id": 0})],
)

blob = np.random.rand(6, 3, 48, 320).astype(np.float32)
feed = {sess.get_inputs()[0].name: blob}
outs = [o.name for o in sess.get_outputs()]
for _ in range(3):
    sess.run(outs, feed)
path = sess.end_profiling()
print("profile 文件:", path)

data = json.loads(Path(path).read_text(encoding="utf-8"))
evts = [e for e in data if e.get("cat") == "Node"]
dur = collections.Counter()
cnt = collections.Counter()
for e in evts:
    op = e.get("args", {}).get("op_name", "?")
    cnt[op] += 1
    dur[op] += e.get("dur", 0)

total = sum(dur.values())
print(f"\n算子总数: {len(evts)}   总耗时: {total / 1000:.1f} ms")
print("\n按算子类型 (数量 / 总耗时):")
for op, c in cnt.most_common(12):
    share = dur[op] / total * 100 if total else 0
    print(f"  {op:<24}{c:>4}  {dur[op] / 1000:>8.1f} ms  ({share:.0f}%)")
