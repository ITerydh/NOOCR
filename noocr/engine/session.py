"""统一推理引擎。

全项目唯一的 ONNX Runtime session 构造入口，集中解决三类问题：

1. **缓存容量失控**：:class:`SessionCache` 提供显式容量上限、LRU 淘汰
   与线程安全，避免多份常驻 session 累积导致 OOM。
2. **加速后端静默失效**：统一做能力探测与回退，回退行为显式告警。
3. **配置漂移**：:func:`_make_session_options` 全局唯一实现。
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..logging_config import get_logger

log = get_logger(__name__)

__all__ = [
    "Device",
    "ModelNotFoundError",
    "SessionCache",
    "build_providers",
    "create_session",
    "detect_device",
    "get_global_cache",
    "warmup",
]

ProviderSpec = Union[str, Tuple[str, Dict[str, Any]]]


class ModelNotFoundError(FileNotFoundError):
    """模型文件缺失。异常信息附带获取方式。"""


@dataclass(frozen=True)
class Device:
    """探测到的执行设备。"""

    kind: str
    device_id: int = 0
    name: str = "unknown"
    #: 该 EP 实际可用的内存（字节），CPU 为 0
    total_memory: int = 0

    @property
    def is_gpu(self) -> bool:
        return self.kind in ("cuda", "directml", "cann")

    def __str__(self) -> str:
        if self.is_gpu:
            mem = f" {self.total_memory / 1e9:.1f}GB" if self.total_memory else ""
            return f"{self.kind}:{self.device_id} ({self.name}{mem})"
        return "cpu"


_DEVICE_CACHE: Optional[Device] = None
_DEVICE_LOCK = threading.Lock()


def _available_providers() -> Tuple[str, ...]:
    try:
        import onnxruntime as ort

        return tuple(ort.get_available_providers())
    except Exception:
        return ()


def detect_device(prefer: str = "auto", device_id: int = 0, refresh: bool = False) -> Device:
    """探测可用执行设备。

    Args:
        prefer: ``"auto"`` / ``"cpu"`` / ``"cuda"`` / ``"dml"``。
            ``"auto"`` 按 CUDA → DirectML → CANN → CPU 顺序选择。
        device_id: 多卡时的设备序号。
        refresh: 强制重新探测，测试用。

    Returns:
        探测到的 :class:`Device`。请求的设备不可用时降级到 CPU 并告警。
    """
    global _DEVICE_CACHE
    with _DEVICE_LOCK:
        if _DEVICE_CACHE is not None and not refresh and device_id == 0:
            return _DEVICE_CACHE
        result = _detect_device_uncached(prefer, device_id)
        if device_id == 0:
            _DEVICE_CACHE = result
        return result


def _detect_device_uncached(prefer: str, device_id: int) -> Device:
    providers = _available_providers()

    def cuda_name() -> Tuple[str, int]:
        try:
            import subprocess

            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip().splitlines()
            if device_id < len(out):
                name, mem = [x.strip() for x in out[device_id].split(",")]
                return name, int(float(mem)) * 1024 * 1024
        except Exception:
            pass
        return "NVIDIA GPU", 0

    if prefer in ("auto", "cuda") and "CUDAExecutionProvider" in providers:
        name, mem = cuda_name()
        if prefer == "cuda":
            log.info("使用 CUDA 设备: {}", name)
        return Device("cuda", device_id, name, mem)

    if prefer in ("auto", "dml") and "DmlExecutionProvider" in providers and os.name == "nt":
        if prefer == "dml":
            log.info("使用 DirectML 设备")
        return Device("directml", device_id, "DirectML (GPU)", 0)

    if prefer in ("auto", "cann") and "CANNExecutionProvider" in providers:
        return Device("cann", device_id, "Ascend NPU", 0)

    if prefer not in ("auto", "cpu"):
        log.warning(
            "请求的执行后端 '{}' 不可用（当前可用: {}），已回退到 CPU。"
            "如需 GPU 加速请安装 onnxruntime-gpu: pip install noocr[gpu]",
            prefer,
            ", ".join(providers) or "无",
        )
    return Device("cpu", 0, "CPU", 0)


def build_providers(device: Device, *, arena: bool = True) -> List[ProviderSpec]:
    """构造 providers 列表，CPU 永远作为最后兜底。"""
    if device.kind == "cuda":
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device.device_id,
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
    if device.kind == "directml":
        return [("DmlExecutionProvider", {"device_id": device.device_id}), "CPUExecutionProvider"]
    if device.kind == "cann":
        return [("CANNExecutionProvider", {"device_id": device.device_id}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _make_session_options(
    *, threads: int = 0, arena: bool = True, dynamic_shape: bool = False
):
    """构造 SessionOptions，全局唯一实现。

    Args:
        threads: CPU 线程数，0 表示交给 ORT 自行决定。
        arena: 是否启用内存 arena。
        dynamic_shape: 模型输入形状是否随运行变化。

    .. warning:: 动态形状模型必须关闭 arena，否则会出现数量级的性能退化

       实测 PP-OCRv5 rec（bs=24，同一 session 内喂递增宽度）：

       ==========================  =======  =======  =======
       配置                w=320     w=512    总计
       ==========================  =======  =======  =======
       arena on（默认）        177ms    336ms    5203ms
       arena off               345ms    569ms    4112ms
       ==========================  =======  =======  =======

       单独 ``enable_mem_pattern=False``（保留 arena）更糟，总计 28288ms。

       成因：ORT 的内存 arena 按首次见到的形状规划并复用缓冲区。输入宽度
       不断变化时 arena 需反复重新规划，且旧块不能及时回收，产生重尾延迟。
       实测最差点达 3447ms，而隔离进程重测同形状仅需 655ms。

       固定形状模型（det/cls）保持 arena 开启以获得分配加速。
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if dynamic_shape:
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
    else:
        opts.enable_cpu_mem_arena = arena
        opts.enable_mem_pattern = True

    if threads and threads > 0:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = max(1, threads // 2)
    opts.log_severity_level = 3
    return opts


def _coerce_device(device: Optional[Union[Device, str]]) -> Device:
    """把 ``"cpu"`` / ``"cuda"`` 之类的字符串转成 :class:`Device`。

    字符串入参是公开 API 的一部分，各后端的 ``device`` 配置项一路传下来
    都是字符串，因此在入口统一做归一化。
    """
    if isinstance(device, Device):
        return device
    if device is None:
        return detect_device()
    return detect_device(str(device))


def create_session(
    model_path: Union[str, Path],
    device: Optional[Device] = None,
    *,
    providers: Optional[Sequence[ProviderSpec]] = None,
    threads: int = 0,
    arena: bool = True,
    dynamic_shape: bool = False,
    warmup_shape: Optional[Tuple[int, ...]] = None,
):
    """创建 ONNX Runtime session，全项目唯一的构造入口。

    Args:
        model_path: 模型文件路径。
        device: 目标设备，可为 :class:`Device` 或 ``"cpu"``/``"cuda"``/
            ``"auto"`` 等字符串，None 时自动探测。
        providers: 显式指定 providers，覆盖 device 推导。
        threads: CPU 线程数，0 表示交给 ORT 自行决定。
        arena: 是否启用内存 arena，``dynamic_shape=True`` 时强制关闭。
        dynamic_shape: 输入形状是否随运行变化。宽度动态的模型必须传 True，
            否则会遭遇内存 arena 引起的性能退化，详见
            :func:`_make_session_options`。
        warmup_shape: 给定则做一次 dummy 推理预热，形状必须是 int 元组。

    Returns:
        ``onnxruntime.InferenceSession``

    Raises:
        ModelNotFoundError: 模型不存在，异常信息含获取方式。
    """
    import onnxruntime as ort

    path = Path(model_path)
    if not path.is_file():
        raise ModelNotFoundError(
            f"模型文件不存在: {path}\n"
            f"请先下载模型: python -m noocr models"
        )

    dev = _coerce_device(device)
    sess_providers = list(providers) if providers is not None else build_providers(dev, arena=arena)
    opts = _make_session_options(threads=threads, arena=arena, dynamic_shape=dynamic_shape)

    t0 = _now()
    try:
        session = ort.InferenceSession(
            str(path), sess_options=opts, providers=sess_providers
        )
    except Exception as e:
        if dev.is_gpu:
            log.warning("在 {} 上加载 {} 失败（{}），回退到 CPU 推理", dev, path.name, e)
            session = ort.InferenceSession(
                str(path),
                sess_options=_make_session_options(threads=threads, dynamic_shape=dynamic_shape),
                providers=["CPUExecutionProvider"],
            )
            dev = Device("cpu")
        else:
            raise

    log.info(
        "已加载模型 {} ({:.2f}s) 设备={} 实际EP={}",
        path.name,
        (_now() - t0) / 1000,
        dev,
        session.get_providers(),
    )

    if warmup_shape is not None:
        warmup(session, warmup_shape)
    return session


def warmup(session, shape: Tuple[int, ...], dtype: str = "float32") -> None:
    """预热推理。

    ORT 的算子选择与内存规划是惰性的，首次调用可能比后续慢 2 至 10 倍。
    对延迟敏感的服务端，初始化时预热一次很划算。

    Raises:
        ValueError: shape 含非整数，常见于直接透传 argparse 的 float 参数。
    """
    import numpy as np

    if any(not isinstance(d, int) or isinstance(d, bool) for d in shape):
        raise ValueError(
            f"预热形状必须全部是 int，收到 {shape}。"
            f"若来自命令行解析的 float 参数，请先 int() 转换。"
        )
    feeds = {}
    for node in session.get_inputs():
        target = [d if isinstance(d, int) and d > 0 else 1 for d in shape]
        feeds[node.name] = np.zeros(target, dtype=dtype)
    try:
        session.run(None, feeds)
    except Exception as e:
        log.debug("预热跳过（{}）: {}", shape, e)


def input_names(session) -> List[str]:
    return [n.name for n in session.get_inputs()]


def output_names(session) -> List[str]:
    return [n.name for n in session.get_outputs()]


class SessionCache:
    """线程安全、容量受限的 session 缓存。

    - **有界**：超过 ``max_size`` 按 LRU 淘汰并显式释放。
    - **线程安全**：``check-then-act`` 在锁内完成，避免并发重复加载。
    - **按路径与设备键控**：不同设备、不同模型互不污染。
    """

    def __init__(
        self,
        max_size: int = 4,
        device: Optional[Union[Device, str]] = None,
        threads: int = 0,
    ):
        self._max_size = max(1, max_size)
        self._device = _coerce_device(device) if device is not None else None
        self._threads = threads
        self._store: "OrderedDict[Tuple[str, str], Any]" = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
            }

    def get_or_create(self, model_path: Union[str, Path], **kwargs):
        raw_dev = kwargs.pop("device", None)
        dev = (
            _coerce_device(raw_dev)
            if raw_dev is not None
            else (self._device or detect_device())
        )
        path = str(Path(model_path).resolve())
        dyn = bool(kwargs.get("dynamic_shape", False))
        key = (path, f"{dev.kind}:{'dyn' if dyn else 'fix'}")

        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._hits += 1
                return self._store[key]
            self._misses += 1

        session = create_session(model_path, dev, threads=self._threads, **kwargs)

        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
            self._store[key] = session
            while len(self._store) > self._max_size:
                evicted_key, evicted = self._store.popitem(last=False)
                log.info(
                    "缓存已满，淘汰模型 {}（上限 {}）",
                    Path(evicted_key[0]).name,
                    self._max_size,
                )
                del evicted
        return session

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            log.debug("session 缓存已清空")

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


_GLOBAL_CACHE: Optional[SessionCache] = None
_GLOBAL_CACHE_LOCK = threading.Lock()


def get_global_cache(max_size: int = 4) -> SessionCache:
    """进程级共享 session 缓存。

    多后端共存时共享，避免同一模型被重复加载多份（每份数百 MB）。
    """
    global _GLOBAL_CACHE
    with _GLOBAL_CACHE_LOCK:
        if _GLOBAL_CACHE is None:
            _GLOBAL_CACHE = SessionCache(max_size=max_size)
        return _GLOBAL_CACHE


def _now() -> float:
    import time

    return time.perf_counter() * 1000
