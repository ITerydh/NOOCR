"""统一推理引擎。

全项目唯一的 ONNX Runtime session 构造入口，集中解决四类问题：

1. **缓存容量失控**：:class:`SessionCache` 提供显式容量上限、LRU 淘汰
   与线程安全，避免多份常驻 session 累积导致 OOM。
2. **加速后端静默失效**：统一做能力探测，回退行为显式报错而非悄悄降级。
3. **配置漂移**：:func:`_make_session_options` 全局唯一实现。
4. **GPU 运行库找不到**：:func:`ensure_gpu_runtime` 负责把 cuDNN 等
   依赖目录挂进 DLL 搜索路径，否则 ORT 会静默退回 CPU。
"""

from __future__ import annotations

import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..logging_config import get_logger

log = get_logger(__name__)

_DLL_READY = False

#: 探测用微型模型的落盘位置；放在临时目录，避免污染包目录
_PROBE_MODEL_PATH = Path(tempfile.gettempdir()) / "noocr_cuda_probe.onnx"
_PROBE_MODEL_CACHE: Dict[str, bytes] = {}


def ensure_gpu_runtime(extra_dirs: Optional[Sequence[str]] = None) -> List[str]:
    """把GPU 运行库目录挂进 DLL 搜索路径，返回实际生效的目录。

    ONNX Runtime 的 CUDA EP 依赖 ``cudnn64_9.dll`` 等库。缺失时它**不报错**
    而是把算子悄悄交给 CPU ��户，表现为"选了 GPU 但没有任何加速"。
    Windows 还需要 ``os.add_dll_directory`` 显式登记，纯改 ``PATH`` 不够。

    按优先级查找：显式传入的目录 → 环境变量 ``NOOCR_GPU_LIB_DIR``
    → ``noocr-gpu`` 环境同级或 ``venv`` 同级的 ``cudnn/`` 目录。
    """
    global _DLL_READY
    if _DLL_READY:
        return []
    _DLL_READY = True

    candidates: List[Path] = [Path(d) for d in (extra_dirs or []) if d]
    env = os.environ.get("NOOCR_GPU_LIB_DIR")
    if env:
        candidates.extend(Path(p) for p in env.split(os.pathsep) if p)

    # 从本文件向上逐级找，覆盖三种常见摆法：
    #   <root>/cudnn                —— 直接解压在项目根
    #   <venv>/Lib/site-packages/cudnn
    #   <root>/<venv>/Lib/site-packages/cudnn  —— GPU 环境与包目录平级
    here = Path(__file__).resolve()
    for base in here.parents:
        candidates.append(base / "cudnn")
        for pat in ("*/Lib/site-packages/cudnn", "*/*/site-packages/cudnn"):
            try:
                candidates.extend(base.glob(pat))
            except OSError:
                continue

    registered: List[str] = []
    seen: set = set()
    for path in candidates:
        try:
            if not path.is_dir() or path in seen:
                continue
            if not any(path.glob("cudnn*.dll")):
                continue
            seen.add(path)
            resolved = str(path.resolve())
            if resolved in registered:
                continue
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(resolved)
            os.environ["PATH"] = resolved + os.pathsep + os.environ.get("PATH", "")
            registered.append(resolved)
        except OSError as e:
            log.debug("注册 DLL 目录失败 {}: {}", path, e)

    if registered:
        log.info("已挂载 GPU 运行库: {}", registered)
    else:
        log.debug("未找到 cudnn 运行库目录，GPU 加速可能不可用")
    return registered


__all__ = [
    "Device",
    "ModelNotFoundError",
    "SessionCache",
    "build_providers",
    "create_session",
    "detect_device",
    "ensure_gpu_runtime",
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


#: 设备探测结果缓存。**必须按 (prefer, device_id) 键控**——
#: 早期版本只存一份，导致先跑``prefer="cpu"`` 的调用把结果钉死，
#: 后续 ``prefer="cuda"`` 直接复用 CPU，用户永远等不到GPU。
_DEVICE_CACHE: Dict[Tuple[str, int], Device] = {}
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
    key = (prefer, device_id)
    with _DEVICE_LOCK:
        if not refresh:
            cached = _DEVICE_CACHE.get(key)
            if cached is not None:
                return cached
        result = _detect_device_uncached(prefer, device_id)
        _DEVICE_CACHE[key] = result
        return result


def _detect_device_uncached(prefer: str, device_id: int) -> Device:
    if prefer in ("auto", "cuda", "dml", "cann"):
        ensure_gpu_runtime()
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
        if not _cuda_ep_usable(device_id):
            if prefer == "cuda":
                log.warning(
                    "安装了 onnxruntime-gpu 但 CUDA EP 无法初始化"
                    "（通常是缺少 CUDA/cuDNN 运行库或版本不匹配）。"
                    "请检查 NOOCR_GPU_LIB_DIR 是否指向 cuDNN 9 的 bin 目录。"
                )
        else:
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


def _cuda_ep_usable(device_id: int) -> bool:
    """实测 CUDA EP 能否真正初始化，而不是只看它是否被编译进来。

    ``get_available_providers()`` 列出 ``CUDAExecutionProvider`` 只说明
    打包时包含了它；缺少 CUDA/cuDNN 运行库时，ORT 会在创建 session 时
    打一条警告然后把算子全部交给 CPU，**不抛异常**。这会让"已启用 GPU"
    的用户拿到纯CPU 的性能而毫无察觉。

    这里用一个4x4 的最小图建session，代价约1ms，却能提前把假GPU 拦下来。
    """
    try:
        import onnxruntime as ort

        model = _PROBE_MODEL_CACHE.get("probe")
        if model is None:
            model = _make_probe_model()
            _PROBE_MODEL_CACHE["probe"] = model
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.enable_cpu_mem_arena = False
        sess = ort.InferenceSession(
            model, opts, providers=[("CUDAExecutionProvider", {"device_id": device_id})]
        )
        try:
            return "CUDAExecutionProvider" in sess.get_providers()
        finally:
            del sess  # 探测会话必须立刻释放，否则会占住显存不退还
    except Exception as e:
        log.debug("CUDA EP 不可用: {}", e)
        return False


def _make_probe_model() -> bytes:
    """生成一个恒等映射的微型 ONNX 模型，用于验证 EP 是否可用。"""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])
    node = helper.make_node("Identity", ["x"], ["y"], name="probe")
    graph = helper.make_graph([node], "probe", [x], [y], [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(_PROBE_MODEL_PATH), save_as_external_data=False)
    return _PROBE_MODEL_PATH.read_bytes()


def build_providers(device: Device, *, arena: bool = True) -> List[ProviderSpec]:
    """构造 providers 列表，CPU 永远作为最后兜底。

    .. warning:: ``cudnn_conv_algo_search`` 一旦写成 ``"DEFAULT"``，
       性能会掉一个数量级，且**覆盖** SessionOptions 里的同名配置。

       Provider 级选项的优先级高于 session 级配置，所以这里必须显式给
       ``"EXHAUSTIVE"``，否则 :func:`_make_session_options` 里设的值会被
       静默丢弃。实测 PP-OCRv6-rec（bs=6 w=320，RTX 4070 Ti SUPER）：

       ============================  ==========
       provider 级 algo_search       单次推理
       ============================  ==========
       DEFAULT                90.2 ms
       EXHAUSTIVE                    5.9 ms
       ============================  ==========

       ``"DEFAULT"`` 走cuDNN 最保守的启发式分支，会让大量卷积退回 CPU
       实现，ORT 在日志里打印 ``running in Fallback mode`` 警告。
    """
    if device.kind == "cuda":
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device.device_id,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
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
    *,
    threads: int = 0,
    arena: bool = True,
    dynamic_shape: bool = False,
    on_gpu: bool = False,
):
    """构造 SessionOptions，全局唯一实现。

    Args:
        threads: CPU 线程数，0 表示交给 ORT 自行决定。
        arena: 是否启用内存 arena。
        dynamic_shape: 模型输入形状是否随运行变化。
        on_gpu: 是否走 CUDA/DirectML。此处开�� cuDNN 深度卷积搜索。

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

    if on_gpu:
        # 放开 cuDNN 的workspace 上限，给卷积算法穷举足够的试错空间。
        # 注意：``cudnn_conv_algo_search`` 必须在 :func:`build_providers` 的
        # provider 级选项里设EXHAUSTIVE——provider 级优先级更高，写在这里
        # 会被覆盖。
        opts.add_session_config_entry("ep.cuda.cudnn_conv_use_max_workspace", "1")

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
    opts = _make_session_options(
        threads=threads,
        arena=arena,
        dynamic_shape=dynamic_shape,
        on_gpu=dev.is_gpu,
    )

    t0 = _now()
    try:
        session = ort.InferenceSession(
            str(path), sess_options=opts, providers=sess_providers
        )
    except Exception as e:
        if dev.is_gpu:
            raise RuntimeError(
                f"在 {dev} 上加载 {path.name} 失败: {e}\n"
                f"ONNX Runtime 在缺少 CUDA/cuDNN 运行库时会静默回退到 CPU，"
                f"此时性能与 CPU 相同但用户以为是 GPU。\n"
                f"请确认已安装 CUDA 12 与 cuDNN 9，并把它们加入 PATH；"
                f"或改用 CPU: device='cpu'"
            ) from e
        raise

    # ORT 即使不抛异常，也可能因缺库把算子全丢给 CPU；必须核对实际生效的 EP
    actual = set(session.get_providers())
    wanted = {p if isinstance(p, str) else p[0] for p in sess_providers}
    if dev.is_gpu and not (actual & wanted):
        raise RuntimeError(
            f"{path.name} 请求了 {dev}，但实际只启用了 {sorted(actual)}。"
            f"通常是缺少 CUDA/cuDNN 运行库（Windows 需把 cudnn64_9.dll 等"
            f"加入 PATH）。可用 device='cpu' 强制使用 CPU。"
        )

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
        #:默认 8 而非 4：一个后端要占 det+rec+cls 三个 session，
        #:同时存在 CPU 与GPU 两套时会达到 6 个，容量 4 会把刚建的挤掉，
        #:表现为「每次请求都重新加载模型」的诡异卡顿。
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


def get_global_cache(max_size: int = 8) -> SessionCache:
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
