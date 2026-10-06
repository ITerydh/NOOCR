"""后端注册表。

设计原则是**声明式**：每个后端以工厂函数登记并附能力标签，
供 CLI / API / GUI 按需选取。新增后端只需在 :data:`BACKEND_REGISTRY` 中加一行。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from ..engine.base import OCRBackend

__all__ = ["BACKEND_REGISTRY", "get_backend", "list_backends", "create_backend"]


def _make_ppocrv5(**opts) -> OCRBackend:
    from .ppocr import PPOCRBackend

    return PPOCRBackend(**opts)


def _make_ppocrv6_tiny(**opts) -> OCRBackend:
    from .ppocrv6 import PPOCRv6Backend

    opts["tier"] = "tiny"
    return PPOCRv6Backend(**opts)


def _make_ppocrv6_small(**opts) -> OCRBackend:
    from .ppocrv6 import PPOCRv6Backend

    opts["tier"] = "small"
    return PPOCRv6Backend(**opts)


def _make_ppocrv6_medium(**opts) -> OCRBackend:
    from .ppocrv6 import PPOCRv6Backend

    opts["tier"] = "medium"
    return PPOCRv6Backend(**opts)


#: 后端名 -> 工厂
#:
#: :data:`_ALIASES` 里的名字**不进这张表**。别名与规范名指向同一个
#: 工厂，若同时登记，:func:`list_backends` 会把同一档位列出两次，
#: 且两行的 label 完全相同——GUI 的下拉框里就出现两个一样的
#: "PP-OCRv6 small"，用户会以为是两个不同的模型。
BACKEND_REGISTRY: Dict[str, Callable[..., OCRBackend]] = {
    "ppocrv5": _make_ppocrv5,
    "ppocrv6-tiny": _make_ppocrv6_tiny,
    "ppocrv6-small": _make_ppocrv6_small,
    "ppocrv6-medium": _make_ppocrv6_medium,
}

#: 别名 -> 规范名。``ppocrv6`` 与 ``ppocrv6-small`` 是同一档。
_ALIASES: Dict[str, str] = {"ppocrv6": "ppocrv6-small"}

#: 可传入的名字全集（规范名 + 别名），供报错信息与帮助使用。
#: 与 :data:`BACKEND_REGISTRY` 分开维护：报错要告诉用户别名可用，
#: 但列举可选后端时不能把别名混进去。
_ACCEPTABLE: Tuple[str, ...] = tuple(sorted(set(BACKEND_REGISTRY) | set(_ALIASES)))

#: 各后端的**静态**能力摘要，供 ``list_backends`` 零成本列出。
#: 不能靠实例化拿到——那会真的加载模型。
#:
#: ``det_mb`` / ``rec_mb`` 是 ``models/`` 里实际文件的大小（MB，MiB），
#: 由 ``python -m noocr models`` 量出来。改了权重就要同步这里，
#: 否则 CLI的 ``noocr backends`` 与 WebUI 下拉里的体积会骗人。
_CAPABILITY_SUMMARY: Dict[str, Dict[str, object]] = {
    "ppocrv5": {
        "label": "PP-OCRv5",
        "det_mb": 4.6,
        "rec_mb": 15.9,
        "notes": "上一代通用OCR，兼容旧项目",
    },
    "ppocrv6-tiny": {
        "label": "PP-OCRv6 tiny",
        "det_mb": 1.7,
        "rec_mb": 4.3,
        "notes": "1.5M参数，最快，精度较低",
    },
    "ppocrv6-small": {
        "label": "PP-OCRv6 small",
        "det_mb": 9.5,
        "rec_mb": 20.3,
        "notes": "7.7M参数，默认档，精度与速度均衡",
    },
    "ppocrv6-medium": {
        "label": "PP-OCRv6 medium",
        "det_mb": 59.2,
        "rec_mb": 73.0,
        "notes": "34.5M参数，服务器档，版面复杂时精度最高",
    },
}

#: 默认后端。v6-small 在官方指标上优于 v5 且速度相当。
DEFAULT_BACKEND = "ppocrv6-small"


def get_backend(name: Optional[str] = None, **opts) -> OCRBackend:
    """按名创建后端实例。

    Raises:
        KeyError: 后端名未注册。消息中列出全部可用名。
    """
    key = _ALIASES.get(name or DEFAULT_BACKEND, name or DEFAULT_BACKEND)
    factory = BACKEND_REGISTRY.get(key)
    if factory is None:
        raise KeyError(
            f"未知后端 {key!r}；可用: {', '.join(_ACCEPTABLE)}"
        )
    return factory(**opts)


#: 兼容别名
create_backend = get_backend


def list_backends() -> List[Dict[str, object]]:
    """列出全部已注册后端。

    **不实例化**，因为实例化会真正把 ONNX 模型载入内存。
    能力信息取自 :data:`_CAPABILITY_SUMMARY` 静态表。
    """
    out: List[Dict[str, object]] = []
    for name in sorted(BACKEND_REGISTRY):
        canon = _ALIASES.get(name, name)
        info = dict(_CAPABILITY_SUMMARY.get(canon, {}))
        info["name"] = name
        info["requires_gpu"] = False
        info["default"] = name == DEFAULT_BACKEND
        out.append(info)
    return out
