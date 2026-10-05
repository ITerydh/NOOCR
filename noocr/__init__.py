"""NOOCR —— 全功能 OCR 系统。

一句话定位：**一个内核，四种后端，五种交付形态。**

    from noocr import ocr

    result = ocr("扫描件.png")            # 自动选后端
    print(result.text)
    print(result.to_markdown())           # 文档直接转 Markdown

设计要点：

- **统一返回契约**：任何后端、任何模式都返回 :class:`~noocr.types.OCRResult`，
  调用方无需分支处理。
- **能力声明式编排**：后端用 :class:`~noocr.types.BackendCapabilities` 声明能力，
  pipeline 负责组合，而非在类里做互斥分支。
- **有界资源**：session 缓存有容量上限与 LRU 淘汰，避免多份常驻模型耗尽内存。
- **可预期的性能**：预处理算子白名单化、框排序 O(n log n)、归一化单次分配。
"""

from __future__ import annotations

__version__ = "0.1.0"

from .types import (
    AxisAlignedBox,
    BackendCapabilities,
    BoundingBox,
    LayoutRegion,
    OCRResult,
    PageResult,
    RegionType,
    Table,
    TableCell,
    TextLine,
    Word,
)

__all__ = [
    "__version__",
    "AxisAlignedBox",
    "BackendCapabilities",
    "BoundingBox",
    "LayoutRegion",
    "OCRResult",
    "PageResult",
    "RegionType",
    "Table",
    "TableCell",
    "TextLine",
    "Word",
    "ocr",
    "list_backends",
    "OCRPipeline",
]


def __getattr__(name: str):
    """惰性导出重对象。

    让 ``import noocr`` 保持轻量——不触碰 onnxruntime 这类重依赖，
    只有真正用到 pipeline 时才加载。
    """
    if name in ("ocr", "OCRPipeline"):
        from .pipeline import OCRPipeline

        return OCRPipeline() if name == "ocr" else OCRPipeline
    if name == "list_backends":
        from .backends import list_backends

        return list_backends
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
