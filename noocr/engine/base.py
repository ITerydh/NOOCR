"""后端抽象。

组合优于分支：每个后端是一个独立类，实现统一的 :class:`OCRBackend` 接口，
能力差异由 :class:`BackendCapabilities` 声明，跨后端编排交由 pipeline 负责。
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..logging_config import get_logger
from ..types import BackendCapabilities, OCRResult, PageResult
log = get_logger(__name__)

__all__ = ["OCRBackend", "BackendError", "BackendUnavailable"]


class BackendError(RuntimeError):
    """后端执行失败。"""


class BackendUnavailable(BackendError):
    """后端不可用：缺依赖、缺模型、缺原生库等。消息应包含自助解决方式。"""


class OCRBackend(abc.ABC):
    """所有 OCR 后端的基类。

    生命周期由 :meth:`load` / :meth:`unload` 显式管理。隐式加载会让多实例场景
    下的原生资源（如 DLL 句柄）无法回收。
    """

    #: 能力声明，子类覆盖
    capabilities: BackendCapabilities

    def __init__(self, name: str, model_dir: Optional[Union[str, Path]] = None, **options: Any):
        self.name = name
        self.model_dir = Path(model_dir) if model_dir else None
        self.options: Dict[str, Any] = dict(options)
        self._loaded = False

    # ---------- 生命周期 ----------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abc.abstractmethod
    def load(self) -> None:
        """加载模型。必须可重复调用而不泄漏资源。"""

    @abc.abstractmethod
    def unload(self) -> None:
        """释放模型与原生资源。必须可重复调用。"""

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ---------- 推理 ----------

    @abc.abstractmethod
    def recognize_image(self, image: np.ndarray, page_index: int = 0) -> PageResult:
        """识别单张 BGR 图像。

        Args:
            image: OpenCV BGR 数组，形状 ``(H, W, 3)``。
            page_index: 页码，仅用于结果标记。

        Returns:
            :class:`PageResult`——统一结构，与后端无关。
        """

    def recognize_images(
        self, images: Sequence[np.ndarray], source_name: str = ""
    ) -> OCRResult:
        """识别多张图像（多页文档）。默认逐页串行。

        子类若支持批处理或线程并行可覆盖本方法以提升吞吐。
        """
        import time

        self.ensure_loaded()
        t0 = time.perf_counter()
        pages: List[PageResult] = []
        for i, img in enumerate(images):
            pages.append(self.recognize_image(img, page_index=i))
        return OCRResult(
            pages=pages,
            backend=self.name,
            total_time=time.perf_counter() - t0,
            source_name=source_name,
            source_type="images",
            page_count=len(pages),
        )

    # ---------- 工具 ----------

    def describe(self) -> Dict[str, Any]:
        """供 CLI / API 展示的后端信息。"""
        caps = self.capabilities
        return {
            "name": self.name,
            "loaded": self._loaded,
            "word_level": caps.word_level,
            "layout_analysis": caps.layout_analysis,
            "table_structure": caps.table_structure,
            "markdown": caps.markdown,
            "requires_gpu": caps.requires_gpu,
            "requires_native": caps.requires_native,
            "model_dir": str(self.model_dir) if self.model_dir else None,
        }

    def __enter__(self) -> "OCRBackend":
        self.ensure_loaded()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.unload()

    def __repr__(self) -> str:
        state = "loaded" if self._loaded else "unloaded"
        return f"<{type(self).__name__} name={self.name!r} {state}>"
