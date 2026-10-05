"""编排层：把「输入解析 → 后端推理 → 结果组装」串成一条可复用的流水线。

这是库层唯一的推荐入口。CLI 与 Web 服务都走它，避免三处各写一遍编排逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..backends import DEFAULT_BACKEND, get_backend
from ..engine.base import OCRBackend
from ..inputs.loader import load_document
from ..logging_config import get_logger
from ..models import ensure_models
from ..types import OCRResult

log = get_logger(__name__)

__all__ = ["OCRPipeline"]


class OCRPipeline:
    """一次配置、多次复用。

    实例本身不持有模型；首次调用时才加载，之后常驻，
    因此长驻服务应当缓存一个实例而不是每次请求都新建。

    Args:
        backend: 后端名，默认 :data:`~noocr.backends.DEFAULT_BACKEND`。
        model_dir: 权重根目录，``None`` 表示用全局默认。
        auto_download: 权重缺失时是否自动从 ModelScope 拉取。
        **options: 透传给后端，如 ``rec_batch_size`` / ``use_cls`` / ``device``。
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        model_dir: Optional[Union[str, Path]] = None,
        *,
        auto_download: bool = True,
        **options: Any,
    ):
        self.backend_name = backend or DEFAULT_BACKEND
        self.model_dir = Path(model_dir) if model_dir else None
        self.auto_download = auto_download
        self.options: Dict[str, Any] = dict(options)
        self._backend: Optional[OCRBackend] = None

    @property
    def backend(self) -> OCRBackend:
        """已加载的后端，首次访问时初始化。"""
        if self._backend is None:
            self._backend = self._build()
        return self._backend

    def _build(self) -> OCRBackend:
        if self.auto_download:
            try:
                ensure_models(self.backend_name, quiet=True)
            except Exception as e:
                log.warning("权重自动下载失败（{}），继续尝试本地权重", e)

        opts = dict(self.options)
        if self.model_dir:
            opts["model_dir"] = self.model_dir
        backend = get_backend(self.backend_name, **opts)
        backend.load()
        return backend

    def read(
        self,
        source: Union[str, Path, bytes, Any],
        *,
        dpi: int = 200,
        max_pages: int = 0,
    ) -> OCRResult:
        """识别任意支持的输入，统一返回 :class:`OCRResult`。

        Args:
            source: 文件路径、URL、图片字节或 numpy 图像。
            dpi: PDF 渲染分辨率，小字号文档建议 300。
            max_pages: 最多处理页数，0 表示不限。

        Returns:
            :class:`~noocr.types.OCRResult`，含逐页文本、坐标与置信度。
        """
        doc = load_document(source, dpi=dpi, max_pages=max_pages)
        backend = self.backend

        pages: List[Any] = []
        for image in doc:
            pages.append(backend.recognize_image(image, page_index=len(pages)))

        return OCRResult(
            pages=pages,
            backend=backend.name,
            total_time=sum(p.processing_time for p in pages),
            source_name=doc.name,
            source_type=doc.kind,
            page_count=len(pages),
            native_text=doc.native_text,
            warnings=list(doc.warnings),
        )

    def read_batch(self, sources: List[Any], **kwargs: Any) -> List[OCRResult]:
        """依次识别多个文件，复用同一个后端实例。"""
        return [self.read(s, **kwargs) for s in sources]

    def __call__(self, source: Any, **kwargs: Any) -> OCRResult:
        """:meth:`read` 的别名，使实例可直接当函数用。"""
        return self.read(source, **kwargs)

    def unload(self) -> None:
        """释放模型。"""
        if self._backend is not None:
            self._backend.unload()
            self._backend = None

    def __enter__(self) -> "OCRPipeline":
        self.backend
        return self

    def __exit__(self, *exc: Any) -> None:
        self.unload()

    def __repr__(self) -> str:
        state = "loaded" if self._backend is not None else "lazy"
        return f"<OCRPipeline backend={self.backend_name!r} {state}>"
