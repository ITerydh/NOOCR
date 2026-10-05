"""Web 服务层：REST API 与Web UI。

两个入口共享同一份 :class:`BackendPool`，模型只加载一次并复用：

- :func:`create_app` 返回FastAPI 应用，挂载 ``/api/*`` 与 ``/`` 界面。
- ``python -m noocr serve`` 直接启动。
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..backends import DEFAULT_BACKEND, get_backend, list_backends
from ..engine.base import BackendError
from ..inputs.loader import load_document
from ..logging_config import get_logger
from ..types import OCRResult

log = get_logger(__name__)

__all__ = ["create_app", "BackendPool", "run"]

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATE_DIR = Path(__file__).parent / "templates"

#: 示例图允许的扩展名。
_SAMPLE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".pdf"}


class BackendPool:
    """按需加载并复用后端实例，模型只常驻一份。

    设备（CPU/CUDA）也是实例键的一部分：同一个后端在两种设备上
    是两套独立的 ONNX session，混用会导致「传了 GPU 却拿到 CPU 模型」。
    """

    def __init__(self, device: str = "auto") -> None:
        self._lock = threading.Lock()
        self._device = device or "auto"
        self._instances: Dict[str, Any] = {}

    @property
    def device(self) -> str:
        """当前绑定的设备偏好。"""
        return self._device

    def get(self, name: str = DEFAULT_BACKEND) -> Any:
        with self._lock:
            backend = self._instances.get(name)
            if backend is None:
                backend = get_backend(name, device=self._device)
                backend.load()
                self._instances[name] = backend
            return backend

    def warm(self, names: Optional[List[str]] = None) -> Dict[str, str]:
        """预加载后端，返回 ``{后端名: 状态或错误信息}``。"""
        targets = names or [DEFAULT_BACKEND]
        out: Dict[str, str] = {}
        for n in targets:
            try:
                self.get(n)
                out[n] = "ok"
            except Exception as e:
                out[n] = str(e)
        return out

    def shutdown(self) -> None:
        with self._lock:
            for backend in self._instances.values():
                try:
                    backend.unload()
                except Exception:
                    log.exception("卸载后端失败")
            self._instances.clear()


class DocCache:
    """源文件与渲染页图的有界缓存。

    识别接口把上传内容先存进来拿到 ``doc_id``，之后前端翻页时
    再按页取图，避免把整份文档的渲染结果一次性塞进 JSON。
    超出上限时按先进先出淘汰，防止长驻服务把内存吃满。
    """

    def __init__(self, max_docs: int = 8, max_pages_per_doc: int = 64) -> None:
        self._lock = threading.Lock()
        self._max_docs = max_docs
        self._max_pages = max_pages_per_doc
        self._docs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def put_source(self, data: bytes) -> str:
        """存入源文件字节，返回文档标识。"""
        doc_id = uuid.uuid4().hex
        with self._lock:
            self._docs[doc_id] = {"source": data, "pages": {}}
            self._docs.move_to_end(doc_id)
            while len(self._docs) > self._max_docs:
                self._docs.popitem(last=False)
        return doc_id

    def get_source(self, doc_id: str) -> bytes:
        """取源文件字节，不存在时抛 :class:`HTTPException`。"""
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                raise HTTPException(status_code=404, detail="文档不存在或已过期")
            self._docs.move_to_end(doc_id)
            return doc["source"]

    def store_page(self, doc_id: str, index: int, image: Any) -> None:
        """记录某一页的渲染图。"""
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None or index >= self._max_pages:
                return
            doc["pages"][index] = image
            self._docs.move_to_end(doc_id)

    def get_page(self, doc_id: str, index: int) -> Any:
        """取某一页的渲染图，没有则返回 ``None``。"""
        with self._lock:
            doc = self._docs.get(doc_id)
            return None if doc is None else doc["pages"].get(index)


def _result_payload(result: Any, boxes: bool) -> Dict[str, Any]:
    """把 :class:`OCRResult` 转成前端/客户端都通用的 JSON 结构。"""
    pages: List[Dict[str, Any]] = []
    for page in result.pages:
        lines: List[Dict[str, Any]] = []
        for ln in page.lines:
            item: Dict[str, Any] = {
                "text": ln.text,
                "confidence": round(float(ln.confidence), 4),
                "angle": float(ln.angle),
                "line_id": ln.line_id,
            }
            if boxes and ln.box:
                item["box"] = [[round(x, 1), round(y, 1)] for x, y in ln.box.points]
            lines.append(item)
        pages.append(
            {
                "page_index": page.page_index,
                "width": page.width,
                "height": page.height,
                "text": page.text,
                "lines": lines,
                "processing_time": round(page.processing_time, 3),
                "has_image": page.image is not None,
                # 分阶段耗时，用于界面展示慢在哪一步
                "stages": page.debug.get("stage_ms") if page.debug else None,
            }
        )
    return {
        "backend": result.backend,
        "source": result.source_name,
        "source_type": result.source_type,
        "page_count": result.page_count,
        "total_time": round(result.total_time, 3),
        "native_text": result.native_text,
        "warnings": list(result.warnings),
        "markdown": result.to_markdown(),
        "pages": pages,
    }


def create_app(pool: Optional[BackendPool] = None, device: str = "auto") -> FastAPI:
    """构造 FastAPI 应用。

    Args:
        pool: 复用已有后端池。为空时按 ``device`` 新建。
        device: 设备偏好，见 :func:`run`。
    """
    backend_pool = pool or BackendPool(device=device)
    cache = DocCache()
    app = FastAPI(
        title="NOOCR",
        version=__version__,
        description="ONNX 全功能 OCR 服务",
    )
    app.state.pool = backend_pool

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ---------------------------------------------------------------- 页面

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        page = TEMPLATE_DIR / "index.html"
        if not page.is_file():
            return HTMLResponse("<h1>NOOCR</h1><p>界面文件缺失，请使用 API。</p>")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/backends")
    def api_backends() -> Dict[str, Any]:
        return {
            "default": DEFAULT_BACKEND,
            "items": list_backends(),
            "loaded": sorted(backend_pool._instances),
        }

    @app.get("/api/device")
    def api_device() -> Dict[str, Any]:
        """返回设备偏好与实际生效的设备。

        前端据此显示「正在用GPU / CPU」——GPU 被静默降级时用户必须看得见。
        """
        from ..engine.session import detect_device

        prefer = backend_pool.device
        info: Dict[str, Any] = {"prefer": prefer, "options": ["auto", "cpu", "cuda"]}
        try:
            dev = detect_device(prefer)
            info["kind"] = dev.kind
            info["name"] = dev.name
            info["label"] = str(dev)
            info["gpu_active"] = dev.is_gpu
        except Exception as e:
            info["kind"] = "cpu"
            info["label"] = f"探测失败: {e}"
            info["gpu_active"] = False
        return info

    # ---------------------------------------------------------------- 识别

    def _run(doc_id: str, backend_name: str, dpi: int, max_pages: int) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            backend = backend_pool.get(backend_name)
        except BackendError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:
            log.exception("后端加载失败")
            raise HTTPException(status_code=500, detail=f"后端加载失败: {e}") from e

        try:
            doc = load_document(cache.get_source(doc_id), dpi=dpi, max_pages=max_pages)
            pages = []
            for image in doc:
                pr = backend.recognize_image(image, page_index=len(pages))
                pr.image = image
                pages.append(pr)
                cache.store_page(doc_id, len(pages) - 1, image)
            result = _assemble(backend, doc, pages, time.perf_counter() - started)
            payload = _result_payload(result, boxes=True)
            payload["doc_id"] = doc_id
            return payload
        except HTTPException:
            raise
        except Exception as e:
            log.error("识别失败: {}\n{}", e, traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"识别失败: {e}") from e

    @app.post("/api/ocr")
    async def api_ocr(
        file: UploadFile = File(...),
        backend: str = Form(DEFAULT_BACKEND),
        dpi: int = Form(200),
        max_pages: int = Form(0),
    ) -> Dict[str, Any]:
        """上传文件并识别。"""
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="空文件")
        return _run(cache.put_source(data), backend, dpi, max_pages)

    @app.post("/api/ocr/path")
    def api_ocr_path(
        path: str,
        backend: str = DEFAULT_BACKEND,
        dpi: int = 200,
        max_pages: int = 0,
    ) -> Dict[str, Any]:
        """识别服务器本地路径上的文件。"""
        target = Path(path).expanduser()
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"文件不存在: {path}")
        return _run(cache.put_source(target.read_bytes()), backend, dpi, max_pages)

    @app.post("/api/sample/{name}")
    def api_sample(
        name: str,
        backend: str = DEFAULT_BACKEND,
        dpi: int = 200,
        max_pages: int = 0,
    ) -> Dict[str, Any]:
        """识别内置示例图。

        示例图以**磁盘路径**解析，因此不受 Web 路由前缀影响。
        """
        safe = Path(name).name
        target = STATIC_DIR / safe
        if not target.is_file() or target.suffix.lower() not in _SAMPLE_SUFFIXES:
            raise HTTPException(status_code=404, detail=f"示例图不存在: {name}")
        return _run(cache.put_source(target.read_bytes()), backend, dpi, max_pages)

    @app.get("/api/page/{doc_id}/{index}")
    def api_page_image(doc_id: str, index: int) -> Response:
        """取某一页的渲染图（PNG），供多页文档翻页显示。"""
        img = cache.get_page(doc_id, index)
        if img is None:
            raise HTTPException(status_code=404, detail="页不存在或已过期")
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise HTTPException(status_code=500, detail="页面编码失败")
        return Response(content=buf.tobytes(), media_type="image/png")

    @app.post("/api/warmup")
    def api_warmup(backend: str = DEFAULT_BACKEND) -> Dict[str, Any]:
        return backend_pool.warm([backend])

    @app.on_event("shutdown")
    def _shutdown() -> None:
        backend_pool.shutdown()

    return app


def _assemble(backend: Any, doc: Any, pages: List[Any], elapsed: float) -> OCRResult:
    """把逐页结果打包成统一的 :class:`OCRResult`。"""
    return OCRResult(
        pages=pages,
        backend=backend.name,
        total_time=elapsed,
        source_name=doc.name,
        source_type=doc.kind,
        page_count=len(pages),
        native_text=doc.native_text,
    )


def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    backend: str = DEFAULT_BACKEND,
    reload: bool = False,
    device: str = "auto",
) -> None:
    """启动 Web 服务。

    Args:
        device: ``auto`` / ``cpu`` / ``cuda``。``auto`` 会在有 CUDA 的机器上
            自动启用 GPU，但**必须真的能用**：缺 CUDA/cuDNN 时会明确报错，
            而不是悄悄退回 CPU 让人误以为在用显卡。
    """
    import uvicorn

    from ..engine.session import detect_device

    pool = BackendPool(device=device)
    dev = detect_device(device)
    print(f"[device] 请求={device} 实际={dev}", flush=True)
    if backend:
        for name, status in pool.warm([backend]).items():
            msg = f"[warmup] {name} 已就绪" if status == "ok" else f"[warmup] {name} 加载失败: {status}"
            print(msg, flush=True)

    print(f"NOOCR {__version__} 已启动 ->  http://{host}:{port}", flush=True)
    print(f"  界面 http://{host}:{port}/", flush=True)
    print(f"  文档 http://{host}:{port}/docs", flush=True)
    uvicorn.run(create_app(pool), host=host, port=port, log_level="info")
