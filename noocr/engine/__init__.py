"""推理引擎层：统一的 session 管理、设备探测与后端抽象。"""

from .base import BackendError, BackendUnavailable, OCRBackend
from .imageops import (
    crop_quad,
    imread,
    imwrite,
    normalize_db,
    order_quad,
    resize_for_det,
    resize_keep_ratio,
    sort_reading_order,
    to_bgr,
)
from .session import (
    Device,
    ModelNotFoundError,
    SessionCache,
    build_providers,
    create_session,
    detect_device,
    get_global_cache,
    warmup,
)

__all__ = [
    "BackendError",
    "BackendUnavailable",
    "OCRBackend",
    "Device",
    "ModelNotFoundError",
    "SessionCache",
    "build_providers",
    "create_session",
    "detect_device",
    "get_global_cache",
    "warmup",
    "crop_quad",
    "imread",
    "imwrite",
    "normalize_db",
    "order_quad",
    "resize_for_det",
    "resize_keep_ratio",
    "sort_reading_order",
    "to_bgr",
]
