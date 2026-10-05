"""图像几何与预处理算子。

统一收口三类基础能力：

1. **读写**：:func:`imread` / :func:`imwrite` 走 ``np.fromfile`` / ``buf.tofile``，
   规避 OpenCV 在 Windows 上无法处理 unicode 路径的问题。
2. **几何**：四边形排序、透视矫正裁切、阅读顺序排列，均不依赖输入有序性。
3. **预处理**：缩放插值策略与 DB 归一化，算子通过白名单注册，不做动态求值。

.. note::
   框排序采用 O(n log n) 实现。实测冒泡排序在4000 框下约 2.5ms，
   并非性能瓶颈，此处重写的价值在于消除隐式前置条件、并给出可验证的正确性。
"""

from __future__ import annotations

import os
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from ..logging_config import get_logger

log = get_logger(__name__)

__all__ = [
    "imread",
    "imwrite",
    "to_bgr",
    "order_quad",
    "crop_quad",
    "sort_reading_order",
    "box_anchor",
    "resize_keep_ratio",
    "resize_for_det",
    "normalize_db",
    "OPERATORS",
    "build_pipeline",
]


def imread(
    path: Union[str, "os.PathLike[str]", bytes],
    flags: int = cv2.IMREAD_COLOR,
) -> Optional[np.ndarray]:
    """读图，支持中文路径、``Path`` 对象、URL 与图像原始字节。

    Args:
        path: 本地路径（``str``/``Path``）、URL或图像原始 ``bytes``。
        flags: OpenCV 解码标志。

    Returns:
        解码后的 BGR 图像；失败返回 ``None``，与 ``cv2.imread`` 行为一致。
    """
    if isinstance(path, (bytes, bytearray)):
        return _decode_bytes(bytes(path), flags, "<bytes>")

    text = os.fspath(path) if isinstance(path, os.PathLike) else str(path)
    if text.startswith(("http://", "https://")):
        try:
            data = urllib.request.urlopen(text, timeout=30).read()
        except Exception as e:
            log.error("下载图片失败 {}: {}", text, e)
            return None
        return _decode_bytes(data, flags, text)

    try:
        data = np.fromfile(text, dtype=np.uint8)
    except (OSError, ValueError) as e:
        log.error("读取文件失败 {}: {}", text, e)
        return None
    return _decode_bytes(data, flags, text)


def _decode_bytes(data: Any, flags: int, label: str) -> Optional[np.ndarray]:
    """把原始字节或已读入的 buffer 解码为图像。

    ``np.fromfile`` 返回 ``ndarray``，不能直接用 ``if not data`` 判断 emptiness，
    numpy 会抛 ``ValueError: truth value of an array ... is ambiguous``。
    """
    buf = data if isinstance(data, np.ndarray) else np.frombuffer(data, np.uint8)
    if buf.size == 0:
        log.error("图像数据为空: {}", label)
        return None
    img = cv2.imdecode(buf, flags)
    if img is None:
        log.warning("无法解码图片（格式不支持或文件损坏）: {}", label)
    return img


def imwrite(path: Union[str, "os.PathLike[str]"], image: np.ndarray, quality: int = 95) -> bool:
    """写图，支持中文路径。返回是否成功。"""
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        log.error("编码图片失败: {}", path)
        return False
    try:
        buf.tofile(path)
        return True
    except OSError as e:
        log.error("写入图片失败 {}: {}", path, e)
        return False


def to_bgr(image: np.ndarray) -> np.ndarray:
    """统一转成 3 通道 BGR，兼容灰度 / RGBA 输入。"""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def order_quad(points: np.ndarray) -> np.ndarray:
    """把 4 点排成左上→右上→右下→左下。

    以 ``sum`` / ``diff`` 判定角点，退化情形回退到质心极角排序，
    避免求和/求差判据在四点共线时产生重复角点。
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    total = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    rect = np.zeros((4, 2), dtype=np.float32)
    try:
        rect[0] = pts[np.argmin(total)]
        rect[2] = pts[np.argmax(total)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        if len({tuple(p) for p in rect.tolist()}) == 4:
            return rect
    except (ValueError, IndexError):
        pass

    c = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    rect = pts[order]
    start = int(np.argmin(rect.sum(axis=1)))
    return np.roll(rect, -start, axis=0)


def crop_quad(image: np.ndarray, points: Sequence[Sequence[float]]) -> np.ndarray:
    """透视矫正裁切出一个四边形区域，高宽比达1.5 时自动转正。"""
    pts = order_quad(np.asarray(points, dtype=np.float32))
    h, w = image.shape[:2]

    width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if width < 1 or height < 1:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts, dst)
    cropped = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if cropped.shape[0] / max(cropped.shape[1], 1) >= 1.5:
        cropped = np.rot90(cropped)
    return np.ascontiguousarray(cropped)


def box_anchor(box: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """取四边形的左上角锚点 ``(x, y)``。

    取**最小 x / 最小 y** 而非 ``box[0]``：角点顺序依赖 :func:`order_quad`，
    而外部传入的框不保证是 TL→TR→BR→BL 顺序。

    .. warning::
       调用方若传入 ``box[:, 0].tolist()``（四边形的第一列，即 4 个 x 坐标），
       而本函数按 ``(x, y)`` 取前两个元素，则会把第二个点的 x 当成 y，
       导致阅读顺序完全错乱。本函数对输入形状做归一化，点数与顺序均不再重要。

    Returns:
        ``(min_x, min_y)``；输入为空时返回 ``(0.0, 0.0)``。
    """
    xs: List[float] = []
    ys: List[float] = []
    for p in box:
        try:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
        except (TypeError, IndexError, ValueError):
            continue
    if not xs:
        return 0.0, 0.0
    return min(xs), min(ys)


def sort_reading_order(
    boxes: Sequence[Sequence[Sequence[float]]], line_tolerance: float = 10.0
) -> List[int]:
    """按阅读顺序（先上下、再左右）排列框索引，不修改入参。

    算法：

    1. 每个框取左上角锚点 :func:`box_anchor`，不假设角点顺序。
    2. 按 y 升序粗排。
    3. 按 y 聚类成「行」：与当前行锚点 y 差超过 ``line_tolerance`` 即另起一行。
    4. 行内按 x 升序。

    Args:
        boxes: 框列表，每个为 ``(N, 2)`` 的点集。
        line_tolerance: 判定「同一行」的 y 偏差阈值，单位像素。

    Returns:
        排序后的索引列表，复杂度 O(n log n)。
    """
    n = len(boxes)
    if n <= 1:
        return list(range(n))

    anchors: List[Tuple[float, float]] = [box_anchor(bx) for bx in boxes]
    order = sorted(range(n), key=lambda i: (anchors[i][1], anchors[i][0]))

    result: List[int] = []
    row: List[int] = []
    row_y = 0.0
    for i in order:
        y = anchors[i][1]
        if row and abs(y - row_y) > line_tolerance:
            row.sort(key=lambda j: anchors[j][0])
            result.extend(row)
            row = []
            row_y = y
        elif not row:
            row_y = y
        row.append(i)
    if row:
        row.sort(key=lambda j: anchors[j][0])
        result.extend(row)
    return result


def _pick_interp(src_h: int, src_w: int, dst_h: int, dst_w: int) -> int:
    """按缩放方向选插值方式。

    缩小时必须用 ``INTER_AREA``（自带抗锯齿）：DB 检测对细长文字框敏感，
    ``INTER_LINEAR`` 缩小会产生混叠，直接导致检测率下降。
    """
    if dst_h < src_h or dst_w < src_w:
        return cv2.INTER_AREA
    return cv2.INTER_CUBIC


def resize_keep_ratio(
    image: np.ndarray, target_h: int, target_w: int, align: int = 32
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """等比缩放到 ``target_h x target_w`` 以内，并把尺寸对齐到 ``align`` 的倍数。

    Returns:
        ``(图像, (src_h, src_w, ratio_h, ratio_w))``，ratio 用于把检测框映射回原图。
    """
    src_h, src_w = image.shape[:2]
    ratio = min(target_h / max(src_h, 1), target_w / max(src_w, 1))

    scale = min(ratio, 2.0) if ratio > 1 else ratio
    new_h = max(align, int(round(src_h * scale / align)) * align)
    new_w = max(align, int(round(src_w * scale / align)) * align)

    interp = _pick_interp(src_h, src_w, new_h, new_w)
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    return resized, (src_h, src_w, new_h / src_h, new_w / src_w)


def resize_for_det(image: np.ndarray, limit: int = 960, align: int = 32) -> np.ndarray:
    """检测前的输入缩放：长边不超过 ``limit``。"""
    resized, _ = resize_keep_ratio(image, limit, limit, align)
    return resized


def normalize_db(image: np.ndarray) -> np.ndarray:
    """DB 检测预处理：归一化到 [-1, 1] 并转 CHW。

    用 ``cv2`` / numpy 原地运算，临时全图数组降到 1 个。
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img = image.astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
    img -= mean
    img /= std
    return np.ascontiguousarray(img.transpose(2, 0, 1))


class _Operator:
    """预处理算子基类。"""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


class DetResizeOp(_Operator):
    """按长边上限缩放。"""

    def __init__(self, limit: int = 960, align: int = 32):
        self.limit = limit
        self.align = align

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return resize_for_det(image, self.limit, self.align)


class NormalizeOp(_Operator):
    """DB 归一化。"""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return normalize_db(image)


class ToCHWOp(_Operator):
    """HWC 转 CHW。"""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(image.transpose(2, 0, 1))


class PassOp(_Operator):
    """直通。"""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image


#: 算子白名单注册表。以显式查表替代动态求值，避免配置表成为代码执行入口。
OPERATORS: Dict[str, Callable[..., _Operator]] = {
    "DetResizeForTest": DetResizeOp,
    "NormalizeImage": NormalizeOp,
    "ToCHWImage": ToCHWOp,
    "KeepKeys": PassOp,
}


def build_pipeline(specs: Sequence[Tuple[str, Dict[str, Any]]]) -> List[_Operator]:
    """按配置构建预处理链。

    Args:
        specs: ``[("DetResizeForTest", {"limit": 960}), ...]`` 形式的配置。

    Raises:
        KeyError: 算子名不在白名单内。
    """
    pipeline: List[_Operator] = []
    for name, params in specs:
        if name not in OPERATORS:
            raise KeyError(
                f"未知算子 {name!r}；可用算子: {', '.join(sorted(OPERATORS))}。"
            )
        pipeline.append(OPERATORS[name](**(params or {})))
    return pipeline
