"""PP-OCRv5 后端：CPU 友好的通用文字 OCR。

由三个 ONNX 模型组成：det（DBNet）定位文本区域、rec（SVTR_LCNet + CTC）
识别文本内容、cls（角度分类）纠正 0/180 度翻转。

两处关键设计：

1. **固定宽度分桶**：CTC 时间步数 = 输入宽度 / 4，若padding 宽度取「本批
   最宽者」，同一张图在不同 ``rec_batch_size`` 下会得到不同的行数与文本。
   :meth:`PPOCRBackend._assign_width_buckets` 让输入宽度只取决于文本自身宽度。

2. **方向分类结果真正生效**：分类器判定180 度后实际旋转裁剪图，
   并把角度透出到 :class:`~noocr.types.TextLine`。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..engine.base import BackendUnavailable, OCRBackend
from ..engine.imageops import (
    crop_quad,
    normalize_db,
    resize_keep_ratio,
    sort_reading_order,
    to_bgr,
)
from ..engine.session import Device, detect_device, get_global_cache
from ..models import model_path
from ..types import BackendCapabilities, BoundingBox, PageResult, TextLine
from .decode import CTCDecoder
from .postprocess import DBPostProcess

__all__ = ["PPOCRBackend"]


def _as_probabilities(preds: np.ndarray) -> np.ndarray:
    """把模型输出规范化为概率，已是概率则原样返回。

    cls 模型输出节点名为 ``softmax_0.tmp_0``，本身即 softmax 之后的概率，
    实测每行和为 1.0（正立 ``[0.958, 0.042]``、倒置 ``[0.093, 0.907]``）。
    对其再套一次 softmax 会把倒置图的 180 度分数从 0.907 压到 0.693，
    恰好跌破 0.9 阈值而使纠正失效。

    Args:
        preds: ``(B, C)`` 的输出。

    Returns:
        ``(B, C)`` 的概率数组，输入本身是概率时不做任何改动。
    """
    arr = np.asarray(preds, dtype=np.float32)
    row_sums = arr.sum(axis=-1, keepdims=True)
    is_prob = bool(np.all(np.abs(row_sums - 1.0) < 1e-2) and np.all(arr >= -1e-6))
    if is_prob:
        return arr
    shifted = arr - arr.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


class PPOCRBackend(OCRBackend):
    """PP-OCRv5 通用 OCR。

    Args:
        model_dir: 模型根目录，默认取全局 ``models/``。
        device: ``"auto"`` / ``"cpu"`` / ``"cuda"`` / ``"dml"``。
        det_limit: 检测输入长边上限。
        det_db_thresh / det_db_box_thresh / det_db_unclip_ratio: DB 参数。
        rec_batch_size: 识别批大小。**不影响输出结果**——输入宽度由
            :meth:`_assign_width_buckets` 的固定档位决定，与分批无关。
        drop_score: 置信度低于此值的文本行被丢弃。
        use_cls: 是否启用 180 度翻转纠正。
        use_dilation: DB 膨胀预处理。
    """

    capabilities = BackendCapabilities(
        name="ppocrv5",
        word_level=False,
        layout_analysis=False,
        table_structure=False,
        markdown=False,
        text_angle=False,
        input_formats=["image", "pdf", "docx", "xlsx"],
        requires_gpu=False,
        speed_score=1.0,
    )

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        *,
        device: str = "auto",
        det_limit: int = 960,
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5,
        det_db_unclip_ratio: float = 1.6,
        rec_batch_size: int = 6,
        rec_img_h: int = 48,
        rec_img_w: int = 320,
        drop_score: float = 0.5,
        use_cls: bool = True,
        use_dilation: bool = False,
        threads: int = 0,
    ):
        super().__init__("ppocrv5", model_dir)
        self.device_pref = device
        #: :meth:`load` 解析出的**真实**设备。与 ``device_pref`` 区分：
        #: ``auto`` 在无 CUDA 的机器上会落到 CPU，只回显偏好会让用户
        #: 以为自己在用显卡。
        self._resolved_device: Optional[Device] = None
        self.det_limit = det_limit
        self.rec_batch_size = max(1, rec_batch_size)
        self.rec_img_h = rec_img_h
        self.rec_img_w = rec_img_w
        self.drop_score = drop_score
        self.use_cls = use_cls
        self.threads = threads
        #: 复用的识别输入缓冲，见 :meth:`_acquire_blob`
        self._blob_cache: Optional[np.ndarray] = None
        #: 复用的方向分类输入缓冲，形状固定故只分配一次
        self._cls_blob: Optional[np.ndarray] = None

        self._det_path = model_path("ppocrv5", "det.onnx")
        self._rec_path = model_path("ppocrv5", "rec.onnx")
        self._cls_path = model_path("ppocrv5", "cls.onnx")
        self._dict_path = model_path("ppocrv5", "ppocrv5_dict.txt")

        self.db = DBPostProcess(
            thresh=det_db_thresh,
            box_thresh=det_db_box_thresh,
            unclip_ratio=det_db_unclip_ratio,
            use_dilation=use_dilation,
        )
        self.decoder = CTCDecoder(self._dict_path)
        self._cls_session = None
        self._cls_labels = ["0", "180"]
        self._cls_thresh = 0.9

    # ---------------------------------------------------------------- 生命周期

    def load(self) -> None:
        if self._loaded:
            return

        for path, name in ((self._det_path, "det"), (self._rec_path, "rec"), (self._dict_path, "dict")):
            if not Path(path).is_file():
                raise BackendUnavailable(
                    f"缺少 {name} 模型: {path}\n"
                    f"请先下载: python -m noocr models --backend ppocrv5"
                )

        dev = detect_device(self.device_pref)
        self._resolved_device = dev
        cache = get_global_cache()
        self._det_session = cache.get_or_create(
            self._det_path,
            device=dev,
            warmup_shape=(1, 3, self.det_limit, self.det_limit),
            dynamic_shape=False,
        )
        self._rec_session = cache.get_or_create(
            self._rec_path,
            device=dev,
            warmup_shape=(1, 3, self.rec_img_h, self.rec_img_w),
            dynamic_shape=True,
        )
        self._det_in = self._det_session.get_inputs()[0].name
        self._det_out = [o.name for o in self._det_session.get_outputs()]
        self._rec_in = self._rec_session.get_inputs()[0].name
        self._rec_out = [o.name for o in self._rec_session.get_outputs()]

        # 方向分类器为可选组件，缺失时静默降级为不纠正
        if self.use_cls and Path(self._cls_path).is_file():
            try:
                self._cls_session = cache.get_or_create(
                    self._cls_path, device=dev,
                    warmup_shape=(1, 3, 48, 192),
                )
                self._cls_in = self._cls_session.get_inputs()[0].name
                self._cls_out = [o.name for o in self._cls_session.get_outputs()]
            except Exception as e:
                self._cls_session = None
                from ..logging_config import get_logger

                get_logger(__name__).warning("方向分类器加载失败，已禁用: {}", e)
        elif self.use_cls:
            from ..logging_config import get_logger

            get_logger(__name__).info("未找到方向分类模型，180 度纠正不可用（属正常情况）")
            self._cls_session = None

        self._loaded = True

    def unload(self) -> None:
        self._det_session = None
        self._rec_session = None
        self._cls_session = None
        self._loaded = False

    # ---------------------------------------------------------------- 推理

    def recognize_image(self, image: np.ndarray, page_index: int = 0) -> PageResult:
        t0 = time.perf_counter()
        self.ensure_loaded()
        image = to_bgr(image)
        h, w = image.shape[:2]

        # --- 检测 ---
        t_pre = time.perf_counter()
        resized, shape = resize_keep_ratio(image, self.det_limit, self.det_limit)
        blob = normalize_db(resized)[np.newaxis, ...]
        t_det0 = time.perf_counter()
        prob = self._det_session.run(self._det_out, {self._det_in: blob})[0]
        t_det1 = time.perf_counter()
        det_boxes = self.db(prob, shape)

        page = PageResult(
            page_index=page_index,
            width=w,
            height=h,
            backend=self.name,
        )
        stage = {
            "det_preprocess": (t_det0 - t_pre) * 1000,
            "det_infer": (t_det1 - t_det0) * 1000,
        }
        if not det_boxes:
            page.processing_time = time.perf_counter() - t0
            page.debug = {"stage_ms": stage, "device": str(self._resolved_device or self.device_pref)}
            return page

        # --- 排序 + 裁剪 ---
        t_det2 = time.perf_counter()
        order = sort_reading_order([b for b in det_boxes])
        det_boxes = [det_boxes[i] for i in order]

        crops = [crop_quad(image, b) for b in det_boxes]
        t_det3 = time.perf_counter()
        stage["det_postprocess"] = (t_det3 - t_det2) * 1000

        t_rec0 = time.perf_counter()
        angles = self._classify_angles(crops)
        t_rec1 = time.perf_counter()
        rec_results = self._recognize(crops)
        t_rec2 = time.perf_counter()
        stage["cls"] = (t_rec1 - t_rec0) * 1000
        stage["rec"] = (t_rec2 - t_rec1) * 1000
        stage["crops"] = len(crops)

        # --- 组装 + 过滤 ---
        lines: List[TextLine] = []
        for box, rec, angle in zip(det_boxes, rec_results, angles):
            text, score = rec
            if score < self.drop_score:
                continue
            lines.append(
                TextLine(
                    text=text,
                    box=BoundingBox(points=[(float(x), float(y)) for x, y in box]),
                    confidence=score,
                    angle=angle,
                    line_id=len(lines),
                )
            )
        page.lines = lines
        page.text = "\n".join(ln.text for ln in lines)
        page.processing_time = time.perf_counter() - t0
        page.debug = {"stage_ms": stage, "device": str(self._resolved_device or self.device_pref)}
        return page

    # ---------------------------------------------------------------- 内部

    def _recognize(self, crops: Sequence[np.ndarray]) -> List[Tuple[str, float]]:
        """批量识别裁剪图，按固定宽度分桶合批。

        .. important::
           分桶的目的不是省算力，而是保证**结果可复现**。CTC 时间步数等于
           输入宽度除以 4，若padding 宽度取本批最宽者，同一张图在
           ``rec_batch_size=1`` 与 ``=32`` 下会得到不同的行数与文本。
           批大小本是性能参数，不应影响输出。

        padding 填充值为 ``+1.0``（归一化后的白底）而非 ``0.0``，
        后者相当于给模型补了一条灰边。
        """
        n = len(crops)
        if n == 0:
            return []

        prepared = [self._prepare_crop(c) for c in crops]
        widths = [p.shape[2] for p in prepared]
        # 每条文本的输入宽度：仅由自身宽度决定 -> 与分批方式无关
        target_widths = self._assign_width_buckets(widths)

        results: List[Tuple[str, float]] = [("", 0.0)] * n
        # 按目标宽度分组；同组内宽度完全相同，可安全合批
        groups: Dict[int, List[int]] = {}
        for i, w in enumerate(target_widths):
            groups.setdefault(w, []).append(i)

        for max_w in sorted(groups):
            indices = groups[max_w]
            for start in range(0, len(indices), self.rec_batch_size):
                chunk = indices[start : start + self.rec_batch_size]
                if not chunk:
                    continue
                blob = self._acquire_blob(len(chunk), max_w)
                for slot, i in enumerate(chunk):
                    arr = prepared[i]
                    w = arr.shape[2]
                    if w > max_w:  # 防御：正常不会发生
                        arr = arr[:, :, :max_w]
                        w = max_w
                    np.copyto(blob[slot, :, :, :w], arr)

                preds = self._rec_session.run(self._rec_out, {self._rec_in: blob})[0]
                for i, res in zip(chunk, self.decoder(preds)):
                    results[i] = res
        return results

    def _acquire_blob(self, batch: int, width: int) -> np.ndarray:
        """取一块已填好padding 值的输入缓冲，避免每批重新分配。

        宽度单调递增（档位由小到大），因此按 ``(宽度, 容量)`` 缓存已分配
        的最大块：大批复用小块会导致尾部数据残留，必须同时校验容量。
        填 ``+1.0`` 而非 ``0.0``——后者相当于给模型补了一条灰边。
        """
        cached = self._blob_cache
        if cached is not None and cached.shape[1:] == (3, self.rec_img_h, width):
            if cached.shape[0] < batch:
                cached = None
            else:
                cached = cached[:batch]
        else:
            cached = None

        if cached is None:
            cached = np.empty((max(batch, self.rec_batch_size), 3, self.rec_img_h, width), np.float32)
            self._blob_cache = cached

        cached.fill(1.0)
        return cached

    #: 识别输入宽度档位（常量），间隔约 1.4x：足够密以省 padding，
    #: 又足够疏以控制分组数量。
    _WIDTH_TIERS: Tuple[int, ...] = (
        32, 48, 64, 96, 128, 160, 200, 256, 320, 416, 544, 704, 896, 1152, 1472, 1920,
    )

    def _assign_width_buckets(self, widths: Sequence[int]) -> List[int]:
        """把每条文本映射到一个固定宽度档，档位是自身宽度的纯函数。"""
        tiers = self._WIDTH_TIERS
        out: List[int] = []
        for w in widths:
            for t in tiers:
                if w <= t:
                    out.append(t)
                    break
            else:
                # 超长文本：对齐到 32 的倍数，仍只取决于自身宽度
                out.append(max(int(math.ceil(w / 32) * 32), tiers[-1]))
        return out

    def _prepare_crop(self, crop: np.ndarray) -> np.ndarray:
        """把裁剪图缩放到识别高度并归一化到 [-1, 1]。

        归一化按 ``(x/255 - 0.5) / 0.5`` 合并为 ``x * (1/127.5) - 1`` 一步完成，
        避免"除 255 → 减 0.5 → 除 0.5"三趟内存遍历。
        通道维用 :func:`cv2.split` 直接填充，省掉 ``transpose`` 之后
        的一次大数组拷贝。

        .. important::
           系数取 ``1/127.5`` 而非 ``1/255`` 再做减法：若先转整型再减128，
           会引入 1/255 量级的截断误差（实测最大 0.0039），
           这足以改变模型的输入分布。
        """
        if crop.size == 0:
            return np.zeros((3, self.rec_img_h, 8), np.float32)
        h, w = crop.shape[:2]
        target_w = max(8, int(math.ceil(w * self.rec_img_h / max(h, 1))))
        interp = cv2.INTER_AREA if target_w < w else cv2.INTER_CUBIC
        resized = cv2.resize(crop, (target_w, self.rec_img_h), interpolation=interp)

        # cv2.split 免去 transpose 造成的不连续视图
        out = np.empty((3, self.rec_img_h, target_w), np.float32)
        for c, plane in enumerate(cv2.split(resized)):
            np.multiply(plane, np.float32(1.0 / 127.5), out=out[c])
            out[c] -= 1.0
        return out

    def _classify_angles(self, crops: List[np.ndarray]) -> List[float]:
        """180 度翻转分类，返回每张图被判定的角度（0.0 / 180.0）。

        判定为 180 度时**实际旋转裁剪图**：实测倒置文本的识别质量显著下降，
        不能只旋转图像却丢弃角度信息。

        Args:
            crops: 裁剪图列表，原地可能被替换为旋转后的版本。
        """
        if self._cls_session is None or not crops:
            return [0.0] * len(crops)

        # 批大小 32 而非 8：cls 输入固定 192 宽、算量极小，
        # 实测 bs=1 与 bs=32 单次推理同为 ~1.7ms，批大不涨耗时。
        step = 32
        img_h, img_w = 48, 192
        angles: List[float] = [0.0] * len(crops)
        for start in range(0, len(crops), step):
            chunk = crops[start : start + step]
            if self._cls_blob is None or self._cls_blob.shape[0] < len(chunk):
                self._cls_blob = np.empty((step, 3, img_h, img_w), np.float32)
            blob = self._cls_blob[: len(chunk)]
            blob.fill(1.0)
            for i, c in enumerate(chunk):
                if c.size == 0:
                    continue
                interp = cv2.INTER_AREA if img_w < c.shape[1] else cv2.INTER_CUBIC
                r = cv2.resize(c, (img_w, img_h), interpolation=interp)
                for ch, plane in enumerate(cv2.split(r)):
                    np.multiply(plane, np.float32(1.0 / 127.5), out=blob[i, ch])
                    blob[i, ch] -= 1.0

            # 模型输出已经是概率，不可再套 softmax，否则倒置图的180 度
            # 分数会从 0.907 被压到 0.693，跌破 0.9 阈值而使纠正永不生效。
            preds = self._cls_session.run(self._cls_out, {self._cls_in: blob})[0]
            probs = _as_probabilities(preds)

            for i, p in enumerate(probs):
                idx = int(np.argmax(p))
                conf = float(p[idx])
                # 阈值 0.9 取自官方实现：宁可漏纠也不误纠，
                # 误旋转会把正立的字变成倒字，代价远大于漏纠。
                if idx == 1 and conf > self._cls_thresh:
                    angles[start + i] = 180.0
                    crops[start + i] = cv2.rotate(chunk[i], cv2.ROTATE_180)
        return angles
