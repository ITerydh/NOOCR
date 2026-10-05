"""PP-OCRv6 后端：CPU 上的提速主路径。

独立成后端而非给 v5 加参数，因为二者在三处不兼容，混在一起会让两个版本的
行为互相污染：

===================  ==========================  ==========================
环节                 PP-OCRv5                   PP-OCRv6
===================  ==========================  ==========================
DB 阈值              thresh 0.3 / box 0.5 / 1.6  thresh 0.2 / box 0.45 / 1.4
候选框上限1000                        3000
检测输入尺寸         长边≤960（长方形）           短边自适应 736，不补白边
字典                 6623 类中英混排             18710 类（small）/ 6906 类（tiny）
===================  ==========================  ==========================

官方 ``inference.yml`` 中 ``DetResizeForTest: null``，即不再补白边到正方形，
而是短边缩放到 736、长边按比例（上限 4000）。沿用 v5 的「长边 960 + 对齐 32」
会导致检测框偏移。

实测档位（见 ``tests/bench_models.py``）：

=========  ======  ======  =======  ==========================
档位det  rec    字典类别  适用场景
=========  ======  ======  =======  ==========================
``tiny``  1.8MB  4.3MB   6906     边缘/移动端，追求极致速度
``small`` 9.5MB  21MB    18710    默认档，平衡速度与精度
=========  ======  ======  =======  ==========================
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..engine.base import BackendUnavailable, OCRBackend
from ..engine.imageops import crop_quad, normalize_db, sort_reading_order, to_bgr
from ..engine.session import detect_device, get_global_cache
from ..models import model_path
from ..types import BackendCapabilities, BoundingBox, PageResult, TextLine
from .decode import CTCDecoder
from .postprocess import DBPostProcess

__all__ = ["PPOCRv6Backend", "V6_TIERS"]

#: 可用的 v6 档位 -> (models 清单键, 展示名)
V6_TIERS: Dict[str, str] = {
    "tiny": "ppocrv6-tiny",
    "small": "ppocrv6-small",
}


class PPOCRv6Backend(OCRBackend):
    """PP-OCRv6 通用文字 OCR。

    Args:
        tier: ``"tiny"`` 或 ``"small"``。
        device: ``"auto"`` / ``"cpu"`` / ``"cuda"`` / ``"dml"``。
        rec_batch_size: 识别批大小。**不影响输出结果**。
        drop_score: 置信度低于此值的文本行被丢弃。
        use_cls: 是否启用 180 度翻转纠正（复用 v5 的 cls 模型）。
    """

    capabilities = BackendCapabilities(
        name="ppocrv6",
        word_level=False,
        layout_analysis=False,
        table_structure=False,
        markdown=False,
        text_angle=True,
        input_formats=["image", "pdf", "docx", "xlsx"],
        requires_gpu=False,
        # 官方数据：检测较 v5 +4.6%、识别 +5.1%
        speed_score=2.0,
    )

    #: 官方 inference.yml：PreProcess 与 DBPostProcess 的取值
    _DET_SHORT_SIDE = 736
    _DET_MAX_LONG_SIDE = 4000
    #: 检测输入高宽必须对齐的倍数。
    #:
    #: **必须是 32，不能是 16**——这是实测踩过的坑。初版按 16 对齐，
    #: 遇到 976（=16x61）依然崩溃：
    #:
    #:     Shape mismatch attempting to re-use buffer.
    #:     {1,64,46,61} != {1,64,46,62}
    #:
    #: 解读：736/16 = 46（第一层下采样），976/16 = 61（第二层）。
    #: 61 是**质数**，无法被 2 整除，而模型内部还有一次 2 倍上采样，
    #: 于是 61 与 62 两条路径撞在同一 buffer 上。
    #:
    #: 实测（det_tiny，独立 session 逐个验证）：
    #:   - 736x736 / 736x1152 / 704x704 等**32 倍数**全部通过
    #:   - 736x976（16 倍数但非 32 倍数）**必崩**，与
    #:     ``enable_cpu_mem_arena`` 开关**无关**
    #:
    #: 详见 :meth:`_resize_for_det`。
    _DET_ALIGN = 32
    _DB_THRESH = 0.2
    _DB_BOX_THRESH = 0.45
    _DB_UNCLIP = 1.4
    _DB_MAX_CANDIDATES = 3000

    #: 识别输入宽度档位（与 v5 相同，保证两个后端结果可逐项对比）
    _WIDTH_TIERS: Tuple[int, ...] = (
        32, 48, 64, 96, 128, 160, 200, 256, 320, 416, 544, 704, 896, 1152, 1472, 1920,
    )

    #: 方向分类的批大小。该模型输入固定且算量极小，32 与 1 同速。
    _CLS_BATCH = 32

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        *,
        tier: str = "small",
        device: str = "auto",
        rec_batch_size: int = 6,
        rec_img_h: int = 48,
        drop_score: float = 0.5,
        use_cls: bool = True,
        threads: int = 0,
    ):
        if tier not in V6_TIERS:
            raise ValueError(
                f"tier 必须是 {sorted(V6_TIERS)} 之一，收到 {tier!r}"
            )
        key = V6_TIERS[tier]
        super().__init__(f"ppocrv6-{tier}", model_dir)
        self.tier = tier
        self.device_pref = device
        self.rec_batch_size = max(1, rec_batch_size)
        self.rec_img_h = rec_img_h
        self.drop_score = drop_score
        self.use_cls = use_cls
        self.threads = threads
        #: 复用的识别输入缓冲，见 :meth:`_acquire_blob`
        self._blob_cache: Optional[np.ndarray] = None
        #: 复用的方向分类输入缓冲，形状固定故只分配一次
        self._cls_blob: Optional[np.ndarray] = None

        self._det_path = model_path(key, f"PP-OCRv6_det_{tier}.onnx")
        self._rec_path = model_path(key, f"PP-OCRv6_rec_{tier}.onnx")
        self._dict_path = model_path(
            key,
            "ppocrv6_tiny_dict.txt" if tier == "tiny" else "ppocrv6_dict.txt",
        )
        self._cls_path = model_path(key, "cls.onnx")

        self.db = DBPostProcess(
            thresh=self._DB_THRESH,
            box_thresh=self._DB_BOX_THRESH,
            unclip_ratio=self._DB_UNCLIP,
            max_candidates=self._DB_MAX_CANDIDATES,
        )
        self.decoder = CTCDecoder(self._dict_path)
        self._cls_session = None

    # ---------------------------------------------------------------- 生命周期

    def load(self) -> None:
        if self._loaded:
            return
        for path, name in (
            (self._det_path, "det"),
            (self._rec_path, "rec"),
            (self._dict_path, "dict"),
        ):
            if not Path(path).is_file():
                raise BackendUnavailable(
                    f"缺少 PP-OCRv6-{self.tier} 的 {name} 模型: {path}\n"
                    f"下载: python -m noocr models --backend {V6_TIERS[self.tier]}"
                )

        dev = detect_device(self.device_pref)
        cache = get_global_cache()

        self._det_session = cache.get_or_create(
            self._det_path, device=dev,
            # 短边 736 是固定值，长边随图变化但下界固定
            warmup_shape=(1, 3, self._DET_SHORT_SIDE, self._DET_SHORT_SIDE),
            dynamic_shape=True,
        )
        self._rec_session = cache.get_or_create(
            self._rec_path, device=dev,
            warmup_shape=(1, 3, self.rec_img_h, 320),
            dynamic_shape=True,
        )
        self._det_in = self._det_session.get_inputs()[0].name
        self._det_out = [o.name for o in self._det_session.get_outputs()]
        self._rec_in = self._rec_session.get_inputs()[0].name
        self._rec_out = [o.name for o in self._rec_session.get_outputs()]

        # 目录一致性检查：CTCDecoder 的 num_classes 已含 blank(索引0) 与
        # 追加的 space，因此必须与模型输出维度**严格相等**。
        # 字典文件本身少 2 行（blank + space 是加载器加的），
        # 所以「字典行数 + 2 == 模型维度」才是判据。
        model_c = self._rec_session.get_outputs()[0].shape[-1]
        if isinstance(model_c, int) and model_c != self.decoder.num_classes:
            from ..logging_config import get_logger

            get_logger(__name__).error(
                "PP-OCRv6-{tier} 字典({d} 类含 blank+space)与模型({m} 类)"
                "不匹配，解码结果必然错乱。请检查字典文件是否为 {tier} 档专用版本"
                "（tiny 与 small 字典不同）",
                tier=self.tier,
                d=self.decoder.num_classes,
                m=model_c,
            )
            raise BackendUnavailable(
                f"PP-OCRv6-{self.tier} 字典与模型不匹配: "
                f"字典 {self.decoder.num_classes} 类 vs 模型 {model_c} 类"
            )

        if self.use_cls and Path(self._cls_path).is_file():
            self._cls_session = cache.get_or_create(
                self._cls_path, device=dev, warmup_shape=(1, 3, 48, 192)
            )
            self._cls_in = self._cls_session.get_inputs()[0].name
            self._cls_out = [o.name for o in self._cls_session.get_outputs()]

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

        # --- 检测（v6 专用预处理：短边 736，不补白边）---
        t_pre = time.perf_counter()
        resized, shape = self._resize_for_det(image)
        blob = normalize_db(resized)[np.newaxis, ...]
        t_det0 = time.perf_counter()
        prob = self._run_det(blob)
        t_det1 = time.perf_counter()
        det_boxes = self.db(prob, shape)

        page = PageResult(
            page_index=page_index, width=w, height=h, backend=self.name
        )
        stage = {
            "det_preprocess": (t_det0 - t_pre) * 1000,
            "det_infer": (t_det1 - t_det0) * 1000,
        }
        if not det_boxes:
            page.processing_time = time.perf_counter() - t0
            page.debug = {"stage_ms": stage, "device": str(self.device_pref)}
            return page

        # --- 排序 + 裁剪 ---
        t_det2 = time.perf_counter()
        order = sort_reading_order([b for b in det_boxes])
        det_boxes = [det_boxes[i] for i in order]
        crops = [crop_quad(image, b) for b in det_boxes]
        t_det3 = time.perf_counter()
        stage["det_postprocess"] = (t_det3 - t_det2) * 1000

        # --- 方向纠正 + 识别 ---
        t_rec0 = time.perf_counter()
        angles = self._classify_angles(crops)
        t_rec1 = time.perf_counter()
        rec_results = self._recognize(crops)
        t_rec2 = time.perf_counter()
        stage["cls"] = (t_rec1 - t_rec0) * 1000
        stage["rec"] = (t_rec2 - t_rec1) * 1000
        stage["crops"] = len(crops)

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
        page.debug = {"stage_ms": stage, "device": str(self.device_pref)}
        return page

    # ---------------------------------------------------------------- 内部

    def _resize_for_det(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """v6 检测预处理：**短边**缩放到 736，长边按比例且不超过 4000。

        官方 ``inference.yml`` 里 ``DetResizeForTest: null``，即 v6
        **不再补白边到正方形**。这与 v5 的「长边 960 + 对齐 32」是两个
        不同的几何变换——沿用 v5 会让所有检测框的坐标映射出错。

        .. warning:: 尺寸必须对齐到 32 的倍数，否则**必崩**

           PP-OCRv6 的 ONNX 导出图在特征图上采样阶段把**两次倍频**
           （16x 与 2x）烘焙进了形状推导。模型声明的输出维度是动态符号
           （``ConvTranspose_459_o0__d2`` / ``__d3``），看起来支持任意尺寸，
           实则有硬性约束::

               Shape mismatch attempting to re-use buffer.
               {1,64,46,61} != {1,64,46,62}

           ``46 = 736/16``、``61 = 976/16``；61 是质数，无法被 2 整除，
           而模型内部还有一次 2 倍上采样，两条路径因此撞在同一 buffer 上。

           实测：736x736、736x1152、704x704（均为 32 倍数）全部通过；
           736x976（16 倍数但非 32 倍数）**必崩**，且与
           ``enable_cpu_mem_arena`` / ``enable_mem_pattern`` 的开关**无关**——
           这是模型图的固有约束，不是 ORT 配置问题。

           因此这里按 32 向上取整。``ratio`` 必须用 resize 后的**真实**
           高宽计算，不能用目标值，否则所有检测框坐标会偏移。
        """
        src_h, src_w = image.shape[:2]
        short = min(src_h, src_w)
        long_side = max(src_h, src_w)

        # 放大倍数上限 2.0：避免小图被过度插值后只增加算力不增信息
        scale = min(self._DET_SHORT_SIDE / max(short, 1), 2.0)
        if long_side * scale > self._DET_MAX_LONG_SIDE:
            scale = self._DET_MAX_LONG_SIDE / long_side

        # 关键：对齐到 32 的倍数（见上方 warning）
        align = self._DET_ALIGN
        new_h = max(align, int(math.ceil(src_h * scale / align)) * align)
        new_w = max(align, int(math.ceil(src_w * scale / align)) * align)

        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
        return resized, (src_h, src_w, new_h / src_h, new_w / src_w)

    def _run_det(self, blob: np.ndarray) -> np.ndarray:
        """执行检测推理，形状非法时自动降级重试。

        :meth:`_resize_for_det` 已保证输入是 32 的倍数，但这个约束来自
        **ONNX 导出图的实现细节**，不是官方文档承诺——将来换权重版本
        可能就变了。因此这里再加一层运行期兜底：捕获形状类异常并把
        输入补到最近的安全尺寸后重试，而不是让整个OCR 调用崩掉。

        Args:
            blob: ``(1, 3, H, W)`` 的 float32 输入。

        Returns:
            ``(1, 1, h, w)`` 的概率图，``h = H//4``、``w = W//4``。
        """
        try:
            return self._det_session.run(self._det_out, {self._det_in: blob})[0]
        except Exception as e:
            msg = str(e)
            if "Shape mismatch" not in msg and "Resize" not in msg:
                raise

        from ..logging_config import get_logger

        log = get_logger(__name__)
        _, _, h, w = blob.shape
        safe_h = max(32, int(math.ceil(h / 32) * 32))
        safe_w = max(32, int(math.ceil(w / 32) * 32))
        log.warning(
            "检测模型拒绝了 {h}x{w} 的输入（形状约束可能已变更），"
            "已补齐到 {sh}x{sw} 重试",
            h=h, w=w, sh=safe_h, sw=safe_w,
        )
        padded = np.ones((1, 3, safe_h, safe_w), np.float32)
        padded[:, :, :h, :w] = blob
        return self._det_session.run(self._det_out, {self._det_in: padded})[0]

    def _assign_width_buckets(self, widths: Sequence[int]) -> List[int]:
        """把每条文本映射到**固定**宽度档（与 v5 同源）。"""
        tiers = self._WIDTH_TIERS
        out: List[int] = []
        for w in widths:
            for t in tiers:
                if w <= t:
                    out.append(t)
                    break
            else:
                out.append(max(int(math.ceil(w / 32) * 32), tiers[-1]))
        return out

    def _prepare_crop(self, crop: np.ndarray) -> np.ndarray:
        """把裁剪图缩放到识别高度并归一化到 [-1, 1]。

        归一化按 ``(x/255 - 0.5) / 0.5`` 合并为 ``x * (1/127.5) - 1`` 一步完成，
        避免"除 255 → 减 0.5 → 除 0.5"三趟大数组遍历。通道维用
        :func:`cv2.split` 直接填充，省掉 ``transpose`` 之后的一次拷贝。

        .. important::
           系数取 ``1/127.5`` 而非先转整型再减 128：后者会引入
           1/255 量级（0.0039）的截断误差，足以改变模型输入分布。
        """
        if crop.size == 0:
            return np.zeros((3, self.rec_img_h, 8), np.float32)
        h, w = crop.shape[:2]
        target_w = max(8, int(math.ceil(w * self.rec_img_h / max(h, 1))))
        interp = cv2.INTER_AREA if target_w < w else cv2.INTER_CUBIC
        resized = cv2.resize(crop, (target_w, self.rec_img_h), interpolation=interp)

        out = np.empty((3, self.rec_img_h, target_w), np.float32)
        for c, plane in enumerate(cv2.split(resized)):
            np.multiply(plane, np.float32(1.0 / 127.5), out=out[c])
            out[c] -= 1.0
        return out

    def _acquire_blob(self, batch: int, width: int) -> np.ndarray:
        """取一块已填好 padding 值的输入缓冲，避免每批重新分配。

        实测（bs=6 w=1152）``np.full`` 每次要 0.77ms，复用后仅 0.08ms。
        GPU 上更关键：新建的数组不是 pinned 内存，``session.run``
        每次都要额外走一趟可 pageable 的 H2D 拷贝。

        缓存键必须同时校验宽度**与容量**：只按宽度命中会让大批复用
        小批的缓冲，尾部残留上一批的数据。
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
            cached = np.empty(
                (max(batch, self.rec_batch_size), 3, self.rec_img_h, width), np.float32
            )
            self._blob_cache = cached

        # 白底 = +1.0。填 0.0 相当于给模型补一条灰边，会诱发幻觉字符
        cached.fill(1.0)
        return cached

    def _recognize(self, crops: Sequence[np.ndarray]) -> List[Tuple[str, float]]:
        n = len(crops)
        if n == 0:
            return []
        prepared = [self._prepare_crop(c) for c in crops]
        target = self._assign_width_buckets([p.shape[2] for p in prepared])

        results: List[Tuple[str, float]] = [("", 0.0)] * n
        groups: Dict[int, List[int]] = {}
        for i, w in enumerate(target):
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
                    w = min(arr.shape[2], max_w)
                    np.copyto(blob[slot, :, :, :w], arr[:, :, :w])

                preds = self._rec_session.run(self._rec_out, {self._rec_in: blob})[0]
                for i, res in zip(chunk, self.decoder(preds)):
                    results[i] = res
        return results

    def _classify_angles(self, crops: List[np.ndarray]) -> List[float]:
        """180 度纠正。复用 v5 的 cls 模型（v6 未自带分类器）。

        批大小取 32 而非 8：实测 cls 模型 bs=1 与 bs=32 单次推理都是
        ~1.7ms（输入固定 192 宽、算量极小），批大一倍不涨耗时，
        却能把串行调用次数压到 1/4。
        """
        if self._cls_session is None or not crops:
            return [0.0] * len(crops)

        from .ppocr import _as_probabilities

        angles: List[float] = [0.0] * len(crops)
        img_h, img_w = 48, 192
        step = self._CLS_BATCH
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

            preds = self._cls_session.run(self._cls_out, {self._cls_in: blob})[0]
            for i, p in enumerate(_as_probabilities(preds)):
                idx = int(np.argmax(p))
                if idx == 1 and float(p[idx]) > 0.9:
                    angles[start + i] = 180.0
                    crops[start + i] = cv2.rotate(chunk[i], cv2.ROTATE_180)
        return angles
