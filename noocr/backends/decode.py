"""识别结果解码。

热路径完全向量化：先argmax 拿到索引与概率，再用「与前一个不同」的布尔
掩码一次性剔除连续重复，避免 Python 逐字符循环。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from ..logging_config import get_logger

log = get_logger(__name__)

__all__ = ["CTCDecoder", "load_character_dict"]


def load_character_dict(
    path: Union[str, Path], use_space: bool = True
) -> Tuple[List[str], int]:
    """读取字符字典。

    Returns:
        ``(字符列表, blank 的索引)``。blank 固定为 0（CTC 约定）。
    """
    chars: List[str] = []
    with open(path, "rb") as f:
        for line in f:
            chars.append(line.decode("utf-8").strip("\n").strip("\r\n"))
    if use_space and " " not in chars:
        chars.append(" ")
    # blank 占索引 0
    return ["<blank>"] + chars, 0


class CTCDecoder:
    """CTC 贪心解码 + 置信度计算。

    置信度定义为有效字符概率的算术平均，下游 ``drop_score`` 阈值按此口径校准。
    """

    def __init__(
        self,
        character_dict_path: Union[str, Path],
        use_space: bool = True,
        blank_id: int = 0,
    ):
        self.charset, self.blank_id = load_character_dict(character_dict_path, use_space)
        # 预先构造 numpy 版本，避免逐字符 Python 查表
        self._charset_array = np.array(self.charset, dtype=object)
        self._num_classes = len(self.charset)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def __call__(self, preds: np.ndarray) -> List[Tuple[str, float]]:
        """解码一批识别结果。

        Args:
            preds: ``(B, T, C)`` 的logits/概率。

        Returns:
            ``[(text, confidence), ...]``，长度与 batch 一致。
        """
        if preds.ndim != 3:
            raise ValueError(f"期望 (B, T, C) 三维输入，收到 {preds.shape}")

        # argmax + max 一次性拿到索引和概率
        indices = preds.argmax(axis=2)
        probs = preds.max(axis=2)

        results: List[Tuple[str, float]] = []
        for b in range(preds.shape[0]):
            results.append(self._decode_one(indices[b], probs[b]))
        return results

    def _decode_one(self, indices: np.ndarray, probs: np.ndarray) -> Tuple[str, float]:
        """解码单行。

        步骤：去掉 blank 与连续重复 → 取有效字符概率均值。
        连续重复的去除用 ``np.flatnonzero(np.diff(idx))`` 向量化，
        避免 Python 逐字符循环。
        """
        # 1) 去掉 blank，同时记录有效位置
        keep = indices != self.blank_id
        kept_idx = indices[keep]
        kept_prob = probs[keep]
        if kept_idx.size == 0:
            return "", 0.0

        # 2) 去掉连续重复：保留「与前一个不同」的位置，以及第一个
        if kept_idx.size > 1:
            change = np.empty(kept_idx.size, dtype=bool)
            change[0] = True
            np.not_equal(kept_idx[1:], kept_idx[:-1], out=change[1:])
            kept_idx = kept_idx[change]
            kept_prob = kept_prob[change]

        if kept_idx.size == 0:
            return "", 0.0

        # 3) 越界保护：模型输出维度与字典不一致时跳过非法字符，不中断整批
        valid = kept_idx < self._num_classes
        if not valid.all():
            bad = int((~valid).sum())
            log.debug("跳过 {} 个越界字符索引（模型输出维度与字典不匹配）", bad)
            kept_idx = kept_idx[valid]
            kept_prob = kept_prob[valid]
            if kept_idx.size == 0:
                return "", 0.0

        text = "".join(self._charset_array[i] for i in kept_idx)
        return text, float(kept_prob.mean())

    def decode_with_positions(
        self, preds: np.ndarray
    ) -> List[Tuple[str, float, List[int]]]:
        """额外返回每行有效字符的时间步位置。

        用于「把识别结果对齐回检测框」——一些下游任务（如逐字高亮、
        关键词定位）需要知道每个字符出现在序列的哪个位置。
        """
        if preds.ndim != 3:
            raise ValueError(f"期望 (B, T, C) 三维输入，收到 {preds.shape}")
        indices = preds.argmax(axis=2)
        probs = preds.max(axis=2)

        out = []
        for b in range(preds.shape[0]):
            idx_row, prob_row = indices[b], probs[b]
            keep = np.flatnonzero(idx_row != self.blank_id)
            if keep.size == 0:
                out.append(("", 0.0, []))
                continue
            kept_idx = idx_row[keep]
            kept_prob = prob_row[keep]
            if kept_idx.size > 1:
                change = np.empty(kept_idx.size, dtype=bool)
                change[0] = True
                np.not_equal(kept_idx[1:], kept_idx[:-1], out=change[1:])
                kept_idx, kept_prob, keep = kept_idx[change], kept_prob[change], keep[change]
            valid = kept_idx < self._num_classes
            kept_idx, kept_prob, keep = kept_idx[valid], kept_prob[valid], keep[valid]
            text = "".join(self._charset_array[i] for i in kept_idx)
            out.append((text, float(kept_prob.mean()) if kept_prob.size else 0.0, keep.tolist()))
        return out
