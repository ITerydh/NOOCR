"""统一数据模型。

任何后端、任何模式统一返回 :class:`OCRResult`，调用方无需分支处理。
几何字段名显式标注坐标系，避免歧义。
"""

from __future__ import annotations

import math
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "AxisAlignedBox",
    "BackendCapabilities",
    "BoundingBox",
    "LayoutRegion",
    "OCRResult",
    "PageResult",
    "Point",
    "QuadBox",
    "RegionType",
    "Table",
    "TableCell",
    "TextLine",
    "Word",
]

Point = Tuple[float, float]


class RegionType(str, Enum):
    """版面区域类型。取值对齐 PP-DocLayout 标签体系，便于跨后端映射。"""

    TEXT = "text"
    TITLE = "title"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"
    HEADER = "header"
    FOOTER = "footer"
    SEAL = "seal"
    LIST = "list"
    CODE = "code"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """多边形包围盒。

    像素坐标，原点位于图像左上角，x 向右、y 向下。
    ``points`` 至少 3 点，四边形为标准 OCR 输出。
    """

    points: List[Point] = Field(..., min_length=3, max_length=8)

    @field_validator("points")
    @classmethod
    def _finite(cls, v: List[Point]) -> List[Point]:
        for x, y in v:
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                raise ValueError(f"坐标必须是数值，收到 {v}")
            if x != x or y != y:
                raise ValueError(f"坐标不能是 NaN，收到 {v}")
        return [(float(x), float(y)) for x, y in v]

    @property
    def is_quad(self) -> bool:
        return len(self.points) == 4

    @property
    def xmin(self) -> float:
        return min(p[0] for p in self.points)

    @property
    def ymin(self) -> float:
        return min(p[1] for p in self.points)

    @property
    def xmax(self) -> float:
        return max(p[0] for p in self.points)

    @property
    def ymax(self) -> float:
        return max(p[1] for p in self.points)

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        """鞋带公式计算多边形面积。"""
        pts = self.points
        n = len(pts)
        if n < 3:
            return 0.0
        s = 0.0
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return abs(s) * 0.5

    @property
    def center(self) -> Point:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def as_axis_aligned(self) -> "AxisAlignedBox":
        return AxisAlignedBox(x1=self.xmin, y1=self.ymin, x2=self.xmax, y2=self.ymax)

    def iou(self, other: "BoundingBox") -> float:
        """轴对齐 IoU。多边形相交场景下的实用近似。"""
        a, b = self.as_axis_aligned(), other.as_axis_aligned()
        ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
        ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = a.width * a.height + b.width * b.height - inter
        return inter / union if union > 0 else 0.0

    def to_array(self) -> List[float]:
        """展平为 ``[x1,y1,x2,y2,...]``，对齐 PaddleOCR 的 8 元素习惯。

        点数不足 4 时用最后一个点补齐，保证下游数组形状稳定。
        """
        pts = list(self.points)
        while len(pts) < 4:
            pts.append(pts[-1])
        out: List[float] = []
        for x, y in pts[:4]:
            out.extend((x, y))
        return out

    def sort_clockwise(self) -> "BoundingBox":
        """把点排成左上→右上→右下→左下，供透视变换使用。

        图像坐标系 y 轴向下，视觉顺时针对应 ``atan2(dy, dx)`` 递增，
        与数学坐标系相反。
        """
        pts = self.points
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        ordered = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        start = min(range(len(ordered)), key=lambda i: (ordered[i][1], ordered[i][0]))
        ordered = ordered[start:] + ordered[:start]
        return BoundingBox(points=ordered)


class QuadBox(BoundingBox):
    """四边形框，序列化时精简为 8 元素数组以减小 JSON 体积。"""

    @field_validator("points")
    @classmethod
    def _exactly_four(cls, v: List[Point]) -> List[Point]:
        if len(v) != 4:
            raise ValueError(f"QuadBox 必须恰好 4 个点，收到 {len(v)}")
        return [(float(x), float(y)) for x, y in v]


class AxisAlignedBox(BaseModel):
    """轴对齐框(x1,y1,x2,y2)，用于版面检测等输出矩形框的场景。

    不继承 :class:`BoundingBox`：父类的 ``points`` 是 pydantic 字段，
    子类以同名 property 覆盖会与字段定义冲突，故改为独立模型加显式转换。
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def to_polygon(self) -> BoundingBox:
        return BoundingBox(
            points=[(self.x1, self.y1), (self.x2, self.y1), (self.x2, self.y2), (self.x1, self.y2)]
        )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def iou(self, other: "AxisAlignedBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class Word(BaseModel):
    """单词级结果。仅部分后端提供。"""

    model_config = {"protected_namespaces": ()}

    text: str
    box: Optional[BoundingBox] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TextLine(BaseModel):
    """文本行——OCR 的基本输出单元。"""

    model_config = {"protected_namespaces": ()}

    text: str
    box: Optional[BoundingBox] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    words: List[Word] = Field(default_factory=list)
    #: 0=正常, 90/180/270=已纠正的旋转角度
    angle: float = 0.0
    #: 引擎内部 ID，便于跨步骤引用
    line_id: int = -1
    @property
    def has_words(self) -> bool:
        return len(self.words) > 0


class TableCell(BaseModel):
    """表格单元格。"""

    text: str = ""
    row: int = 0
    col: int = 0
    row_span: int = 1
    col_span: int = 1
    box: Optional[BoundingBox] = None
    confidence: float = 0.0


class Table(BaseModel):
    """表格结构化结果。"""

    #: 二维字符串表格，None 表示合并单元格产生的占位
    cells: List[List[str]] = Field(default_factory=list)
    #: 保留结构信息的单元格列表（含合并跨度）
    cell_details: List[TableCell] = Field(default_factory=list)
    box: Optional[BoundingBox] = None
    #: HTML 格式，便于前端直接渲染
    html: str = ""
    confidence: float = 0.0

class LayoutRegion(BaseModel):
    """版面区域。"""

    region_type: RegionType = RegionType.UNKNOWN
    box: BoundingBox
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    #: 区域序号，由版面分析给出的自然阅读顺序
    order: int = -1
    #: 关联的文本行索引（指向 PageResult.lines）
    line_indices: List[int] = Field(default_factory=list)
    table: Optional[Table] = None


class PageResult(BaseModel):
    """单页/单图结果。"""

    page_index: int = 0
    width: int = 0
    height: int = 0
    lines: List[TextLine] = Field(default_factory=list)
    regions: List[LayoutRegion] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    #: 整页文本，行优先拼接
    text: str = ""
    #: 整图倾斜角（弧度），部分后端可提供
    text_angle: Optional[float] = None
    processing_time: float = 0.0
    backend: str = ""
    #: 该页的渲染图（PDF/Office 场景用于多页浏览），不参与序列化
    image: Optional[Any] = Field(default=None, exclude=True, repr=False)
    #: 中间产物，调试用；序列化时按需排除
    debug: Dict[str, Any] = Field(default_factory=dict, exclude=True)


class OCRResult(BaseModel):
    """统一返回契约——所有后端、所有模式的唯一出口。"""

    pages: List[PageResult] = Field(default_factory=list)
    backend: str = ""
    total_time: float = 0.0
    #: 输入文档信息
    source_name: str = ""
    source_type: str = ""
    page_count: int = 0
    #: docx/xlsx/pptx 等格式直接提取的文本，准确率高于 OCR
    native_text: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """全部页面文本拼接。"""
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def all_lines(self) -> List[TextLine]:
        return [ln for p in self.pages for ln in p.lines]

    @property
    def all_tables(self) -> List[Table]:
        return [t for p in self.pages for t in p.tables]

    def to_markdown(self) -> str:
        """转为 Markdown。

        优先输出原生文本（docx/xlsx/pptx 自带文本层，准确率高于 OCR），
        否则按版面区域类型映射标题层级，表格以 HTML 嵌入。
        """
        if self.native_text:
            return self.native_text

        heading_marks = {
            RegionType.TITLE: "#",
            RegionType.HEADER: "##",
            RegionType.FOOTER: "##",
        }
        chunks: List[str] = []
        for page in self.pages:
            if len(self.pages) > 1:
                chunks.append(f"## 第 {page.page_index + 1} 页\n")
            for region in sorted(page.regions, key=lambda r: r.order):
                if not region.text:
                    continue
                mark = heading_marks.get(region.region_type)
                chunks.append(f"{mark} {region.text}\n" if mark else region.text + "\n")
            for table in page.tables:
                if table.html:
                    chunks.append(table.html + "\n")
            if not page.regions and page.text:
                chunks.append(page.text + "\n")
        return "\n".join(chunks).strip()

    def to_dict(self) -> Dict[str, Any]:
        """紧凑字典形式，供 JSON API 使用。"""
        return {
            "backend": self.backend,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "page_count": self.page_count,
            "total_time": round(self.total_time, 4),
            "text": self.text,
            "pages": [
                {
                    "page_index": p.page_index,
                    "width": p.width,
                    "height": p.height,
                    "text": p.text,
                    "processing_time": round(p.processing_time, 4),
                    "lines": [
                        {
                            "text": ln.text,
                            "confidence": round(ln.confidence, 4),
                            "box": ln.box.to_array() if ln.box and ln.box.is_quad else None,
                            "angle": ln.angle,
                            "words": [
                                {
                                    "text": w.text,
                                    "confidence": round(w.confidence, 4),
                                    "box": w.box.to_array() if w.box and w.box.is_quad else None,
                                }
                                for w in ln.words
                            ],
                        }
                        for ln in p.lines
                    ],
                    "tables": [
                        {"html": t.html, "confidence": round(t.confidence, 4)} for t in p.tables
                    ],
                    "regions": [
                        {
                            "type": r.region_type.value,
                            "order": r.order,
                            "confidence": round(r.confidence, 4),
                            "box": r.box.to_array() if r.box.is_quad else None,
                        }
                        for r in p.regions
                    ],
                }
                for p in self.pages
            ],
        }


class BackendCapabilities(BaseModel):
    """后端能力声明。

    管线据此自动编排：后端不支持表格结构化时，才启用本地表格模型补齐。
    """

    name: str
    #: 能给出单词级框
    word_level: bool = False
    #: 能检测版面区域
    layout_analysis: bool = False
    #: 能输出表格 HTML
    table_structure: bool = False
    #: 能直接输出 Markdown
    markdown: bool = False
    #: 能给出整图倾斜角
    text_angle: bool = False
    #: 支持的输入格式
    input_formats: List[str] = Field(default_factory=lambda: ["image"])
    #: 是否需要GPU
    requires_gpu: bool = False
    #: 是否需要额外原生依赖
    requires_native: bool = False
    #: 大致相对速度，1.0 为基准，仅用于 UI 展示
    speed_score: float = 1.0


def now_ms() -> float:
    """单调递增的毫秒计时器。"""
    return time.perf_counter() * 1000.0
