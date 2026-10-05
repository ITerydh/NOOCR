"""性能脚本的公共引导。

统一处理两件事，避免每个脚本各写一份 ``sys.path`` 魔法：

1. 把项目根塞进 ``sys.path``，使脚本能从任意cwd 启动；
2. 提供 :data:`SAMPLES` 等常用路径常量。

被 ``bench_*`` / ``ab_*`` / ``prof_*`` 脚本import，**不要**单独运行。
"""

from __future__ import annotations

import sys
from pathlib import Path

#: 项目根目录（本文件位于 ``<root>/scripts/perf/_boot.py``）
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 内置示例图目录
SAMPLES = ROOT / "noocr" / "web" / "static"

#: 参与基准的示例图。名称与图片内容一一对应，便于人工核对识别质量。
IMAGES = [
    "ticket_train.jpg",
    "receipt_bank_statement.jpg",
    "medical_lab_report.jpg",
    "id_card_china.jpg",
    "scene_bank_branch.jpg",
]

__all__ = ["ROOT", "SAMPLES", "IMAGES"]
