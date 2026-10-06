"""把 docs/images 下的 PNG 统一压成 JPEG。

截图产出的是 PNG（无损、体积600KB 上下），直接提交会让 README 首屏
变慢。这里按固定参数批量转一遍，使图例总重控制在 1MB 以内。

用法::

    python scripts/shrink_images.py                # 压缩 docs/images 下全部 PNG
    python scripts/shrink_images.py --quality 92   # 需要更清晰时提高质量

已存在的同名 JPEG 会被覆盖，转换后删除源 PNG。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

IMAGES = Path(__file__).resolve().parent.parent / "docs" / "images"


def main() -> int:
    ap = argparse.ArgumentParser(description="压缩 docs/images 下的截图")
    ap.add_argument("--quality", type=int, default=88, help="JPEG 质量，默认 88")
    ap.add_argument("--dir", type=Path, default=IMAGES, help="图片目录")
    args = ap.parse_args()

    if not args.dir.is_dir():
        print(f"目录不存在: {args.dir}")
        return 1

    total_before = total_after = 0
    for png in sorted(args.dir.glob("*.png")):
        jpg = png.with_suffix(".jpg")
        before = png.stat().st_size
        # 转 RGB：截图可能带 alpha 通道，JPEG 不支持，保留会报OSError
        Image.open(png).convert("RGB").save(
            jpg, "JPEG", quality=args.quality, optimize=True, progressive=True
        )
        after = jpg.stat().st_size
        png.unlink()
        total_before += before
        total_after += after
        print(f"{png.name} -> {jpg.name}  {before // 1024}KB -> {after // 1024}KB")

    if total_before:
        print(
            f"\n合计 {total_before // 1024}KB -> {total_after // 1024}KB"
            f"（省{(1 - total_after / total_before) * 100:.0f}%）"
        )
    else:
        print("没有需要转换的 PNG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
