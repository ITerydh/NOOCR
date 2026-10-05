"""验证 180 度方向纠正是否真正生效（修复前只记录角度不旋转图像）。"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2, numpy as np
from noocr.backends.ppocr import PPOCRBackend
from noocr.engine.imageops import imread, imwrite

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "noocr/web/static"

src = STATIC / "exam_chinese_primary.jpg"
img = imread(src)

# 造一张 180 度旋转的图（模拟翻拍/倒置扫描）
rot = cv2.rotate(img, cv2.ROTATE_180)
out_path = ROOT / "tests/_rot180.jpg"
imwrite(out_path, rot)
print(f"原图 {img.shape[1]}x{img.shape[0]}  旋转后已存{out_path.name}")

be = PPOCRBackend(device="cpu", rec_batch_size=6, use_cls=True)
be.load()

for label, path in (("正立", src), ("倒置 180°", out_path)):
    im = imread(path)
    t = time.perf_counter()
    page = be.recognize_image(im)
    dt = (time.perf_counter() - t) * 1000
    lines = [ln for ln in page.lines if ln.text.strip()]
    n180 = sum(1 for ln in page.lines if ln.angle == 180.0)
    avg = sum(ln.confidence for ln in lines) / len(lines) if lines else 0.0
    print(f"\n{label}: {len(page.lines)} 行, 其中判定180度={n180} 行, "
          f"平均置信={avg:.3f}, 耗时={dt:.0f}ms")
    for ln in lines[:6]:
        print(f"    {ln.text[:44]}")

print("\n判读：倒置图应有相当比例的行被判为 180 度，且置信度应接近正立图。")
print("若倒置图 0 行被判 180 度 -> 方向分类器未生效，需检查模型或阈值。")

out_path.unlink(missing_ok=True)