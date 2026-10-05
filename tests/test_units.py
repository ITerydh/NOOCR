"""单元测试：不依赖模型，快速验证几何/排序/解码等纯逻辑。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import noocr.engine.imageops as iops

FAILED = []


def _shoelace(q):
    """多边形面积（鞋带公式）。q 可为 (N,2) ndarray 或点列表。"""
    pts = [(float(p[0]), float(p[1])) for p in q]
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return s / 2


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILED.append(name)


print("=== box_anchor ===")
check("标准四边形 TL",
      iops.box_anchor([[10, 20], [110, 22], [112, 52], [12, 50]]) == (10.0, 20.0))
check("角点乱序仍取最小",
      iops.box_anchor([[110, 22], [12, 50], [10, 20], [112, 52]]) == (10.0, 20.0))
check("空输入", iops.box_anchor([]) == (0.0, 0.0))
check("numpy 数组输入",
      iops.box_anchor(np.array([[5, 6], [7, 8]], np.float32)) == (5.0, 6.0))

print("\n=== sort_reading_order（本次修复的核心）===")
# 2 行 2 列，y 递增、每行内 x 递增
boxes = [
    np.array([[100, 200], [150, 200], [150, 220], [100, 220]], np.float32),  # 行0 右
    np.array([[10, 200], [60, 200], [60, 220], [10, 220]], np.float32),    # 行0 左
    np.array([[100, 300], [150, 300], [150, 320], [100, 320]], np.float32),  # 行1 右
    np.array([[10, 300], [60, 300], [60, 320], [10, 320]], np.float32),    # 行1 左
]
order = iops.sort_reading_order(boxes)
check("阅读顺序 = 行0左,行0右,行1左,行1右", order == [1, 0, 3, 2], f"得到 {order}")

# 关键回归：传 (4,2) 点集而非 (4,) 扁平列表
flat_boxes = [b[:, 0].tolist() for b in boxes]
order_flat = iops.sort_reading_order(flat_boxes)
check("旧式扁平输入不崩溃", isinstance(order_flat, list) and len(order_flat) == 4)

# 完全逆序输入
rev = list(reversed(boxes))
order_rev = iops.sort_reading_order(rev)
check("逆序输入结果一致（内容序）",
      [boxes[i][0][0] for i in order] == [rev[i][0][0] for i in order_rev],
      f"{[boxes[i][0][0] for i in order]} vs {[rev[i][0][0] for i in order_rev]}")

# 角点顺序打乱
shuf = [np.array([[150, 200], [100, 200], [100, 220], [150, 220]], np.float32),
        np.array([[60, 200], [10, 200], [10, 220], [60, 220]], np.float32),
        np.array([[150, 300], [100, 300], [100, 320], [150, 320]], np.float32),
        np.array([[60, 300], [10, 300], [10, 320], [60, 320]], np.float32)]
check("角点顺序不影响结果", iops.sort_reading_order(shuf) == order,
      f"{iops.sort_reading_order(shuf)} vs {order}")

# y 单调性：不同行必须递增
ys = [iops.box_anchor(boxes[i])[1] for i in order]
check("行间 y 递增", all(ys[i] <= ys[i + 1] + 10.0 for i in range(len(ys) - 1)), f"{ys}")

print("\n=== order_quad ===")
q = iops.order_quad(np.array([[110, 20], [110, 50], [10, 50], [10, 20]], np.float32))
check("TL 最小", abs(q[0][0] - 10) < 1e-3 and abs(q[0][1] - 20) < 1e-3, f"{q[0]}")
check("BR 最大", abs(q[2][0] - 110) < 1e-3 and abs(q[2][1] - 50) < 1e-3, f"{q[2]}")

print("\n=== _min_area_quad（postprocess）===")
from noocr.backends.postprocess import _min_area_quad

ret = _min_area_quad(np.array([[10, 20], [110, 22], [112, 52], [12, 50]], np.float32))
q1, short_side = ret  # 返回 (四角坐标, 短边长)
check("返回结构为 (points, short)", isinstance(q1, np.ndarray) and q1.shape == (4, 2),
      f"{type(q1)} {getattr(q1,'shape',None)}")
check("面积接近矩形面积(3000)", abs(abs(_shoelace(q1)) - 3000) < 400,
      f"得到 {abs(_shoelace(q1)):.0f}")
check("短边长约30", abs(short_side - 30) < 3, f"得到 {short_side:.1f}")

print("\n=== 归一化 / 缩放 ===")
img = np.full((100, 200, 3), 255, np.uint8)
n = iops.normalize_db(img)
# DB 用 ImageNet 统计量：(255/255 - mean) / std，白底应得到较大正值
check("输出形状为 CHW", n.shape == (3, 100, 200), f"{n.shape}")
check("白底归一化为正", n.min() > 1.0, f"min={n.min():.2f} max={n.max():.2f}")
check("三通道均值按mean递减(0.485>0.456>0.406)", n[0].mean() < n[1].mean() < n[2].mean(),
      f"{[round(float(n[c].mean()),2) for c in range(3)]}")

r, meta = iops.resize_keep_ratio(img, 50, 50)
src_h, src_w, rh, rw = meta
check("缩放返回4元组meta", len(meta) == 4, f"{meta}")
check("尺寸对齐到32的倍数", r.shape[0] % 32 == 0 and r.shape[1] % 32 == 0, f"{r.shape}")
check("长边不超过目标", max(r.shape[:2]) <= 64, f"{r.shape}")
check("ratio 可反算原尺寸", abs(r.shape[1] / rw - src_w) < 2, f"{r.shape} rw={rw}")

print("\n=== OPERATORS 白名单 ===")
try:
    ops = iops.build_pipeline([("DetResizeForTest", {"limit": 960}), ("NormalizeImage", {})])
    check("白名单算子可构建", len(ops) == 2, f"得到 {len(ops)} 个")
    out = ops[0](img)
    check("算子可执行", out is not None and out.ndim == 3, f"{None if out is None else out.shape}")
except Exception as e:
    check("白名单算子可构建", False, f"{type(e).__name__}: {e}")
try:
    iops.build_pipeline([("__import__('os').system('echo pwned')", {})])
    check("恶意算子被拒绝", False, "竟然执行了！")
except KeyError:
    check("恶意算子被拒绝", True)
except Exception as e:
    check("恶意算子被拒绝", False, f"抛了非 KeyError: {e}")

print("\n=== imread 接受 Path ===")
ROOT = Path(__file__).resolve().parents[1]
test_img = ROOT / "noocr/web/static/exam_chinese_primary.jpg"
r1 = iops.imread(test_img)
check("Path 对象可读", r1 is not None and r1.ndim == 3)
r2 = iops.imread(str(test_img))
check("str 可读且一致", r2 is not None and np.array_equal(r1, r2))
check("不存在的文件返回 None", iops.imread(ROOT / "no_such_file_xyz.png") is None)

print("\n=== _assign_width_buckets（固定档位，保证可复现）===")
from noocr.backends.ppocr import PPOCRBackend

# 该方法是纯函数，不触碰任何权重；绕过 __init__ 以免 CI 缺字典文件时失败
b = object.__new__(PPOCRBackend)
ws = [10, 32, 33, 64, 100, 320, 321, 700, 1920, 5000]
tw = b._assign_width_buckets(ws)
check("长度一致", len(tw) == len(ws))
check("每档>=原宽", all(t >= w for t, w in zip(tw, ws)), f"{list(zip(ws, tw))}")
check("纯函数（同输入同输出）", b._assign_width_buckets(ws) == tw)
# 关键契约：只取决于自身宽度，与同批内容无关
check("宽度不受邻居影响",
      b._assign_width_buckets([100]) == [b._assign_width_buckets([100, 2000])[0]],
      f"{b._assign_width_buckets([100])} vs {b._assign_width_buckets([100,2000])}")
check("多档位被有效合并", len(set(b._assign_width_buckets([100, 110, 120]))) == 1,
      f"{b._assign_width_buckets([100,110,120])}")
check("档位为4的倍数(CTC下采样)", all(t % 4 == 0 for t in tw), f"{tw}")
check("超长文本被处理", tw[-1] >= 5000, f"{tw[-1]}")
check("空输入安全", b._assign_width_buckets([]) == [])

print("\n=== 已移除的方法不应残留 ===")
check("_safe_batch_size 已删除", not hasattr(b, "_safe_batch_size"))
check("_bucket_by_width 已删除", not hasattr(b, "_bucket_by_width"))

print("\n" + "=" * 50)
print(f"失败 {len(FAILED)} 项" + (f": {FAILED}" if FAILED else "，全部通过"))
sys.exit(1 if FAILED else 0)
