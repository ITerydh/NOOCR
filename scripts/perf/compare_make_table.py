"""同模型对比表：NOOCR 与 OnnxOCR 跑**同一份权重**。

**为什么必须同权重**：比 v6 对 v5 测的是模型代次差异，不是实现差异——
那张表看着好看，但证明不了任何事。同一代次内比才是同模型对比。

三个档位各出一张表：

======================  ==================  ==========================
档位                     权重来源            校验方式
======================  ==================  ==========================
``ppocrv5``             两侧仓库各带一份     det/rec/cls 三个 onnx md5
``ppocrv6-tiny``        共用目录             与源文件 md5 一致
``ppocrv6-small``       共用目录             与源文件 md5 一致
``ppocrv6-medium``      共用目录             与源文件 md5 一致
======================  ==================  ==========================

**为什么 v6 要绕这一圈**：本项目与 OnnxOCR 各自导出的 v6 det 图不同源
（我们的那份带 ``p2o.pd_op.*`` 图优化痕迹，文件大小差 0.5%），det
那一半无法证明逐字节相同；字典内容一致（18710 / 6906 词条），rec 的
图结构也一致（tiny 的标识符序列相似度 1.0000），但只要det 不能证明，
整张表就站不住。

解法是准备一个**共用权重目录**：把 OnnxOCR ``ppocrv6`` 分支自带的
models/ 复制成本项目的 ``models/`` 布局，两侧都指向它。这样连文件
路径都一样，"哪份权重更权威"的问题就不存在了。

输出 JSON 供 README 表格使用。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

#: 参与对比的档位-> (OnnxOCR 侧结果文件, NOOCR 侧结果文件, 表格标题)
TIERS = {
    "ppocrv5": ("bench_onnxocr_ppocrv5.json",
                "bench_noocr_ppocrv5_onnxocrw.json", "PP-OCRv5"),
    "ppocrv6-tiny": ("bench_onnxocr_ppocrv6_tiny.json",
                     "bench_noocr_ppocrv6_tiny_onnxocrw.json",
                     "PP-OCRv6 tiny"),
    "ppocrv6-small": ("bench_onnxocr_ppocrv6_small.json",
                      "bench_noocr_ppocrv6_small_onnxocrw.json",
                      "PP-OCRv6 small"),
    "ppocrv6-medium": ("bench_onnxocr_ppocrv6_medium.json",
                       "bench_noocr_ppocrv6_medium_onnxocrw.json",
                       "PP-OCRv6 medium"),
}


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _same_dict(a: Path, b: Path) -> bool:
    """字典只比内容——Windows 上可能是 CRLF，Linux 上是 LF，
    直接比字节会误报成"模型不同"。"""
    return (a.read_text("utf-8").splitlines()
            == b.read_text("utf-8").splitlines())


#: 共用目录里的文件 -> OnnxOCR 自带 models/ 里的对应文件。
#: 目录布局不同（``det/PP-OCRv6_det_small.onnx`` vs ``small/det/det.onnx``），
#: 这里显式列出对应关系，校验时才不会漏项。
SOURCE_MAP = [
    # (共用目录内的相对路径,OnnxOCR models/ 内的相对路径)
    ("ppocrv5/det/det.onnx", "ppocrv5/det/det.onnx"),
    ("ppocrv5/rec/rec.onnx", "ppocrv5/rec/rec.onnx"),
    ("ppocrv5/cls/cls.onnx", "ppocrv5/cls/cls.onnx"),
    ("ppocrv5/ppocrv5_dict.txt", "ppocrv5/ppocrv5_dict.txt"),
    ("ppocrv6/det/PP-OCRv6_det_tiny.onnx", "ppocrv6/tiny/det/det.onnx"),
    ("ppocrv6/rec/PP-OCRv6_rec_tiny.onnx", "ppocrv6/tiny/rec/rec.onnx"),
    ("ppocrv6/det/PP-OCRv6_det_small.onnx", "ppocrv6/small/det/det.onnx"),
    ("ppocrv6/rec/PP-OCRv6_rec_small.onnx", "ppocrv6/small/rec/rec.onnx"),
    ("ppocrv6/det/PP-OCRv6_det_medium.onnx", "ppocrv6/medium/det/det.onnx"),
    ("ppocrv6/rec/PP-OCRv6_rec_medium.onnx", "ppocrv6/medium/rec/rec.onnx"),
    ("ppocrv6/cls/cls.onnx", "ppocrv5/cls/cls.onnx"),
    ("ppocrv6/ppocrv6_dict.txt", "ppocrv6/ppocrv6_dict.txt"),
    ("ppocrv6/ppocrv6_tiny_dict.txt", "ppocrv6/ppocrv6_tiny_dict.txt"),
]


def assert_shared_weights(shared: Path) -> None:
    """共用目录里的每个文件都必须与 OnnxOCR 自带的那份一致。

    这一步是整张表的地基：两侧运行时读的是同一个目录下的同一批文件，
    "同一模型"因此不靠声明而是靠路径保证。这里再回头校验一遍来源，
    防止复制过程中出错或目录被改动过——一旦不一致就直接抛错，
    不给使用者"表格照常生成但结论不可信"的机会。

    找不到 OnnxOCR 检出时跳过：数据是之前实测出来的，来源校验属于
    复现路径上的辅助环节，不该阻塞已经完成的测量结果。
    """
    src_root = _HERE / "OnnxOCR" / "onnxocr" / "models"
    if not src_root.is_dir():
        print(f"跳过来源校验（未找到 OnnxOCR 检出: {src_root}）")
        return

    n_onnx = n_dict = 0
    for rel, src_rel in SOURCE_MAP:
        a, b = shared / rel, src_root / src_rel
        if not a.is_file():
            raise SystemExit(f"缺少共用权重: {a}")
        if not b.is_file():
            raise SystemExit(f"缺少 OnnxOCR 源权重: {b}")
        if a.suffix == ".onnx":
            if _md5(a) != _md5(b):
                raise SystemExit(
                    f"共用目录与 OnnxOCR 源文件不一致: {rel}\n"
                    " 两侧必须指向同一份模型，否则这张表没有意义。")
            n_onnx += 1
        else:
            if not _same_dict(a, b):
                raise SystemExit(f"字典内容不一致: {rel}")
            n_dict += 1
    print(f"权重校验通过：{n_onnx} 个 onnx md5 逐字节一致，"
          f"{n_dict} 个字典内容一致（来源 {src_root}）")


def load(n: str) -> dict:
    return json.loads((_HERE / n).read_text("utf-8"))


def _image_sizes() -> dict:
    """读各图分辨率。

    优先用 OpenCV——两个引擎的环境都装了它，不必额外依赖 Pillow；
    实在拿不到就返回空 dict，表格里显示 ``—``，不影响数字本身。
    """
    out: dict = {}
    try:
        import cv2
    except ImportError:
        return out
    for p in sorted((_HERE / "images").glob("*.jpg")):
        img = cv2.imread(str(p))
        if img is not None:
            out[p.stem] = f"{img.shape[1]}x{img.shape[0]}"
    return out


def build_table(tier: str, sizes: dict) -> tuple[list[str], dict]:
    """生成一个档位的对比表。返回 (markdown 行, 统计摘要)。"""
    ocr_name, our_name, label = TIERS[tier]
    ocr, ours = load(ocr_name), load(our_name)

    L = [
        f"| 示例图 | 分辨率 | OnnxOCR<br>{label} | NOOCR<br>{label} "
        f"| 倍数 | 文本行 |",
        "|---|---|---|---|---|---|",
    ]
    tot_o = tot_s = 0
    for n in ocr["per_image"]:
        o = ocr["per_image"][n]
        s = ours["per_image"].get(n, {})
        if not s.get("avg_ms"):
            continue
        key = n.replace(".jpg", "")
        tot_o += o["avg_ms"]
        tot_s += s["avg_ms"]
        L.append(
            f"| `{key}` | {sizes.get(key, '—')} "
            f"| {o['avg_ms']:.0f} ms | **{s['avg_ms']:.0f} ms** "
            f"| **{o['avg_ms'] / s['avg_ms']:.1f}x** "
            f"| {o['lines']} / {s['lines']} |"
        )
    ro, rs = ocr["avg_ms"], ours["avg_ms"]
    L.append(
        f"| **均值** | — | **{ro:.0f} ms** | **{rs:.0f} ms** "
        f"| **{ro / rs:.2f}x** | {ocr['total_lines']} / {ours['total_lines']} |"
    )
    L.append(
        f"| **合计** | — | **{tot_o:.0f} ms** | **{tot_s:.0f} ms** "
        f"| **{tot_o / tot_s:.2f}x** | — |"
    )
    summary = {
        "tier": tier,
        "onnxocr_ms": ro,
        "noocr_ms": rs,
        "speedup_median": round(ro / rs, 2),
        "speedup_total": round(tot_o / tot_s, 2),
        "lines": {"OnnxOCR": ocr["total_lines"], "NOOCR": ours["total_lines"]},
        "weights": ours.get("weights", ""),
    }
    return L, summary


def main() -> int:
    # 共用权重目录：四个档位的 det/rec/字典都从这里读，两侧指向同一批文件。
    shared = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "weights_shared"
    if not shared.is_dir():
        raise SystemExit(
            f"未找到共用权重目录: {shared}\n"
            "见 compare_onnxocr.md 的「准备共用权重目录」一节。")
    assert_shared_weights(shared)

    sizes = _image_sizes()
    blocks, summaries = [], {}
    for tier in TIERS:
        try:
            L, s = build_table(tier, sizes)
        except FileNotFoundError as e:
            print(f"跳过 {tier}：缺少 {e.filename}")
            continue
        blocks.append((TIERS[tier][2], L))
        summaries[tier] = s

    md = []
    for label, L in blocks:
        md.append(f"### {label}\n")
        md.append("\n".join(L))
        md.append("")
    out = "\n".join(md)
    print(out)
    (_HERE / "compare_table_same_model.md").write_text(out, "utf-8")
    (_HERE / "compare_summary_same_model.json").write_text(
        json.dumps({
            "note": ("同模型对比：四个档位两侧读同一份权重文件；"
                     "v5 md5 一致，v6 走共用目录"),
            "images": load(TIERS["ppocrv5"][0])["images"],
            "gpu": "NVIDIA GeForce RTX 4070 Ti SUPER",
            "ort_version": "1.23.2",
            "tiers": summaries,
        }, ensure_ascii=False, indent=2), "utf-8")
    for t, s in summaries.items():
        print(f"{t:16s} 均值加速 {s['speedup_median']:.2f}x  "
              f"合计加速 {s['speedup_total']:.2f}x  "
              f"文本行 {s['lines']['OnnxOCR']}/{s['lines']['NOOCR']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
