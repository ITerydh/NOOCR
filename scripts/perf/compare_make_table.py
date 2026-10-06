"""把两侧的基准结果合成一张对比表。

只做数据搬运与算术，不重新测，也不修饰结论——
快就是快，慢就是慢，差异以中位数为准。
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((_HERE / name).read_text("utf-8"))


def main() -> int:
    ocr = load("bench_onnxocr.json")
    sml = load("bench_noocr_ppocrv6_small.json")
    tiny = load("bench_noocr_ppocrv6_tiny.json")

    # 尺寸从文件实测，不手抄——手抄的表一旦和图对不上就没意义
    from PIL import Image
    sizes = {p.stem: f"{Image.open(p).size[0]}x{Image.open(p).size[1]}"
             for p in sorted((_HERE / "images").glob("*.jpg"))}

    names = list(ocr["per_image"].keys())

    rows = []
    for n in names:
        o = ocr["per_image"][n]
        s = sml["per_image"].get(n, {})
        t = tiny["per_image"].get(n, {})
        rows.append((n, o, s, t))

    L = []
    L.append("| 示例图 | 分辨率 | OnnxOCR<br>ppocrv5 | NOOCR<br>v6 small | 倍数 | NOOCR<br>v6 tiny | 倍数 |")
    L.append("|---|---|---|---|---|---|---|")
    for n, o, s, t in rows:
        key = n.replace(".jpg", "")
        sp = o["median_ms"] / s["median_ms"] if s.get("median_ms") else 0
        tp = o["median_ms"] / t["median_ms"] if t.get("median_ms") else 0
        L.append(
            f"| `{key}` | {sizes.get(key, '—')} "
            f"| {o['median_ms']:.0f} ms / {o['lines']} 行 "
            f"| {s.get('median_ms',0):.0f} ms / {s.get('lines','—')} 行 "
            f"| **{sp:.1f}x** "
            f"| {t.get('median_ms',0):.0f} ms / {t.get('lines','—')} 行 "
            f"| **{tp:.1f}x** |"
        )

    sp_all = ocr["median_ms"] / sml["median_ms"]
    tp_all = ocr["median_ms"] / tiny["median_ms"]

    L.append(
        f"| **中位** | — | **{ocr['median_ms']:.0f} ms** "
        f"| **{sml['median_ms']:.0f} ms** | **{sp_all:.2f}x** "
        f"| **{tiny['median_ms']:.0f} ms** | **{tp_all:.2f}x** |"
    )

    out = "\n".join(L)
    print(out)
    (_HERE / "compare_table.md").write_text(out, "utf-8")

    # 附一份机器可读的对账数据，便于以后复核
    summary = {
        "images": ocr["images"],
        "gpu": "NVIDIA GeForce RTX 4070 Ti SUPER",
        "ort_version": "1.23.2",
        "results": {
            "OnnxOCR(ppocrv5)": {
                "median_ms": ocr["median_ms"], "total_lines": ocr["total_lines"]},
            "NOOCR(ppocrv6-small)": {
                "median_ms": sml["median_ms"], "total_lines": sml["total_lines"]},
            "NOOCR(ppocrv6-tiny)": {
                "median_ms": tiny["median_ms"], "total_lines": tiny["total_lines"]},
        },
        "speedup_vs_onnxocr": {
            "ppocrv6-small": round(sp_all, 2),
            "ppocrv6-tiny": round(tp_all, 2),
        },
    }
    (_HERE / "compare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(f"\nsmall {sp_all:.2f}x   tiny {tp_all:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
