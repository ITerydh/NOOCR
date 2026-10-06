"""把实测产物里的表格同步进三份 README。

三份README 的数字必须同源。手抄表格迟早会出现「简中改了就忘了繁中」
或某个数字抄错半位的情况，而这类错误没人会看出来——表格照常渲染，
只是结论错了。

所以这里从 JSON 产物重新生成表格并就地替换：
- **对比表**：``scripts/perf/compare_table_same_model.md``（由
  ``compare_make_table.py`` 产出，已做权重 MD5 校验）
- **性能表**：``device_cpu.json`` / ``device_cuda.json``

用法::

    python scripts/sync_readme_tables.py
    python scripts/sync_readme_tables.py --check   # 只校验，不写入
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / "scripts" / "perf"
READMES = ("README.md", "README_EN.md", "README_TW.md")

#: 图文件名 -> 三语README 里用的展示名。key 是 ``per_image`` 的键。
LABELS = {
    "doc_comparison_table": ("对比表格", "Table", "對比表格"),
    "exam_chinese_primary": ("小学试卷", "Exam sheet", "小學試卷"),
    "id_card_china": ("身份证", "ID card", "身份證"),
    "medical_lab_report": ("化验单", "Lab report", "化驗單"),
    "product_spec_sheet": ("规格书", "Spec sheet", "規格書"),
    "receipt_bank_statement": ("银行流水", "Statement", "銀行流水"),
    "scene_vertical_plaque": ("竖式牌匾", "Plaque", "豎式牌匾"),
    "ticket_train": ("火车票", "Train ticket", "火車票"),
}


def build_perf_table() -> str:
    """由device_cpu.json / device_cuda.json 生成性能表（返回各语版本）。"""
    cpu = json.loads((PERF / "device_cpu.json").read_text("utf-8"))
    gpu = json.loads((PERF / "device_cuda.json").read_text("utf-8"))
    heads = {
        "zh": ("| 图片 | 行数 | CPU | GPU | 加速比 |",
               "|---|---|---|---|---|"),
        "en": ("| Image | Lines | CPU | GPU | Speedup |",
               "|---|---|---|---|---|"),
        "tw": ("| 圖片 | 行數 | CPU | GPU | 加速比 |",
               "|---|---|---|---|---|"),
    }
    rows = {k: [h[0], h[1]] for k, h in heads.items()}
    tc = tg = 0
    for key, cv in cpu["per_image"].items():
        gv = gpu["per_image"][key]["avg_ms"]
        lines = cv.get("lines", "")
        tc += cv["avg_ms"]
        tg += gv
        for lang, idx in (("zh", 0), ("en", 1), ("tw", 2)):
            name = LABELS[key][idx]
            unit = "ms" if lang != "zh" else " ms"
            bold = "**" if lang == "zh" else ""
            rows[lang].append(
                f"| {name} | {lines} | {cv['avg_ms']:.0f}{unit} | "
                f"{bold}{gv:.0f}{unit}{bold} | {cv['avg_ms'] / gv:.1f}x |")
    for lang, tail in (("zh", f"| **合计** | — | **{tc:.0f} ms** | **{tg:.0f} ms** | **{tc / tg:.1f}x** |"),
                       ("en", f"| **Total** | — | **{tc:.0f} ms** | **{tg:.0f} ms** | **{tc / tg:.1f}x** |"),
                       ("tw", f"| **合計** | — | **{tc:.0f} ms** | **{tg:.0f} ms** | **{tc / tg:.1f}x** |")):
        rows[lang].append(tail)
    return {k: "\n".join(v) for k, v in rows.items()}


def build_compare_tables() -> dict:
    """从 compare_table_same_model.md 切出四档表格（按语种给head）。"""
    md = (PERF / "compare_table_same_model.md").read_text("utf-8")
    out = {}
    for tier, key in (("ppocrv5", "PP-OCRv5"),
                      ("ppocrv6-tiny", "PP-OCRv6 tiny"),
                      ("ppocrv6-small", "PP-OCRv6 small"),
                      ("ppocrv6-medium", "PP-OCRv6 medium")):
        m = re.search(
            r"### " + re.escape(key) + r"\n\n(\|.*?)\n(?=\n|\Z)",
            md, re.S)
        if not m:
            raise SystemExit(f"产物里找不到 {key} 的表格，请先跑 compare_make_table.py")
        out[tier] = m.group(1)
    return out


def replace_table(text: str, new: str, lang: str) -> str:
    """替换性能表：定位表头行到空行之间的整块表格。"""
    first = new.splitlines()[0]
    lines = text.splitlines()
    idx = next((i for i, ln in enumerate(lines) if ln.strip() == first), None)
    if idx is None:
        raise SystemExit(f"README 里找不到性能表头: {first}")
    end = idx + 1
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return "\n".join(lines[:idx] + new.splitlines() + lines[end:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验不写入")
    args = ap.parse_args()

    perf = build_perf_table()
    tables = build_compare_tables()
    langs = {"README.md": "zh", "README_EN.md": "en", "README_TW.md": "tw"}

    changed = []
    for name, lang in langs.items():
        p = ROOT / name
        src = p.read_text("utf-8")
        lines = src.splitlines()

        # 逐块替换：先定位所有 `<summary><b>档位名</b>` 所在的折叠块，
        # 再把该块里紧跟 summary 的表格整段换掉。
        #
        # 不写成「一边遍历一边 sub」：那样前一次替换会改变后续正则的
        # 匹配位置（`.*?` 能跨过已替换的块），四档里会有一两档静默
        # 保持旧数据——表格照常渲染，只是数字是错的，很难发现。
        for tier, label in (("ppocrv5", "PP-OCRv5"),
                            ("ppocrv6-tiny", "PP-OCRv6 tiny"),
                            ("ppocrv6-small", "PP-OCRv6 small"),
                            ("ppocrv6-medium", "PP-OCRv6 medium")):
            idx = next((i for i, ln in enumerate(lines)
                        if ln.startswith("<summary><b>") and label in ln), None)
            if idx is None:
                continue
            j = idx + 1
            while j < len(lines) and not lines[j].startswith("|"):
                j += 1
            if j >= len(lines):
                continue
            k = j
            while k < len(lines) and lines[k].startswith("|"):
                k += 1
            lines[j:k] = tables[tier].splitlines()

        text = "\n".join(lines)
        dst = replace_table(text, perf[lang], lang)

        if dst != src:
            # Windows 上 README 是 CRLF。直接按 splitlines() 的结果写回会
            # 把 CRLF 统一成 LF，让 --check 永远报「有更新」——明明内容
            # 一模一样。统一以 CRLF 落盘，与仓库现有文件保持一致。
            if not args.check:
                p.write_bytes(dst.replace("\n", "\r\n").encode("utf-8"))
            changed.append(name)

    if args.check:
        print("需要更新的README:" if changed else "三份 README 均已是最新")
        for c in changed:
            print(f"  {c}")
        return 1 if changed else 0
    print(f"已更新: {', '.join(changed) if changed else '无变化'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
