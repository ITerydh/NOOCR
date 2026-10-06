"""校验多语言README 的结构一致性。

三份README 是同一份内容的三个语种，靠人工同步迟早漏改。这里做机器校验：
章节数、代码块配对、表格结构必须一致，且语言切换链接齐全。

用法::

    python scripts/check_readme.py

退出码 0 表示通过。任何不一致都会打印具体行号便于定位。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Windows 控制台默认编码是 cp1252，打印中文会抛 UnicodeEncodeError。
# CI 的 windows-latest 用 pwsh，默认编码同样不是 UTF-8。强制切到 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover - Python 3.7 及以下
    import io

    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

# (文件名, 语言名, 该语言里语言切换链接应有的写法)
FILES: list[tuple[str, str, str]] = [
    ("README.md", "简体中文", "简体中文"),
    ("README_EN.md", "English", "English"),
    ("README_TW.md", "繁體中文", "繁體中文"),
]

PROBLEMS: list[str] = []


def fail(msg: str) -> None:
    PROBLEMS.append(msg)


def headings(lines: list[str]) -> list[tuple[int, str, str]]:
    """抽取真正的 Markdown 标题，剔除代码块内的 # 注释行。"""
    out: list[tuple[int, str, str]] = []
    in_fence = False
    for i, line in enumerate(lines, 1):
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if m:
            out.append((i, len(m.group(1)), m.group(2)))
    return out


def main() -> int:
    docs: dict[str, list[str]] = {}
    for name, _, _ in FILES:
        path = ROOT / name
        if not path.is_file():
            fail(f"{name} 不存在")
            continue
        docs[name] = path.read_text(encoding="utf-8").splitlines()

    if len(docs) < 2:
        print("至少需要两份 README 才能比对")
        return 1

    base_name = FILES[0][0]
    base = docs.get(base_name, [])

    # ---------------------------------------------------------- 单文件检查
    for name, _lang, self_label in FILES:
        lines = docs.get(name)
        if lines is None:
            continue

        # 同一张表内所有行的列数必须一致。这一条比跨文件比对更敏感：
        # 某一列整体漏掉时表还在，行数不变，只有这里能发现。
        for t_idx, group in enumerate(table_groups(lines), 1):
            if len(set(group)) > 1:
                first = group[0]
                bad_rows = [
                    idx
                    for idx, line in enumerate(lines, 1)
                    if line.startswith("|") and line.count("|") != first
                ][:3]
                fail(
                    f"{name}: 第 {t_idx} 张表格列数不一致（{sorted(set(group))}），"
                    f"疑似行 {bad_rows}"
                )

        # 代码围栏必须成对
        fences = [
            idx for idx, line in enumerate(lines, 1) if line.startswith("```")
        ]
        if len(fences) % 2:
            fail(f"{name}: 代码块围栏数为奇数({len(fences)})，最后一个在第 {fences[-1]} 行")

        # 语言切换行：三种语言名必须都出现，且含自身（高亮当前语言）
        head = "\n".join(lines[:12])
        for _, _, label in FILES:
            if f"[{label}]" not in head:
                fail(f"{name}: 语言切换缺少 [{label}]")
        if f"[{self_label}]" not in head:
            fail(f"{name}: 语言切换未高亮当前语言 [{self_label}]")

        # 相对链接不得跨到不存在的文件
        for link in re.findall(r"\]\((README[^)#]*\.md)\)", head):
            if not (ROOT / link).is_file():
                fail(f"{name}: 语言切换指向不存在的文件 {link}")

        # 不该出现其它语种独有的标题写法（英文版混中文标题等）
        if name.endswith("_EN.md"):
            for i, _, text in headings(lines):
                if re.search(r"[\u4e00-\u9fff]", text) and "README" not in text:
                    fail(f"{name}: 第 {i} 行标题含中文 —— {text}")

    # ---------------------------------------------------------- 跨文件比对
    base_heads = [(lvl, txt) for _, lvl, txt in headings(base)]
    for entry in FILES[1:]:
        name = entry[0]
        lines = docs.get(name)
        if lines is None:
            continue
        other = [(lvl, txt) for _, lvl, txt in headings(lines)]

        if len(other) != len(base_heads):
            fail(
                f"{name}: 标题数 {len(other)} != {base_name} 的 {len(base_heads)}"
            )
            continue

        # 标题只需层级与数量对应。标题文本必须翻译，强行逐字比对没有
        # 意义——「性能」译成「效能」是对的，译成「Performance」也对。
        for idx, ((bl, _bt), (ol, _ot)) in enumerate(zip(base_heads, other)):
            if bl != ol:
                fail(f"{name}: 第 {idx + 1} 个标题层级 {ol} != {base_name} 的 {bl}")

        # 表格列数逐张比对
        bt_rows = table_shapes(base)
        ot_rows = table_shapes(lines)
        if len(bt_rows) != len(ot_rows):
            fail(f"{name}: 表格数 {len(ot_rows)} != {base_name} 的 {len(bt_rows)}")
        else:
            for k, (b, o) in enumerate(zip(bt_rows, ot_rows)):
                if b != o:
                    fail(f"{name}: 第 {k + 1} 张表格列数 {o} != {base_name} 的 {b}")

        # 命令示例必须逐字一致：里面的文件名、参数、URL 都不该被翻译。
        # 但示例里出现中文文件名（发票.jpg）是合理的本地化，跳过含中文的块。
        def norm_code(block: str) -> str:
            return block

        b_code = [norm_code(b) for b in code_blocks(base)]
        o_code = [norm_code(o) for o in code_blocks(lines)]
        if len(b_code) != len(o_code):
            fail(f"{name}: 代码块数 {len(o_code)} != {base_name} 的 {len(b_code)}")
        else:
            for k, (b, o) in enumerate(zip(b_code, o_code), 1):
                if b == o:
                    continue
                # 示例含中文（本地化的文件名/注释）则跳过，只比纯 ASCII 的
                has_cjk = re.search(r"[\u4e00-\u9fff]", b + o)
                if not has_cjk:
                    fail(f"{name}: 第 {k} 个代码块与 {base_name} 不一致（命令不应改写）")

    # ---------------------------------------------------------------- 汇总
    print(f"检查 {len(docs)} 份README：{', '.join(docs)}")
    if PROBLEMS:
        print(f"\n发现 {len(PROBLEMS)} 个问题：")
        for p in PROBLEMS:
            print(f"  - {p}")
        return 1
    base_heads = len(headings(base))
    print(f"  标题 {base_heads} 个 / 代码块 {len(code_blocks(base))} 个 / 表格 {len(table_shapes(base))} 张，三份一致")
    print("全部通过")
    return 0


def code_blocks(lines: list[str]) -> list[str]:
    """抽取所有围栏代码块的内容，忽略语言标记。"""
    out: list[str] = []
    cur: list[str] | None = None
    for line in lines:
        if line.startswith("```"):
            if cur is None:
                cur = []
            else:
                out.append("\n".join(cur).strip())
                cur = None
            continue
        if cur is not None:
            cur.append(line)
    return out


def table_groups(lines: list[str]) -> list[list[int]]:
    """把连续的 ``|`` 行按表分组，返回每张表各行的竖线数量。"""
    groups: list[list[int]] = []
    cur: list[int] = []
    in_fence = False
    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
            if cur:
                groups.append(cur)
                cur = []
            continue
        if in_fence or not line.startswith("|"):
            if cur:
                groups.append(cur)
                cur = []
            continue
        cur.append(line.count("|"))
    if cur:
        groups.append(cur)
    return groups


def table_shapes(lines: list[str]) -> list[tuple[int, int]]:
    """返回每张表的 (行数, 列数)。

    列数取表内**众数**而非首行：Markdown 表格允许单元格内含 ``|``（需转义），
    首行不一定有代表性。用众数能容忍个别异常行，又能在整列缺失时报警。
    """
    out: list[tuple[int, int]] = []
    rows: list[int] = []
    in_fence = False

    def flush() -> None:
        if rows:
            # 众数；并列时取最小，避免把异常行当成主流
            common = max(set(rows), key=lambda c: (rows.count(c), -c))
            out.append((len(rows), common))

    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
            flush()
            rows = []
            continue
        if in_fence or not line.startswith("|"):
            flush()
            rows = []
            continue
        rows.append(line.count("|"))
    flush()
    return out


if __name__ == "__main__":
    sys.exit(main())
