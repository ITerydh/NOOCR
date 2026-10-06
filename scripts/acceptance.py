"""开箱即用验收：模拟用户照README 操作的全流程，逐项检查。

与 ``tests/test_units.py`` 的分工：

- ``test_units.py`` —— 纯逻辑（几何/排序/解码/缓存），不碰模型，快；
- 本脚本 —— 端到端，真加载模型、真起 Web 服务，验证「装完就能用」。

用法::

    # 在目标环境里跑（默认用当前解释器）
    python scripts/acceptance.py

    # 指定解释器，例如验证另一个虚拟环境
    python scripts/acceptance.py --python D:\\path\\to\\env\\Scripts\\python.exe

    # 只跑 CLI 与库层，跳过需要拉起服务的项
    python scripts/acceptance.py --no-serve

退出码 0 表示全部通过。任何一项失败都会打印实际输出以便定位。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Windows 控制台默认 cp1252，打印中文会抛 UnicodeEncodeError。
# CI 的 windows-latest 用 pwsh 同样不是 UTF-8，不做这一步会直接崩在
# 第一条 check 的输出上。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover - Python 3.7 及以下
    import io

    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMG = "noocr/web/static/ticket_train.jpg"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  {mark}  {name}" + (f"  {detail}" if detail else ""))


class Env:
    """在指定解释器上跑 noocr CLI。"""

    def __init__(self, python: Path) -> None:
        self.python = python

    def run(self, *args: str, timeout: int = 600) -> tuple[int, str]:
        p = subprocess.run(
            [str(self.python), "-X", "utf8", "-m", "noocr", *args],
            cwd=ROOT,
            capture_output=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout.decode("utf-8", "replace") + p.stderr.decode(
            "utf-8", "replace"
        )

    @staticmethod
    def first(text: str, needle: str) -> str:
        """取出包含 needle 的第一行，用于断言设备行/耗时行。"""
        for line in text.splitlines():
            if needle in line:
                return line.strip()
        return ""


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    ap = argparse.ArgumentParser(description="NOOCR 开箱即用验收")
    ap.add_argument("--python", type=Path, default=None, help="目标解释器，默认当前")
    ap.add_argument("--image", default=DEFAULT_IMG, help="用于识别的示例图")
    ap.add_argument("--no-serve", action="store_true", help="跳过需要起服务的检查")
    args = ap.parse_args()

    env = Env((args.python or Path(sys.executable)).resolve())
    img = args.image
    print(f"解释器: {env.python}")

    # ---------------------------------------------------------------- 设备
    section("设备：同一环境 CPU 与 GPU 都要能用")
    rc, txt = env.run(img, "-d", "cpu", "-f", "text")
    line = env.first(txt, "实际=")
    check("-d cpu 可用", rc == 0 and "行" in txt, line)
    check("cpu 未误用 GPU", "cuda" not in line, line)

    rc, txt = env.run(img, "-d", "cuda", "-f", "text")
    line = env.first(txt, "实际=")
    cuda_used = "cuda" in line
    check("-d cuda 不报错", rc == 0 and "行" in txt, line)
    # 缺 CUDA/cuDNN 时会明确警告并回退 CPU，这属于正确行为
    check(
        "-d cuda 生效或明确降级",
        cuda_used or "不可用" in txt,
        "GPU 生效" if cuda_used else "已降级到 CPU（缺 CUDA/cuDNN）",
    )

    rc, txt = env.run(img, "-f", "text")
    check("默认（auto）可识别", rc == 0 and "行" in txt, env.first(txt, "行 /"))

    # ---------------------------------------------------------------- CLI
    section("CLI 基本能力")
    rc, txt = env.run("--version")
    check("--version", rc == 0 and "0." in txt, txt.strip()[:40])
    rc, txt = env.run("--help")
    check("--help 含 --device", rc == 0 and "--device" in txt)
    rc, txt = env.run("backends")
    # 四档都要在列，且每档都要报体积——少一档说明注册表漏登记了
    check("backends 四档齐全",
          rc == 0 and all(n in txt for n in ("ppocrv5", "ppocrv6-tiny",
                                          "ppocrv6-small", "ppocrv6-medium")))
    check("backends 报体积", "MB" in txt)
    rc, txt = env.run("models")
    check("models 权重状态", rc == 0 and "ppocrv6" in txt)
    rc, txt = env.run("x.jpg", "-d", "nonsense")
    check("非法 device 被拒", rc != 0 and "device" in txt, txt.strip()[-56:])

    # ------------------------------------------------------------ 输出格式
    section("输出文件")
    tmp = Path(os.environ.get("TEMP", "/tmp"))
    # 断言特征选「必然出现」的那些：JSON 有 lines 数组，文本有已知首行。
    # 不能用 Markdown 标题做判据——票据类识别结果本来就没有标题。
    cases = (
        ("json", ".json", ('"lines"', '"confidence"')),
        ("md", ".md", ("78.com",)),
        ("text", ".txt", ("78.com",)),
    )
    for fmt, ext, needles in cases:
        out = tmp / f"noocr_acc{ext}"
        if out.exists():
            out.unlink()
        rc, txt = env.run(img, "-o", str(out), "-f", fmt)
        body = out.read_text(encoding="utf-8") if out.is_file() else ""
        missing = [n for n in needles if n not in body]
        check(f"输出 {fmt.upper()}", rc == 0 and not missing,
              f"rc={rc}" + (f" 缺 {missing}" if missing else f" {len(body)} 字符"))

    # ------------------------------------------------------------ 后端档位
    section("四档后端")
    for be in ("ppocrv6-tiny", "ppocrv6-small", "ppocrv6-medium", "ppocrv5"):
        rc, txt = env.run(img, "-b", be, "-f", "text")
        check(f"后端 {be}", rc == 0 and "行" in txt, env.first(txt, "行 /"))

    section("性能剖析")
    rc, txt = env.run("bench", img)
    check("bench 剖析", rc == 0 and "det" in txt.lower(), f"rc={rc}")

    section("库层按次覆盖")
    code = (
        "from noocr import ocr\n"
        f"f = {img!r}\n"
        "a = ocr(f)\n"
        "b = ocr(f, device='cpu')\n"
        "c = ocr(f, backend='ppocrv5')\n"
        "d = ocr(f)\n"
        "assert a.backend == 'ppocrv6-small', a.backend\n"
        "assert c.backend == 'ppocrv5', c.backend\n"
        "assert d.backend == 'ppocrv6-small', d.backend\n"
        "assert b.page_count == a.page_count == d.page_count\n"
        # 打标记前缀，避免与日志行混淆
        "print('RESULT', a.backend, c.backend, d.backend)\n"
    )
    p = subprocess.run(
        [str(env.python), "-X", "utf8", "-c", code],
        cwd=ROOT,
        capture_output=True,
        timeout=900,
    )
    out = p.stdout.decode("utf-8", "replace")
    tail = [x for x in out.splitlines() if x.startswith("RESULT")]
    detail = tail[0][7:].strip() if tail else (
        p.stderr.decode("utf-8", "replace").strip().splitlines()[-1][:60] if p.stderr else ""
    )
    check("ocr() 支持 device / backend 按次覆盖", p.returncode == 0, detail)

    # ---------------------------------------------------------------- 服务
    if not args.no_serve:
        section("Web 服务")
        import urllib.error
        import urllib.request

        port = 8077
        proc = subprocess.Popen(
            [str(env.python), "-X", "utf8", "-m", "noocr", "serve", "--port", str(port)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            ready = False
            for _ in range(45):
                if proc.poll() is not None:
                    break
                try:
                    urllib.request.urlopen(base + "/health", timeout=2).read()
                    ready = True
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(1)
            if not ready:
                tail = ""
                if proc.poll() is not None:
                    tail = proc.stdout.read().decode("utf-8", "replace")[-200:]
                check("serve 启动", False, tail or "45s 内未就绪")
            else:
                check("serve /health", True)
                dev = json.load(urllib.request.urlopen(base + "/api/device", timeout=30))
                check(
                    "GET /api/device",
                    dev.get("prefer") == "auto" and "usable" in dev,
                    f"gpu_available={dev.get('gpu_available')}",
                )
                page = urllib.request.urlopen(base + "/", timeout=10).read().decode("utf-8")
                check("首页含设备切换控件", "devPill" in page and "devPop" in page)
                req = urllib.request.Request(
                    base + "/api/device", data=b"device=cpu", method="POST"
                )
                req.add_header("Content-Type", "application/x-www-form-urlencoded")
                d2 = json.load(urllib.request.urlopen(req, timeout=120))
                check(
                    "POST 切到 CPU",
                    d2.get("kind") == "cpu" and bool(d2.get("warmup")),
                    f"warmup={d2.get('warmup')}",
                )
                # 无 GPU 环境应明确拒绝，而不是静默假切换
                if not dev.get("gpu_available"):
                    req = urllib.request.Request(
                        base + "/api/device", data=b"device=cuda", method="POST"
                    )
                    req.add_header("Content-Type", "application/x-www-form-urlencoded")
                    try:
                        urllib.request.urlopen(req, timeout=30)
                        check("无 GPU 时拒绝切 cuda", False, "竟返回成功")
                    except urllib.error.HTTPError as e:
                        check("无 GPU 时拒绝切 cuda", e.code == 400, f"HTTP {e.code}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    # ---------------------------------------------------------------- 汇总
    fails = [n for n, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 56)
    print(f"共 {len(RESULTS)} 项，失败 {len(fails)} 项")
    if fails:
        for n in fails:
            print(f"  - {n}")
    else:
        print("全部通过")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
