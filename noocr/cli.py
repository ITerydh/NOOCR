"""NOOCR 命令行入口。

设计原则：**默认就要好用**。``noocr a.jpg`` 一条命令直接出结果；
所有高级选项都有��合默认值，不要求用户先读文档。

常用::

    noocr 扫描件.jpg                      # 识别并打印文本
    noocr 扫描件.jpg -o result.json       # 输出结构化 JSON
    noocr 论文.pdf -o out.md --format md  # PDF 转 Markdown
    noocr models                          # 列出各后端权重状态
    noocr models --get ppocrv6-tiny       # 下载 PP-OCRv6 tiny 权重
    noocr backends                        # 列出可用后端
    noocr bench 扫描件.jpg                # 看各阶段耗时
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__


def _print_backends() -> None:
    from .backends import list_backends

    print(f"NOOCR {__version__} —— 可用后端\n")
    print(f"{'名称':<20}{'说明':<44}{'模型体积':>10}")
    print("-" * 76)
    for b in list_backends():
        name = b["name"] + ("  [默认]" if b.get("default") else "")
        size = b.get("det_mb", 0) + b.get("rec_mb", 0)
        print(f"{name:<20}{b.get('notes', ''):<44}{size:8.1f}MB")
    print(
        "\n提示: 极速选 ppocrv6-tiny（0.58x 速度，精度略降）；"
        "高精度选 ppocrv6-small（可简写为 ppocrv6）；上一代兼容选 ppocrv5。"
    )


def _print_models() -> None:
    from .models import print_status

    print_status()


def _get_models(backend: str) -> int:
    from .models import ensure_models

    try:
        got = ensure_models(backend)
    except Exception as e:
        print(f"下载失败: {e}", file=sys.stderr)
        return 1
    print(f"已就绪 {len(got)} 个文件")
    for p in got:
        print(f"  + {p}")
    return 0


def _run_serve(args) -> int:
    """启动 Web 界面与 REST API。"""
    try:
        from .web import run
    except ImportError as e:
        # requirements.txt 已含web 依赖，所以这里更可能是用户手动精简过
        # 安装。给出可直接复制执行的两条命令，而不是只丢一个 extras 名。
        missing = getattr(e, "name", None) or "fastapi/uvicorn"
        print(
            f"启动 Web 服务缺少依赖: {missing}\n"
            f"  完整安装:  pip install -r requirements.txt\n"
            f"  仅补Web:  pip install fastapi uvicorn[standard] python-multipart",
            file=sys.stderr,
        )
        return 5
    try:
        run(
            host=args.host,
            port=args.port,
            backend=args.backend or "",
            device=args.device,
        )
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


def _resolve_device(prefer: str) -> str:
    """把设备偏好解析成实际设备字符串，失败时返回错误摘要。"""
    try:
        from .engine.session import detect_device

        return str(detect_device(prefer))
    except Exception as e:
        return f"探测失败: {e}"


def _run_ocr(args) -> int:
    from .backends import get_backend
    from .inputs.loader import load_document

    path = Path(args.path)
    if not path.exists():
        print(f"文件不存在: {path}", file=sys.stderr)
        return 1

    backend = get_backend(
        args.backend,
        rec_batch_size=args.batch,
        use_cls=args.use_cls,
        device=args.device,
    )
    try:
        backend.ensure_loaded()
    except Exception as e:
        print(f"后端 {args.backend or '默认'} 不可用: {e}", file=sys.stderr)
        return 2

    print(f"[device] {args.device}  实际={_resolve_device(args.device)}", flush=True)

    try:
        doc = load_document(path)
    except Exception as e:
        print(f"无法读取文件: {e}", file=sys.stderr)
        return 4

    # docx/xlsx/pptx 的原生文本无需 OCR，直接给出
    if doc.native_text and not args.force_ocr:
        if args.output:
            Path(args.output).write_text(doc.native_text, encoding="utf-8")
            print(f"已写入 {args.output}（原生文本，未经过 OCR）")
        else:
            print(doc.native_text)
        return 0

    pages = []
    try:
        for i, img in enumerate(doc):
            if img is None:
                continue
            pages.append(backend.recognize_image(img, page_index=i))
    except Exception as e:
        print(f"识别失败: {e}", file=sys.stderr)
        return 3

    if not pages:
        print("未从文件中解出任何图像", file=sys.stderr)
        return 4
    for w in doc.warnings:
        print(f"[警告] {w}", file=sys.stderr)

    total_lines = sum(len(p.lines) for p in pages)
    total_time = sum(p.processing_time for p in pages)

    if args.output:
        out = Path(args.output)
        if args.format == "json":
            payload = {
                "source": str(path),
                "backend": backend.name,
                "pages": [
                    {
                        "index": p.page_index,
                        "width": p.width,
                        "height": p.height,
                        "text": p.text,
                        "lines": [
                            {
                                "text": ln.text,
                                "confidence": round(ln.confidence, 4),
                                "angle": ln.angle,
                                "box": [[round(x, 1), round(y, 1)] for x, y in ln.box.points],
                            }
                            for ln in p.lines
                        ],
                    }
                    for p in pages
                ],
            }
            out.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            out.write_text("\n\n".join(p.text for p in pages), encoding="utf-8")
        print(f"已写入 {out}")
    else:
        for p in pages:
            if len(pages) > 1:
                print(f"--- 第 {p.page_index + 1} 页 ---")
            print(p.text)

    print(
        f"\n{total_lines} 行 / {len(pages)} 页 / {total_time * 1000:.0f}ms "
        f"({total_time * 1000 / max(total_lines, 1):.1f}ms每行)",
        file=sys.stderr,
    )
    return 0


def _run_bench(args) -> int:
    """把单张图的分阶段耗时打到 stdout。"""
    from .backends import get_backend
    from .engine.imageops import (
        crop_quad,
        imread,
        normalize_db,
        sort_reading_order,
        to_bgr,
    )

    img = imread(args.path)
    if img is None:
        print(f"无法读取: {args.path}", file=sys.stderr)
        return 1
    b = get_backend(args.backend, device=args.device)
    b.ensure_loaded()
    print(f"[device] 请求={args.device}  实际={_resolve_device(args.device)}", flush=True)
    img = to_bgr(img)

    stages: dict = {}

    t = time()
    if hasattr(b, "_resize_for_det"):
        resized, shape = b._resize_for_det(img)
    else:
        from .engine.imageops import resize_keep_ratio

        resized, shape = resize_keep_ratio(img, b.det_limit, b.det_limit)
    blob = normalize_db(resized)[None]
    stages["预处理"] = time() - t

    t = time()
    prob = b._det_session.run(b._det_out, {b._det_in: blob})[0]
    stages["检测推理"] = time() - t

    t = time()
    boxes = b.db(prob, shape)
    stages["检测后处理"] = time() - t

    if boxes:
        order = sort_reading_order([x for x in boxes])
        boxes = [boxes[i] for i in order]
        t = time()
        crops = [crop_quad(img, x) for x in boxes]
        stages["裁剪"] = time() - t

        t = time()
        b._classify_angles(crops)
        stages["方向分类"] = time() - t

        t = time()
        b._recognize(crops)
        stages["识别(含解码)"] = time() - t
    else:
        crops = []

    total = sum(stages.values())
    print(f"\n{b.name}  {args.path}  {len(crops)} 行\n")
    print(f"{'阶段':<18}{'ms':>10}{'占比':>9}")
    print("-" * 37)
    for k, v in stages.items():
        print(f"{k:<18}{v * 1000:10.1f}{100 * v / max(total, 1e-9):8.1f}%")
    print("-" * 37)
    print(f"{'合计':<18}{total * 1000:10.1f}{100.0:8.1f}%")
    return 0


def time():
    import time as _t

    return _t.perf_counter()


class _Args:
    """CLI 参数。刻意不用 argparse——见 :func:`parse_argv` 的说明。"""

    __slots__ = ("command", "path", "backend", "output", "format",
                 "batch", "use_cls", "force_ocr", "get", "host", "port", "device")

    def __init__(self) -> None:
        self.command = "ocr"
        self.path: Optional[str] = None
        self.backend: Optional[str] = None
        self.output: Optional[str] = None
        self.format = "json"
        self.batch = 6
        self.use_cls = True
        self.force_ocr = False
        self.get: Optional[str] = None
        self.host = "127.0.0.1"
        self.port = 8000
        self.device = "auto"


def _print_help() -> None:
    print(f"""NOOCR {__version__} —— 全功能 OCR 系统

用法:
  noocr <文件>                识别图片/PDF/Office 并打印文本
  noocr <文件> -o out.json     输出结构化 JSON（含坐标与置信度）
  noocr serve                 启动 Web 界面与 REST API
  noocr bench <文件>           分阶段性能剖析，看时间花在哪
  noocr backends               列出可用后端
  noocr models                 查看各后端权重状态
  noocr models --get <后端>    下载权重

选项:
  -b, --backend <名>    后端（默认 ppocrv6-small）
                        ppocrv6-tiny  最快，体积 6MB
                        ppocrv6-small  默认，精度最高
                        ppocrv5        上一代，兼容性最好
  -o, --output <路径>   输出文件
  -f, --format <格式>   json（默认）/ text / md
      --batch <n>       识别批大小（不影响结果，仅影响速度）
  -d, --device <设备>   auto（默认）/ cpu / cuda / dml / cann
                        auto 会在有 CUDA 时自动用 GPU；缺 CUDA/cuDNN
                        会明确报错，不会悄悄退回 CPU
      --host <地址>     serve 监听地址，默认 127.0.0.1
      --port <端口>     serve 监听端口，默认 8000
      --no-cls          关闭 180 度纠正
      --force-ocr       即使文件含原生文本也走 OCR
  -h, --help            显示本帮助
  -V, --version         显示版本

示例:
  noocr 发票.jpg -b ppocrv6-tiny
  noocr 论文.pdf -o 论文.md -f md
  noocr serve --port 8080 --device cuda
  noocr 发票.jpg -d cuda
  noocr bench 试卷.jpg -b ppocrv6 -d cuda
""")


def parse_argv(argv: List[str]) -> _Args:
    """手写参数解析。

    为什么不用 argparse
    -----------------
    本 CLI 要求同时支持``noocr a.jpg``（裸文件）与
    ``noocr bench a.jpg``（子命令 + 文件）。argparse 一旦注册
    ``add_subparsers``，顶层位置参数就只能匹配子命令名——
    裸文件路径会被拒；而若不注册子命令，``bench`` 之类又会被当成文件名。
    两边都有理，且互斥。

    与其和框架斗智，不如用50 行明确的分词：识别命令只认**第一个**裸词，
    其余裸词归文件。行为可预测，错误信息也能自己写清楚。
    """
    a = _Args()
    cmds = {"backends", "models", "bench", "serve", "ocr", "help"}
    i = 0
    n = len(argv)

    # 1) 命令：只看第一个 token
    if i < n and not argv[i].startswith("-"):
        if argv[i] in cmds:
            a.command = argv[i]
            i += 1
        else:
            a.command = "ocr"

    # 2) 其余：选项 + 位置参数
    takes_value = {
        "-b": "backend", "--backend": "backend",
        "-o": "output", "--output": "output",
        "-f": "format", "--format": "format",
        "--batch": "batch", "--get": "get",
        "--host": "host", "--port": "port",
        "-d": "device", "--device": "device",
    }
    positional: List[str] = []
    while i < n:
        tok = argv[i]
        if tok in ("-h", "--help"):
            a.command = "help"
        elif tok in ("-V", "--version"):
            a.command = "version"
        elif tok in takes_value:
            key = takes_value[tok]
            if i + 1 >= n:
                raise SystemExit(f"noocr: {tok} 需要一个值")
            val = argv[i + 1]
            if key == "batch":
                try:
                    a.batch = max(1, int(val))
                except ValueError as e:
                    raise SystemExit(f"noocr: --batch 需要整数，收到 {val!r}") from e
            elif key == "port":
                try:
                    a.port = int(val)
                except ValueError as e:
                    raise SystemExit(f"noocr: --port 需要整数，收到 {val!r}") from e
            else:
                setattr(a, key, val)
            i += 2
            continue
        elif tok == "--no-cls":
            a.use_cls = False
        elif tok == "--force-ocr":
            a.force_ocr = True
        elif tok.startswith("--get="):
            a.get = tok.split("=", 1)[1]
        elif tok.startswith("-") and tok != "-":
            raise SystemExit(f"noocr: 未知选项 {tok}（用 --help 看用法）")
        else:
            positional.append(tok)
        i += 1

    if a.format not in ("json", "text", "md"):
        raise SystemExit(f"noocr: --format 只能是 json/text/md，收到 {a.format!r}")
    if a.device not in ("auto", "cpu", "cuda", "dml", "cann"):
        raise SystemExit(
            f"noocr: --device 只能是 auto/cpu/cuda/dml/cann，收到 {a.device!r}"
        )
    if positional:
        a.path = positional[0]
    return a


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        _print_help()
        return 0
    args = parse_argv(raw)

    if args.command == "help":
        _print_help()
        return 0
    if args.command == "version":
        print(f"NOOCR {__version__}")
        return 0
    if args.command == "backends":
        _print_backends()
        return 0
    if args.command == "models":
        if args.get:
            return _get_models(args.get)
        _print_models()
        return 0
    if args.command == "serve":
        return _run_serve(args)
    if args.command == "bench":
        if not args.path:
            _print_help()
            return 1
        return _run_bench(args)
    if not args.path:
        _print_help()
        return 1
    return _run_ocr(args)


if __name__ == "__main__":
    raise SystemExit(main())
