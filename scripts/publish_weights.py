"""把models/ 下的权重上传到 ModelScope 仓库。

用法::

    python scripts/publish_weights.py --token <token>

按 :data:`noocr.models.BACKEND_MODELS` 的清单逐一上传，
保持仓库目录结构与 ``rel_path`` 完全一致——这样用户端的下载逻辑
就是「快照 + 按需拷贝」，无需任何路径映射。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noocr.models import BACKEND_MODELS, MODELSCOPE_REPO, MODELS_ROOT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="上传权重到 ModelScope")
    ap.add_argument("--token", default=os.getenv("MODELSCOPE_TOKEN"), help="ModelScope 访问令牌")
    ap.add_argument("--repo", default=MODELSCOPE_REPO, help="目标仓库 ID")
    ap.add_argument("--base-dir", default=str(MODELS_ROOT), help="权重本地根目录")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不实际上传")
    args = ap.parse_args()

    if not args.dry_run and not args.token:
        print("缺少访问令牌：用 --token 或环境变量 MODELSCOPE_TOKEN", file=sys.stderr)
        return 1

    base = Path(args.base_dir).resolve()
    seen: set[str] = set()
    plan: list[Path] = []
    for backend, specs in BACKEND_MODELS.items():
        for spec in specs:
            rel = spec.rel_path
            if rel in seen:
                continue
            seen.add(rel)
            if (base / rel).is_file():
                plan.append(base / rel)
            else:
                print(f"[跳过] {rel}（本地不存在）")

    total = sum(p.stat().st_size for p in plan)
    print(f"\n仓库: {args.repo}")
    print(f"待上传 {len(plan)} 个文件，合计 {total / 1e6:.1f}MB\n")
    for p in plan:
        print(f"  {p.stat().st_size / 1e6:8.1f}MB  {p.relative_to(base).as_posix()}")

    if args.dry_run:
        return 0

    from modelscope.hub.api import HubApi

    api = HubApi()
    print(f"\n登录中…")
    api.login(args.token)
    print(f"创建/检查仓库 {args.repo}")
    try:
        api.create_model(args.repo, visibility=1, chinese_name="NOOCR ONNX 权重")
        print("仓库已创建")
    except Exception as e:
        print(f"仓库创建跳过（可能已存在）: {e}")

    for p in plan:
        rel = p.relative_to(base).as_posix()
        print(f"\n上传 {rel} …")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=rel,
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"  完成 {rel}")

    print(f"\n全部完成，仓库地址: https://www.modelscope.cn/models/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())