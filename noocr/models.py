"""模型下载与管理。

按**后端**组织权重清单，每个后端只拉自己需要的文件，支持最小化下载
（如 ``--backend ppocrv6-tiny`` 仅约 6MB）。默认路径与下载列表从同一份
:data:`BACKEND_MODELS` 生成，避免两处清单漂移导致 404。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .logging_config import get_logger

log = get_logger(__name__)

__all__ = [
    "ModelSpec",
    "BACKEND_MODELS",
    "model_path",
    "missing_models",
    "ensure_models",
    "download_from_hub",
    "print_status",
    "MODELS_ROOT",
    "MODELSCOPE_REPO",
]


def _default_models_root() -> Path:
    """模型根目录。优先级：环境变量 > 项目根/models > 包同级 models。"""
    env = os.getenv("NOOCR_MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "models").resolve()


MODELS_ROOT = _default_models_root()


@dataclass(frozen=True)
class ModelSpec:
    """一个模型文件的清单项。"""

    #: 相对 ``MODELS_ROOT`` 的路径
    rel_path: str
    #: 是否为「主模型」——缺失时后端不可用
    required: bool = True

    @property
    def path(self) -> Path:
        return MODELS_ROOT / self.rel_path


#: 各后端所需的模型。**这是模型清单的唯一真相来源**：
#: 下载逻辑与运行期路径解析都从这里派生，不会再出现两处不一致。
BACKEND_MODELS: Dict[str, List[ModelSpec]] = {
    "ppocrv5": [
        ModelSpec("ppocrv5/det/det.onnx"),
        ModelSpec("ppocrv5/rec/rec.onnx"),
        ModelSpec("ppocrv5/cls/cls.onnx", required=False),
        ModelSpec("ppocrv5/ppocrv5_dict.txt"),
    ],
    # ---- PP-OCRv6：官方新一代，检测 +4.6% / 识别 +5.1% ----
    # tiny: 1.8MB det + 4.3MB rec（1.5M 参数，边缘设备）
    # small: 9.5MB det + 21MB  rec（7.7M 参数，默认档）
    # medium: 34.5M 参数（服务器档，未包含）
    "ppocrv6-tiny": [
        ModelSpec("ppocrv6/det/PP-OCRv6_det_tiny.onnx"),
        ModelSpec("ppocrv6/rec/PP-OCRv6_rec_tiny.onnx"),
        ModelSpec("ppocrv6/cls/cls.onnx", required=False),
        ModelSpec("ppocrv6/ppocrv6_tiny_dict.txt"),
    ],
    "ppocrv6-small": [
        ModelSpec("ppocrv6/det/PP-OCRv6_det_small.onnx"),
        ModelSpec("ppocrv6/rec/PP-OCRv6_rec_small.onnx"),
        ModelSpec("ppocrv6/cls/cls.onnx", required=False),
        ModelSpec("ppocrv6/ppocrv6_dict.txt"),
    ],
    "plate": [
        ModelSpec("license_plate/car_plate_detect.onnx"),
        ModelSpec("license_plate/plate_rec.onnx"),
    ],
    "orientation": [
        ModelSpec("orientation/rapid_orientation.onnx", required=False),
    ],
    "layout": [
        ModelSpec("layout/layout_cdla.onnx", required=False),
        ModelSpec("layout/layout_publaynet.onnx", required=False),
    ],
    "table": [
        ModelSpec("table/slanet-plus.onnx", required=False),
    ],
}

#: 官方权重仓库（ModelScope）。所有后端的权重统一托管于此，
#: 目录结构与 ``rel_path`` 一致，因此下载就是一次快照 + 按需拷贝。
MODELSCOPE_REPO = "iterhui/noocr-onnx"

#: 单文件直链前缀，用于不走SDK 的最小下载。
MODELSCOPE_RESOLVE = f"https://www.modelscope.cn/models/{MODELSCOPE_REPO}/resolve/master"


def download_v6(backend: str = "ppocrv6-small", *, quiet: bool = False) -> List[Path]:
    """下载权重到 ``models/``（兼容旧调用名，实际不分版本）。

    Args:
        backend: :data:`BACKEND_MODELS` 的键。
        quiet: 静默模式。

    Returns:
        实际下载（或已存在而跳过）的文件列表。

    Raises:
        KeyError: 后端名不在清单中。
    """
    return download_from_hub(backend, quiet=quiet)


def model_path(backend: str, name: str) -> Path:
    """取某后端下指定模型的绝对路径。

    Raises:
        KeyError: 后端名或模型名不在清单中，消息中列出全部可用项。
    """
    specs = BACKEND_MODELS.get(backend)
    if specs is None:
        raise KeyError(f"未知后端 {backend!r}；可选: {', '.join(sorted(BACKEND_MODELS))}")
    for spec in specs:
        if spec.rel_path.endswith(name):
            return spec.path
    available = ", ".join(s.rel_path for s in specs)
    raise KeyError(f"后端 {backend!r} 中没有模型 {name!r}；可用: {available}")


def missing_models(backend: str, include_optional: bool = False) -> List[Path]:
    """列出缺失的模型文件。"""
    out: List[Path] = []
    for spec in BACKEND_MODELS.get(backend, []):
        if not spec.required and not include_optional:
            continue
        if not spec.path.is_file():
            out.append(spec.path)
    return out




def ensure_models(backend: str = "ppocrv6-small", *, quiet: bool = False) -> List[Path]:
    """确保某后端的模型就位，缺失则下载。

    Args:
        backend: :data:`BACKEND_MODELS` 的键。
        quiet: 静默模式。

    Returns:
        实际安装的文件列表；为空表示本就完整。

    Raises:
        KeyError: 后端名不在清单中。
        RuntimeError: 全部下载源均失败。
    """
    if backend not in BACKEND_MODELS:
        raise KeyError(
            f"未知后端 {backend!r}；可选: {', '.join(sorted(BACKEND_MODELS))}"
        )

    missing = missing_models(backend, include_optional=True)
    if not missing:
        if not quiet:
            log.info("后端 {} 的模型已就绪", backend)
        return []

    if not quiet:
        log.info("后端 {} 缺少 {} 个文件，开始下载", backend, len(missing))

    installed: List[Path] = []
    errors: List[str] = []
    for spec in BACKEND_MODELS[backend]:
        if spec.path.is_file() and spec.path.stat().st_size > 0:
            installed.append(spec.path)
            continue
        try:
            _fetch_one(spec, quiet=quiet)
            installed.append(spec.path)
        except Exception as e:
            if spec.required:
                raise
            errors.append(f"{spec.rel_path}: {e}")

    still = missing_models(backend, include_optional=True)
    if still:
        names = "\n".join(f"  - {p.relative_to(MODELS_ROOT)}" for p in still)
        raise RuntimeError(
            f"权重下载不完整，缺少:\n{names}\n"
            f"请手动从 {MODELSCOPE_REPO} 下载后放入: {MODELS_ROOT}"
        )
    if errors and not quiet:
        for msg in errors:
            log.warning("可选文件跳过 {}", msg)
    return installed


def _fetch_one(spec: ModelSpec, quiet: bool = False) -> None:
    """下载单个权重文件到目标路径。

    优先走 ModelScope SDK：它能正确处理鉴权、镜像与完整性校验。
    SDK 不可用时退回到 HTTPS 直链。
    """
    dst = spec.path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        log.info("下载 {} …", spec.rel_path)

    try:
        _fetch_via_sdk(spec, dst, quiet=quiet)
    except ImportError:
        _fetch_via_http(spec, dst, quiet=quiet)
    except Exception as e:
        if not quiet:
            log.warning("SDK 下载失败（{}），改用直链", e)
        _fetch_via_http(spec, dst, quiet=quiet)

    if not quiet:
        log.info("已安装 {} ({:.1f}MB)", spec.rel_path, dst.stat().st_size / 1e6)


def _fetch_via_sdk(spec: ModelSpec, dst: Path, quiet: bool = False) -> None:
    """用 ModelScope SDK 拉取单个文件到指定路径。"""
    from modelscope.hub.file_download import model_file_download

    with tempfile.TemporaryDirectory() as tmp:
        got = model_file_download(
            MODELSCOPE_REPO,
            spec.rel_path,
            local_dir=tmp,
        )
        if not Path(got).is_file():
            raise RuntimeError(f"SDK 未取到文件: {got}")
        shutil.copy2(got, dst)


def _fetch_via_http(spec: ModelSpec, dst: Path, quiet: bool = False) -> None:
    """用 HTTPS 直链拉取单个文件，先写 ``.part`` 再原子替换。"""
    import urllib.request

    url = f"{MODELSCOPE_RESOLVE}/{spec.rel_path}"
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=300) as r:
            tmp.write_bytes(r.read())
        tmp.replace(dst)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"下载 {spec.rel_path} 失败: {e}\n"
            f"URL: {url}\n"
            f"可手动从 {MODELSCOPE_REPO} 下载后放入 {dst}"
        ) from e


def download_from_hub(backend: str = "ppocrv6-small", *, quiet: bool = False) -> List[Path]:
    """强制从ModelScope 重新拉取某后端的全部权重。

    与 :func:`ensure_models` 的区别是忽略本地已存在的文件。
    """
    if backend not in BACKEND_MODELS:
        raise KeyError(
            f"未知后端 {backend!r}；可选: {', '.join(sorted(BACKEND_MODELS))}"
        )
    installed: List[Path] = []
    for spec in BACKEND_MODELS[backend]:
        if spec.path.is_file():
            spec.path.unlink()
        _fetch_one(spec, quiet=quiet)
        installed.append(spec.path)
    return installed


def print_status() -> None:
    """打印各后端模型状态。"""
    print(f"模型目录: {MODELS_ROOT}")
    print(f"权重仓库: {MODELSCOPE_REPO}\n")
    for backend, specs in BACKEND_MODELS.items():
        size = sum(s.path.stat().st_size for s in specs if s.path.is_file()) / 1e6
        print(f"[{backend}]  {size:.1f}MB")
        for spec in specs:
            exists = spec.path.is_file()
            mb = spec.path.stat().st_size / 1e6 if exists else 0
            mark = "OK" if exists else ("可选" if not spec.required else "缺失")
            print(f"  {mark:<4} {spec.rel_path:<44}{mb:7.1f}MB")
        print()
    print("下载: noocr models --get <后端>")
