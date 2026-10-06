"""日志配置。

采用「惰性安装 + 引用记账」：只有显式调用 :func:`setup` 时才改动全局状态，
重复调用不会叠加 sink。库代码默认不配置日志，交给宿主应用决定。
"""

from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Any, Union

from loguru import logger

__all__ = ["logger", "setup", "add_file_sink", "remove_sink", "get_logger", "ensure_stdout_utf8"]


def ensure_stdout_utf8() -> None:
    """把标准输出/错误切到 UTF-8，避免非ASCII 文本触发编码异常。

    Windows 控制台默认编码是 cp1252（CI 的 pwsh 同理），而本项目的日志与
    CLI 输出含大量中文。缺了这层防护，``noocr backends`` 这类命令会直接
    抛 ``UnicodeEncodeError`` 而中断——用户看到的是崩溃，而不是信息。

    只在标准流支持 ``reconfigure`` 时改动；被测试替换成``io.StringIO``
    的场景会自然跳过。``errors="replace"`` 兜底极少数无法映射的字符
    （如 emoji），保证「显示为问号」好过「整个进程崩掉」。
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # 非 TextIO（例如被替换成 BytesIO）会抛 ValueError/OSError，
        # 那种情况直接跳过即可，不影响主流程
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


ensure_stdout_utf8()

#: 本模块添加过的 sink 记账，避免重复添加并支持精确卸载。
_SINKS: dict[str, int] = {}

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <7}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {name}:{line} - {message}"


class _InterceptHandler(logging.Handler):
    """把标准库 logging（含第三方库日志）转发到 loguru。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: Union[str, int] = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame: Any = logging.currentframe()
        depth = 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup(
    level: str = "INFO",
    *,
    colorize: bool = True,
    intercept_stdlib: bool = True,
    force: bool = False,
) -> None:
    """配置日志。仅应用入口应调用。

    Args:
        level: 日志级别，如 ``"DEBUG"`` / ``"INFO"``。
        colorize: 是否彩色输出，重定向到文件时应关闭。
        intercept_stdlib: 是否接管标准库 logging。
        force: 是否先移除本模块此前添加的 sink。

    .. note::
       接管标准库日志时只追加 handler，不使用 ``force=True``，
       避免清空宿主应用自己的 root handler。
    """
    if force:
        for key in list(_SINKS):
            remove_sink(key)

    if "console" not in _SINKS:
        _SINKS["console"] = logger.add(
            sys.stderr,
            format=_CONSOLE_FORMAT,
            level=level.upper(),
            colorize=colorize,
        )

    if intercept_stdlib and "intercept" not in _SINKS:
        root = logging.getLogger()
        root.setLevel(logging.WARNING)
        root.addHandler(_InterceptHandler())
        _SINKS["intercept"] = -1


def add_file_sink(path: Union[str, Path], level: str = "INFO", *, rotation: str = "10 MB") -> int:
    """添加文件日志，返回可传给 :func:`remove_sink` 的句柄。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return logger.add(
        str(p),
        format=_FILE_FORMAT,
        level=level.upper(),
        encoding="utf-8",
        rotation=rotation,
    )


def remove_sink(key: Union[str, int]) -> None:
    """移除由本模块添加的 sink。"""
    if isinstance(key, str):
        handle = _SINKS.pop(key, None)
        if handle is None or handle < 0:
            return
        logger.remove(handle)
    else:
        logger.remove(key)


def get_logger(name: str):
    """获取绑定统一 name 的 logger，用法与 loguru 一致。"""
    return logger.bind(name=name)
