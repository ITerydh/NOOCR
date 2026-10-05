"""Web 服务层：REST API 与 Web UI。

两个入口共享同一份 :class:`BackendPool`，模型只加载一次并复用：

- :func:`create_app` 返回 FastAPI 应用，同时挂载 ``/api/*`` 与 ``/`` 界面。
- :func:`run` 供 ``python -m noocr serve`` 直接启动。
"""

from __future__ import annotations

from .app import BackendPool, create_app, run

__all__ = ["BackendPool", "create_app", "run"]