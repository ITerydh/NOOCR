"""检查与停止 NOOCR 的 Web 服务；启动交给调用方在后台执行。

**为什么脚本不负责启动**：本机上由 shell 拉起的子进程（Git Bash 后台、
``cmd /c start``、``DETACHED_PROCESS``、``subprocess.Popen`` 四种方式都
试过）都会在发起它的那条命令结束时被一并回收。表现为「刚启动探测 200，
几十秒后端口就没了」，而服务日志里什么错都没有——进程根本没走到能打
日志的那一步。所以启动必须走宿主的后台任务机制，由它维持进程存活。

改完代码后的固定流程::

    # 1) 停掉旧进程。必须做：模板与静态资源在服务启动时已读进内存，
    #    不重启的话浏览器看到的还是旧页面，会得出「改了没生效」的错误结论
    python scripts/restart.py --stop

    # 2) 在后台起服务（两条命令，分别对应 GPU 与 CPU）
    noocr-gpu/Scripts/python.exe -m noocr serve --port 8940
    noocr-env/Scripts/python.exe -m noocr serve --port 8812 --device cpu

    # 3) 确认就绪
    python scripts/restart.py --check

两个服务必须用**各自的解释器**，不能混：GPU 那个用错解释器会静默退回
CPU（ORT 在 CUDA provider 初始化失败时不抛异常，只打warning），
而 ``/api/backends`` 一样返回 200，从接口上完全看不出设备变了。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: 端口 -> (解释器目录, 设备参数, 标签)
#: 解释器用相对 ROOT 的路径，换机器不用改脚本。
SERVICES: dict[int, tuple[str, str, str]] = {
    8940: ("noocr-gpu", "", "GPU"),
    8812: ("noocr-env", "--device cpu", "CPU"),
}

#: 探测用的路径。选 /api/backends 而不是 /：它是本项目独有的端点，
#: 用 / 的话别的程序占用同一端口也会返回 200，脚本会误报「已就绪」。
PROBE = "/api/backends"


def find_pids(port: int) -> list[int]:
    """找出占用指定端口的进程 PID。

    走 ``netstat`` 而非遍历所有进程：我们要的是「谁占着这个端口」，
    而同一条serve 命令可能留下两个 python 进程（venv 启动器与它拉起
    的解释器），只有端口能唯一确定该杀谁。

    输出按字节读再自行解码：``text=True`` 会用 locale 编码解码，
    而 netstat 在中文 Windows 上输出 GBK，直接用会抛
    ``UnicodeDecodeError``，``stdout`` 拿到的是 None。
    """
    try:
        proc = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    pids: list[int] = []
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        if f":{port} " not in line or "LISTENING" not in line:
            continue
        cols = line.split()
        if cols and cols[-1].isdigit():
            pid = int(cols[-1])
            if pid not in pids:
                pids.append(pid)
    return pids


def probe(port: int, timeout: float = 4.0) -> str:
    """探测服务状态，返回可读描述。"""
    url = f"http://127.0.0.1:{port}{PROBE}"
    # 显式不走代理：本机回环请求若被 http_proxy 接管，连不上会被
    # 报成 502，掩盖真实原因。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as r:
            body = r.read()
        # 数"name" 出现次数即后端数量——比解析 JSON 轻，够判断是否
        # 真的起来了（返回 200 但内容不对的情况也见过）
        tiers = body.count(b'"name"')
        return f"{r.status}（{tiers} 个后端）"
    except urllib.error.HTTPError as e:
        return f"{e.code}（已启动但报错）"
    except urllib.error.URLError:
        return "无响应"
    except OSError as e:
        return f"无响应（{e.strerror or e}）"


def stop(port: int) -> list[int]:
    """停掉占用端口的进程，返回被杀的 PID。"""
    pids = find_pids(port)
    if not pids:
        return []
    for pid in pids:
        # taskkill /T 连子进程一起杀：Windows 上 python.exe 会派生子进程，
        # 只杀父进程的话子进程继续占着端口
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid), "/T"],
            capture_output=True, timeout=30,
        )
    # 端口从 LISTENING 消失要一点时间，立刻启会撞「地址已在使用」
    for _ in range(20):
        if not find_pids(port):
            break
        time.sleep(0.25)
    return pids


def report(targets: list[int]) -> None:
    print("端口".rjust(6) + "类型".rjust(6) + "PID".rjust(10) + "  探测")
    for port in targets:
        _, _, label = SERVICES[port]
        pids = find_pids(port)
        pid_txt = ",".join(str(p) for p in pids) if pids else "—"
        print(f"{port:>6}{label:>6}{pid_txt:>10}  {probe(port)}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="检查/ 停止 NOOCR Web 服务（启动需在后台执行）")
    ap.add_argument("--stop", action="store_true",
                    help="停掉所有服务")
    ap.add_argument("--check", action="store_true",
                    help="只看状态与探测结果（默认行为）")
    ap.add_argument("--only", choices=["gpu", "cpu"],
                    help="只处理其中一个，默认两个都处理")
    args = ap.parse_args()

    targets = list(SERVICES)
    if args.only:
        targets = [8940 if args.only == "gpu" else 8812]

    if args.stop:
        for port in targets:
            _, _, label = SERVICES[port]
            pids = stop(port)
            state = "PID " + ",".join(map(str, pids)) if pids else "本就未运行"
            print(f"  {port} {label:3} {state}")
        # 停完再报一次：确认端口真的空了，而不是杀完就返回
        print()
        report(targets)
        return 0

    report(targets)
    down = [p for p in targets if "200" not in probe(p)]
    if down:
        print("\n以下端口未就绪，需在后台启动：")
        for p in down:
            env_dir, device, label = SERVICES[p]
            print(f"  {ROOT / env_dir / 'Scripts' / 'python.exe'} "
                  f"-m noocr serve --port {p} {device}".rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
