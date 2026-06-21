from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
from threading import Thread
from typing import Any


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, int(port))) != 0


def _find_port(host: str, start_port: int) -> int:
    for port in range(int(start_port), int(start_port) + 100):
        if _port_available(host, port):
            return port
    raise RuntimeError(f"No free local port found from {start_port} to {start_port + 99}.")


@dataclass
class DashboardHandle:
    host: str
    port: int
    data_dir: Path
    server: Any
    thread: Thread | None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=5)


def start_dashboard(
    *,
    data_dir: str | Path = "runs",
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    auto_port: bool = False,
    background: bool = True,
) -> DashboardHandle:
    if host == "0.0.0.0":
        raise ValueError("marl-dashboard refuses to bind 0.0.0.0 by default; pass an explicit safe host.")
    if not _port_available(host, int(port)):
        if not auto_port:
            raise RuntimeError(f"Port {port} is already in use on {host}. Pass auto_port=True to pick a free port.")
        port = _find_port(host, int(port) + 1)

    import uvicorn

    from marl_dashboard.backend.app import create_app

    app = create_app(data_dir=data_dir)
    config = uvicorn.Config(app, host=host, port=int(port), log_level="warning")
    server = uvicorn.Server(config)
    thread = None
    url = f"http://{host}:{port}"
    print(f"Dashboard running at {url}")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    if background:
        thread = Thread(target=server.run, name="marl-dashboard-server", daemon=True)
        thread.start()
    else:
        server.run()
    return DashboardHandle(host=host, port=int(port), data_dir=Path(data_dir).expanduser().resolve(), server=server, thread=thread)
