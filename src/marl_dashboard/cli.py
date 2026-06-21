from __future__ import annotations

import argparse
from pathlib import Path

from marl_dashboard.backend.server import start_dashboard
from marl_dashboard.demo.generate_demo_run import generate_demo_run


def _serve(args: argparse.Namespace) -> None:
    start_dashboard(
        data_dir=args.data_dir,
        host=args.host,
        port=args.port,
        auto_port=args.auto_port,
        open_browser=args.open_browser,
        background=False,
    )


def _demo(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    run_id = generate_demo_run(
        data_dir=data_dir,
        run_id=args.run_id,
        vpp_count=args.vpp_count,
        epochs=args.epochs,
        days=args.days,
        steps_per_day=args.steps_per_day,
        async_writer=False,
    )
    print(f"Demo run generated: {run_id}")
    if args.no_serve:
        return
    start_dashboard(
        data_dir=data_dir,
        host=args.host,
        port=args.port,
        auto_port=args.auto_port,
        open_browser=args.open_browser,
        background=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local MARL VPP dashboard.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Serve an existing dashboard data directory.")
    serve.add_argument("--data-dir", default="runs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--auto-port", action="store_true")
    serve.add_argument("--open-browser", action="store_true")
    serve.set_defaults(func=_serve)

    demo = subparsers.add_parser("demo", help="Generate a synthetic multi-VPP run and serve it.")
    demo.add_argument("--data-dir", default="runs")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8765)
    demo.add_argument("--auto-port", action="store_true")
    demo.add_argument("--open-browser", action="store_true")
    demo.add_argument("--run-id", default="demo_vpp_marl")
    demo.add_argument("--vpp-count", type=int, default=5)
    demo.add_argument("--epochs", type=int, default=3)
    demo.add_argument("--days", type=int, default=35)
    demo.add_argument("--steps-per-day", type=int, default=24)
    demo.add_argument("--no-serve", action="store_true", help="Generate demo data without starting the local server.")
    demo.set_defaults(func=_demo)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
