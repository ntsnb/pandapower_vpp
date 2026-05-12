from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.dashboard.app import create_dashboard_app, load_dashboard_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Dash dashboard for simulation outputs.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "outputs" / "dashboard_data"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check that dashboard CSV frames can be loaded; do not start Dash.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = load_dashboard_frames(args.data_dir)
    loaded = {name: len(frame) for name, frame in frames.items()}
    if args.check:
        print(f"dashboard_data_dir={args.data_dir}")
        print(f"loaded_frames={loaded}")
        return

    try:
        app = create_dashboard_app(frames=frames)
    except ImportError as exc:
        print(str(exc))
        print('Install visualization extras with: pip install -e ".[viz]"')
        raise SystemExit(1) from exc

    print(f"dashboard_url=http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

