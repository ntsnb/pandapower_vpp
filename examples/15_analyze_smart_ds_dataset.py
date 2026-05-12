from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vpp_dso_sim.data_sources.smart_ds import export_smart_ds_analysis  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze downloaded SMART-DS OpenDSS data.")
    parser.add_argument(
        "--root",
        default=str(
            PROJECT_ROOT
            / "data"
            / "external"
            / "raw"
            / "smart_ds"
            / "v1.0"
            / "2018"
            / "AUS"
            / "P1U"
            / "base_timeseries"
            / "opendss"
        ),
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "smart_ds_analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = export_smart_ds_analysis(args.root, args.output_dir)
    for key, path in paths.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
