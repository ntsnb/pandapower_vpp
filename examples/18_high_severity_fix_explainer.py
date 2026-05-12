from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.visualization.high_severity_fix_explainer import (
    build_high_severity_fix_explainer_html,
)


def main() -> None:
    output = build_high_severity_fix_explainer_html(
        Path("outputs") / "high_severity_fix_explainer.html"
    )
    print(f"high_severity_fix_explainer={output}")


if __name__ == "__main__":
    main()
