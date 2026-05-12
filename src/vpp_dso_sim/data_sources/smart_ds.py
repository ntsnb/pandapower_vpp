from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
from typing import Any

import pandas as pd

from vpp_dso_sim.utils.io import ensure_dir, write_json


ELEMENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "load": re.compile(r"^\s*new\s+load\.", re.IGNORECASE),
    "line": re.compile(r"^\s*new\s+line\.", re.IGNORECASE),
    "transformer": re.compile(r"^\s*new\s+transformer\.", re.IGNORECASE),
    "pvsystem": re.compile(r"^\s*new\s+pvsystem\.", re.IGNORECASE),
    "storage": re.compile(r"^\s*new\s+storage\.", re.IGNORECASE),
    "capacitor": re.compile(r"^\s*new\s+capacitor\.", re.IGNORECASE),
    "regcontrol": re.compile(r"^\s*new\s+regcontrol\.", re.IGNORECASE),
    "loadshape": re.compile(r"^\s*new\s+loadshape\.", re.IGNORECASE),
}


@dataclass(frozen=True)
class SmartDSSuite:
    suite_id: str
    name: str
    topology_source: str
    profile_source: str
    vpp_design: str
    train_split: str
    eval_split: str
    holdout_split: str
    use_case: str
    missing_adapters: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _count_elements(path: Path) -> dict[str, int]:
    counts = {key: 0 for key in ELEMENT_PATTERNS}
    text = _safe_read_text(path)
    for line in text.splitlines():
        for key, pattern in ELEMENT_PATTERNS.items():
            counts[key] += int(bool(pattern.search(line)))
    return counts


def _is_primary_feeder_dir(path: Path) -> bool:
    return bool(re.fullmatch(r"p\d+uhs\d+_1247", path.name))


def _is_distribution_transformer_dir(path: Path) -> bool:
    return "--p1udt" in path.name.lower()


def smart_ds_file_inventory(root: str | Path) -> pd.DataFrame:
    base = Path(root)
    rows: list[dict[str, Any]] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        rows.append(
            {
                "relative_path": str(relative),
                "suffix": path.suffix.lower() or "<none>",
                "size_bytes": int(path.stat().st_size),
                "parent": str(relative.parent),
                "is_dss": path.suffix.lower() == ".dss",
            }
        )
    return pd.DataFrame(rows)


def smart_ds_feeder_inventory(root: str | Path) -> pd.DataFrame:
    base = Path(root)
    rows: list[dict[str, Any]] = []
    for feeder_dir in sorted([path for path in base.iterdir() if path.is_dir() and _is_primary_feeder_dir(path)]):
        dss_files = list(feeder_dir.rglob("*.dss"))
        nested_dt_dirs = [path for path in feeder_dir.rglob("*") if path.is_dir() and _is_distribution_transformer_dir(path)]
        counts = {key: 0 for key in ELEMENT_PATTERNS}
        for dss_file in dss_files:
            file_counts = _count_elements(dss_file)
            for key, value in file_counts.items():
                counts[key] += value
        rows.append(
            {
                "feeder_id": feeder_dir.name,
                "relative_path": str(feeder_dir.relative_to(base)),
                "dss_file_count": len(dss_files),
                "nested_distribution_transformer_dirs": len(nested_dt_dirs),
                "total_size_bytes": int(sum(path.stat().st_size for path in feeder_dir.rglob("*") if path.is_file())),
                **{f"{key}_count": value for key, value in counts.items()},
            }
        )
    return pd.DataFrame(rows)


def smart_ds_dataset_suites(root: str | Path) -> pd.DataFrame:
    base = Path(root)
    return pd.DataFrame(
        [
            SmartDSSuite(
                suite_id="smart_ds_full_feeder_ctde",
                name="SMART-DS full feeder CTDE training suite",
                topology_source=str(base / "Master.dss"),
                profile_source=str(base / "LoadShapes.dss"),
                vpp_design=(
                    "Map each primary feeder to one or more VPPs; use downstream transformer folders as "
                    "candidate multi-node portfolios and root feeder folders as single-PCC portfolios."
                ),
                train_split="AUS/P1U feeders p1uhs0-p1uhs14",
                eval_split="AUS/P1U feeders p1uhs15-p1uhs19",
                holdout_split="AUS/P1U feeders p1uhs20-p1uhs24 plus high-PV scenario when downloaded",
                use_case="Topology-scale CTDE/MATD3 stress training and single-PCC vs multi-node VPP generalization.",
                missing_adapters="OpenDSS-to-pandapower converter, feeder split manager, DER ownership synthesis.",
            ).to_dict(),
            SmartDSSuite(
                suite_id="smart_ds_lv_portfolio_suite",
                name="SMART-DS distribution-transformer portfolio suite",
                topology_source=str(base),
                profile_source=str(base / "LoadShapes.dss"),
                vpp_design=(
                    "Treat nested distribution-transformer directories as low-voltage/taiqu-like portfolio scopes; "
                    "assign PV/storage/EV/flexible loads to transformer-level VPPs."
                ),
                train_split="Transformer folders sampled from p1uhs0-p1uhs12",
                eval_split="Transformer folders sampled from p1uhs13-p1uhs18",
                holdout_split="Transformer folders from unseen feeders p1uhs19-p1uhs24",
                use_case="Slow portfolio agent and multi-node VPP membership/reweighting experiments.",
                missing_adapters="LV-equivalent extraction, transformer-to-bus mapping, portfolio event generator.",
            ).to_dict(),
            SmartDSSuite(
                suite_id="hybrid_der_market_suite",
                name="Hybrid DER-market suite",
                topology_source=str(base / "Master.dss"),
                profile_source="SMART-DS load shapes + NREL EULP + ACN-Data + OpenEI/CAISO prices",
                vpp_design=(
                    "Use SMART-DS topology as the electrical backbone, EULP for building/HVAC profiles, "
                    "ACN-Data for EVCS sessions, and OpenEI/CAISO for retail/market price signals."
                ),
                train_split="Chronological profile split by date after adapters are added",
                eval_split="Different weather/load/EV days",
                holdout_split="Reverse-flow, high-price, cloudy-PV and feeder-transfer stress cases",
                use_case="Paper-grade VPP profit, flexibility service and grid-safety benchmark.",
                missing_adapters="EULP, ACN and tariff adapters; market settlement integration into simulator.step.",
            ).to_dict(),
        ]
    )


def _profile_root_for_opendss_root(base: Path) -> Path:
    parts = list(base.parts)
    try:
        index = parts.index("base_timeseries")
    except ValueError:
        return base.parent / "profiles"
    return Path(*parts[:index]) / "profiles"


def _loadshape_profile_references(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"(?:q?mult)\s*=\s*\(\s*file\s*=\s*([^)]+\.csv)\s*\)", re.IGNORECASE)
    for path in root.rglob("LoadShapes.dss"):
        text = _safe_read_text(path)
        for line in text.splitlines():
            if "new loadshape" not in line.lower():
                continue
            for ref in pattern.findall(line):
                clean = ref.replace("\\", "/").strip().strip('"').strip("'")
                rows.append(
                    {
                        "loadshape_file": str(path.relative_to(root)),
                        "profile_reference": clean,
                        "profile_filename": Path(clean).name,
                        "profile_class": "residential"
                        if "res" in clean.lower()
                        else "commercial"
                        if "com" in clean.lower()
                        else "unknown",
                    }
                )
    return pd.DataFrame(rows)


def analyze_smart_ds_dataset(root: str | Path, profiles_root: str | Path | None = None) -> dict[str, Any]:
    base = Path(root)
    profile_base = Path(profiles_root) if profiles_root is not None else _profile_root_for_opendss_root(base)
    files = smart_ds_file_inventory(base)
    feeders = smart_ds_feeder_inventory(base)
    suites = smart_ds_dataset_suites(base)
    profile_refs = _loadshape_profile_references(base)
    profile_files = smart_ds_file_inventory(profile_base) if profile_base.exists() else pd.DataFrame()
    available_profile_names = (
        set(Path(value).name for value in profile_files["relative_path"])
        if not profile_files.empty and "relative_path" in profile_files
        else set()
    )
    if not profile_refs.empty:
        profile_refs["available_locally"] = profile_refs["profile_filename"].isin(available_profile_names)
    else:
        profile_refs["available_locally"] = []
    suffix_counts = (
        files.groupby("suffix", dropna=False)
        .agg(file_count=("relative_path", "count"), total_size_bytes=("size_bytes", "sum"))
        .reset_index()
        .sort_values(["file_count", "suffix"], ascending=[False, True])
    )
    summary = {
        "dataset_id": "smart_ds_aus_p1u_base_opendss",
        "root": str(base),
        "exists": bool(base.exists()),
        "file_count": int(len(files)),
        "dss_file_count": int(files["is_dss"].sum()) if not files.empty else 0,
        "total_size_bytes": int(files["size_bytes"].sum()) if not files.empty else 0,
        "total_size_mib": round(float(files["size_bytes"].sum()) / 2**20, 3) if not files.empty else 0.0,
        "primary_feeder_count": int(len(feeders)),
        "distribution_transformer_dir_count": int(
            feeders["nested_distribution_transformer_dirs"].sum()
        )
        if not feeders.empty
        else 0,
        "total_load_count": int(feeders["load_count"].sum()) if "load_count" in feeders else 0,
        "total_line_count": int(feeders["line_count"].sum()) if "line_count" in feeders else 0,
        "total_transformer_count": int(feeders["transformer_count"].sum()) if "transformer_count" in feeders else 0,
        "suite_count": int(len(suites)),
        "profiles_root": str(profile_base),
        "profiles_root_exists": bool(profile_base.exists()),
        "profile_file_count": int(len(profile_files)),
        "profile_total_size_mib": round(float(profile_files["size_bytes"].sum()) / 2**20, 3)
        if not profile_files.empty and "size_bytes" in profile_files
        else 0.0,
        "loadshape_profile_reference_count": int(len(profile_refs)),
        "unique_loadshape_profile_reference_count": int(profile_refs["profile_filename"].nunique())
        if not profile_refs.empty
        else 0,
        "available_profile_reference_count": int(profile_refs["available_locally"].sum())
        if not profile_refs.empty
        else 0,
        "missing_profile_reference_count": int((~profile_refs["available_locally"]).sum())
        if not profile_refs.empty
        else 0,
    }
    return {
        "summary": summary,
        "files": files,
        "profile_files": profile_files,
        "profile_references": profile_refs,
        "suffix_counts": suffix_counts,
        "feeders": feeders,
        "suites": suites,
    }


def export_smart_ds_analysis(root: str | Path, output_dir: str | Path) -> dict[str, Path]:
    out = ensure_dir(output_dir)
    result = analyze_smart_ds_dataset(root)
    paths: dict[str, Path] = {}
    for name in ("files", "profile_files", "profile_references", "suffix_counts", "feeders", "suites"):
        path = out / f"smart_ds_{name}.csv"
        result[name].to_csv(path, index=False)
        paths[name] = path
    summary_path = out / "smart_ds_summary.json"
    write_json(summary_path, result["summary"])
    paths["summary"] = summary_path
    report_path = out / "smart_ds_dataset_report.md"
    report_path.write_text(
        "# SMART-DS Dataset Analysis\n\n"
        f"```json\n{json.dumps(result['summary'], indent=2)}\n```\n\n"
        "## Recommended Suites\n\n"
        + result["suites"].to_markdown(index=False),
        encoding="utf-8",
    )
    paths["report"] = report_path
    return paths


__all__ = [
    "analyze_smart_ds_dataset",
    "export_smart_ds_analysis",
    "smart_ds_dataset_suites",
    "smart_ds_feeder_inventory",
    "smart_ds_file_inventory",
]
