from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vpp_dso_sim.utils.config import project_root, resolve_project_path


def load_profile_csv(path: str | Path, horizon_steps: int = 288) -> list[float]:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Profile CSV not found: {path}")
    frame = pd.read_csv(resolved)
    if "value" not in frame.columns:
        raise ValueError(f"Profile CSV must contain a 'value' column: {resolved}")
    values = frame["value"].astype(float).to_list()
    if len(values) >= horizon_steps:
        return values[:horizon_steps]
    repeats = int(np.ceil(horizon_steps / len(values)))
    return (values * repeats)[:horizon_steps]


def default_load_profile(horizon_steps: int = 288) -> list[float]:
    x = np.linspace(0, 2 * np.pi, horizon_steps, endpoint=False)
    return (0.95 + 0.30 * np.sin(x - np.pi / 2) + 0.15 * np.sin(2 * x)).clip(0.5, 1.4).tolist()


def default_pv_profile(horizon_steps: int = 288) -> list[float]:
    hours = np.arange(horizon_steps) / 4.0
    pv = np.exp(-0.5 * ((hours - 12.0) / 3.0) ** 2)
    pv[(hours < 5.0) | (hours > 19.0)] = 0.0
    return pv.clip(0.0, 1.0).tolist()


def default_price_profile(horizon_steps: int = 288) -> list[float]:
    hours = np.arange(horizon_steps) / 4.0
    price = 50 + 40 * ((hours >= 7) & (hours <= 11)) + 70 * ((hours >= 17) & (hours <= 21))
    return price.astype(float).tolist()


def benchmark_profile_pack(
    horizon_steps: int = 288,
    *,
    dt_hours: float = 0.25,
    seed: int = 2026,
    variant: str = "train_mixed",
) -> dict[str, list[float]]:
    """Return non-repeating synthetic benchmark profiles.

    This is still synthetic data, not a replacement for public datasets such as
    SimBench, NREL EULP, Pecan Street, or ACN-Data. It is intentionally richer
    than the demo 96-point CSV replay: daily load multipliers, cloud events,
    evening EV-like peaks, and scarcity price adders differ across days and
    holdout variants.
    """

    rng = np.random.default_rng(seed + sum(ord(ch) for ch in variant))
    steps_per_day = max(1, int(round(24.0 / dt_hours)))
    index = np.arange(horizon_steps)
    hour = (index * dt_hours) % 24.0
    day = index // steps_per_day
    n_days = int(day.max()) + 1 if horizon_steps else 1

    base_daily = np.array([0.94, 1.04, 1.13, 0.98, 1.08, 1.18, 1.02], dtype=float)
    daily = np.resize(base_daily, n_days)
    if "holdout_reverseflow" in variant:
        daily = daily - np.linspace(0.10, 0.18, n_days)
    elif "holdout_peak" in variant:
        daily = daily + np.linspace(0.06, 0.16, n_days)
    elif "holdout_light" in variant:
        daily = daily - 0.05
    elif "holdout_cloudy" in variant:
        daily = daily + 0.03
    daily = daily * rng.normal(1.0, 0.025, size=n_days)

    morning = np.exp(-0.5 * ((hour - 7.8) / 1.8) ** 2)
    evening = np.exp(-0.5 * ((hour - 19.2) / 2.4) ** 2)
    commercial = np.exp(-0.5 * ((hour - 13.0) / 3.0) ** 2)
    ev_peak = np.exp(-0.5 * ((hour - 21.0) / 1.7) ** 2)
    load = 0.62 + 0.16 * morning + 0.30 * evening + 0.14 * commercial + 0.10 * ev_peak
    load = load * daily[day]
    load += 0.015 * np.sin(2 * np.pi * index / max(steps_per_day * 2, 1))
    load = np.clip(load, 0.48, 1.55)

    solar = np.sin(np.pi * (hour - 5.7) / 13.8)
    solar = np.where((hour >= 5.7) & (hour <= 19.5), np.maximum(0.0, solar), 0.0)
    cloud = np.ones(horizon_steps)
    cloud_daily = rng.uniform(0.82, 1.02, size=n_days)
    if "holdout_reverseflow" in variant:
        cloud_daily *= np.linspace(1.05, 1.18, n_days)
    elif "holdout_cloudy" in variant:
        cloud_daily *= np.linspace(0.62, 0.78, n_days)
    elif "holdout_peak" in variant:
        cloud_daily *= np.linspace(0.88, 0.72, n_days)
    for d in range(n_days):
        day_mask = day == d
        event_center = rng.uniform(10.5, 15.5)
        event_width = rng.uniform(0.9, 2.2)
        event_depth = rng.uniform(0.10, 0.35)
        if "holdout_reverseflow" in variant:
            event_depth *= 0.35
        elif "holdout_cloudy" in variant:
            event_depth += 0.20
        cloud[day_mask] = cloud_daily[d] * (
            1.0 - event_depth * np.exp(-0.5 * ((hour[day_mask] - event_center) / event_width) ** 2)
        )
    pv = np.clip(solar**1.25 * cloud, 0.0, 1.0)

    scarcity = np.maximum(0.0, load - 1.02) * 95.0 + np.maximum(0.0, 0.35 - pv) * 12.0
    price = 38.0 + 22.0 * morning + 52.0 * evening + scarcity
    if "holdout_reverseflow" in variant:
        midday_discount = 18.0 * ((hour >= 10.0) & (hour <= 15.0))
        price = price - midday_discount
    if "holdout_peak" in variant:
        price += 18.0 * (hour >= 17.0) * (hour <= 22.0)
    price = np.clip(price, 18.0, 185.0)

    return {
        "load": load.astype(float).tolist(),
        "pv": pv.astype(float).tolist(),
        "price": price.astype(float).tolist(),
    }


def smart_ds_austin_profile_pack(
    horizon_steps: int = 288,
    *,
    dt_hours: float = 0.25,
    seed: int = 2026,
    variant: str = "train_mixed",
    profiles_root: str | Path | None = None,
    load_profile_count: int = 32,
    solar_profile_count: int = 12,
) -> dict[str, list[float] | dict[str, object]]:
    """Build long-horizon profiles from the locally downloaded SMART-DS Austin data.

    SMART-DS gives annual 15-minute load-shape CSVs and Austin solar-shape
    CSVs, but it does not provide the exact market-price process required by
    this VPP/DSO simulator. The returned price is therefore a transparent
    scarcity proxy derived from load, PV, and peak hours; the load and PV
    trajectories themselves are sampled from local SMART-DS files when present.
    If the dataset is missing, the function falls back to ``benchmark_profile_pack``
    and marks the source in metadata so experiments remain runnable.
    """

    root = Path(profiles_root) if profiles_root is not None else (
        project_root()
        / "data"
        / "external"
        / "raw"
        / "smart_ds"
        / "v1.0"
        / "2018"
        / "AUS"
        / "P1U"
        / "profiles"
    )
    if not root.exists():
        pack = benchmark_profile_pack(horizon_steps, dt_hours=dt_hours, seed=seed, variant=variant)
        pack["metadata"] = {
            "source": "synthetic_fallback_benchmark_profile_pack",
            "reason": f"SMART-DS profiles root not found: {root}",
            "variant": variant,
        }
        return pack

    rng = np.random.default_rng(seed + sum(ord(ch) for ch in variant))
    res_files = sorted(root.glob("res_kw_*_pu.csv"))
    com_files = sorted(root.glob("com_kw_*_pu.csv"))
    solar_files = sorted(root.glob("AUS_*.csv"))
    if not res_files and not com_files:
        pack = benchmark_profile_pack(horizon_steps, dt_hours=dt_hours, seed=seed, variant=variant)
        pack["metadata"] = {
            "source": "synthetic_fallback_benchmark_profile_pack",
            "reason": "No SMART-DS res_kw/com_kw profiles found.",
            "variant": variant,
        }
        return pack

    def choose(files: list[Path], count: int) -> list[Path]:
        if not files:
            return []
        count = min(max(1, int(count)), len(files))
        indices = rng.choice(len(files), size=count, replace=False)
        return [files[int(idx)] for idx in np.sort(indices)]

    def load_series(path: Path) -> np.ndarray:
        series = pd.read_csv(path, header=None).iloc[:, 0].astype(float).to_numpy()
        if len(series) == 0:
            return np.zeros(horizon_steps, dtype=float)
        max_start = max(0, len(series) - horizon_steps)
        if "eval" in variant or "holdout" in variant:
            start = int(0.65 * max_start) if max_start else 0
            start = min(max_start, start + int(rng.integers(0, max(1, max_start // 6 + 1))))
        else:
            start = int(rng.integers(0, max(1, max_start + 1))) if max_start else 0
        window = series[start : start + horizon_steps]
        if len(window) < horizon_steps:
            repeats = int(np.ceil(horizon_steps / max(1, len(series))))
            window = np.resize(np.tile(series, repeats), horizon_steps)
        return np.asarray(window[:horizon_steps], dtype=float)

    selected_res = choose(res_files, max(1, int(round(load_profile_count * 0.75))))
    selected_com = choose(com_files, max(1, int(round(load_profile_count * 0.25)))) if com_files else []
    load_components = [load_series(path) for path in [*selected_res, *selected_com]]
    load_raw = np.mean(np.stack(load_components, axis=0), axis=0) if load_components else np.ones(horizon_steps)
    load_mean = float(np.mean(load_raw)) if np.mean(load_raw) > 1e-9 else 1.0
    load = load_raw / load_mean

    selected_solar = choose(solar_files, solar_profile_count)
    if selected_solar:
        solar_components = [load_series(path) for path in selected_solar]
        pv_raw = np.mean(np.stack(solar_components, axis=0), axis=0)
        pv = pv_raw / max(float(np.max(pv_raw)), 1e-9)
    else:
        pv = np.asarray(default_pv_profile(horizon_steps), dtype=float)

    if "holdout_peak" in variant:
        load = load * 1.10
        pv = pv * 0.82
    elif "holdout_cloudy" in variant:
        load = load * 1.04
        pv = pv * 0.62
    elif "holdout_reverseflow" in variant:
        load = load * 0.90
        pv = np.clip(pv * 1.18, 0.0, 1.0)

    hour = (np.arange(horizon_steps) * dt_hours) % 24.0
    morning = ((hour >= 7.0) & (hour <= 10.5)).astype(float)
    evening = ((hour >= 17.0) & (hour <= 22.0)).astype(float)
    net_load = np.maximum(0.0, load - 0.45 * pv)
    price = 34.0 + 48.0 * net_load + 18.0 * morning + 42.0 * evening
    price += 20.0 * np.maximum(0.0, load - 1.22)
    price -= 14.0 * ((hour >= 10.0) & (hour <= 15.0)).astype(float) * pv
    if "holdout_peak" in variant:
        price += 16.0 * evening
    price = np.clip(price, 12.0, 205.0)

    return {
        "load": np.clip(load, 0.35, 2.10).astype(float).tolist(),
        "pv": np.clip(pv, 0.0, 1.0).astype(float).tolist(),
        "price": price.astype(float).tolist(),
        "metadata": {
            "source": "smart_ds_austin_profiles_local",
            "profiles_root": str(root),
            "variant": variant,
            "selected_residential_load_profiles": [path.name for path in selected_res[:8]],
            "selected_commercial_load_profiles": [path.name for path in selected_com[:8]],
            "selected_solar_profiles": [path.name for path in selected_solar[:8]],
            "price_source": "derived_scarcity_proxy_not_observed_market_price",
        },
    }


def profile_quality_summary(
    load_profile: list[float],
    pv_profile: list[float],
    price_profile: list[float],
    *,
    dt_hours: float = 0.25,
) -> pd.DataFrame:
    """Summarize daily diversity so repeated-day profiles are visible."""

    steps_per_day = max(1, int(round(24.0 / dt_hours)))
    rows: list[dict[str, float | int]] = []
    load = np.asarray(load_profile, dtype=float)
    pv = np.asarray(pv_profile, dtype=float)
    price = np.asarray(price_profile, dtype=float)
    n_days = int(np.ceil(len(load) / steps_per_day)) if len(load) else 0
    for day_idx in range(n_days):
        start = day_idx * steps_per_day
        end = min((day_idx + 1) * steps_per_day, len(load))
        if start >= end:
            continue
        rows.append(
            {
                "day": int(day_idx),
                "load_mean": float(load[start:end].mean()),
                "load_peak": float(load[start:end].max()),
                "pv_energy_pu_h": float(pv[start:end].sum() * dt_hours),
                "price_mean": float(price[start:end].mean()),
                "price_peak": float(price[start:end].max()),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) >= 2:
        frame["load_mean_delta_from_day0"] = frame["load_mean"] - float(frame.iloc[0]["load_mean"])
        frame["pv_energy_delta_from_day0"] = frame["pv_energy_pu_h"] - float(frame.iloc[0]["pv_energy_pu_h"])
    return frame
