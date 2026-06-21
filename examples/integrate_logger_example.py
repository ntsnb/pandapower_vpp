from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marl_dashboard.logging import ExperimentLogger, start_dashboard


DATASET_UNITS = {
    "electricity_price": "currency/MWh",
    "ev_charging_load": "kW",
    "storage_power": "kW",
    "storage_soc": "%",
    "pv_power": "kW",
    "wind_power": "kW",
    "base_load": "kW",
    "net_load": "kW",
}

FORMULAS = {
    "net_load": "P^{net}_{i,t}=P^{base}_{i,t}+P^{EV}_{i,t}-P^{PV}_{i,t}-P^{wind}_{i,t}",
    "profit_reward": "r^{profit}_{i,t}=revenue_{i,t}-cost_{i,t}",
    "grid_balance_reward": "r^{balance}_{i,t}=-|P^{net}_{i,t}-P^{schedule}_{i,t}|",
    "constraint_violation_penalty": "r^{violation}_{i,t}=-\\lambda_v N^{violation}_{i,t}",
    "total_reward": "R_{i,t}=\\sum_k r^k_{i,t}",
    "energy_purchase_cost": "C^{buy}_{i,t}=p_t E^{buy}_{i,t}",
    "storage_degradation_cost": "C^{deg}_{i,t}=c^{deg}|P^{storage}_{i,t}|",
    "total_cost": "C_{i,t}=\\sum_k C^k_{i,t}",
    "actor_loss": "\\mathcal{L}_{actor}",
    "critic_loss": "\\mathcal{L}_{critic}",
    "total_loss": "\\mathcal{L}=\\mathcal{L}_{actor}+\\mathcal{L}_{critic}",
}


def _variable_dictionary() -> list[dict[str, Any]]:
    variables = []
    for name, unit in DATASET_UNITS.items():
        variables.append(
            {
                "name": name,
                "display_name": name.replace("_", " ").title(),
                "symbol": name,
                "unit": unit,
                "group": "dataset",
                "physical_meaning": "Example physical dataset signal; replace with de-normalized training values.",
                "formula_latex": FORMULAS.get(name),
                "source": "examples/integrate_logger_example.py",
                "notes": "Integration example only.",
            }
        )
    for name in ("profit_reward", "grid_balance_reward", "constraint_violation_penalty", "total_reward"):
        variables.append(
            {
                "name": name,
                "display_name": name.replace("_", " ").title(),
                "symbol": name,
                "unit": "score",
                "group": "reward",
                "physical_meaning": "Example reward term; replace with the exact project reward formula.",
                "formula_latex": FORMULAS.get(name),
                "source": "training reward hook",
                "notes": "Confirm sign convention before publication.",
            }
        )
    return variables


def _dataset_values(time_index: int) -> dict[str, float]:
    pv_power = max(0.0, 120.0 - abs(time_index - 1) * 30.0)
    wind_power = 45.0 + 3.0 * time_index
    base_load = 240.0 + 20.0 * time_index
    ev_charging_load = 55.0 + 8.0 * time_index
    storage_power = -18.0 if time_index == 0 else 22.0
    return {
        "electricity_price": 72.0 + 4.0 * time_index,
        "ev_charging_load": ev_charging_load,
        "storage_power": storage_power,
        "storage_soc": 58.0 - 2.0 * time_index,
        "pv_power": pv_power,
        "wind_power": wind_power,
        "base_load": base_load,
        "net_load": base_load + ev_charging_load - pv_power - wind_power,
    }


def write_example_run(data_dir: Path, run_id: str) -> None:
    logger = ExperimentLogger(
        run_id=run_id,
        data_dir=str(data_dir),
        config={
            "algorithm": "integration_example",
            "environment": "minimal_vpp_loop",
            "seed": 7,
            "episode_horizon_steps": 2,
            "dashboard_integration_mode": "adapter_logger_hook",
        },
        variable_dictionary=_variable_dictionary(),
        formulas=FORMULAS,
        metadata={"source": "examples/integrate_logger_example.py", "demo": True},
        async_writer=False,
    )
    try:
        logger.log_event("train_start", {"message": "integration example started"}, epoch_id=0, episode_id=1)
        for time_index in range(2):
            context = {
                "epoch_id": 0,
                "episode_id": 1,
                "env_id": "env_0",
                "vpp_id": "vpp_example",
                "agent_id": "vpp_example_dispatch",
                "policy_id": "shared_policy",
                "date": "2026-01-01",
                "time_index": time_index,
                "timestamp": f"2026-01-01T{time_index:02d}:00:00Z",
                "global_env_step": time_index + 1,
            }
            values = _dataset_values(time_index)
            logger.log_dataset(values=values, units=DATASET_UNITS, formulas=FORMULAS, **context)
            reward_terms = {
                "profit_reward": 4.0 + time_index,
                "grid_balance_reward": -0.2 * time_index,
                "constraint_violation_penalty": 0.0,
                "total_reward": 3.8 + 0.8 * time_index,
            }
            cost_terms = {
                "energy_purchase_cost": 12.5 + time_index,
                "storage_degradation_cost": abs(values["storage_power"]) * 0.02,
                "constraint_violation_cost": 0.0,
                "total_cost": 12.5 + time_index + abs(values["storage_power"]) * 0.02,
            }
            logger.log_reward_terms(terms=reward_terms, units={name: "score" for name in reward_terms}, formulas=FORMULAS, **context)
            logger.log_cost_terms(terms=cost_terms, units={name: "cost" for name in cost_terms}, formulas=FORMULAS, **context)
        logger.log_loss_terms(
            epoch_id=0,
            episode_id=1,
            gradient_step=1,
            global_env_step=2,
            vpp_id="aggregate",
            agent_id="aggregate",
            policy_id="shared_policy",
            optimizer_name="adam",
            network_name="actor_critic",
            terms={"actor_loss": 0.12, "critic_loss": 0.34, "total_loss": 0.46},
            formulas=FORMULAS,
        )
        logger.log_scalar("episode_return", 8.4, epoch_id=0, episode_id=1, global_env_step=2, vpp_id="vpp_example")
        logger.log_scalar("episode_length", 2, epoch_id=0, episode_id=1, global_env_step=2, vpp_id="vpp_example")
        logger.log_event("episode_end", {"message": "integration example episode finished"}, epoch_id=0, episode_id=1)
    except Exception:
        logger.close(status="error")
        raise
    else:
        logger.close(status="finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal MARL dashboard logger integration example.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--run-id", default="integration_example_run")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--auto-port", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only write example logs; do not start the dashboard service.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    write_example_run(data_dir, args.run_id)
    print(f"run_id={args.run_id}")
    print(f"data_dir={data_dir}")
    print(f"dry_run={str(bool(args.dry_run)).lower()}")
    if args.dry_run:
        return
    dashboard = start_dashboard(data_dir=data_dir, host=args.host, port=args.port, auto_port=args.auto_port)
    print("Press Ctrl+C to stop the example dashboard.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        dashboard.stop()


if __name__ == "__main__":
    main()
