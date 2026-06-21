from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vpp_dso_sim.dso.envelope.safe_decoder import decode_operating_envelope
from vpp_dso_sim.dso.models.bipartite_attention_actor import BipartiteSensitivityDSOActor
from vpp_dso_sim.dso.observation.structured_bipartite import encode_dso_observation_structured
from vpp_dso_sim.dso.sensitivity.finite_difference import compute_finite_difference_sensitivity_tensor
from vpp_dso_sim.dso.sensitivity.selectors import build_action_units, select_critical_network_objects
from vpp_dso_sim.network.powerflow import run_powerflow
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator
from vpp_dso_sim.utils.io import ensure_dir, write_json


def config_hash(config_path: str | Path) -> str:
    data = Path(config_path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:12]


def _write_structured_envelope_artifacts(records: list[dict[str, Any]], out: Path) -> dict[str, str]:
    selected_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    actor_rows: list[dict[str, Any]] = []
    decoded_rows: list[dict[str, Any]] = []
    for row in records:
        step = int(row.get("step", -1))
        vpp_id = str(row.get("vpp_id", ""))
        for object_index, object_id in enumerate(row.get("selected_network_objects", []) or []):
            selected_rows.append(
                {
                    "step": step,
                    "vpp_id": vpp_id,
                    "object_index": int(object_index),
                    "network_object_id": str(object_id),
                }
            )
        for action_index, action_unit_id in enumerate(row.get("action_units", []) or []):
            action_rows.append(
                {
                    "step": step,
                    "vpp_id": vpp_id,
                    "action_index": int(action_index),
                    "action_unit_id": str(action_unit_id),
                }
            )
        shape = row.get("active_sensitivity_edges_shape", ())
        if isinstance(shape, str):
            shape_text = shape
        else:
            shape_text = "x".join(str(int(value)) for value in shape) if shape else ""
        sensitivity_rows.append(
            {
                "step": step,
                "vpp_id": vpp_id,
                "active_sensitivity_edges_shape": shape_text,
                "sensitivity_confidence": float(row.get("sensitivity_confidence", row.get("confidence", 0.0)) or 0.0),
                "action_unit_count": len(row.get("action_units", []) or []),
                "network_object_count": len(row.get("selected_network_objects", []) or []),
            }
        )
        raw_outputs = row.get("dso_actor_raw_outputs", {}) or {}
        centers = list(raw_outputs.get("center_ratio", []) or [])
        widths = list(raw_outputs.get("width_ratio", []) or [])
        strengths = list(raw_outputs.get("guidance_strength", []) or [])
        directions = list(row.get("direction_probs", []) or [])
        for action_index, center in enumerate(centers):
            direction = directions[action_index] if action_index < len(directions) else [0.0, 0.0, 0.0]
            actor_rows.append(
                {
                    "step": step,
                    "vpp_id": vpp_id,
                    "action_index": int(action_index),
                    "center_ratio": float(center),
                    "width_ratio": float(widths[action_index]) if action_index < len(widths) else 0.0,
                    "guidance_strength_lambda": float(strengths[action_index]) if action_index < len(strengths) else 0.0,
                    "direction_absorb": float(direction[0]) if len(direction) > 0 else 0.0,
                    "direction_balanced": float(direction[1]) if len(direction) > 1 else 0.0,
                    "direction_inject": float(direction[2]) if len(direction) > 2 else 0.0,
                }
            )
        for decoded in row.get("decoded_operating_envelope", []) or []:
            decoded_rows.append({"step": step, "parent_vpp_id": vpp_id, **dict(decoded)})

    paths = {
        "selected_network_objects": str(out / "selected_network_objects.csv"),
        "action_units": str(out / "action_units.csv"),
        "sensitivity_edges": str(out / "sensitivity_edges.csv"),
        "dso_actor_outputs": str(out / "dso_actor_outputs.csv"),
        "decoded_operating_envelope": str(out / "decoded_operating_envelope.csv"),
    }
    pd.DataFrame(selected_rows).to_csv(paths["selected_network_objects"], index=False)
    pd.DataFrame(action_rows).to_csv(paths["action_units"], index=False)
    pd.DataFrame(sensitivity_rows).to_csv(paths["sensitivity_edges"], index=False)
    pd.DataFrame(actor_rows).to_csv(paths["dso_actor_outputs"], index=False)
    pd.DataFrame(decoded_rows).to_csv(paths["decoded_operating_envelope"], index=False)
    return paths


def run_smoke_rollout(
    *,
    config_path: str | Path,
    seed: int = 0,
    steps: int = 2,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    scenario = load_scenario(config_path)
    scenario.config.setdefault("simulation", {})["seed"] = int(seed)
    if output_dir is None:
        output_dir = Path("outputs") / f"dso_sensitivity_smoke_{Path(config_path).stem}_seed{seed}"
    out = ensure_dir(output_dir)
    simulator = Simulator(scenario)
    rows: list[dict[str, Any]] = []
    for step in range(max(1, int(steps))):
        result = simulator.step(step)
        rows.append(
            {
                "step": int(step),
                "converged": bool(result.get("converged", False)),
                "total_cost": float(result.get("reward_components", {}).get("total_cost", 0.0)),
                "total_reward": float(result.get("reward_components", {}).get("reward", 0.0)),
            }
        )
    pd.DataFrame(rows).to_csv(out / "smoke_step_metrics.csv", index=False)
    envelope_records = simulator.records.get("dso_operating_envelope", [])
    pd.DataFrame(envelope_records).to_csv(
        out / "dso_operating_envelope.csv",
        index=False,
    )
    structured_artifacts = _write_structured_envelope_artifacts(envelope_records, out)
    summary = {
        "config": str(config_path),
        "config_hash": config_hash(config_path),
        "seed": int(seed),
        "steps": int(steps),
        "output_dir": str(out),
        "envelope_policy": str(scenario.config.get("dso", {}).get("envelope_policy", "rule_v0")),
        "records": {key: len(value) for key, value in simulator.records.items()},
        "nan_or_inf_detected": bool(pd.DataFrame(rows).select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any()),
        "structured_artifacts": structured_artifacts,
    }
    write_json(out / "smoke_summary.json", summary)
    return summary


def _rule_targets_for_units(rule_envelope: dict[str, Any], units) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    agg_width = max(1e-9, float(rule_envelope["p_max_mw"]) - float(rule_envelope["p_min_mw"]))
    center = (float(rule_envelope["preferred_target_p_mw"]) - float(rule_envelope["p_min_mw"])) / agg_width
    width = (float(rule_envelope["preferred_p_max_mw"]) - float(rule_envelope["preferred_p_min_mw"])) / agg_width
    service = str(rule_envelope.get("service_request", "")).lower()
    direction = 1
    if "absorb" in service or "charge" in service:
        direction = 0
    elif "export" in service or "reduce" in service:
        direction = 2
    return (
        np.full((len(units),), np.clip(center, 0.0, 1.0), dtype=np.float32),
        np.full((len(units),), np.clip(width, 0.0, 1.0), dtype=np.float32),
        np.full((len(units),), int(direction), dtype=np.int64),
    )


def run_short_training_sanity(
    *,
    config_path: str | Path,
    seed: int = 0,
    steps: int = 16,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(int(seed))
    scenario = load_scenario(config_path)
    scenario.config.setdefault("simulation", {})["seed"] = int(seed)
    if output_dir is None:
        output_dir = Path("outputs") / f"dso_sensitivity_short_train_{Path(config_path).stem}_seed{seed}"
    out = ensure_dir(output_dir)
    simulator = Simulator(scenario)
    assert run_powerflow(scenario.net), "base power flow must converge for short training sanity"
    vpp = scenario.vpps[0]
    step = 0
    price = simulator._profile_value(scenario.price_profile, step)
    bid = vpp.day_ahead_bid(step, price_hint=price)
    fr = compute_static_feasible_region(vpp, step, scope="bus_vector")
    rule_envelope = simulator._build_dso_operating_envelope(
        vpp,
        step,
        bid,
        fr,
        price,
        grid_state={**scenario.dso.compute_network_state(), "pre_dispatch_powerflow_converged": True},
    )
    units = build_action_units(vpp, fr, t=step, granularity="vpp_bus")
    objects = select_critical_network_objects(
        scenario.net,
        voltage_limits=scenario.dso.voltage_limits,
        line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
        trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
        topk_low_voltage_buses=2,
        topk_high_voltage_buses=2,
        topk_lines=2,
        topk_trafos=1,
    )
    edges = compute_finite_difference_sensitivity_tensor(scenario.net, units, objects)
    observation = encode_dso_observation_structured(
        step=step,
        dt_hours=scenario.dt_hours,
        voltage_limits=scenario.dso.voltage_limits,
        line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
        trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
        action_units=units,
        network_objects=objects,
        sensitivity_edges=edges,
        max_action_units=max(1, len(units)),
        max_network_objects=max(1, len(objects)),
    )
    actor_cfg = dict(scenario.config.get("dso", {}).get("actor", {}))
    actor = BipartiteSensitivityDSOActor(
        global_feature_dim=int(observation.global_features.shape[-1]),
        action_token_dim=int(observation.action_tokens.shape[-1]),
        object_token_dim=int(observation.object_tokens.shape[-1]),
        edge_feature_dim=int(observation.sensitivity_edges.shape[-1]),
        d_model=int(actor_cfg.get("d_model", 64)),
        num_heads=int(actor_cfg.get("num_heads", 4)),
        num_layers=int(actor_cfg.get("num_layers", 1)),
        action_self_attention_layers=int(actor_cfg.get("action_self_attention_layers", 1)),
        dropout=float(actor_cfg.get("dropout", 0.0)),
        min_width_ratio=float(actor_cfg.get("min_width_ratio", 0.10)),
        max_width_ratio=float(actor_cfg.get("max_width_ratio", 1.00)),
    )
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    target_center, target_width, target_direction = _rule_targets_for_units(rule_envelope, units)
    tensors = {
        "global_features": torch.tensor(observation.global_features, dtype=torch.float32).unsqueeze(0),
        "action_tokens": torch.tensor(observation.action_tokens, dtype=torch.float32).unsqueeze(0),
        "object_tokens": torch.tensor(observation.object_tokens, dtype=torch.float32).unsqueeze(0),
        "sensitivity_edges": torch.tensor(observation.sensitivity_edges, dtype=torch.float32).unsqueeze(0),
        "action_mask": torch.tensor(observation.action_mask, dtype=torch.bool).unsqueeze(0),
        "object_mask": torch.tensor(observation.object_mask, dtype=torch.bool).unsqueeze(0),
        "edge_mask": torch.tensor(observation.edge_mask, dtype=torch.bool).unsqueeze(0),
    }
    target_center_t = torch.tensor(target_center, dtype=torch.float32).unsqueeze(0)
    target_width_t = torch.tensor(target_width, dtype=torch.float32).unsqueeze(0)
    target_direction_t = torch.tensor(target_direction, dtype=torch.long)
    real_actions = len(units)
    loss_rows: list[dict[str, float | int]] = []
    nan_or_inf = False
    for train_step in range(max(1, int(steps))):
        outputs = actor(**tensors)
        center_loss = F.mse_loss(outputs["center_ratio"][:, :real_actions], target_center_t)
        width_loss = F.mse_loss(outputs["width_ratio"][:, :real_actions], target_width_t)
        direction_loss = F.nll_loss(
            torch.log(outputs["direction_probs"][:, :real_actions, :].reshape(real_actions, 3).clamp_min(1e-8)),
            target_direction_t,
        )
        loss = center_loss + width_loss + 0.25 * direction_loss
        optimizer.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5).detach().cpu().item())
        optimizer.step()
        values = [float(loss.detach()), float(center_loss.detach()), float(width_loss.detach()), float(direction_loss.detach()), grad_norm]
        if not all(np.isfinite(values)):
            nan_or_inf = True
            break
        loss_rows.append(
            {
                "update_step": int(train_step),
                "bc_loss": values[0],
                "center_loss": values[1],
                "width_loss": values[2],
                "direction_loss": values[3],
                "grad_norm": values[4],
            }
        )
    pd.DataFrame(loss_rows).to_csv(out / "dso_sensitivity_attention_short_train_loss_metrics.csv", index=False)
    torch.save(actor.state_dict(), out / "dso_sensitivity_attention_actor.pt")
    with torch.no_grad():
        final_outputs = actor(**tensors)
    decoded = decode_operating_envelope(
        action_unit_ids=[unit.id.action_unit_id for unit in units],
        vpp_ids=[unit.id.vpp_id for unit in units],
        pcc_ids=[unit.id.pcc_id for unit in units],
        bus_ids=[unit.id.bus_id for unit in units],
        p_hard_min_mw=[unit.p_min_mw for unit in units],
        p_hard_max_mw=[unit.p_max_mw for unit in units],
        center_ratio=final_outputs["center_ratio"].squeeze(0).detach().cpu().numpy()[:real_actions],
        width_ratio=final_outputs["width_ratio"].squeeze(0).detach().cpu().numpy()[:real_actions],
        direction_probs=final_outputs["direction_probs"].squeeze(0).detach().cpu().numpy()[:real_actions],
        guidance_strength=final_outputs["guidance_strength"].squeeze(0).detach().cpu().numpy()[:real_actions],
    )
    pd.DataFrame([record.to_dict() for record in decoded]).to_csv(out / "decoded_operating_envelope.csv", index=False)
    summary = {
        "config": str(config_path),
        "config_hash": config_hash(config_path),
        "seed": int(seed),
        "steps_requested": int(steps),
        "steps_completed": len(loss_rows),
        "output_dir": str(out),
        "loss_metrics": str(out / "dso_sensitivity_attention_short_train_loss_metrics.csv"),
        "checkpoint": str(out / "dso_sensitivity_attention_actor.pt"),
        "nan_or_inf_detected": bool(nan_or_inf),
        "initial_loss": float(loss_rows[0]["bc_loss"]) if loss_rows else None,
        "final_loss": float(loss_rows[-1]["bc_loss"]) if loss_rows else None,
    }
    write_json(out / "short_train_summary.json", summary)
    return summary
