from __future__ import annotations

import importlib.util

import pytest

from vpp_dso_sim.dso.sensitivity.finite_difference import SENSITIVITY_CHANNELS


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_bipartite_attention_actor_handles_variable_action_and_object_masks() -> None:
    import torch

    from vpp_dso_sim.dso.models.bipartite_attention_actor import BipartiteSensitivityDSOActor

    model = BipartiteSensitivityDSOActor(
        global_feature_dim=6,
        action_token_dim=13,
        object_token_dim=10,
        edge_feature_dim=len(SENSITIVITY_CHANNELS),
        d_model=32,
        num_heads=4,
        num_layers=1,
        action_self_attention_layers=1,
        dropout=0.0,
        min_width_ratio=0.10,
        max_width_ratio=1.0,
    )
    batch = 2
    max_actions = 4
    max_objects = 5
    outputs = model(
        global_features=torch.zeros(batch, 6),
        action_tokens=torch.randn(batch, max_actions, 13),
        object_tokens=torch.randn(batch, max_objects, 10),
        sensitivity_edges=torch.randn(batch, max_objects, max_actions, len(SENSITIVITY_CHANNELS)),
        action_mask=torch.tensor([[True, True, False, False], [True, True, True, False]]),
        object_mask=torch.tensor([[True, True, True, False, False], [True, False, False, False, False]]),
        edge_mask=torch.ones(batch, max_objects, max_actions, dtype=torch.bool),
    )

    assert outputs["center_ratio"].shape == (batch, max_actions)
    assert outputs["width_ratio"].shape == (batch, max_actions)
    assert outputs["direction_probs"].shape == (batch, max_actions, 3)
    assert outputs["guidance_strength"].shape == (batch, max_actions)
    assert torch.isfinite(outputs["center_ratio"]).all()
    assert torch.all((outputs["width_ratio"] >= 0.10) & (outputs["width_ratio"] <= 1.0))
    assert torch.allclose(outputs["direction_probs"].sum(dim=-1), torch.ones(batch, max_actions), atol=1e-5)
