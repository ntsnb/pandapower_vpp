from __future__ import annotations

import pytest

from vpp_dso_sim.learning.deep_rl import torch_available


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_set_attention_dispatch_actor_preserves_output_contract():
    import torch

    from vpp_dso_sim.learning.ctde_networks import (
        VPP_DISPATCH_CONTEXT_DIM,
        VPP_DISPATCH_TOKEN_DIM,
        build_privacy_separated_ctde_modules,
    )

    max_der = 4
    modules, metadata = build_privacy_separated_ctde_modules(
        torch=torch,
        nn=torch.nn,
        dso_input_dim=3,
        vpp_input_dim=VPP_DISPATCH_CONTEXT_DIM + max_der * VPP_DISPATCH_TOKEN_DIM,
        portfolio_input_dim=5,
        critic_input_dim=7,
        action_summary_dim=2,
        action_dim=2,
        der_action_dim=max_der,
        hidden_dim=32,
        dispatch_actor_encoder_type="set_attention_v1",
    )

    actor = modules["vpp_dispatch_actor"]
    x = torch.zeros(2, VPP_DISPATCH_CONTEXT_DIM + max_der * VPP_DISPATCH_TOKEN_DIM)
    x[:, :VPP_DISPATCH_CONTEXT_DIM] = torch.linspace(0.0, 1.0, VPP_DISPATCH_CONTEXT_DIM)
    x[:, VPP_DISPATCH_CONTEXT_DIM : VPP_DISPATCH_CONTEXT_DIM + 2 * VPP_DISPATCH_TOKEN_DIM] = 0.25

    aggregate_mean, aggregate_log_std, der_mean, der_log_std = actor(x)

    assert metadata["vpp_encoder_type"] == "set_attention_v1_masked_self_attention"
    assert metadata["architecture_version"] == "ctde_v3_set_attention_action_conditioned"
    assert aggregate_mean.shape == (2, 1)
    assert aggregate_log_std.shape == (2, 1)
    assert der_mean.shape == (2, max_der)
    assert der_log_std.shape == (2, max_der)
    assert torch.isfinite(aggregate_mean).all()
    assert torch.isfinite(der_mean).all()
    assert float(aggregate_mean.abs().max()) <= 1.0
    assert float(der_mean.abs().max()) <= 1.0


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_legacy_deepset_dispatch_actor_remains_default():
    import torch

    from vpp_dso_sim.learning.ctde_networks import (
        VPP_DISPATCH_CONTEXT_DIM,
        VPP_DISPATCH_TOKEN_DIM,
        build_privacy_separated_ctde_modules,
    )

    _, metadata = build_privacy_separated_ctde_modules(
        torch=torch,
        nn=torch.nn,
        dso_input_dim=3,
        vpp_input_dim=VPP_DISPATCH_CONTEXT_DIM + 2 * VPP_DISPATCH_TOKEN_DIM,
        portfolio_input_dim=5,
        critic_input_dim=7,
        action_summary_dim=2,
        action_dim=2,
        der_action_dim=2,
        hidden_dim=16,
    )

    assert metadata["vpp_encoder_type"] == "deep_sets_shared_token_mlp"
    assert metadata["architecture_version"] == "ctde_v2_deepsets_action_conditioned"
