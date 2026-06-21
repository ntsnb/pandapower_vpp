from __future__ import annotations

import pytest

from vpp_dso_sim.learning.deep_rl import _target_from_normalized_scalar


def test_dispatch_target_decoder_intersects_preferred_and_hard_bounds() -> None:
    obs = {
        "aggregate_bounds": {"p_min_mw": -0.20, "p_max_mw": 0.30},
        "operating_envelope": {
            "preferred_p_min_mw": -1.00,
            "preferred_p_max_mw": 1.00,
        },
    }

    assert _target_from_normalized_scalar(-1.0, obs, 1.0) == pytest.approx(-0.20)
    assert _target_from_normalized_scalar(1.0, obs, 1.0) == pytest.approx(0.30)
    assert -0.20 <= _target_from_normalized_scalar(0.25, obs, 1.0) <= 0.30


def test_dispatch_target_decoder_uses_preferred_interval_inside_hard_bounds() -> None:
    obs = {
        "aggregate_bounds": {"p_min_mw": -1.00, "p_max_mw": 1.00},
        "operating_envelope": {
            "preferred_p_min_mw": -0.25,
            "preferred_p_max_mw": 0.50,
        },
    }

    assert _target_from_normalized_scalar(-1.0, obs, 1.0) == pytest.approx(-0.25)
    assert _target_from_normalized_scalar(1.0, obs, 1.0) == pytest.approx(0.50)
