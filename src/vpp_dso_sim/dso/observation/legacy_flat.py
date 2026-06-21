from __future__ import annotations

from typing import Any

import numpy as np

from vpp_dso_sim.learning.deep_rl import encode_dso_observation


def encode_dso_observation_legacy(
    obs: dict[str, Any],
    vpp_ids: list[str],
    max_vpps: int | None = None,
) -> np.ndarray:
    """Preserve the existing flat DSO observation encoder.

    For the current 3-VPP paper setup this remains the legacy 26-dimensional input.
    """

    return encode_dso_observation(obs, vpp_ids, max_vpps=max_vpps)
