from __future__ import annotations


class NormalizeObservationWrapper:
    # TODO(v0.2): add running-stat normalization for RL training once observation
    # ranges are finalized across randomized scenarios.
    def __init__(self, env):
        self.env = env

