from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int | None) -> np.random.Generator:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    return np.random.default_rng(seed)

