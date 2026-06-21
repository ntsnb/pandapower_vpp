from __future__ import annotations

import os


NUMERIC_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def configure_numeric_thread_limits(*, default_threads: int = 8) -> dict[str, str]:
    """Set conservative numeric-library thread defaults for server experiments."""

    thread_count = str(int(default_threads))
    configured: dict[str, str] = {}
    for name in NUMERIC_THREAD_ENV_VARS:
        os.environ.setdefault(name, thread_count)
        configured[name] = str(os.environ[name])
    return configured
