"""Helpers for distributing per-variable computations across jobs.

`_parallelize_with_joblib` is adapted from scirpy (BSD 3-clause).
"""

import math
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
from joblib import Parallel
from tqdm.auto import tqdm

from pertpy._logger import logger

# Upper bound for densified blocks of variables, ~64 MB in float64.
_MAX_BLOCK_ELEMENTS = 8_000_000


def _parallelize_with_joblib(delayed_objects: Iterable[Any], *, total: int | None = None, **kwargs) -> Iterator[Any]:
    """Run `delayed_objects` through `joblib.Parallel`, with a progress bar if the backend supports it."""
    try:
        return iter(tqdm(Parallel(return_as="generator", **kwargs)(delayed_objects), total=total))
    except ValueError:
        logger.info(
            "Backend doesn't support return_as='generator'. No progress bar will be shown. "
            "Consider setting verbosity in joblib.parallel_config"
        )
        return iter(Parallel(return_as="list", **kwargs)(delayed_objects))


def _block_slices(n_items: int, *, n_blocks: int = 1, max_block_size: int | None = None) -> list[slice]:
    """Split `range(n_items)` into contiguous blocks, of `max_block_size` items at most."""
    if n_items <= 0:
        return []
    block_size = math.ceil(n_items / max(1, min(n_blocks, n_items)))
    if max_block_size is not None:
        block_size = min(block_size, max(1, max_block_size))
    return [slice(start, min(start + block_size, n_items)) for start in range(0, n_items, block_size)]


def _spawn_rngs(rng, n: int) -> list[np.random.Generator | None]:
    """Derive `n` independent generators from a seed, generator or None, deterministically."""
    if rng is None:
        return [None] * n
    if isinstance(rng, np.random.BitGenerator):
        rng = np.random.default_rng(rng)
    if isinstance(rng, np.random.Generator):
        return list(rng.spawn(n))
    if isinstance(rng, np.random.SeedSequence):
        return [np.random.default_rng(seed) for seed in rng.spawn(n)]
    if isinstance(rng, np.random.RandomState):
        return [np.random.default_rng(int(seed)) for seed in rng.randint(0, 2**31 - 1, size=n)]
    return [np.random.default_rng(seed) for seed in np.random.SeedSequence(rng).spawn(n)]
