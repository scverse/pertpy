"""Helpers for the optional JAX backend."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_JAX_STACK = frozenset({"jax", "jaxlib", "numpyro", "ott", "flax", "optax"})


@contextmanager
def jax_import(feature: str) -> Iterator[None]:
    """Turn a missing JAX dependency into an install hint naming ``feature``.

    Import errors from anything outside the JAX stack propagate unchanged.
    """
    try:
        yield
    except ImportError as error:
        root = (error.name or "").split(".")[0]
        if root not in _JAX_STACK:
            raise
        raise ImportError(
            f"{feature} requires pertpy's optional JAX backend, but {root!r} is not installed. "
            "Install it with: pip install 'pertpy[jax]'"
        ) from error
