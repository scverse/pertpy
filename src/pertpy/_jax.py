"""Helpers for the optional JAX backend.

The JAX stack is an optional extra so that a plain ``pip install pertpy`` stays free of a deep
learning runtime.
Tools that need it import it lazily and wrap the import in :func:`jax_import` so that a missing
dependency surfaces as an actionable message instead of a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_JAX_STACK = frozenset({"jax", "jaxlib", "numpyro", "ott", "flax", "optax"})


@contextmanager
def jax_import(feature: str) -> Iterator[None]:
    """Translate a missing JAX dependency into an actionable install hint.

    Args:
        feature: User facing name of the tool or method that needs the JAX backend.

    Raises:
        ImportError: If a module of the JAX stack is missing.
            Unrelated import errors are re-raised unchanged.
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
