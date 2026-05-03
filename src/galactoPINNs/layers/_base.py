"""Base types and utilities for neural network layers."""

from typing import Protocol

from flax import nnx
from jaxtyping import Array


class ExternalPytree(nnx.Variable):
    """Variable wrapper for external pytrees (like equinox modules)."""


class ActivationFn(Protocol):
    """Callable activation function interface."""

    def __call__(self, x: Array, /) -> Array:
        """Apply the activation function to the input array."""
        ...
