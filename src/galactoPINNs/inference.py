"""Inference utilities for trained models."""

__all__ = ("apply_model",)

from collections.abc import Mapping
from typing import Any, Protocol

from flax import nnx
from jaxtyping import Array


class _NNXModelLike(Protocol):
    """Protocol for NNX modules that can be called directly."""

    def __call__(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]: ...


def apply_model(
    model: nnx.Module | _NNXModelLike,
    x_scaled: Array,
) -> dict[str, Any]:
    """Apply a model to scaled Cartesian inputs and return standardized outputs.

    Works for both static and time-dependent models.

    Parameters
    ----------
    model
        NNX model that can be called directly. The call must
        return a mapping containing (at minimum) keys ``"potential"`` and
        ``"acceleration"``.
    x_scaled
        Scaled Cartesian inputs. Expected shape ``(N, 3)`` for batch evaluation
        or ``(3,)`` for a single point.

    Returns
    -------
    out
        Dictionary with standardized keys:
        - ``"u_pred"``: scaled predicted potential (shape typically ``(N,)``
          or ``(1,)``)
        - ``"a_pred"``: scaled predicted acceleration (shape typically
          ``(N, 3)`` or ``(3,)``)
        - ``"outputs"``: auxiliary output payload if present under
          ``predictions["outputs"]``, otherwise ``None``.

    """
    predictions = model(x_scaled)

    # Required outputs
    return {
        "u_pred": predictions["potential"],
        "a_pred": predictions["acceleration"],
        "outputs": predictions.get("outputs", None),
    }
