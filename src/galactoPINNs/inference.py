"""Inference utilities for trained models."""

__all__ = ("apply_model",)

from collections.abc import Mapping
from typing import Any, Protocol, TypedDict

from jaxtyping import Array

Params = Any


##### Protocols and types #####
class _ModelLike(Protocol):
    """Protocol for Flax-like modules exposing an `.apply(...)` method."""

    def apply(
        self, variables: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> Mapping[str, Any]: ...


class ApplyModelOutput(TypedDict, total=False):
    """Standardized return schema for model application helpers."""

    u_pred: Array
    a_pred: Array
    outputs: Any


##########


def apply_model(model: _ModelLike, params: Params, x_scaled: Array) -> dict[str, Any]:
    """Apply a model to scaled Cartesian inputs and return standardized outputs.

    Works for both static and time-dependent models.

    Parameters
    ----------
    model
        Flax-like model exposing ``model.apply(...)``. The apply call must
        return a mapping containing (at minimum) keys ``"potential"`` and
        ``"acceleration"``.
    params
        Parameter pytree passed as ``{"params": params}`` into ``model.apply``.
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
    predictions = model.apply({"params": params}, x_scaled)

    # Required outputs
    u_pred = predictions["potential"]
    a_pred = predictions["acceleration"]
    outputs = predictions.get("outputs", None)

    return {
        "u_pred": u_pred,
        "a_pred": a_pred,
        "outputs": outputs,
    }
