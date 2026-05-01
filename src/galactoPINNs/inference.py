"""Inference utilities for trained models."""

__all__ = ("apply_model",)

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from jaxtyping import Array

Params: TypeAlias = Any


def apply_model(
    model_or_params: Callable[..., Mapping[str, Any]] | Params,
    x_or_apply_fn: Array | Callable[..., Mapping[str, Any]],
    x_scaled: Array | None = None,
) -> dict[str, Any]:
    """Apply a model to scaled inputs and return standardized outputs.

    This function works in two modes depending on the arguments provided:
    1.  **Object Mode**: `apply_model(model, x_scaled, ...)`
        Accepts a "live" NNX module instance.
    2.  **Functional Mode**: `apply_model(params, apply_fn, x_scaled, ...)`
        Accepts the model's parameters and its functional apply method.

    Parameters
    ----------
    model_or_params
        In object mode, the live NNX model.
        In functional mode, the model's parameters pytree.
    x_or_apply_fn
        In object mode, the scaled input array `x_scaled`.
        In functional mode, the model's functional `apply` method.
    x_scaled
        In functional mode, the scaled input array. Must be provided.

    Returns
    -------
    out
        Dictionary with standardized keys: "u_pred", "a_pred", "outputs".

    """
    if callable(x_or_apply_fn):  # Functional Mode
        params = model_or_params
        apply_fn = x_or_apply_fn
        x = x_scaled
        if x is None:
            raise ValueError(
                "In functional mode, `x_scaled` must be provided as the third argument."
            )
        predictions = apply_fn(params, x)

    else:  # Object Mode
        model = model_or_params
        x = x_or_apply_fn
        if x_scaled is not None:
            raise ValueError(
                "In object mode, do not provide the third "
                "positional argument `x_scaled`."
            )
            raise ValueError(msg)
        predictions = model(x)

    return {
        "u_pred": predictions["potential"],
        "a_pred": predictions["acceleration"],
        "outputs": predictions.get("outputs", None),
    }
