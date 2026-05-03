"""Scale functions and scaling layer for neural network potentials."""

import functools as ft
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

from ._base import ExternalPytree

# Scale functions
# implementations module-level, bound via ft.partial
# Shared final signature after partial application: (x_cart, r, r_s, t) -> Array


def _scale_one(x_cart: Array, r: Array, r_s: float, t: Any) -> Array:  # noqa: ARG001
    return jnp.ones_like(r)


def _scale_power(x_cart: Array, r: Array, r_s: float, t: Any, *, power: float) -> Array:  # noqa: ARG001
    return jnp.power(1.0 / r, power)


def _scale_nfw(
    x_cart: Array,  # noqa: ARG001
    r: Array,
    r_s: float,
    t: Any,  # noqa: ARG001
    *,
    x_transformer: Any,
) -> Array:
    r_s_scaled = x_transformer.transform(r_s)
    return jnp.log(1.0 + r / r_s_scaled) / r


def _scale_precomputed(*_: Any, precomputed: Array) -> Array:
    return precomputed


def _scale_external(
    x_cart: Array,
    r: Array,
    r_s: float,  # noqa: ARG001
    t: Any,
    *,
    external_scale: ExternalPytree,
    r_ref: float | None,
    eps_frac: float,
    clip_min: float,
    clip_max: float,
    reciprocal: bool,
) -> Array:
    xB = jnp.atleast_2d(x_cart)
    ext_potential = external_scale.value

    u = ext_potential.potential(xB, t=t).squeeze()
    s_raw = jnp.abs(u)

    if r_ref is None:
        s_norm = s_raw
    else:
        r_safe = jnp.maximum(r, 1e-12)
        x_dir = xB / r_safe[:, None]
        u_ref = ext_potential.potential(x_dir * r_ref, t=t).squeeze()
        s_ref = jnp.abs(u_ref)
        eps = eps_frac * jnp.maximum(s_ref, 1e-12)
        s_norm = (s_raw + eps) / (s_ref + eps)

    s_out = 1.0 / jnp.maximum(s_norm, 1e-12) if reciprocal else s_norm
    return jnp.clip(s_out, clip_min, clip_max)


# Type alias for the fully-bound scale callable.
_ScaleFn = Callable[[Array, Array, float, Any], Array]


class ScaleNNPotentialLayer(nnx.Module):
    """Apply an analytic, radius-dependent prefactor to a proxy potential."""

    scale_fn: _ScaleFn

    def __init__(
        self,
        config: Mapping[str, Any],
        external_scale: ExternalPytree | None = None,
    ) -> None:
        """Initialize the scaling layer, resolving all config at construction time.

        Parameters
        ----------
        config
            Configuration dict. The ``"scale"`` key selects the mode:
            - ``str``: ``"one"``, ``"power"``, or ``"nfw"`` for built-in modes.
            - ``Array``: pre-computed scale values used directly.
            - Anything else: external potential mode; pass the wrapped object via
              ``external_scale``.
        external_scale
            An ``ExternalPytree``-wrapped galax potential for dynamic scaling.
            Required when ``config["scale"]`` is not a string or array.

        """
        scale_val = config.get("scale", "one")

        if isinstance(scale_val, str):
            mode = scale_val
        elif isinstance(scale_val, jax.Array):
            mode = "precomputed"
        else:
            mode = "external"

        # Keep as a named attribute so NNX can track the Variable.
        if mode == "external" and external_scale is None:
            raise ValueError(
                "scale mode is 'external' but no external_scale was provided."
            )

        self._default_r_s: float = float(config.get("r_s", 1.0))

        # Bind the correct implementation once — __call__ has zero branching.
        match mode:
            case "one":
                self.scale_fn: _ScaleFn = _scale_one
            case "power":
                self.scale_fn = ft.partial(
                    _scale_power, power=float(config.get("power", 1.0))
                )
            case "nfw":
                self.scale_fn = ft.partial(
                    _scale_nfw, x_transformer=config["x_transformer"]
                )
            case "precomputed":
                self.scale_fn = ft.partial(_scale_precomputed, precomputed=scale_val)
            case _:  # "external"
                self.scale_fn = ft.partial(
                    _scale_external,
                    external_scale=external_scale,
                    r_ref=config.get("scale_r_ref", None),
                    eps_frac=float(config.get("scale_eps_frac", 1e-6)),
                    clip_min=float(config.get("scale_clip_min", 1e-3)),
                    clip_max=float(config.get("scale_clip_max", 1e3)),
                    reciprocal=bool(config.get("scale_reciprocal", True)),
                )

    def __call__(
        self, x_cart: Array, u_nn: Array, /, *, r_s: float | None = None, t: Any = 0
    ) -> Array:
        """Apply scaling to the NN potential."""
        r = jnp.linalg.norm(x_cart, axis=-1)
        r_s_ = r_s if r_s is not None else self._default_r_s
        return self.scale_fn(x_cart, r, r_s_, t) * u_nn
