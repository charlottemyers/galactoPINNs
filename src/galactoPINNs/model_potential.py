"""Galax potential backed by a learned galactoPINN model."""

__all__ = (
    "ModelPotential",
    "make_galax_potential",
)

from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY
from typing import Any, Protocol, TypeAlias, final

import equinox as eqx
import jax.numpy as jnp
import unxt as u
from flax import nnx
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractPotential
from jaxtyping import Array
from unxt import unitsystems
from unxt.quantity import AbstractQuantity
from xmmutablemap import ImmutableMap

Config: TypeAlias = Mapping[str, Any]
PositionInput: TypeAlias = Array | AbstractQuantity


# --- typing for the analytic potential ---
try:
    from galax.potential import AbstractPotential as GalaxPotential
except Exception:  # noqa: BLE001
    class GalaxPotential(Protocol):
        def potential(self,    positions: Any, *, t: Any = ...) -> Any: ...
        def acceleration(self, positions: Any, *, t: Any = ...) -> Any: ...


@final
class ModelPotential(AbstractPotential):
    """A galax-compatible potential backed by a pure NNX potential function.

    This class is a wrapper that makes a trained neural network model, represented
    by a pure function and its parameters, compatible with the `galax` library.
    It handles the transformation from physical units to the model's scaled
    units and back.

    The acceleration is not defined directly but is derived by `galax` via
    automatic differentiation of the `_potential` method.

    """

    potential_fn: Callable[[Any, Array, Any], Array] = eqx.field(static=True)
    acceleration_fn: Callable[[Any, Array], Array] = eqx.field(static=True)
    # The pytree of trained model parameters
    params: Any = eqx.field()
    # Static configuration (transformers, etc.)
    config: dict = eqx.field(static=True)
    # The analytic potential, if any, to be passed to the model
    analytic_potential: Any | None = eqx.field(static=True, default=None)


    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    def _as_batched_xyz(self, q: PositionInput) -> tuple[Array, bool]:
        x = jnp.asarray(getattr(q, "value", q))
        was_batched = x.ndim == 2
        x = eqx.error_if(
            x,
            (x.ndim < 1) | (x.ndim > 2) | (x.shape[-1] != 3),
            "Expected position shape (3,) or (N, 3).",
        )
        return jnp.atleast_2d(x), was_batched


    def _potential(self, q: PositionInput, _: Any) -> Array:
        """Return physical potential at positions ``q``."""
        x_batched, batched = self._as_batched_xyz(q)
        x_batched = x_batched.astype(jnp.float32)

        x_scaled = self.config["x_transformer"].transform(x_batched)
        # Call the pure potential function.
        u_scaled = self.potential_fn(self.params, x_scaled)
        u_phys = self.config["u_transformer"].inverse_transform(u_scaled)

        return jnp.squeeze(u_phys, axis=0) if not batched else jnp.ravel(u_phys)


    def _acceleration(self, q: PositionInput, _: Any) -> Array:
        """Return physical acceleration at positions ``q``."""
        x_batched, batched = self._as_batched_xyz(q)
        x_batched = x_batched.astype(jnp.float32)

        x_scaled = self.config["x_transformer"].transform(x_batched)
        # Call the pure acceleration function
        a_scaled = self.acceleration_fn(self.params, x_scaled)
        a_phys = self.config["a_transformer"].inverse_transform(a_scaled)

        return jnp.squeeze(a_phys, axis=0) if not batched else a_phys

    def _gradient(self, q: PositionInput, t: Any) -> Array:
        """Return spatial gradient of the potential at ``q``."""
        return -self._acceleration(q, t)


def make_galax_potential(
    model: nnx.Module, analytic_baseline_potential: GalaxPotential | None = None, *, units: u.AbstractUnitSystem = unitsystems.galactic  # noqa: E501
) -> ModelPotential:
    """Wrap a trained NNX model as a Galax `AbstractPotential`.

    This function extracts pure functions and parameters from the NNX model
    to create a fully-decoupled, JAX-compatible Galax potential.
    """
    # 1. Split the model into its static definition and trained parameters
    graph_def, state0 = nnx.split(model)

    def potential_fn(st: Any, x_scaled: Array) -> Array:
        caller = nnx.call((graph_def, st))
        out, _ = caller(
            x_scaled,
            mode="potential",
            analytic_potential=analytic_baseline_potential,
        )
        return out["potential"]

    def acceleration_fn(st: Any, x_scaled: Array) -> Array:
        """Pure acceleration function using the functional API."""
        caller = nnx.call((graph_def, st))
        out, _ = caller(
            x_scaled,
            mode="acceleration",
            analytic_potential=analytic_baseline_potential,
        )
        return out["acceleration"]

    return ModelPotential(
            potential_fn=potential_fn,
            acceleration_fn=acceleration_fn,
            params=state0,                      # params is an nnx.State
            config=model.config,
            units=units
            )
