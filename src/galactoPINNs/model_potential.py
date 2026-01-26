"""Gravitational potential model definitions."""

__all__ = (
    "ModelPotential",
    "make_galax_potential",
)

from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY
from typing import Any, TypeAlias, final

import equinox as eqx
import jax.numpy as jnp
import unxt as u
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractPotential
from jaxtyping import Array
from unxt import unitsystems
from unxt.quantity import AbstractQuantity
from xmmutablemap import ImmutableMap

Config: TypeAlias = Mapping[str, Any]
PositionInput: TypeAlias = Array | AbstractQuantity


@final
class ModelPotential(AbstractPotential):
    """Galax potential backed by a learned galactoPINN model.

    Parameters
    ----------
    model_fn : callable
        Callable that accepts ``(x_scaled)`` and returns a dict containing
        keys ``"potential"`` and ``"acceleration"``.
    config : dict
        Must contain the transformers listed in the module docstring.

    """

    model_fn: Callable[[Array], dict[str, Array]] = eqx.field(static=True)
    config: dict = eqx.field(static=True)

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    def _as_batched_xyz(self, q: PositionInput) -> tuple[Array, bool]:
        """Normalize input positions to shape ``(N, 3)``.

        Parameters
        ----------
        q
            Position input. May be:
            - raw array-like
            - quantity-like with a ``.value`` attribute

        Returns
        -------
        x_batched
            Array of shape ``(N, 3)``.
        was_batched
            True if the input was already batched, False if a single point.

        """
        x = jnp.asarray(getattr(q, "value", q))
        was_batched = x.ndim == 2

        # Validate and batch in one step
        x = eqx.error_if(
            x,
            (x.ndim < 1) | (x.ndim > 2) | (x.shape[-1] != 3),
            "Expected position shape (3,) or (N, 3).",
        )
        return jnp.atleast_2d(x), was_batched

    def _acceleration(self, q: PositionInput, _: Any) -> Array:
        """Return physical acceleration at positions ``q``.

        Returns
        -------
        a
            - shape ``(3,)`` if ``q`` is a single point
            - shape ``(N, 3)`` if ``q`` is batched

        """
        x_batched, batched = self._as_batched_xyz(q)
        x_batched = x_batched.astype(jnp.float32)

        x_scaled = self.config["x_transformer"].transform(x_batched)
        a_scaled = self.model_fn(x_scaled)["acceleration"]
        a_phys = self.config["a_transformer"].inverse_transform(a_scaled)

        return jnp.squeeze(a_phys, axis=0) if not batched else a_phys

    def _potential(self, q: PositionInput, _: Any) -> Array:
        """Return physical potential at positions ``q``.

        Returns
        -------
        u
            - scalar if ``q`` is a single point
            - shape ``(N,)`` if ``q`` is batched

        """
        x_batched, batched = self._as_batched_xyz(q)
        x_batched = x_batched.astype(jnp.float32)

        x_scaled = self.config["x_transformer"].transform(x_batched)
        u_scaled = self.model_fn(x_scaled)["potential"]
        u_phys = self.config["u_transformer"].inverse_transform(u_scaled)

        return jnp.squeeze(u_phys, axis=0) if not batched else jnp.ravel(u_phys)

    def _gradient(self, q: PositionInput, t: Any) -> Array:
        """Return spatial gradient of the potential at ``q``.

        Notes
        -----
        Defined as ``-acceleration`` to ensure consistency.

        """
        return -self._acceleration(q, t)


def make_galax_potential(
    model: Any, *, units: u.AbstractUnitSystem = unitsystems.galactic
) -> ModelPotential:
    """Wrap a trained NNX model as a Galax `AbstractPotential`.

    Parameters
    ----------
    model : nnx.Module
        A Flax NNX module with a `.config` attribute. The model must be
        callable and return a dict with ``"potential"`` and ``"acceleration"``.
    units : unxt.AbstractUnitSystem, optional
        Unit system for the Galax potential (defaults to galactic).

    Returns
    -------
    pot : ModelPotential
        A Galax-compatible potential.

    """
    if model.config is None:
        raise ValueError(
            "make_galax_potential requires `model` to have a `.config` attribute."
        )

    return ModelPotential(model_fn=model, config=model.config, units=units)
