import jax.numpy as jnp
from unxt import unitsystems
from dataclasses import KW_ONLY
from typing import Any, final
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractPotential
import equinox as eqx
from xmmutablemap import ImmutableMap
import unxt as u
from unxt.quantity import AbstractQuantity

__all__ = (
    "ModelPotential",
    "make_galax_potential",
)

@final
class ModelPotential(AbstractPotential):
    """
    Galax potential backed by a learned model.

    Parameters
    ----------
    apply_fn : callable
        Callable compatible with Flax apply, typically `model.apply`.
        Must accept `(variables, x_scaled)` and return a dict containing keys
Provide "potential" and "acceleration" keys.
    params : Any
        Parameter pytree to pass as `{"params": params}` into apply_fn.
    config : dict
        Must contain the transformers listed in the module docstring.

    Notes
    -----
    - Inputs `q` may be raw arrays or quantity-like objects with a `.value` attribute.
    - Expected position shape is (3,) for a single point or (N, 3) for batched evaluation.
    """

    apply_fn: Any = eqx.field(static=True)
    params: Any = eqx.field(static=True)
    config: dict = eqx.field(static=True)

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    def _as_batched_xyz(self, q):
        """
        Normalize input positions to shape (N, 3) and return (x_batched, was_batched).

        Raises ValueError with a clear message if the input shape is not supported.
        """
        x = jnp.asarray(getattr(q, "value", q))
        if x.ndim == 1:
            if x.shape[0] != 3:
                raise ValueError(f"Expected position shape (3,), got {x.shape}.")
            return x.reshape(1, 3), False
        if x.ndim == 2:
            if x.shape[1] != 3:
                raise ValueError(f"Expected position shape (N, 3), got {x.shape}.")
            return x, True
        raise ValueError(f"Expected position ndim 1 or 2, got ndim={x.ndim}, shape={x.shape}.")

    def _acceleration(self, q, t):
        """
        Return physical acceleration at positions q.

        Returns
        -------
        a : jax.Array
            Shape (3,) if q was a single point, or (N, 3) if q was batched.
        """
        x_batched, batched = self._as_batched_xyz(q)
        x_batched = x_batched.astype(jnp.float32)

        x_scaled = self.config["x_transformer"].transform(x_batched)
        a_scaled = self.apply_fn({"params": self.params}, x_scaled)["acceleration"]
        a_phys = self.config["a_transformer"].inverse_transform(a_scaled)

        return jnp.squeeze(a_phys, axis=0) if not batched else a_phys

    def _potential(self, q, t):
        """
        Return physical potential at positions q.

        Returns
        -------
        u : jax.Array
            Scalar if q was a single point, or shape (N,) if q was batched.
        """
        x_batched, batched = self._as_batched_xyz(q)
        x_batched = x_batched.astype(jnp.float32)

        x_scaled = self.config["x_transformer"].transform(x_batched)
        u_scaled = self.apply_fn({"params": self.params}, x_scaled)["potential"]
        u_phys = self.config["u_transformer"].inverse_transform(u_scaled)

        # Standardize shapes: () for single, (N,) for batch.
        if not batched:
            return jnp.squeeze(u_phys, axis=0)
        return jnp.ravel(u_phys)

    def _gradient(self, q, t):
        """Return ∇Φ; Galax defines gradient as negative acceleration for gravitational potentials."""
        return -self._acceleration(q, t)


def make_galax_potential(model, params, *, units=unitsystems.galactic):
    """
    Wrap a trained model as a Galax `AbstractPotential`.

    Parameters
    ----------
    model : object or callable
        Either a Flax module with attribute `.apply` and `.config`, or a callable apply function.
    params : Any
        Parameter pytree used by the model.
    units : unxt.AbstractUnitSystem, optional
        Unit system for the Galax potential (defaults to galactic).

    Returns
    -------
    pot : ModelPotential
        A Galax-compatible potential.
    """
    apply_fn = model.apply
    config = model.config #if hasattr(model, "config") else None
    if config is None:
        raise ValueError("make_galax_potential requires `model` to have a `.config` attribute.")

    return ModelPotential(
        apply_fn=apply_fn,
        params=params,
        config=config,
        units=units,
    )
