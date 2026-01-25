import jax
import jax.numpy as jnp
from unxt import unitsystems
from dataclasses import KW_ONLY
from typing import Any, final, Tuple, Mapping, Union
from typing_extensions import TypeAlias

from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractPotential
import equinox as eqx
from xmmutablemap import ImmutableMap
import unxt as u
from unxt.quantity import AbstractQuantity

Array: TypeAlias = jax.Array
Params: TypeAlias = Any
ApplyFn: TypeAlias = Any
Config: TypeAlias = Mapping[str, Any]
PositionInput: TypeAlias = Union[
    Array,
    AbstractQuantity,
]

__all__ = (
    "ModelPotential",
    "make_galax_potential",
)

@final
class ModelPotential(AbstractPotential):
    """
    Galax potential backed by a learned galactoPINN model.

    Parameters
    ----------
    apply_fn : callable
        Callable compatible with Flax apply.
        Must accept `(variables, x_scaled)` and return a dict containing keys
    params : Any
        Parameter pytree to pass as `{"params": params}` into apply_fn.
    config : dict
        Must contain the transformers listed in the module docstring.

    """

    apply_fn: Any = eqx.field(static=True)
    params: Any = eqx.field(static=True)
    config: dict = eqx.field(static=True)

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )

    def _as_batched_xyz(
        self,
        q: PositionInput,
    ) -> Tuple[Array, bool]:
        """
        Normalize input positions to shape ``(N, 3)``.

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

        if x.ndim == 1:
            if x.shape[0] != 3:
                raise ValueError(
                    f"Expected position shape (3,), got {x.shape}."
                )
            return x.reshape(1, 3), False

        if x.ndim == 2:
            if x.shape[1] != 3:
                raise ValueError(
                    f"Expected position shape (N, 3), got {x.shape}."
                )
            return x, True

        raise ValueError(
            f"Expected position ndim 1 or 2, got ndim={x.ndim}, shape={x.shape}."
        )

    def _acceleration(
        self,
        q: PositionInput,
        t: Any,
    ) -> Array:
        """
        Return physical acceleration at positions ``q``.

        Returns
        -------
        a
            - shape ``(3,)`` if ``q`` is a single point
            - shape ``(N, 3)`` if ``q`` is batched
        """
        x_batched, batched = self._as_batched_xyz(q)
        x_batched = x_batched.astype(jnp.float32)

        x_scaled = self.config["x_transformer"].transform(x_batched)
        a_scaled = self.apply_fn(
            {"params": self.params},
            x_scaled,
        )["acceleration"]
        a_phys = self.config["a_transformer"].inverse_transform(a_scaled)

        return jnp.squeeze(a_phys, axis=0) if not batched else a_phys

    def _potential(
        self,
        q: PositionInput,
        t: Any,
    ) -> Array:
        """
        Return physical potential at positions ``q``.

        Returns
        -------
        u
            - scalar if ``q`` is a single point
            - shape ``(N,)`` if ``q`` is batched
        """
        x_batched, batched = self._as_batched_xyz(q)
        x_batched = x_batched.astype(jnp.float32)

        x_scaled = self.config["x_transformer"].transform(x_batched)
        u_scaled = self.apply_fn(
            {"params": self.params},
            x_scaled,
        )["potential"]
        u_phys = self.config["u_transformer"].inverse_transform(u_scaled)

        return jnp.squeeze(u_phys, axis=0) if not batched else jnp.ravel(u_phys)


    def _gradient(
        self,
        q: PositionInput,
        t: Any,
    ) -> Array:
        """
        Return spatial gradient of the potential at ``q``.

        Notes
        -----
        Defined as ``-acceleration`` to ensure consistency.
        """
        return -self._acceleration(q, t)


def make_galax_potential(
    model: Any,
    params: Any,
    *,
    units: u.AbstractUnitSystem = unitsystems.galactic,
) -> ModelPotential:
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
    config = model.config
    if config is None:
        raise ValueError("make_galax_potential requires `model` to have a `.config` attribute.")

    return ModelPotential(
        apply_fn=apply_fn,
        params=params,
        config=config,
        units=units,
    )
