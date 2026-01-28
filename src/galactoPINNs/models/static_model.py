"""Static gravitational potential model implementations."""

from collections.abc import Mapping
from typing import Any, Literal, Protocol, TypedDict

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

from galactoPINNs.layers import (
    CartesianToModifiedSphericalLayer,
    FuseandBoundary,
    FuseModelsLayer,
    ScaleNNPotentialLayer,
    SmoothMLP,
    TrainableGalaxPotential,
)

Mode = Literal["full", "potential", "acceleration", "density"]


class StaticOutputs(TypedDict, total=False):
    """Standardized outputs returned by StaticModel.__call__."""

    potential: Array
    acceleration: Array
    laplacian: Array
    outputs: dict[str, Any]


# --- Typing for the analytic potential ---
try:
    from galax.potential import AbstractPotential as GalaxPotential
except Exception:  # noqa: BLE001
    class GalaxPotential(Protocol):
        def potential(self,    positions: Any, *, t: Any = ...) -> Any: ...
        def acceleration(self, positions: Any, *, t: Any = ...) -> Any: ...

# ----------------------------
# Static model
# ----------------------------

class StaticModel(nnx.Module):
    """A Flax NNX module for a static gravitational potential model.

    This class defines a flexible model for a gravitational potential, which can
    be a pure neural network, a known analytic potential, or a hybrid of the two.
    It computes the potential and its derivatives, and is designed to be highly
    configurable through a dictionary (config).

    Parameters
    ----------
    config
        A dictionary containing the configuration for the model's architecture
        and behavior. Key options include:

        - ``"include_analytic"`` (bool): If True, add the analytic baseline
          potential to the NN potential.
        - ``"trainable"`` (bool): If True, use a ``TrainableGalaxPotential``
          layer for the analytic component.
        - ``"analytic_only"`` (bool): If True, the model output is solely the
          analytic potential.
        - ``"x_transformer"``, ``"u_transformer"``, ``"a_transformer"``: Objects
          for transforming coordinates and potential between physical and scaled
          units.
        - ``"depth"``, ``"width"``, ``"activation"``: Hyperparameters for
          ``SmoothMLP``.
        - Configuration for sub-layers like ``FuseandBoundary`` (``r_trans``,
          ``k_smooth``, etc.).
    in_features
        The number of input features for the MLP. Typically 5 for modified spherical
        coordinates or 3 for Cartesian.
    trainable_analytic_layer
        An instance of a trainable analytic potential layer, passed if
        ``config['trainable']`` is True.
    rngs
        Random number generator state for parameter initialization.

    """

    def __init__(
        self,
        config: Mapping[str, Any],
        in_features: int = 5,
        trainable_analytic_layer: TrainableGalaxPotential | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the static model layers."""
        self.config = config
        self.trainable_analytic_layer = trainable_analytic_layer

        # Initialize coordinate transform layer
        self.cart_to_sph_layer = CartesianToModifiedSphericalLayer(
            clip=config.get("clip", 1.0)
        )

        # Initialize scaling and fusion layers
        self.scale_layer = ScaleNNPotentialLayer(config=config)
        self.fuse_layer = FuseModelsLayer()
        self.fuse_boundary_layer = FuseandBoundary(config=config)

        # Initialize MLP if not disabled
        self.nn_off = config.get("nn_off", False)
        if not self.nn_off:
            depth = config.get("depth", 4)
            width = config.get("width", 128)
            act = config.get("activation", None)

            mlp_kwargs: dict[str, Any] = {
                "in_features": in_features,
                "width": width,
                "depth": depth,
                "rngs": rngs,
            }
            if act is not None:
                mlp_kwargs["act"] = act  # must be callable
            self.mlp = SmoothMLP(**mlp_kwargs)
        else:
            self.mlp = None

    def compute_potential(self, cart_x: Array) -> Array:
        """Evaluate the model potential at Cartesian position(s).

        Parameters
        ----------
        cart_x
            Input positions. Typically shape ``(3,)`` for a single point or
            ``(N, 3)`` for a batch.

        Returns
        -------
        potential
            Potential values. Shape is ``(N,)``, or ``(N, 1)`` for batched input.

        """
        return self(cart_x, mode="potential")["potential"]


    def compute_laplacian(self, cart_x: Array) -> Array:
        """Compute the Laplacian of the potential.

        Parameters
        ----------
        cart_x
            Batched Cartesian positions, shape ``(N, 3)``.

        Returns
        -------
        laplacian
            Batched Laplacian values, shape ``(N,)``.

        Notes
        -----
        This computes the full Hessian per point via ``jax.hessian`` and takes its
        trace. This is substantially more expensive than gradients.

        """

        def potential_fn(x: Array) -> Array:
            # Ensure scalar output for hessian()
            # This now correctly calls the potential computation from __call__
            return self(x, mode="potential")["potential"].squeeze()

        def laplacian_single(x: Array) -> Array:
            hess = jax.hessian(potential_fn)(x)  # (3,3)
            return jnp.trace(hess)

        laplacian_fn = jax.vmap(laplacian_single)
        return laplacian_fn(cart_x)

    def __call__(
        self,
        cart_x: Array,
        mode: Mode = "full",
        analytic_potential: GalaxPotential | None = None,
    ) -> StaticOutputs:
        """Forward pass for the static model.

        Parameters
        ----------
        cart_x
            Cartesian inputs. Typically shape (3,) for a single point or
            (N, 3) for a batch. Assumed to be in the model's scaled space.
        mode
            Controls what is computed/returned:
            - "full": return at least potential and acceleration (and any aux outputs).
            - "potential": compute/return only the potential pathway.
            - "acceleration": compute/return acceleration.
            - "density": compute/return potential, acceleration, and Laplacian.
        analytic_potential : galax.potential.AbstractPotential, optional
            An external, non-trainable analytic potential object. This is only
            used if ``config["include_analytic"]`` is True and
            ``config["trainable"]`` is False.

        Returns
        -------
        outputs
            A dict-like object containing keys "potential" and "acceleration".

        """
        outputs: dict[str, Any] = {}

        if self.config.get("convert_to_spherical", True):
            x = self.cart_to_sph_layer(cart_x)
        else:
            x = cart_x

        u_nn = 0.0 if self.nn_off or self.mlp is None else self.mlp(x)

        analytic_potential_scaled = 0.0
        r_s_learned = self.config["r_s"]
        if self.config.get("include_analytic", False):
            # 1. Evaluate the external analytic potential in physical units
            x_phys = self.config["x_transformer"].inverse_transform(cart_x)
            u_phys = 0.0
            if analytic_potential is not None:
                u_phys = analytic_potential.potential(x_phys, t=0) # Assuming t=0
            elif self.config.get("trainable", False):
                u_phys, r_s_learned = self.trainable_analytic_layer(x_phys)

            # 2. Transform to scaled units
            analytic_potential_scaled = self.config["u_transformer"].transform(
                    u_phys
                )

        scaled_potential = self.scale_layer(cart_x, u_nn, r_s_learned=r_s_learned)
        fused_potential = self.fuse_layer(scaled_potential, analytic_potential_scaled)
        fuse_boundary_potential = self.fuse_boundary_layer(cart_x, scaled_potential, analytic_potential_scaled)


        if self.config.get("analytic_only", False):
            potential = analytic_potential_scaled
        elif self.config.get("enforce_boundary", False):
            potential = fuse_boundary_potential
        elif self.config.get("include_analytic", True):
            potential = fused_potential
        else:
            potential = scaled_potential

        outputs["final"] = potential

        if mode == "potential":
            return {"potential": potential}

        def potential_for_grad(x_arg: Array) -> Array:
            return self(x_arg, mode="potential", analytic_potential=analytic_potential)[
                "potential"
            ].squeeze()

        acceleration = -jax.vmap(jax.grad(potential_for_grad))(cart_x)

        if mode == "density":
            def laplacian_single(x_arg: Array) -> Array:
                hess = jax.hessian(potential_for_grad)(x_arg)
                return jnp.trace(hess)

            laplacian = jax.vmap(laplacian_single)(cart_x)
            return {
                "laplacian": laplacian,
                "acceleration": acceleration,
                "potential": potential,
            }

        return {
            "acceleration": acceleration,
            "potential": potential,
            "outputs": outputs,
        }
