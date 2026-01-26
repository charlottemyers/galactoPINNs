"""Static gravitational potential model implementations."""

from collections.abc import Mapping
from typing import Any, Literal, TypedDict

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

from galactoPINNs.layers import (
    AnalyticModelLayer,
    CartesianToModifiedSphericalLayer,
    FuseModelsLayer,
    ScaleNNPotentialLayer,
    SmoothMLP,
    TrainableGalaxPotential,
)

Mode = Literal["full", "potential", "acceleration"]


class StaticOutputs(TypedDict, total=False):
    """Standardized outputs returned by StaticModel.__call__."""

    potential: Array
    acceleration: Array
    laplacian: Array
    outputs: dict[str, Any]  # optional auxiliary dict to store intermediate outputs


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
        The number of input features for the MLP. Typically 3 for spherical
        coordinates or 3 for Cartesian if not converting.
    trainable_analytic_layer
        An instance of a trainable analytic potential layer, passed if
        ``config['trainable']`` is True.
    rngs
        Random number generator state for parameter initialization.

    """

    def __init__(
        self,
        config: Mapping[str, Any],
        in_features: int = 3,
        trainable_analytic_layer: TrainableGalaxPotential | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the static model layers."""
        self.config = nnx.data(config)
        self.trainable_analytic_layer = trainable_analytic_layer

        # Initialize coordinate transform layer
        self.cart_to_sph_layer = CartesianToModifiedSphericalLayer(
            clip=config.get("clip", 1.0)
        )

        # Initialize scaling and fusion layers
        self.scale_layer = ScaleNNPotentialLayer(config=config)
        self.fuse_layer = FuseModelsLayer()
        self.analytic_layer = AnalyticModelLayer(config=config, mode="static")

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

    def compute_acceleration(self, cart_x: Array) -> Array:
        """Compute acceleration: ``a = -grad(Φ)``.

        Parameters
        ----------
        cart_x
            Batched Cartesian positions, shape ``(N, 3)``.

        Returns
        -------
        acceleration
            Batched accelerations, shape ``(N, 3)``.

        Notes
        -----
        This method uses ``jax.grad`` on a scalar-valued potential function and
        then vectorizes across the batch with ``jax.vmap``.

        """

        def potential_fn(x: Array) -> Array:
            # Ensure scalar output for grad()
            return self.compute_potential(x).squeeze()

        acceleration_fn = jax.vmap(jax.grad(potential_fn))
        return -acceleration_fn(cart_x)

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
            return self.compute_potential(x).squeeze()

        def laplacian_single(x: Array) -> Array:
            hess = jax.hessian(potential_fn)(x)  # (3,3)
            return jnp.trace(hess)

        laplacian_fn = jax.vmap(laplacian_single)
        return laplacian_fn(cart_x)

    def __call__(self, cart_x: Array, mode: Mode = "full") -> StaticOutputs:
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

        outputs["u_nn"] = u_nn

        # Fuse models (combine analytic and NN outputs)
        if self.config.get("trainable", False):
            phys_x = self.config["x_transformer"].inverse_transform(cart_x)
            phys_analytic_potential, r_s_learned = self.trainable_analytic_layer(phys_x)
            analytic_potential = self.config["u_transformer"].transform(
                phys_analytic_potential
            )

        else:
            analytic_potential = self.analytic_layer(cart_x)
            r_s_learned = self.config["r_s"]

        scaled_potential = self.scale_layer(cart_x, u_nn, r_s_learned=r_s_learned)
        outputs["scale_nn_potential"] = scaled_potential

        # Fuse unscaled NN output with physical analytic model
        if self.config.get("analytic_only", False):
            fused_potential = analytic_potential
        else:
            fused_potential = self.fuse_layer(scaled_potential, analytic_potential)
        outputs["fuse_models"] = fused_potential

        if self.config["include_analytic"]:
            potential = fused_potential
        else:
            potential = scaled_potential

        outputs["final"] = potential

        if mode == "potential":
            return {"potential": potential}

        acceleration = self.compute_acceleration(cart_x)
        if mode == "density":
            laplacian = self.compute_laplacian(cart_x)
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
