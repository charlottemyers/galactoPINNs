"""Static gravitational potential model implementations."""

from collections.abc import Mapping
from typing import Any, Literal, Optional, TypedDict

import jax
import jax.numpy as jnp
from flax import linen as nn
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


class StaticModel(nn.Module):
    """A Flax module for a static gravitational potential model.

    This class defines a flexible model for a gravitational potential, which can
    be a pure neural network, a known analytic potential, or a hybrid of the two.
    It computes the potential and its derivatives, and is designed to be highly
    configurable through a dictionary (config).

    Attributes
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

    depth
        The number of hidden layers in the ``SmoothMLP``.
    trainable_analytic_layer
        An instance of a trainable analytic potential layer, passed if
        ``config['trainable']`` is True.

    """

    config: Mapping[str, Any]
    trainable_analytic_layer: Optional["TrainableGalaxPotential"] = None
    depth: int = 4

    def setup(self) -> None:
        """Initialize the trainable analytic layer reference."""
        self.trainable_layer = self.trainable_analytic_layer

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
        return self.apply(
            {"params": self.variables["params"]},
            cart_x,
            mode="potential",
        )["potential"]

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

    @nn.compact
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
        outputs = {}
        if self.trainable_analytic_layer is not None:
            # stub parameter ensures that the `trainable_analytic_layer` is
            # correctly registered within the Flax parameter structure, even if it
            # contains no trainable parameters itself
            _stub = self.param("_stub", lambda _: jnp.array(0.0))

        cart_to_sph_layer = CartesianToModifiedSphericalLayer(
            clip=self.config.get("clip", 1.0)
        )

        scale_layer = ScaleNNPotentialLayer(config=self.config)
        fuse_layer = FuseModelsLayer()
        analytic_layer = AnalyticModelLayer(config=self.config, mode="static")

        if self.config.get("convert_to_spherical", True):
            x = cart_to_sph_layer(cart_x)
        else:
            x = cart_x

        depth = self.config.get("depth", 4)
        width = self.config.get("width", 128)
        act = self.config.get("activation", None)
        gelu_approx = self.config.get("gelu_approximate", False)

        if self.config.get("nn_off", False):
            u_nn = 0.0
        else:
            mlp_kwargs = {
                "width": width,
                "depth": depth,
                "gelu_approximate": gelu_approx,
            }
            if act is not None:
                mlp_kwargs["act"] = act  # must be callable
            u_nn = SmoothMLP(**mlp_kwargs)(x)

        outputs["u_nn"] = u_nn

        # Fuse models (combine analytic and NN outputs)
        if self.config.get("trainable", False):
            phys_x = self.config["x_transformer"].inverse_transform(cart_x)
            phys_analytic_potential, r_s_learned = self.trainable_analytic_layer(phys_x)
            analytic_potential = self.config["u_transformer"].transform(
                phys_analytic_potential
            )

        else:
            analytic_potential = analytic_layer(cart_x)
            r_s_learned = self.config["r_s"]

        scaled_potential = scale_layer(cart_x, u_nn, r_s_learned=r_s_learned)
        outputs["scale_nn_potential"] = scaled_potential

        # Fuse unscaled NN output with physical analytic model
        if self.config.get("analytic_only", False):
            fused_potential = analytic_potential
        else:
            fused_potential = fuse_layer(scaled_potential, analytic_potential)
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
