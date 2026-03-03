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
    ScaleNNPotentialLayer,
    SmoothMLP,
    TrainableGalaxPotential,
)

Mode = Literal["full", "potential", "acceleration", "density"]


class ExternalPytree(nnx.Variable):
    """Variable wrapper for external pytrees (like equinox modules).

    This allows galax potentials (which are equinox modules containing JAX arrays)
    to be stored as attributes in NNX modules without triggering pytree inspection errors.
    Access the wrapped object via the `.value` attribute.
    """

    pass


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
        def potential(self, positions: Any, *, t: Any = ...) -> Any: ...
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

        - ``"ab_potential"``: A galax potential object for the analytic baseline.
        - ``"scale"``: Scaling mode - either a string ("one", "power", "nfw"),
          a precomputed array, or a galax potential object.
        - ``"include_analytic"`` (bool): If True, add the analytic baseline
          potential to the NN potential.
        - ``"trainable"`` (bool): If True, use a ``TrainableGalaxPotential``
          layer for the analytic component.
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
        # --- Handle analytic baseline potential ---
        raw_ab_potential = config.get("ab_potential", None)
        if isinstance(raw_ab_potential, ExternalPytree):
            self.ab_potential = raw_ab_potential
        elif raw_ab_potential is not None:
            self.ab_potential = ExternalPytree(raw_ab_potential)
        else:
            self.ab_potential = None

        # --- Prepare cleaned config dicts ---
        config_without_ab = {
            k: v for k, v in config.items()
            if k != "ab_potential"
        }
        config_without_externals = {
            k: v for k, v in config.items()
            if k not in ("ab_potential", "scale")
        }

        self.config = config_without_externals
        self.trainable_analytic_layer = trainable_analytic_layer

        # --- Initialize coordinate transform layer ---
        self.cart_to_sph_layer = CartesianToModifiedSphericalLayer(
            clip=config.get("clip", 1.0)
        )

        # --- Determine external scale potential for ScaleNNPotentialLayer ---
        raw_scale = config.get("scale", "one")

        if isinstance(raw_scale, str):
            # String mode ("one", "power", "nfw") - handled internally by ScaleNNPotentialLayer
            wrapped_scale_potential = None
        elif isinstance(raw_scale, (jnp.ndarray, jax.Array)):
            # Precomputed array - handled internally by ScaleNNPotentialLayer
            wrapped_scale_potential = None
        else:
            # raw_scale is an external potential-like object
            if isinstance(raw_scale, ExternalPytree):
                wrapped_scale_potential = raw_scale
            elif raw_scale is raw_ab_potential:
                wrapped_scale_potential = self.ab_potential
            else:
                wrapped_scale_potential = ExternalPytree(raw_scale)



        self.scale_layer = ScaleNNPotentialLayer(
            config=config_without_ab,
            external_scale=wrapped_scale_potential,
        )

        # --- Initialize fusion/boundary layer ---
        self.fuse_boundary_layer = FuseandBoundary(config=config_without_externals)

        # --- Initialize MLP ---
        self.nn_off = config.get("nn_off", False)
        if not self.nn_off:
            depth = config.get("depth", 4)
            width = config.get("width", 128)
            activation = config.get("activation", None)

            mlp_kwargs: dict[str, Any] = {
                "in_features": in_features,
                "width": width,
                "depth": depth,
                "rngs": rngs,
            }
            if activation is not None:
                mlp_kwargs["act"] = activation
            self.mlp = SmoothMLP(**mlp_kwargs)
        else:
            self.mlp = None


    def compute_potential(
    self,
    cart_x: Array,
    *,
    trainable_analytic_layer: TrainableGalaxPotential | None = None,
    ) -> Array:
        """Evaluate the model potential at Cartesian position(s).

        Parameters
        ----------
        cart_x
            Input positions, shape ``(N, 3)`` for a batch or ``(3,)`` for a
            single point. Assumed to be in the model's scaled space.
        trainable_analytic_layer
            Optional trainable analytic layer to use in place of
            ``self.trainable_analytic_layer``. Required when
            ``config["trainable"]`` is ``True`` and the layer is passed
            externally (e.g. during SVI).

        Returns
        -------
        potential
            Squeezed potential values, shape ``(N,)`` for batched input.

        """
        return self(cart_x, mode="potential", trainable_analytic_layer=trainable_analytic_layer)["potential"].squeeze()


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
            return self(x, mode="potential")["potential"].squeeze()

        def laplacian_single(x: Array) -> Array:
            hess = jax.hessian(potential_fn)(x)  # (3, 3)
            return jnp.trace(hess)

        return jax.vmap(laplacian_single)(cart_x)

    def __call__(
        self,
        cart_x: Array,
        mode: Mode = "full",
        trainable_analytic_layer: TrainableGalaxPotential | None = None,  # <-- ADD
    ) -> StaticOutputs:
        """Forward pass for the static model.

        Parameters
        ----------
        cart_x
            Cartesian inputs. Typically shape (3,) for a single point or
            (N, 3) for a batch. Assumed to be in the model's scaled space.
        mode
            Controls what is computed/returned:
            - "full": return potential, acceleration, and auxiliary outputs.
            - "potential": compute/return only the potential.
            - "acceleration": compute/return acceleration.
            - "density": compute/return potential, acceleration, and Laplacian.

        Returns
        -------
        outputs
            A dict-like object containing keys "potential" and "acceleration".

        """
        aux_outputs: dict[str, Any] = {}

        # --- Coordinate transformation ---
        if self.config.get("convert_to_spherical", True):
            x_transformed = self.cart_to_sph_layer(cart_x)
        else:
            x_transformed = cart_x

        # --- Neural network potential ---
        u_nn = 0.0 if self.nn_off or self.mlp is None else self.mlp(x_transformed)

        # --- Analytic baseline potential ---
        analytic_potential_scaled = 0.0
        r_s_learned = self.config.get("r_s", 1.0)

        if self.config.get("include_analytic", False):
            # Transform to physical coordinates
            x_phys = self.config["x_transformer"].inverse_transform(cart_x)
            u_phys = 0.0

            if self.ab_potential is not None and not self.config.get("trainable", False):
                u_phys = self.ab_potential.value.potential(x_phys, t=0)
            elif self.config.get("trainable", False):
                layer = trainable_analytic_layer if trainable_analytic_layer is not None else self.trainable_analytic_layer
                if layer is None:
                    raise ValueError("config['trainable']=True but no trainable_analytic_layer was provided.")
                u_phys, r_s_learned = layer(x_phys)

            # Transform potential to scaled units
            analytic_potential_scaled = self.config["u_transformer"].transform(u_phys)

        # --- Combine potentials ---
        scaled_nn_potential = self.scale_layer(cart_x, u_nn, r_s_learned=r_s_learned)
        fused_potential = scaled_nn_potential + analytic_potential_scaled
        boundary_potential = self.fuse_boundary_layer(
            cart_x, scaled_nn_potential, analytic_potential_scaled
        )

        # --- Select final potential based on config ---
        if self.config.get("enforce_boundary", False):
            potential = boundary_potential
        elif self.config.get("include_analytic", True):
            potential = fused_potential
        else:
            potential = scaled_nn_potential

        aux_outputs["final"] = potential

        # --- Return early if only potential requested ---
        if mode == "potential":
            return {"potential": potential}

        # --- Compute acceleration via autodiff ---
        def pot_single(x1: Array) -> Array:
            return self.compute_potential(x1[None, :], trainable_analytic_layer=trainable_analytic_layer).squeeze()

        acceleration = -jax.vmap(jax.grad(pot_single))(cart_x)

        # --- Compute Laplacian if density mode ---
        if mode == "density":
            def laplacian_single(x_arg: Array) -> Array:
                hess = jax.hessian(self.compute_potential)(x_arg)
                return jnp.trace(hess)

            laplacian = jax.vmap(laplacian_single)(cart_x)
            return {
                "potential": potential,
                "acceleration": acceleration,
                "laplacian": laplacian,
            }

        return {
            "potential": potential,
            "acceleration": acceleration,
            "outputs": aux_outputs,
        }
