"""Static gravitational potential model implementations."""

import functools as ft
from collections.abc import Callable
from typing import Any, Final, NotRequired, TypedDict

import galax.potential as gp
import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

from galactoPINNs.layers import (
    MLP,
    CartesianToModifiedSphericalLayer,
    ExternalPytree,
    FuseandBoundary,
    ScaleNNPotentialLayer,
    TrainableGalaxPotential,
    ZeroPotential,
)


class StaticModelConfig(TypedDict):
    """Configuration dictionary for StaticModel.

    All fields are optional and have sensible defaults.
    """

    # Analytic potential configuration
    ab_potential: NotRequired[Any]
    scale: NotRequired[str | Array | Any]

    # Analytic baseline options
    include_analytic: NotRequired[bool]
    trainable: NotRequired[bool]

    # Coordinate and potential transformers
    x_transformer: NotRequired[Any]
    u_transformer: NotRequired[Any]
    a_transformer: NotRequired[Any]

    # MLP hyperparameters
    depth: NotRequired[int]
    width: NotRequired[int]
    activation: NotRequired[Callable[[Array], Array]]

    # Neural network options
    nn_off: NotRequired[bool]

    # Boundary blending configuration (FuseandBoundary)
    enforce_boundary: NotRequired[bool]
    r_trans: NotRequired[float]
    k_smooth: NotRequired[float]
    train_k: NotRequired[bool]
    min_k: NotRequired[float]
    radial_power: NotRequired[float | None]
    r_smooth: NotRequired[float]
    saturation: NotRequired[float]

    # Scale radius
    r_s: NotRequired[float]


DEFAULT_STATIC_MODEL_CONFIG: Final[StaticModelConfig] = {
    "scale": "one",
    "include_analytic": False,
    "trainable": False,
    "nn_off": False,
    "enforce_boundary": False,
    "depth": 4,
    "width": 128,
    "activation": ft.partial(jax.nn.gelu, approximate=False),
    "r_trans": 200.0,
    "k_smooth": 0.5,
    "train_k": False,
    "min_k": 0.01,
    "r_smooth": 150.0,
    "saturation": 1.0,
    "r_s": 1.0,
}

DEFAULT_INPUT_ENCODER: Final = CartesianToModifiedSphericalLayer(clip=1.0)
EXCLUDED_CONFIG_KEYS: Final = frozenset({"ab_potential", "scale"})


class StaticModel(nnx.Module):
    """A Flax NNX module for a static gravitational potential model.

    This class defines a flexible model for a gravitational potential, which can
    be a pure neural network, a known analytic potential, or a hybrid of the two.
    It computes the potential and its derivatives, and is designed to be highly
    configurable through a dictionary (config).

    Parameters
    ----------
    config : StaticModelConfig
        A configuration dictionary (see ``StaticModelConfig`` for all available
        keys and their meanings).
    in_features : int, optional
        The number of input features for the MLP. Typically 5 for modified
        spherical coordinates or 3 for Cartesian. Default: 5.
    trainable_analytic_layer : TrainableGalaxPotential | None, optional
        An instance of a trainable analytic potential layer, passed if
        ``config['trainable']`` is True.
    rngs : nnx.Rngs
        Random number generator state for parameter initialization.

    """

    # --- Configuration ---
    config: dict[str, Any]
    # --- Forward pass (call order) ---
    input_encoder: nnx.Module = DEFAULT_INPUT_ENCODER
    nn_potential: MLP | ZeroPotential
    ab_potential: ExternalPytree
    trainable_analytic_layer: TrainableGalaxPotential | None
    scale_layer: ScaleNNPotentialLayer
    fuse_boundary_layer: FuseandBoundary

    def __init__(
        self,
        config: StaticModelConfig,
        input_encoder: nnx.Module = DEFAULT_INPUT_ENCODER,
        in_features: int = 5,
        trainable_analytic_layer: TrainableGalaxPotential | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the static model layers."""
        # Merge provided config with defaults
        config = DEFAULT_STATIC_MODEL_CONFIG | config

        # --- Prepare cleaned config dicts ---
        self.config = {k: v for k, v in config.items() if k not in EXCLUDED_CONFIG_KEYS}

        # --- Initialize coordinate transform layer ---
        self.input_encoder = input_encoder

        # --- Handle analytic baseline potential ---
        ab_pot = config.get("ab_potential", None)
        if isinstance(ab_pot, ExternalPytree):
            self.ab_potential = ab_pot
        elif ab_pot is not None:
            self.ab_potential = ExternalPytree(ab_pot)
        else:
            self.ab_potential = ExternalPytree(gp.NullPotential())

        self.trainable_analytic_layer = trainable_analytic_layer

        # --- Determine external scale potential for ScaleNNPotentialLayer ---
        raw_scale = config["scale"]

        if isinstance(raw_scale, str):
            # String mode ("one", "power", "nfw") - handled internally by
            # ScaleNNPotentialLayer
            wrapped_scale_potential = None
        elif isinstance(raw_scale, (jnp.ndarray, jax.Array)):
            # Precomputed array - handled internally by ScaleNNPotentialLayer
            wrapped_scale_potential = None
        # raw_scale is an external potential-like object
        elif isinstance(raw_scale, ExternalPytree):
            wrapped_scale_potential = raw_scale
        elif raw_scale is ab_pot:
            wrapped_scale_potential = self.ab_potential
        else:
            wrapped_scale_potential = ExternalPytree(raw_scale)

        self.scale_layer = ScaleNNPotentialLayer(
            config={k: v for k, v in config.items() if k != "ab_potential"},
            external_scale=wrapped_scale_potential,
        )

        # --- Initialize fusion/boundary layer ---
        self.fuse_boundary_layer = FuseandBoundary(config=self.config)

        # --- Initialize MLP ---
        if config["nn_off"]:
            self.nn_potential = ZeroPotential()
        else:
            self.nn_potential = MLP(
                in_features=in_features,
                width=config["width"],
                depth=config["depth"],
                rngs=rngs,
                act=config["activation"],
            )

    def __call__(
        self,
        x_cart: Array,
        /,
        trainable_analytic_layer: TrainableGalaxPotential | None = None,
    ) -> Array:
        """Compute the gravitational potential at the given positions.

        Parameters
        ----------
        x_cart
            Cartesian inputs. Shape ``(3,)`` for a single point or ``(N, 3)``
            for a batch. Assumed to be in the model's scaled space.
        trainable_analytic_layer
            Optional trainable analytic layer to use in place of
            ``self.trainable_analytic_layer``. Required when
            ``config["trainable"]`` is ``True`` and the layer is passed
            externally (e.g. during SVI).

        Returns
        -------
        potential
            Scalar for ``(3,)`` input; shape ``(N,)`` for batched input.

        """
        scalar_input = x_cart.ndim == 1
        if scalar_input:
            x_cart = x_cart[None, :]  # promote to (1, 3)

        # --- Coordinate transformation ---
        x_in = self.input_encoder(x_cart)

        # --- Neural network potential ---
        u_nn = self.nn_potential(x_in)

        # --- Analytic baseline potential ---

        analytic_potential_scaled = 0.0
        r_s_learned = self.config["r_s"]

        if self.config["include_analytic"]:
            # Transform to physical coordinates
            x_phys = self.config["x_transformer"].inverse_transform(x_cart)
            u_phys = 0.0

            if not self.config["trainable"]:
                u_phys = self.ab_potential.value.potential(x_phys, t=0)
            elif self.config["trainable"]:
                layer = (
                    trainable_analytic_layer
                    if trainable_analytic_layer is not None
                    else self.trainable_analytic_layer
                )
                if layer is None:
                    raise ValueError(
                        "config['trainable']=True but no "
                        "trainable_analytic_layer was provided."
                    )
                u_phys = layer(x_phys)
                r_s_learned = layer.r_s

            # Transform potential to scaled units
            analytic_potential_scaled = self.config["u_transformer"].transform(u_phys)

        # --- Combine potentials ---
        scaled_nn_potential = self.scale_layer(x_cart, u_nn, r_s=r_s_learned)
        fused_potential = scaled_nn_potential + analytic_potential_scaled
        boundary_potential = self.fuse_boundary_layer(
            x_cart, scaled_nn_potential, analytic_potential_scaled
        )

        # --- Select final potential based on config ---
        if self.config["enforce_boundary"]:
            potential = boundary_potential
        elif self.config["include_analytic"]:
            potential = fused_potential
        else:
            potential = scaled_nn_potential

        return potential[0] if scalar_input else potential

    def compute_potential(
        self,
        x_cart: Array,
        *,
        trainable_analytic_layer: TrainableGalaxPotential | None = None,
    ) -> Array:
        """Evaluate the model potential at Cartesian position(s).

        Parameters
        ----------
        x_cart
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
        return self(x_cart, trainable_analytic_layer=trainable_analytic_layer).squeeze()

    def compute_laplacian(self, x_cart: Array) -> Array:
        """Compute the Laplacian of the potential.

        Parameters
        ----------
        x_cart
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

        def laplacian_single(x: Array) -> Array:
            hess = jax.hessian(self.compute_potential)(x)  # (3, 3)
            return jnp.trace(hess)

        return jax.vmap(laplacian_single)(x_cart)

    def acceleration(
        self,
        x_cart: Array,
        /,
        trainable_analytic_layer: TrainableGalaxPotential | None = None,
    ) -> Array:
        """Compute the gravitational acceleration at the given positions.

        Parameters
        ----------
        x_cart
            Cartesian inputs, shape ``(N, 3)``. Assumed to be in the model's
            scaled space.
        trainable_analytic_layer
            Optional trainable analytic layer to use in place of
            ``self.trainable_analytic_layer``.

        Returns
        -------
        acceleration
            Acceleration vectors, shape ``(N, 3)``.

        """
        grads = jax.vmap(jax.grad(self), in_axes=(0, None))(
            x_cart, trainable_analytic_layer
        )
        return -grads

    def potential_acceleration(
        self,
        x_cart: Array,
        /,
        trainable_analytic_layer: TrainableGalaxPotential | None = None,
    ) -> tuple[Array, Array]:
        """Compute the potential and acceleration jointly via ``value_and_grad``.

        More efficient than calling :meth:`__call__` and :meth:`acceleration`
        separately because the forward pass is only executed once.

        Parameters
        ----------
        x_cart
            Cartesian inputs, shape ``(N, 3)``. Assumed to be in the model's
            scaled space.
        trainable_analytic_layer
            Optional trainable analytic layer to use in place of
            ``self.trainable_analytic_layer``.

        Returns
        -------
        potential
            Potential values, shape ``(N,)``.
        acceleration
            Acceleration vectors, shape ``(N, 3)``.

        """
        vgfn = jax.vmap(jax.value_and_grad(self), in_axes=(0, None))
        potential_vals, grads = vgfn(x_cart, trainable_analytic_layer)
        return potential_vals, -grads
