import jax
import jax.numpy as jnp
from flax import linen as nn
from astropy import constants as const
import numpy as np

from ..layers import (
    CartesianToModifiedSphericalLayer,
    AnalyticModelLayer,
    ScaleNNPotentialLayer,
    FuseModelsLayer,
    SmoothMLP,
    FuseandBoundary,
    TrainableGalaxPotential
)


class StaticModel(nn.Module):
    """A Flax module for a static gravitational potential model.

    This class defines a flexible model for a gravitational potential, which can
    be a pure neural network, a known analytic potential, or a hybrid of the two.
    It computes the potential and its derivatives (acceleration, Laplacian)
    and is designed to be highly configurable through a dictionary.

    The model can enforce physical boundary conditions by smoothly transitioning
    to an analytic form at large radii.

    Attributes:
        config (dict): A dictionary containing the configuration for the model's
            architecture and behavior. Key options include:
            - "enforce_bc" (bool): If True, use the `FuseandBoundary` layer to
              enforce the analytic potential as a boundary condition.
            - "include_analytic" (bool): If True, add the analytic potential
              to the NN potential.
            - "trainable" (bool): If True, use a `TrainableGalaxPotential` layer
              for the analytic component.
            - "analytic_only" (bool): If True, the model output is solely the
              analytic potential.
            - "x_transformer", "u_transformer", "a_transformer": Objects for
              transforming coordinates and potential between physical and scaled units.
            - "depth", "width", "activation": Hyperparameters for the `SmoothMLP`.
            - Configuration for sub-layers like `FuseandBoundary` (`r_trans`,
              `k_smooth`, etc.).
        depth (int): The number of hidden layers in the `SmoothMLP`.
        trainable_analytic_layer (TrainableGalaxPotential, optional): An instance
            of a trainable analytic potential layer, passed if `config['trainable']` is True.

    """
    config: dict
    depth: int = 4
    trainable_analytic_layer: TrainableGalaxPotential = None

    def setup(self):
        self.log_lambda = self.config.get("init_log_lambda", 0.0)

    def compute_potential(self, cart_x):
        potential = self.apply(
            {"params": self.variables["params"]}, cart_x, mode="potential"
        )["potential"]
        return potential

    def compute_acceleration(self, cart_x):
        def potential_fn(x):
            pot = self.compute_potential(x).squeeze()
            return pot

        acceleration_fn = jax.vmap(jax.grad(potential_fn))
        acceleration = -acceleration_fn(cart_x)
        return acceleration

    def compute_laplacian(self, cart_x):
        def potential_fn(x):
            pot = self.compute_potential(x).squeeze()  # scalar output
            return pot

        def laplacian_single(x):
            hess = jax.hessian(potential_fn)(x)
            laplacian = jnp.trace(hess)
            return laplacian

        laplacian_fn = jax.vmap(laplacian_single)
        laplacian = laplacian_fn(cart_x)

        return laplacian

    def compute_mass(self):
        G_val = const.G.to("kpc^3/Msun/Myr^2").value  # physical G
        eval_pts = self.config["eval_pts"]

        R_scaled = jnp.linalg.norm(eval_pts, axis=1, keepdims=True)  # shape (n_dirs, 1)
        R_phys = self.config["x_transformer"].inverse_transform(
            R_scaled)  # physical radius
        dirs = eval_pts / R_scaled  # normalize directions

        acc = self.compute_acceleration(eval_pts)
        acc_phys = self.config["a_transformer"].inverse_transform(acc)
        a_r = jnp.sum(acc_phys * dirs, axis=1)

        flux = 4.0 * jnp.pi * (R_phys**2) * jnp.mean(a_r)
        pred_mass = -flux / (4 * np.pi * G_val)

        return pred_mass

    @nn.compact
    def __call__(self, cart_x, mode="full"):
        outputs = {}
        if self.trainable_analytic_layer is not None:
            # stub parameter ensures that the `trainable_analytic_layer` is
            # correctly registered within the Flax parameter structure, even if it
            # contains no trainable parameters itself. This prevents the layer
            # from being pruned during model initialization.
            _stub = self.param("_stub", lambda rng: jnp.array(0.0))

        cart_to_sph_layer = CartesianToModifiedSphericalLayer(
            clip=self.config.get("clip", 1.0)
        )

        scale_layer = ScaleNNPotentialLayer(config=self.config)
        fuse_layer = FuseModelsLayer()
        analytic_layer = AnalyticModelLayer(config=self.config, mode = "static")

        if self.config.get("convert_to_spherical", True):
            x = cart_to_sph_layer(cart_x)
        else:
            x = cart_x

        depth = self.config.get("depth", 4)
        width = self.config.get("width", 128)
        act = self.config.get("activation", "softplus")
        u_nn = SmoothMLP(width=width, depth=depth, act=act)(x)
        outputs["u_nn"] = u_nn


        # Fuse models (combine analytic and NN outputs)
        if self.config.get("trainable", False):
            phys_x = self.config["x_transformer"].inverse_transform(cart_x)
            phys_analytic_potential, r_s_learned = self.trainable_analytic_layer(phys_x)
            analytic_potential = self.config["u_transformer"].transform(phys_analytic_potential)

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
        else:
            return {
                "acceleration": acceleration,
                "potential": potential,
                "outputs": outputs,
            }
