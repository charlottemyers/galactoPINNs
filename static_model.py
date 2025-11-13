import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial
from flax.training.train_state import TrainState
from collections.abc import Callable
from astropy import constants as const
import numpy as np
from galactic.trainable_galax_potential import TrainableGalaxPotential

from layers import (
    CartesianToModifiedSphericalLayer,
    AnalyticModelLayer,
    ScaleNNPotentialLayer,
    FuseModelsLayer, BoundaryConditionLayer, FuseandBoundary,
    SmoothMLP)


class Model_with_analytic(nn.Module):
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
            pot = self.compute_potential(x).squeeze()
            return pot

        def laplacian_single(x):
            hess = jax.hessian(potential_fn)(x)
            laplacian = jnp.trace(hess)
            return laplacian
        laplacian_fn = jax.vmap(laplacian_single)
        laplacian = laplacian_fn(cart_x)

        return laplacian

    def compute_mass(self):
        G_val = const.G.to("kpc^3/Msun/Myr^2").value
        eval_pts = self.config["eval_pts"]
        n_dirs = len(eval_pts)

        R_scaled = jnp.linalg.norm(eval_pts, axis=1, keepdims=True)
        R_phys = self.config["x_transformer"].inverse_transform(
            R_scaled)
        dirs = eval_pts / R_scaled

        acc = self.compute_acceleration(eval_pts)
        acc_phys = self.config["a_transformer"].inverse_transform(acc)
        a_r = jnp.sum(acc_phys * dirs, axis=1)

        flux = 4.0 * jnp.pi * (R_phys**2) * jnp.mean(a_r)
        pred_mass = -flux / (4 * np.pi * G_val)

        return pred_mass

    @nn.compact
    def __call__(self, cart_x, mode="full"):
        outputs = {}
        cart_to_sph_layer = CartesianToModifiedSphericalLayer(
            clip=self.config.get("clip", 1.0)
        )

        scale_layer = ScaleNNPotentialLayer(config=self.config)
        fuse_layer = FuseModelsLayer()
        analytic_layer = AnalyticModelLayer(config=self.config)

        if self.config.get("convert_to_spherical", True):
            x = cart_to_sph_layer(cart_x)
        else:
            x = cart_x

        depth = self.config.get("depth", 4)
        width = self.config.get("width", 128)
        act = self.config.get("activation", "softplus")
        if self.config.get("nn_off", False):
            u_nn = 0.0
        else:
            u_nn = SmoothMLP(width=width, depth=depth, act=act)(x)
        outputs["u_nn"] = u_nn

        # Fuse Models (combine analytic and NN outputs
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
        elif self.config.get("weighted_fuse_lambda", None) is not None:
            weighted_lambda = self.config["weighted_fuse_lambda"]
            fused_potential = weighted_lambda* scaled_potential + (1-weighted_lambda)* analytic_potential
        else:
            fused_potential = fuse_layer(scaled_potential, analytic_potential)
        outputs["fuse_models"] = fused_potential

        fuse_and_bound = FuseandBoundary(config=self.config)(
            cart_x, scaled_potential, analytic_potential
        )
        fuse_and_bound_potential = fuse_and_bound["fused_potential"]

        outputs["h"] = fuse_and_bound["h"]
        outputs["g"] = fuse_and_bound["g"]

        if self.config["enforce_bc"]:
            if self.config["include_analytic"]:
                boundary_potential = fuse_and_bound_potential
            else:
                boundary_potential = BoundaryConditionLayer(config=self.config)(
                    cart_x, scaled_potential, analytic_potential
                )

            outputs["enforce_bc"] = boundary_potential
            potential = boundary_potential

        elif self.config["include_analytic"]:
            potential = fused_potential
        elif self.config.get("include_analytic_2", False):
            potential = scale_layer(cart_x, u_nn + analytic_potential)
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

@partial(jax.jit, static_argnames=("i_large", "mode", "target", "metric"))
def train_step(
    state,
    x,
    a_true,
    target="acceleration",
    lap_true=None,
    lambda_rel=1.0,
    metric=False,
):
    def acc_loss_fn(params, lambda_rel):
        a_pred = state.apply_fn({"params": params}, x)["acceleration"]
        diff = a_pred - a_true
        diff_norm = jnp.linalg.norm(diff, axis=1, keepdims=True)
        a_true_norm = jnp.linalg.norm(a_true, axis=1, keepdims=True)
        return jnp.mean(diff_norm + lambda_rel * (diff_norm / a_true_norm))

    def poisson_loss_fn(params):
        laplacian = state.apply_fn({"params": params}, x, mode="density")["laplacian"]
        rhs = lap_true
        diff = jnp.abs(laplacian - rhs)
        return jnp.mean(diff + (diff / jnp.abs(rhs)))

    # Compute loss and gradients
    if target == "acceleration":
        loss, grads = jax.value_and_grad(acc_loss_fn)(
            state.params, lambda_rel=lambda_rel
        )
    elif target == "density":
        loss, grads = jax.value_and_grad(poisson_loss_fn)(state.params)

    if metric:
        state = state.apply_gradients(grads=grads, value=loss)
    else:
        state = state.apply_gradients(grads=grads)
    return state, loss
