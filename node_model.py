import jax
import jax.numpy as jnp
from flax import linen as nn
import diffrax as dfx
from functools import partial
from collections.abc import Callable
from layers import (
    CartesianToModifiedSphericalLayer,
    ScaleNNPotentialLayer,
    FuseModelsLayer,
    AnalyticModelLayerTime,
    SmoothMLP
)

def compute_delta_phi_per_point(tx_sph, apply_fn, t0, tf=None):
    def single_integrator(tx_sph):
        delta_phi0 = jnp.zeros((1, 1))
        tf = tx_sph[0]

        def ode_fn(t, phi, _):
            x_with_t = tx_sph.at[0].set(t)
            return apply_fn(x_with_t[None, :])  # shape (1, 1)

        term = dfx.ODETerm(ode_fn)
        solver = dfx.Heun()
        sol = dfx.diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=tf,
            dt0=(tf - t0) / 20,
            y0=delta_phi0,
            args=None,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-7),
            adjoint=dfx.DirectAdjoint(),
        )
        return sol.ys[0]

    return jax.vmap(single_integrator)(tx_sph).squeeze(-1)


def compute_delta_phi_batch(tx_sph, apply_fn, t0, nsteps=64):
    batch_size = tx_sph.shape[0]
    t_end = tx_sph[0, 0]

    def ode_fn(t, phi, _):
        tx_sph_t = tx_sph.at[:, 0].set(t)
        dphi_dt = apply_fn(tx_sph_t).reshape(batch_size, 1)
        return dphi_dt

    y0 = jnp.zeros((batch_size, 1))

    term = dfx.ODETerm(ode_fn)
    solver = dfx.Heun()
    sol = dfx.diffeqsolve(
        terms=term,
        solver=solver,
        t0=t0,
        t1=t_end,
        dt0=(-1.0 / nsteps),
        y0=y0,
        saveat=dfx.SaveAt(t1=True),
        adjoint=dfx.DirectAdjoint(),
    )
    return sol.ys.squeeze(-1)  # shape: (batch_size,)


def compute_delta_phi_per_point_gl3(tx_sph, apply_fn, t0):
    # 3-point Gauss–Legendre on [t0, tf]
    nodes = jnp.array([-jnp.sqrt(3 / 5), 0.0, jnp.sqrt(3 / 5)])
    weights = jnp.array([5 / 9, 8 / 9, 5 / 9])

    def one(row):
        tf = row[0]
        a = (tf - t0) / 2.0
        b = (tf + t0) / 2.0
        ts = a * nodes + b

        def eval_at(t):
            r = row.at[0].set(t)
            return apply_fn(r[None, :]).squeeze()

        vals = jax.vmap(eval_at)(ts)
        return a * jnp.sum(weights * vals)

    return jax.vmap(one)(tx_sph)


def compute_delta_phi_per_point_gl3panels(tx_sph, apply_fn, t0, M=4):
    tx_sph = jnp.atleast_2d(tx_sph)
    nodes = jnp.array([-jnp.sqrt(3 / 5), 0.0, jnp.sqrt(3 / 5)])
    weights = jnp.array([5 / 9, 8 / 9, 5 / 9])

    def one(row):
        tf = row[0]
        L = (
            tf - t0
        )
        s_edges = jnp.linspace(0.0, 1.0, M + 1)

        def panel(i):
            a, b = s_edges[i], s_edges[i + 1]
            c = (b - a) / 2.0
            d = (b + a) / 2.0
            s = c * nodes + d  # (3,)
            t = t0 + s * L  # (3,)

            def eval_at(ti):
                rw = row.at[0].set(ti)
                return apply_fn(rw[None, :]).squeeze()  # scalar

            vals = jax.vmap(eval_at)(t)  # (3,)
            return L * c * jnp.sum(weights * vals)

        return jnp.sum(jax.vmap(panel)(jnp.arange(M)))

    return jax.vmap(one)(tx_sph)


class ModelWithAnalytic(nn.Module):
    config: dict
    depth: int = 4

    def setup(self):
        act = self.config.get("activation", "softplus")
        self.delta_phi_net = SmoothMLP(
            depth=self.config.get("delta_phi_depth", 4),
            width=self.config.get("delta_phi_width", 128),
            act=act,
        )
        self.intial_correction_net = SmoothMLP(
            depth=self.config.get("intial_correction_depth", 4),
            width=self.config.get("intial_correction_width", 128),
            act=act,
        )

    def compute_potential(self, cart_x):
        potential = self.apply(
            {"params": self.variables["params"]}, cart_x, mode="potential"
        )["potential"]
        return potential

    def compute_acceleration(self, tx_cart):
        def potential_fn(x):
            pot = self.compute_potential(x).squeeze()
            return pot
        acceleration_fn = jax.vmap(jax.grad(potential_fn))  # Batched gradient
        acceleration = -acceleration_fn(tx_cart)
        return acceleration[:, 1:4]  # Exclude time component

    @nn.compact
    def __call__(self, tx_cart, mode="full"):
        outputs = {}
        t0 = 0.0  # Initial time for ODE integration

        cart_to_sph_layer = CartesianToModifiedSphericalLayer(
            clip=self.config.get("clip", 1.0)
        )
        scale_layer = ScaleNNPotentialLayer(config=self.config)
        fuse_layer = FuseModelsLayer()
        analytic_layer = AnalyticModelLayerTime(config=self.config)

        if tx_cart.ndim == 1:
            tx_cart = tx_cart[None, :]

        x_cart = tx_cart[:, 1:4]
        t = tx_cart[:, :1]
        x_sph = cart_to_sph_layer(x_cart)

        tx_sph = jnp.concatenate([t, x_sph], axis=1)
        initial_correction = self.intial_correction_net(x_sph)
        outputs["initial_correction"] = initial_correction

        _ = self.delta_phi_net(tx_sph)
        delta_phi_params = self.variables["params"]["delta_phi_net"]
        apply_fn = lambda tx: self.delta_phi_net.apply({"params": delta_phi_params}, tx)

        integration_method = self.config.get("integration_mode", "gl3")
        if integration_method == "diffrax_batch":
            delta_phi = compute_delta_phi_batch(tx_sph, apply_fn, t0)
        elif integration_method == "diffrax_per_point":
            delta_phi = compute_delta_phi_per_point(
                tx_sph, apply_fn, t0, tf=None
            ).squeeze(-1)
        elif integration_method == "gl3":
            delta_phi = compute_delta_phi_per_point_gl3(tx_sph, apply_fn, t0)

        total_correction = initial_correction + delta_phi

        outputs["delta_phi"] = delta_phi

        scaled_potential = scale_layer(x_cart, total_correction)
        outputs["scale_nn_potential"] = scaled_potential

        # Fuse Models (combine analytic and NN output)
        analytic_potential = analytic_layer(tx_cart)
        outputs["analytic_model_layer"] = analytic_potential
        fused_potential = fuse_layer(scaled_potential, analytic_potential)
        outputs["fuse_models"] = fused_potential
        outputs["fuse_and_bound"] = fused_potential

        if self.config["include_analytic"]:
            potential = fused_potential
        else:
            potential = scaled_potential

        outputs["final"] = potential

        if mode == "potential":
            return {"potential": potential}
        acceleration = self.compute_acceleration(tx_cart)

        return {
            "acceleration": acceleration,
            "potential": potential,
            "outputs": outputs,
        }


@jax.jit
def train_step(state, tx_cart, a_true, lambda_rel=1.0):
    def loss_fn(params):
        outputs = state.apply_fn({"params": params}, tx_cart)
        a_pred = outputs["acceleration"]

        diff = a_pred - a_true
        diff_norm = jnp.linalg.norm(diff, axis=1, keepdims=True)
        a_true_norm = jnp.linalg.norm(a_true, axis=1, keepdims=True)
        return jnp.mean(diff_norm + lambda_rel * (diff_norm / a_true_norm))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    state = state.apply_gradients(grads=grads)
    return state, loss


@partial(jax.jit, static_argnames=("conserve_mass", "lambda_mass"))
def train_step_mass(state, tx_cart, a_true, conserve_mass=False, lambda_mass=1.0):
    def loss_fn(params):
        outputs = state.apply_fn({"params": params}, tx_cart, mode="mass")
        a_pred = outputs["acceleration"]
        enclosed_mass = outputs["enclosed_mass"]

        diff = a_pred - a_true
        diff_norm = jnp.linalg.norm(diff, axis=1)
        a_true_norm = jnp.linalg.norm(a_true, axis=1)
        base_loss = jnp.mean(diff_norm + diff_norm / (a_true_norm + 1e-8))
        mass_mean = jnp.mean(enclosed_mass)
        mass_std = jnp.std(enclosed_mass)
        mass_var_loss = lambda_mass * mass_std

        total = base_loss + (mass_var_loss if conserve_mass else 0.0)

        aux = {
            "base_loss": base_loss,
            "mass_var_loss": mass_var_loss,
            "total_loss": total,
            "mass_mean": mass_mean,
            "mass_std": mass_std,
        }
        return total, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, aux
