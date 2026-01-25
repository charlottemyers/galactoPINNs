from typing import Callable, Literal
import jax
import jax.numpy as jnp
from flax import linen as nn
import diffrax as dfx

from ..layers import (
    CartesianToModifiedSphericalLayer,
    ScaleNNPotentialLayer,
    FuseModelsLayer,
    AnalyticModelLayer,
    SmoothMLP,
)

__all__ = (
    "compute_delta_phi_per_point",
    "compute_delta_phi_batch",
    "compute_delta_phi_per_point_gl3",
    "compute_delta_phi_per_point_gl3panels",
    "NODEModel",
)


# ----------------------------
# Delta-phi integration helpers
# ----------------------------

def compute_delta_phi_per_point(
    tx_sph: jnp.ndarray,
    apply_fn: Callable[[jnp.ndarray], jnp.ndarray],
    t0: float,
    tf: float | None = None,
) -> jnp.ndarray:
    """
    Compute delta_phi for each sample by solving an ODE independently per row.

    Solves:
        dphi/dt = f(t, x)  where f = apply_fn([t, x_sph...])

    Parameters
    ----------
    tx_sph
        Array of shape (N, D) or (D,). Column 0 is the final time for each row unless
        `tf` is provided. Remaining columns are spatial features.
    apply_fn
        Callable mapping an input of shape (N, D) or (1, D) to a scalar derivative per row.
        Expected output is broadcastable to (N, 1) or (N,).
    t0
        Integration start time.
    tf
        Optional global integration end time. If None, per-row end time is tx_sph[i, 0].

    Returns
    -------
    delta_phi
        Array of shape (N,) containing delta_phi at the end time.
    """
    tx_sph = jnp.atleast_2d(tx_sph)

    def single_integrator(row: jnp.ndarray) -> jnp.ndarray:
        # Initial condition delta_phi(t0) = 0
        y0 = jnp.zeros((1, 1))
        t1 = row[0] if tf is None else tf

        def ode_fn(t, phi, _):
            # Replace the time coordinate in the row at evaluation time t.
            row_t = row.at[0].set(t)
            return apply_fn(row_t[None, :]).reshape(1, 1)

        term = dfx.ODETerm(ode_fn)
        solver = dfx.Heun()

        # Provide an initial step guess.
        dt0 = (t1 - t0) / 20.0

        sol = dfx.diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=None,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-7),
            adjoint=dfx.DirectAdjoint(),
        )
        return sol.ys[0, 0, 0]

    return jax.vmap(single_integrator)(tx_sph)


def compute_delta_phi_batch(
    tx_sph: jnp.ndarray,
    apply_fn: Callable[[jnp.ndarray], jnp.ndarray],
    t0: float,
    nsteps: int = 64,
) -> jnp.ndarray:
    """
    Compute delta_phi for a batch by solving one vector ODE system.
    Assumes all samples share the same integration interval end time (taken from tx_sph[0, 0]).

    Parameters
    ----------
    tx_sph
        Array of shape (N, D). Column 0 is time.
        Assumes tx_sph[:, 0] are all identical
    apply_fn
        Callable mapping (N, D) to per-row derivatives.
    t0
        Integration start time.
    nsteps
        Used only to set an initial dt0.

    Returns
    -------
    delta_phi
        Array of shape (N,) with the integrated delta_phi at t_end.
    """
    tx_sph = jnp.atleast_2d(tx_sph)
    batch_size = tx_sph.shape[0]
    t_end = tx_sph[0, 0]

    def ode_fn(t, y, _):
        # Overwrite time column for entire batch at time t.
        tx_sph_t = tx_sph.at[:, 0].set(t)
        dphi_dt = apply_fn(tx_sph_t).reshape(batch_size, 1)
        return dphi_dt

    y0 = jnp.zeros((batch_size, 1))
    term = dfx.ODETerm(ode_fn)
    solver = dfx.Heun()

    # Sign-correct dt0 supports forward or backward integration.
    dt0 = (t_end - t0) / float(nsteps)

    sol = dfx.diffeqsolve(
        terms=term,
        solver=solver,
        t0=t0,
        t1=t_end,
        dt0=dt0,
        y0=y0,
        saveat=dfx.SaveAt(t1=True),
        adjoint=dfx.DirectAdjoint(),
    )

    return sol.ys[0, :, 0]


def compute_delta_phi_per_point_gl3(
    tx_sph: jnp.ndarray,
    apply_fn: Callable[[jnp.ndarray], jnp.ndarray],
    t0: float,
) -> jnp.ndarray:
    """
    Approximate delta_phi using single-panel 3-point Gauss–Legendre quadrature per row.

    Computes:
        int_{t0}^{tf} f(t, x) dt
    with tf = row[0].

    Parameters
    ----------
    tx_sph
        Array of shape (N, D) (or (D,) which will be promoted). Column 0 is tf.
    apply_fn
        Callable mapping (1, D) -> scalar derivative (or shape compatible with scalar).
    t0
        Lower integration limit.

    Returns
    -------
    delta_phi
        Array of shape (N,) quadrature approximation per row.
    """
    tx_sph = jnp.atleast_2d(tx_sph)

    # GL3 nodes/weights on [-1, 1]
    nodes = jnp.array([-jnp.sqrt(3.0 / 5.0), 0.0, jnp.sqrt(3.0 / 5.0)])
    weights = jnp.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    def one(row: jnp.ndarray) -> jnp.ndarray:
        tf = row[0]
        # Map [-1, 1] -> [t0, tf]
        a = (tf - t0) / 2.0
        b = (tf + t0) / 2.0
        ts = a * nodes + b  # (3,)

        def eval_at(t):
            row_t = row.at[0].set(t)
            return apply_fn(row_t[None, :]).squeeze()

        vals = jax.vmap(eval_at)(ts)  # (3,)
        return a * jnp.sum(weights * vals)

    return jax.vmap(one)(tx_sph)


def compute_delta_phi_per_point_gl3panels(
    tx_sph: jnp.ndarray,
    apply_fn: Callable[[jnp.ndarray], jnp.ndarray],
    t0: float,
    M: int = 4,
) -> jnp.ndarray:
    """
    Composite GL3 quadrature over M panels; can improve accuracy over single-panel
    GL3 when the interval is long or f(t, x) varies rapidly.

    Parameters
    ----------
    tx_sph
        Array of shape (N, D) (or (D,), promoted). Column 0 is tf.
    apply_fn
        Callable mapping (1, D) -> scalar derivative.
    t0
        Lower integration limit.
    M
        Number of sub-intervals.

    Returns
    -------
    delta_phi
        Array of shape (N,) composite quadrature approximation per row.
    """
    tx_sph = jnp.atleast_2d(tx_sph)

    nodes = jnp.array([-jnp.sqrt(3.0 / 5.0), 0.0, jnp.sqrt(3.0 / 5.0)])
    weights = jnp.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    def one(row: jnp.ndarray) -> jnp.ndarray:
        tf = row[0]
        L = tf - t0

        # Normalized panel edges in s \in [0, 1]
        s_edges = jnp.linspace(0.0, 1.0, M + 1)

        def panel(i):
            a, b = s_edges[i], s_edges[i + 1]
            c = (b - a) / 2.0
            d = (b + a) / 2.0
            s = c * nodes + d        # (3,)
            t = t0 + s * L           # (3,)

            def eval_at(ti):
                row_t = row.at[0].set(ti)
                return apply_fn(row_t[None, :]).squeeze()

            vals = jax.vmap(eval_at)(t)  # (3,)
            return L * c * jnp.sum(weights * vals)

        return jnp.sum(jax.vmap(panel)(jnp.arange(M)))

    return jax.vmap(one)(tx_sph)



# ----------------------------
# NODE Model
# ----------------------------

class NODEModel(nn.Module):
    """
    Flax module implementing a time-dependent potential correction via integration.

    Parameters
    ----------
    config : dict
        Configuration dict used by layers and integration routing.
        Expected keys include:
          - "activation" (callable, i.e. jax.nn.tanh): activation for MLPs
          - "delta_phi_depth", "delta_phi_width" (optional)
          - "initial_correction_depth", "initial_correction_width" (optional)
          - "integration_mode" (optional): {"gl3", "diffrax_batch", "diffrax_per_point"}
          - "include_analytic" (bool)
          - plus transformer and analytic-function keys used by ScaleNNPotentialLayer
            and AnalyticModelLayer.
    depth : int
        Kept for backwards-compatibility; primary depths are taken from config.

    Notes
    -----
    Input: tx_cart with shape (N, 4) containing [t, x, y, z] in scaled coords.
    Output: dict with "potential" (N,) and "acceleration" (N, 3).
    """

    config: dict
    depth: int = 4

    def setup(self):
        act = self.config.get("activation", None)
        gelu_approx = self.config.get("gelu_approximate", False)

        if act is not None and not callable(act):
            raise TypeError(
                f"config['activation'] must be a callable (e.g. jax.nn.softplus), "
                f"got {type(act)!r}"
            )

        mlp_common = dict(gelu_approximate=gelu_approx)
        if act is not None:
            mlp_common["act"] = act

        self.delta_phi_net = SmoothMLP(
            depth=self.config.get("delta_phi_depth", 4),
            width=self.config.get("delta_phi_width", 128),
            **mlp_common,
        )

        self.initial_correction_net = SmoothMLP(
            depth=self.config.get("initial_correction_depth", 4),
            width=self.config.get("initial_correction_width", 128),
            **mlp_common,
        )


    def compute_potential(self, tx_cart: jnp.ndarray) -> jnp.ndarray:
        """
        Convenience wrapper that returns only the potential.

        Parameters
        ----------
        tx_cart : array, shape (N, 4) or (4,)
            Time+position input.

        Returns
        -------
        potential : array, shape (N,) or ()
        """
        return self.apply({"params": self.variables["params"]}, tx_cart, mode="potential")["potential"]

    def compute_acceleration(self, tx_cart: jnp.ndarray) -> jnp.ndarray:
        """
        Compute acceleration as -grad(Phi) by differentiating the model potential.
        The gradient is taken w.r.t. the full input [t, x, y, z], then the spatial
        components (x, y, z) are returned and the time derivative is discarded.

        Parameters
        ----------
        tx_cart : array, shape (N, 4) or (4,)

        Returns
        -------
        acceleration : array, shape (N, 3)
        """
        tx_cart = jnp.atleast_2d(tx_cart)

        def potential_fn(single_tx):
            return self.compute_potential(single_tx).squeeze()

        grad_fn = jax.vmap(jax.grad(potential_fn))
        grad_tx = grad_fn(tx_cart)          # (N, 4)
        accel = (-grad_tx)[:, 1:4]          # (N, 3)
        return accel

    @nn.compact
    def __call__(self, tx_cart: jnp.ndarray, mode: Literal["full", "potential"] = "full") -> dict:
        """
        Forward pass.

        Parameters
        ----------
        tx_cart : array, shape (N, 4) or (4,)
            Time+position input: [t, x, y, z].
        mode : {"full", "potential"}
            If "potential", returns {"potential": ...} only.

        Returns
        -------
        out : dict
            If mode == "potential":
                {"potential": potential}
            Else:
                {"acceleration": acceleration, "potential": potential, "outputs": outputs}
        """
        outputs: dict = {}
        t0 = 0.0  # Integration start time

        # Layers
        cart_to_sph_layer = CartesianToModifiedSphericalLayer(clip=self.config.get("clip", 1.0))
        scale_layer = ScaleNNPotentialLayer(config=self.config)
        fuse_layer = FuseModelsLayer()
        analytic_layer = AnalyticModelLayer(config=self.config, mode = "time")

        tx_cart = jnp.atleast_2d(tx_cart)

        # Split time and space (as Einstein recoils)
        x_cart = tx_cart[:, 1:4]     # (N, 3)
        t = tx_cart[:, :1]           # (N, 1)

        # Convert to modified spherical features
        x_sph = cart_to_sph_layer(x_cart)

        # Build [t, sph_features...] for delta_phi network
        tx_sph = jnp.concatenate([t, x_sph], axis=1)

        # Initial (spatial) correction term
        initial_correction = self.initial_correction_net(x_sph)  # (N,)
        outputs["initial_correction"] = initial_correction

        # Fix delta_phi_net params and define an apply_fn for integration/quadrature.
        _ = self.delta_phi_net(tx_sph)  # ensures parameter collection exists
        delta_phi_params = self.variables["params"]["delta_phi_net"]
        apply_fn = lambda z: self.delta_phi_net.apply({"params": delta_phi_params}, z)

        # Integrate delta_phi
        method = self.config.get("integration_mode", "gl3")
        if method == "diffrax_batch":
            delta_phi = compute_delta_phi_batch(tx_sph, apply_fn, t0)
        elif method == "diffrax_per_point":
            delta_phi = compute_delta_phi_per_point(tx_sph, apply_fn, t0, tf=None)
        elif method == "gl3":
            delta_phi = compute_delta_phi_per_point_gl3(tx_sph, apply_fn, t0)
        else:
            raise ValueError(
                f"Unknown integration_mode='{method}'. "
                "Use one of {'gl3', 'diffrax_batch', 'diffrax_per_point'}."
            )

        outputs["delta_phi"] = delta_phi

        # Total learned correction
        total_correction = initial_correction + delta_phi

        # Convert correction to a potential term (radial scaling, etc.)
        scaled_potential = scale_layer(x_cart, total_correction)
        outputs["scale_nn_potential"] = scaled_potential

        # Analytic potential term
        analytic_potential = analytic_layer(tx_cart)
        outputs["analytic_model_layer"] = analytic_potential

        # Fuse (NN + analytic)
        fused_potential = fuse_layer(scaled_potential, analytic_potential)
        outputs["fuse_models"] = fused_potential

        # Choose final output
        include_analytic = self.config.get("include_analytic", True)
        potential = fused_potential if include_analytic else scaled_potential
        outputs["final"] = potential

        if mode == "potential":
            return {"potential": potential}

        # Acceleration via gradient
        acceleration = self.compute_acceleration(tx_cart)

        return {
            "acceleration": acceleration,
            "potential": potential,
            "outputs": outputs,
        }
