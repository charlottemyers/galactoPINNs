"""Neural ODE model implementations."""

__all__ = (
    "NODEModel",
    "compute_delta_phi_batch",
    "compute_delta_phi_per_point",
    "compute_delta_phi_per_point_gl3",
    "compute_delta_phi_per_point_gl3panels",
)

from collections.abc import Callable
from typing import Any, Literal, Protocol

import diffrax as dfx
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
)

# --- Typing for the analytic potential ---
try:
    from galax.potential import AbstractPotential as GalaxPotential
except Exception:  # noqa: BLE001
    class GalaxPotential(Protocol):
        def potential(self,    positions: Any, *, t: Any = ...) -> Any: ...
        def acceleration(self, positions: Any, *, t: Any = ...) -> Any: ...

# ----------------------------
# Delta-phi integration helpers
# ----------------------------


def compute_delta_phi_per_point(
    tx_sph: Array,
    apply_fn: Callable[[Array], Array],
    t0: float,
    tf: float | None = None,
) -> Array:
    """Compute delta_phi for each sample by solving an ODE independently per row.

    Solves:
        dphi/dt = f(t, x)  where f = apply_fn([t, x_sph...])

    Parameters
    ----------
    tx_sph
        Array of shape (N, D) or (D,). Column 0 is the final time for each row unless
        `tf` is provided. Remaining columns are spatial features.
    apply_fn
        Callable mapping an input of shape (N, D) or (1, D) to a scalar
        derivative per row. Expected output is broadcastable to (N, 1) or (N,).
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

    def single_integrator(row: Array) -> Array:
        # Initial condition delta_phi(t0) = 0
        y0 = jnp.zeros((1, 1))
        t1 = row[0] if tf is None else tf

        def ode_fn(t: float, *_: Any) -> Array:
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
    tx_sph: Array, apply_fn: Callable[[Array], Array], t0: float, nsteps: int = 64
) -> Array:
    """Compute delta_phi for a batch by solving one vector ODE system.

    Assumes all samples share the same integration interval end time (taken
    from tx_sph[0, 0]).

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

    def ode_fn(t: float, *_: Any) -> Array:
        # Overwrite time column for entire batch at time t.
        tx_sph_t = tx_sph.at[:, 0].set(t)
        return apply_fn(tx_sph_t).reshape(batch_size, 1)

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
    tx_sph: Array, apply_fn: Callable[[Array], Array], t0: float
) -> Array:
    """Approximate delta_phi using single-panel 3-point Gauss-Legendre quadrature.

    Computes per row:
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

    def one(row: Array) -> Array:
        tf = row[0]
        # Map [-1, 1] -> [t0, tf]
        a = (tf - t0) / 2.0
        b = (tf + t0) / 2.0
        ts = a * nodes + b  # (3,)

        def eval_at(t: float) -> Array:
            row_t = row.at[0].set(t)
            return apply_fn(row_t[None, :]).squeeze()

        vals = jax.vmap(eval_at)(ts)  # (3,)
        return a * jnp.sum(weights * vals)

    return jax.vmap(one)(tx_sph)


def compute_delta_phi_per_point_gl3panels(
    tx_sph: Array, apply_fn: Callable[[Array], Array], t0: float, M: int = 4
) -> Array:
    """Compute delta_phi using GL3 quadrature over M panels.

    Can improve accuracy over single-panel GL3 when the interval is long or f(t,
    x) varies rapidly.

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

    def one(row: Array) -> Array:
        tf = row[0]
        L = tf - t0

        # Normalized panel edges in s \in [0, 1]
        s_edges = jnp.linspace(0.0, 1.0, M + 1)

        def panel(i: int) -> Array:
            a, b = s_edges[i], s_edges[i + 1]
            c = (b - a) / 2.0
            d = (b + a) / 2.0
            s = c * nodes + d  # (3,)
            t = t0 + s * L  # (3,)

            def eval_at(ti: float) -> Array:
                row_t = row.at[0].set(ti)
                return apply_fn(row_t[None, :]).squeeze()

            vals = jax.vmap(eval_at)(t)  # (3,)
            return L * c * jnp.sum(weights * vals)

        return jnp.sum(jax.vmap(panel)(jnp.arange(M)))

    return jax.vmap(one)(tx_sph)


# ----------------------------
# NODE Model
# ----------------------------


class NODEModel(nnx.Module):
    """Flax NNX module implementing a time-dependent potential via integration.

    Parameters
    ----------
    config
        Configuration dict used by layers and integration routing.
        Expected keys include:
          - "activation" (callable, i.e. jax.nn.tanh): activation for MLPs
          - "delta_phi_depth", "delta_phi_width" (optional)
          - "initial_correction_depth", "initial_correction_width" (optional)
          - "integration_mode" (optional): {"gl3", "diffrax_batch", "diffrax_per_point"}
          - "include_analytic" (bool)
          - plus transformer and analytic-function keys used by ScaleNNPotentialLayer
            and AnalyticModelLayer.
    in_features
        The number of spatial input features. Typically 5 for modified spherical coordinates.
        The delta_phi network takes (in_features + 1) features for [t, sph_features...].
    rngs
        Random number generator state for parameter initialization.

    Notes
    -----
    Input: tx_cart with shape (N, 4) containing [t, x, y, z] in scaled coords.
    Output: dict with "potential" (N,) and "acceleration" (N, 3).

    """

    def __init__(
        self, config: dict[str, Any], in_features: int = 5, *, rngs: nnx.Rngs
    ) -> None:
        """Initialize the NODEModel layers."""
        self.config = config

        act = config.get("activation")

        if act is not None and not callable(act):
            msg = (
                f"config['activation'] must be a callable (e.g. jax.nn.softplus), "
                f"got {type(act)!r}"
            )
            raise TypeError(msg)

        mlp_common: dict[str, Any] = {"rngs": rngs}
        if act is not None:
            mlp_common["act"] = act

        # delta_phi_net takes [t, sph_features...] so has (in_features + 1) inputs
        self.delta_phi_net = SmoothMLP(
            in_features=in_features + 1,
            depth=config.get("delta_phi_depth", 4),
            width=config.get("delta_phi_width", 128),
            **mlp_common,
        )

        # initial_correction_net takes spatial features only
        self.initial_correction_net = SmoothMLP(
            in_features=in_features,
            depth=config.get("initial_correction_depth", 4),
            width=config.get("initial_correction_width", 128),
            **mlp_common,
        )

        # Initialize other layers
        self.cart_to_sph_layer = CartesianToModifiedSphericalLayer(
            clip=config.get("clip", 1.0)
        )
        self.scale_layer = ScaleNNPotentialLayer(config=config)
        self.fuse_layer = FuseModelsLayer()
        self.fuse_boundary_layer = FuseandBoundary(config=config)

    def compute_potential(self, tx_cart: Array) -> Array:
        """Compute the potential.

        Parameters
        ----------
        tx_cart : array, shape (N, 4) or (4,)
            Time+position input.

        Returns
        -------
        potential : array, shape (N,) or ()

        """
        return self(tx_cart, mode="potential")["potential"]

    def __call__(
        self, tx_cart: Array, mode: Literal["full", "potential"] = "full",
        analytic_potential: GalaxPotential | None = None,
    ) -> dict[str, Any]:
        """Forward pass.

        Parameters
        ----------
        tx_cart : array, shape (N, 4) or (4,)
            Time+position input: [t, x, y, z].
        mode
            If "potential", returns ``{"potential": ...}`` only.
            If "full", computes and returns potential and acceleration.
        analytic_potential : galax.potential.AbstractPotential, optional
            An external, non-trainable analytic potential object. This is only
            used if ``config["include_analytic"]`` is True.


        Returns
        -------
        out : dict
            If mode == "potential":
                {"potential": potential}
            Else:
                {"acceleration": acceleration, "potential": potential,
                 "outputs": outputs}

        """
        outputs: dict[str, Any] = {}
        t0 = 0.0  # Integration start time

        tx_cart = jnp.atleast_2d(tx_cart)

        # Split time and space (as Einstein recoils)
        x_cart = tx_cart[:, 1:4]  # (N, 3)
        t = tx_cart[:, :1]  # (N, 1)

        # Convert to modified spherical features
        x_sph = self.cart_to_sph_layer(x_cart)

        # Build [t, sph_features...] for delta_phi network
        tx_sph = jnp.concatenate([t, x_sph], axis=1)

        # Initial (spatial) correction term
        initial_correction = self.initial_correction_net(x_sph)  # (N,)
        outputs["initial_correction"] = initial_correction

        # Define an apply_fn for integration/quadrature using the delta_phi_net
        def apply_fn(z: Array) -> Array:
            return self.delta_phi_net(z)

        # Integrate delta_phi
        method = self.config.get("integration_mode", "gl3")
        if method == "diffrax_batch":
            delta_phi = compute_delta_phi_batch(tx_sph, apply_fn, t0)
        elif method == "diffrax_per_point":
            delta_phi = compute_delta_phi_per_point(tx_sph, apply_fn, t0, tf=None)
        elif method == "gl3":
            delta_phi = compute_delta_phi_per_point_gl3(tx_sph, apply_fn, t0)
        else:
            msg = (
                f"Unknown integration_mode='{method}'. "
                "Use one of {'gl3', 'diffrax_batch', 'diffrax_per_point'}."
            )
            raise ValueError(msg)

        outputs["delta_phi"] = delta_phi

        # Total learned correction
        total_correction = initial_correction + delta_phi

        # Convert correction to a potential term (radial scaling, etc.)
        scaled_potential = self.scale_layer(x_cart, total_correction)
        outputs["scale_nn_potential"] = scaled_potential

        # Analytic potential term
        analytic_potential_scaled = 0.0
        if self.config.get("include_analytic", False) and analytic_potential is not None:
            # 1. Evaluate the external analytic potential in physical units
            x_phys = self.config["x_transformer"].inverse_transform(x_cart)
            u_phys = analytic_potential.potential(x_phys, t= t)

            # 2. Transform to scaled units
            analytic_potential_scaled = self.config["u_transformer"].transform(
                    u_phys
                )

        # Fuse (NN + analytic)
        fused_potential = self.fuse_layer(scaled_potential, analytic_potential_scaled)
        outputs["fuse_models"] = fused_potential

        # Choose final output
        fuse_boundary_potential = self.fuse_boundary_layer(x_cart, scaled_potential, analytic_potential_scaled)

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

        def potential_for_grad(tx_arg: Array) -> Array:
            return self(
                tx_arg, mode="potential", analytic_potential=analytic_potential
            )["potential"].squeeze()

        # The gradient is taken w.r.t. the full input [t, x, y, z].
        grad_tx = jax.vmap(jax.grad(potential_for_grad))(tx_cart)  # (N, 4)
        # Return the spatial components (x, y, z) and discard the time derivative.
        acceleration = (-grad_tx)[:, 1:4]

        return {
            "acceleration": acceleration,
            "potential": potential,
            "outputs": outputs,
        }
