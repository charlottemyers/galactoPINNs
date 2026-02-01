"""Neural ODE model implementations."""

__all__ = (
    "NODEModel",
    "compute_delta_phi_batch",
    "compute_delta_phi_per_point",
    "compute_delta_phi_per_point_gl3",
    "compute_delta_phi_per_point_gl3panels",
)

from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol, TypedDict

import diffrax as dfx
import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

from galactoPINNs.layers import (
    CartesianToModifiedSphericalLayer,
    FuseandBoundary,
    ScaleNNPotentialLayer,
    SmoothMLP,
)

Mode = Literal["full", "potential", "acceleration", "density"]
class ExternalPytree(nnx.Variable):
    """Variable wrapper for external pytrees (like equinox modules).

    This allows galax potentials (which are equinox modules containing JAX arrays)
    to be stored as attributes in NNX modules without triggering pytree inspection errors.
    Access the wrapped object via the `.value` attribute.
    """



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

class NODEOutputs(TypedDict, total=False):
    """Standardized outputs returned by NODEModel.__call__."""

    potential: Array
    acceleration: Array
    laplacian: Array
    outputs: dict[str, Any]


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
        self,
        config: Mapping[str, Any],
        in_features: int = 5,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the NODEModel layers."""
        # --- Handle analytic baseline potential ---
        # Galax potentials are equinox modules and must be wrapped in ExternalPytree
        raw_ab_potential = config.get("ab_potential", None)
        if raw_ab_potential is not None:
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

        # --- Validate activation ---
        activation = config.get("activation", None)
        if activation is not None and not callable(activation):
            msg = (
                f"config['activation'] must be a callable (e.g. jax.nn.softplus), "
                f"got {type(activation)!r}"
            )
            raise TypeError(msg)

        # --- Initialize coordinate transform layer ---
        self.cart_to_sph_layer = CartesianToModifiedSphericalLayer(
            clip=config.get("clip", 1.0)
        )

        # --- Determine external scale potential for ScaleNNPotentialLayer ---
        # scale can be: string mode, precomputed array, or galax potential object
        raw_scale = config.get("scale", "one")

        if isinstance(raw_scale, str):
            # String mode ("one", "power", "nfw") - handled internally by ScaleNNPotentialLayer
            wrapped_scale_potential = None
        elif isinstance(raw_scale, (jnp.ndarray, jax.Array)):
            wrapped_scale_potential = None
        # Galax potential object - wrap in ExternalPytree
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

        # --- Initialize MLPs ---
        mlp_common: dict[str, Any] = {"rngs": rngs}
        if activation is not None:
            mlp_common["act"] = activation

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

    def compute_potential(self, tx_cart: Array) -> Array:
        """Evaluate the model potential at time+position(s).

        Parameters
        ----------
        tx_cart
            Time+position input. Typically shape ``(4,)`` for a single point or
            ``(N, 4)`` for a batch, containing [t, x, y, z].

        Returns
        -------
        potential
            Potential values with ``.squeeze()`` applied.

        """
        return self(tx_cart, mode="potential")["potential"].squeeze()

    def __call__(
        self,
        tx_cart: Array,
        mode: Mode = "full",
    ) -> NODEOutputs:
        """Forward pass for the time-dependent model.

        Parameters
        ----------
        tx_cart
            Time+position input. Shape (N, 4) or (4,) containing [t, x, y, z]
            in scaled coordinates.
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
        t0 = 0.0  # Integration start time

        tx_cart = jnp.atleast_2d(tx_cart)

        # --- Split time and space ---
        t = tx_cart[:, :1]        # (N, 1)
        x_cart = tx_cart[:, 1:4]  # (N, 3)

        # --- Coordinate transformation ---
        x_sph = self.cart_to_sph_layer(x_cart)

        # --- Build [t, sph_features...] for delta_phi network ---
        tx_sph = jnp.concatenate([t, x_sph], axis=1)

        # --- Initial (spatial) correction term ---
        initial_correction = self.initial_correction_net(x_sph)  # (N,)
        aux_outputs["initial_correction"] = initial_correction

        # --- Integrate delta_phi ---
        def apply_fn(z: Array) -> Array:
            return self.delta_phi_net(z)

        integration_mode = self.config.get("integration_mode", "gl3")
        if integration_mode == "diffrax_batch":
            delta_phi = compute_delta_phi_batch(tx_sph, apply_fn, t0)
        elif integration_mode == "diffrax_per_point":
            delta_phi = compute_delta_phi_per_point(tx_sph, apply_fn, t0, tf=None)
        elif integration_mode == "gl3":
            delta_phi = compute_delta_phi_per_point_gl3(tx_sph, apply_fn, t0)
        else:
            msg = (
                f"Unknown integration_mode='{integration_mode}'. "
                "Use one of {'gl3', 'diffrax_batch', 'diffrax_per_point'}."
            )
            raise ValueError(msg)

        aux_outputs["delta_phi"] = delta_phi

        # --- Total learned correction ---
        total_correction = initial_correction + delta_phi

        # --- Neural network potential (scaled) ---
        scaled_nn_potential = self.scale_layer(x_cart, total_correction)
        aux_outputs["scaled_nn_potential"] = scaled_nn_potential

        # --- Analytic baseline potential ---
        analytic_potential_scaled = 0.0

        if self.config.get("include_analytic", False):  # noqa: SIM102
            if self.ab_potential is not None:
                # Transform to physical coordinates
                x_phys = self.config["x_transformer"].inverse_transform(x_cart)

                # Evaluate analytic potential (access wrapped value)
                u_phys = self.ab_potential.value.potential(x_phys, t=t)

                # Transform potential to scaled units
                analytic_potential_scaled = self.config["u_transformer"].transform(u_phys)

        # --- Combine potentials ---
        fused_potential = scaled_nn_potential + analytic_potential_scaled
        aux_outputs["fused_potential"] = fused_potential

        boundary_potential = self.fuse_boundary_layer(
            x_cart, scaled_nn_potential, analytic_potential_scaled
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
        # Gradient is taken w.r.t. full input [t, x, y, z], then extract spatial components
        grad_tx = jax.vmap(jax.grad(self.compute_potential))(tx_cart)  # (N, 4)
        acceleration = (-grad_tx)[:, 1:4]  # Discard time derivative, keep spatial

        # --- Compute Laplacian if density mode ---
        if mode == "density":
            def laplacian_single(tx_arg: Array) -> Array:
                # Hessian w.r.t. full [t, x, y, z], then trace over spatial components
                hess = jax.hessian(self.compute_potential)(tx_arg)  # (4, 4)
                return jnp.trace(hess[1:4, 1:4])  # Trace over spatial block only

            laplacian = jax.vmap(laplacian_single)(tx_cart)
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
