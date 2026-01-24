from typing import Any, Callable, Mapping, Optional

import jax.random as jr
import numpyro
import numpyro.distributions as dist
import quaxed.numpy as jnp
from numpyro.contrib.module import random_flax_module
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam

from .static_model import StaticModel

__all__ = (
    "model_svi",
    "make_svi",
    "run_window",
)


def model_svi(
    x: jnp.ndarray,
    a_obs: Optional[jnp.ndarray] = None,
    *,
    # priors / noise
    sigma_lambda: float = 0.05,
    sigma_a: float = 2e-4,
    # relative-error weighting in the acc loss
    lambda_rel: float = 0.1,
    parameter_dict: Mapping[str, tuple] | None = None,
    analytic_form: Optional[Callable[[dict[str, dist.Distribution]], Any]] = None,
    config: Optional[dict] = None,
    # optional orbit-energy penalty
    orbit_q: Optional[jnp.ndarray] = None,
    orbit_p: Optional[jnp.ndarray] = None,
    w_orbit: float = 1.0,
    std_weight: float = 1.0,
) -> None:
    """
    NumPyro model for SVI training of a Bayesian galactoPINNs static model.

    Parameters
    ----------
    x : array, shape (N, D)
        Scaled model inputs.
    a_obs : array, shape (N, 3), optional
        Observed accelerations in the *same space* as the model output for the likelihood term.
        (Typically this means scaled accelerations if the model outputs scaled values.)
    sigma_lambda : float
        Prior std for neural network weights used by `random_flax_module`.
    sigma_a : float
        Acceleration noise scale controlling the strength of the data term.
        Used as 1/(2*sigma_a^2) multiplier on the loss inside a `numpyro.factor`.
    lambda_rel : float
        Additional relative-error weighting term in the per-point loss:
            diff_norm + lambda_rel * (diff_norm / ||a_obs||)
    parameter_dict : mapping, optional
        Dict mapping parameter name -> tuple of args for `dist.TruncatedNormal(*param_info)`.
        Used to define priors over analytic potential parameters.
    analytic_form : callable, optional
        Function that takes `parameter_distributions` and returns an object compatible with
        `StaticModel(trainable_analytic_layer=...)`.
    config : dict
        Model configuration passed to `StaticModel(config=...)`.
    orbit_q : array, shape (B, T, 3), optional
        Orbit positions used to build an energy conservation penalty term.
        Expected in the same coordinate space as the model input x (i.e., scaled positions).
    orbit_p : array, shape (B, T, 3), optional
        Orbit velocities/momenta used for kinetic energy.
    w_orbit : float
        Weight for the orbit energy loss factor.
    std_weight : float
        Weight for per-trajectory energy scatter term.

    Returns
    -------
    None
        This is a NumPyro model; it records factors and deterministic sites.
    """
    if config is None:
        raise ValueError("model_svi requires `config` to be provided.")

    # ----- analytic parameter priors (optional) -----
    trainable_analytic_layer = None
    if (parameter_dict is not None) and (analytic_form is not None):
        parameter_distributions: dict[str, dist.Distribution] = {
            name: dist.TruncatedNormal(*info) for name, info in parameter_dict.items()
        }
        trainable_analytic_layer = analytic_form(parameter_distributions)

    # ----- deterministic Flax model definition -----
    net = StaticModel(config=config, trainable_analytic_layer=trainable_analytic_layer)

    # ----- Bayesianize the Flax module weights -----
    bnn = random_flax_module(
        "full_model",
        net,
        prior=dist.Normal(0.0, sigma_lambda),
        input_shape=(x.shape[0], x.shape[1]),
    )

    out = bnn(x)
    a_pred = out["acceleration"]
    u_pred = out["potential"]

    # Record for posterior predictive inspection
    numpyro.deterministic("acceleration", a_pred)
    numpyro.deterministic("potential", u_pred)

    # ----- data term: acceleration loss -----
    if a_obs is not None:
        diff = a_pred - a_obs
        diff_norm = jnp.linalg.norm(diff, axis=1)
        a_true_norm = jnp.linalg.norm(a_obs, axis=1) + 1e-12

        per_point = diff_norm + lambda_rel * (diff_norm / a_true_norm)
        loss = jnp.mean(per_point)

        data_weight = 1.0 / (2.0 * sigma_a**2)
        numpyro.factor("acc_loss", -data_weight * loss)

        # ----- optional orbit energy loss -----
        if (orbit_q is not None) and (orbit_p is not None):
            if orbit_q.ndim != 3 or orbit_q.shape[-1] != 3:
                raise ValueError(f"orbit_q must have shape (B, T, 3); got {orbit_q.shape}")
            if orbit_p.ndim != 3 or orbit_p.shape[-1] != 3:
                raise ValueError(f"orbit_p must have shape (B, T, 3); got {orbit_p.shape}")

            B, T, _ = orbit_q.shape

            # Kinetic energy per step
            T_ke = 0.5 * jnp.sum(orbit_p**2, axis=-1)  # (B, T)

            # Potential energy via model potential evaluated at orbit positions
            q_flat = orbit_q.reshape(B * T, 3)
            phi_flat = bnn(q_flat)["potential"]  # (B*T,) or (B*T,1) depending on model
            phi = jnp.reshape(phi_flat, (B, T))

            E = T_ke + phi  # (B, T)
            dE = E[:, -1] - E[:, 0]  # (B,)

            # Energy fluctuation (trajectory-wise)
            E_cent = E - jnp.mean(E, axis=1, keepdims=True)
            std_E = jnp.std(E_cent, axis=1)  # (B,)

            L_orbit = jnp.mean(dE**2 + std_weight * std_E**2)
            numpyro.factor("orbit_E_loss", -w_orbit * L_orbit)


def make_svi(
    *,
    guide,
    sigma_lambda: float,
    sigma_a: float,
    lambda_rel: float,
    parameter_dict: Mapping[str, tuple] | None,
    analytic_form: Optional[Callable[[dict[str, dist.Distribution]], Any]],
    config: dict,
    orbit_q: Optional[jnp.ndarray] = None,
    orbit_p: Optional[jnp.ndarray] = None,
    w_orbit: float = 1.0,
    std_weight: float = 1.0,
    lr: float = 5e-3,
) -> SVI:
    """
    Construct an SVI object with `model_svi` closed over configuration objects.

    This avoids the common failure mode where `run_window` cannot pass non-array objects
    (e.g. config dict, analytic constructors) through the SVI data signature.

    Returns
    -------
    svi : numpyro.infer.SVI
    """
    def _model(x, a_obs=None):
        return model_svi(
            x,
            a_obs,
            sigma_lambda=sigma_lambda,
            sigma_a=sigma_a,
            lambda_rel=lambda_rel,
            parameter_dict=parameter_dict,
            analytic_form=analytic_form,
            config=config,
            orbit_q=orbit_q,
            orbit_p=orbit_p,
            w_orbit=w_orbit,
            std_weight=std_weight,
        )

    return SVI(_model, guide, Adam(lr), Trace_ELBO())


def run_window(
    prev_result,
    guide,
    *,
    x_train,
    a_train,
    steps: int = 1000,
    lr: float = 5e-3,
    sigma_lambda: float = 0.05,
    sigma_a: float = 2e-4,
    lambda_rel: float = 0.1,
    parameter_dict: Mapping[str, tuple] | None = None,
    analytic_form: Optional[Callable[[dict[str, dist.Distribution]], Any]] = None,
    config: Optional[dict] = None,
    orbit_q: Optional[jnp.ndarray] = None,
    orbit_p: Optional[jnp.ndarray] = None,
    w_orbit: float = 1.0,
    std_weight: float = 1.0,
    rng_key=None,
):
    """
    Run (or continue) SVI optimization.

    Parameters
    ----------
    prev_result : numpyro.infer.SVIState or None
        Previous SVI run result. If provided, uses `prev_result.params` to initialize.
    guide : callable
        NumPyro guide (e.g., AutoDiagonalNormal(model)).
    x_train, a_train : arrays
        Training inputs and observed accelerations.
    steps : int
        Number of SVI steps.
    lr : float
        Adam learning rate.
    sigma_lambda, sigma_a, lambda_rel : float
        Passed through to `model_svi`.
    parameter_dict, analytic_form, config : optional
        Passed through to `model_svi`. `config` is required.
    orbit_q, orbit_p, w_orbit, std_weight : optional
        Orbit energy penalty configuration.
    rng_key : jax.random.PRNGKey, optional
        RNG key for SVI. If None, uses PRNGKey(0).

    Returns
    -------
    result : numpyro.infer.SVIState
        Result of `svi.run`.
    """
    if config is None:
        raise ValueError("run_window requires `config` to be provided.")

    if rng_key is None:
        rng_key = jr.PRNGKey(0)

    svi = make_svi(
        guide=guide,
        sigma_lambda=sigma_lambda,
        sigma_a=sigma_a,
        lambda_rel=lambda_rel,
        parameter_dict=parameter_dict,
        analytic_form=analytic_form,
        config=config,
        orbit_q=orbit_q,
        orbit_p=orbit_p,
        w_orbit=w_orbit,
        std_weight=std_weight,
        lr=lr,
    )

    if prev_result is None:
        return svi.run(rng_key, steps, x_train, a_train)

    return svi.run(rng_key, steps, x_train, a_train, init_params=prev_result.params)
