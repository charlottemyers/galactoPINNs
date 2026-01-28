"""Model evaluation utilities."""

__all__ = (
    "bnn_performance",
    "evaluate_performance",
    "evaluate_performance_node",
)

from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from .inference import apply_model


def evaluate_performance(
    model: Any,
    raw_datadict: Mapping[str, Any],
    num_test: int,
    *,
    analytic_baseline: Any | None = None,
) -> dict[str, Any]:
    """Evaluate a static model on a validation set and compute error metrics.

    This function:
    - transforms physical validation positions `x_val` to the model's scaled
      input space,
    - runs `apply_model(...)` to obtain predicted potential and acceleration
      in scaled space,
    - inverse-transforms predictions back to physical units,
    - computes percent errors against truth,
    - optionally compares against an analytic baseline potential/acceleration.

    Parameters
    ----------
    model : flax.nnx.Module
        A trained static NNX model object with attribute `config`.
        The model is called directly via `model(x_scaled)`.
    raw_datadict : dict
        Dictionary containing (at min.) the following keys:
        - "x_val": array-like, shape (N, 3), physical validation positions
        - "u_val": array-like, shape (N,) or (N, 1), physical true potential
        - "a_val": array-like, shape (N, 3), physical true acceleration
    num_test : int
        Number of validation samples to evaluate (uses the first `num_test` rows).
    analytic_baseline : galax.potential.AbstractPotential, optional
        An optional analytic potential to use for baseline error calculations.


    Required config keys (model.config)
    -----------------------------------
    - "x_transformer": must implement .transform(x_phys) -> x_scaled
    - "u_transformer": must implement .inverse_transform(u_scaled) -> u_phys
    - "a_transformer": must implement .inverse_transform(a_scaled) -> a_phys
    Optional:
    - "include_analytic" (bool): whether to compute analytic baseline errors
      (default True)

    Returns
    -------
    results : dict
        Keys are designed for downstream plotting/analysis. Entries include:
        - "r_eval": np.ndarray, shape (num_test,), radius of each evaluation
          point in physical space
        - "x_val": physical positions used, shape (num_test, 3)
        - "true_u", "predicted_u": physical potentials, shape (num_test,)
        - "true_a", "predicted_a": physical accelerations, shape (num_test, 3)
        - "pot_percent_error": percent potential error, shape (num_test,)
        - "acc_percent_error": percent acceleration error, shape (num_test,)
        If analytic baseline is enabled:
        - "analytic_baseline": analytic baseline potential at t=0
        - "ab_pot_error", "ab_acc_error": baseline percent errors
        - "residual_pot": ab_potential - predicted_u
        - "corrected_pot_percent_error": percent error after applying mean
          residual correction

    """
    true_pot = raw_datadict["u_val"][:num_test]
    true_acc = raw_datadict["a_val"][:num_test]

    config = model.config
    r_eval = jnp.linalg.norm(raw_datadict["x_val"][:num_test], axis=1)
    scaled_x_val = config["x_transformer"].transform(raw_datadict["x_val"][:num_test])
    output = apply_model(model, scaled_x_val, analytic_potential = analytic_baseline)
    predicted_pot = config["u_transformer"].inverse_transform(output["u_pred"])
    predicted_acc = config["a_transformer"].inverse_transform(output["a_pred"])
    acc_percent_error = (
        100
        * jnp.linalg.norm(predicted_acc - true_acc, axis=1)
        / jnp.linalg.norm(true_acc, axis=1)
    )
    pot_percent_error = 100 * jnp.abs((true_pot - predicted_pot) / true_pot)

    fiducial_acc = None
    fiducial_pot = None
    fiducial_acc_error = None
    fiducial_pot_error = None

    if analytic_baseline is not None:
        analytic_baseline_potential = analytic_baseline.potential(
            raw_datadict["x_val"][:num_test], t=0
        )
        analytic_baseline_acc = analytic_baseline.acceleration(
            raw_datadict["x_val"][:num_test], t=0
        )

        ab_pot_error = 100 * jnp.abs(
            (analytic_baseline_potential - true_pot) / true_pot
        )
        ab_acc_error = (
            100
            * jnp.linalg.norm(analytic_baseline_acc - true_acc, axis=1)
            / jnp.linalg.norm(true_acc, axis=1)
        )

        residual_pot = analytic_baseline_potential - predicted_pot
        average_residual_pot = jnp.mean(residual_pot)
        corrected_potential = predicted_pot + average_residual_pot
        corrected_pot_percent_error = 100 * jnp.abs(
            (corrected_potential - true_pot) / true_pot
        )

    else:
        analytic_baseline_potential = None
        ab_pot_error = None
        ab_acc_error = None
        residual_pot = None
        corrected_potential = None
        corrected_pot_percent_error = None

    return {
        "r_eval": r_eval,
        "x_val": raw_datadict["x_val"][:num_test],
        "true_a": true_acc,
        "predicted_a": predicted_acc,
        "true_u": true_pot,
        "predicted_u": predicted_pot,
        "acc_percent_error": acc_percent_error,
        "pot_percent_error": pot_percent_error,
        "residual_pot": residual_pot,
        "corrected_pot_percent_error": corrected_pot_percent_error,
        "analytic_baseline": analytic_baseline_potential,
        "ab_pot_error": ab_pot_error,
        "ab_acc_error": ab_acc_error,
        "fiducial_acc": fiducial_acc,
        "fiducial_pot": fiducial_pot,
        "fiducial_pot_error": fiducial_pot_error,
        "fiducial_acc_error": fiducial_acc_error,
        "avg_percent_error": jnp.mean(acc_percent_error),
    }


def evaluate_performance_node(
    model: Any,
    t_eval: float | Any,
    raw_datadict: Mapping[str, Any],
    num_test: int,
    *,
    analytic_baseline: Any | None = None,
) -> dict[str, Any]:
    """Evaluate a time-dependent model at a single evaluation time.

    This function expects the dataset to be organized by time keys:
        raw_datadict["val"][t_eval] -> dict with keys {"x", "u", "a"}

    It constructs a batched input array `tx_scaled` with columns [t_scaled, x_scaled]
    and uses `apply_model(...)` to generate predictions.

    Parameters
    ----------
    model : _NNXModelLike
        A time-dependent NNX model with attribute `config`.
    t_eval : float or int
        The evaluation time, used as a key for `raw_datadict["val"]`.
    raw_datadict : dict
        A nested dictionary mapping split -> time -> data arrays.
    num_test : int
        Number of validation samples to evaluate from the `t_eval` time slice.
    analytic_baseline : galax.potential.AbstractPotential, optional
        An optional analytic potential for baseline error calculations and to
        pass to the model's forward pass.

    Returns
    -------
    results : dict
        A dictionary of evaluation metrics for the specified time slice.
        Includes predicted and true values, error metrics, and optional
        baseline comparison metrics.

    """
    val_data = raw_datadict["val"][t_eval]
    x_val = val_data["x"][:num_test]

    config = model.config
    r_eval = jnp.linalg.norm(x_val, axis=1)

    true_pot = val_data["u"][:num_test]
    true_acc = val_data["a"][:num_test]

    x_scaled = config["x_transformer"].transform(x_val)
    t_scaled = config["t_transformer"].transform(t_eval) * jnp.ones((x_val.shape[0], 1))
    tx_scaled = jnp.concatenate([t_scaled, x_scaled], axis=1)

    output = apply_model(model, tx_scaled, analytic_potential = analytic_baseline)

    predicted_pot = config["u_transformer"].inverse_transform(output["u_pred"])
    predicted_acc = config["a_transformer"].inverse_transform(output["a_pred"])

    predicted_acc_norm = jnp.linalg.norm(predicted_acc, axis=1, keepdims=True)
    true_acc_norm = jnp.linalg.norm(true_acc, axis=1, keepdims=True)

    acc_percent_error = (
        100
        * jnp.linalg.norm(predicted_acc - true_acc, axis=1)
        / jnp.linalg.norm(true_acc, axis=1)
    )
    pot_percent_error = 100 * jnp.abs((true_pot - predicted_pot) / true_pot)

    if analytic_baseline is not None:
        analytic_baseline_potential = analytic_baseline.potential(x_val, t=t_eval)
        analytic_baseline_acc = analytic_baseline.acceleration(x_val, t=t_eval)
        analytic_baseline_0 = analytic_baseline.potential(x_val, t=0)
        analytic_baseline_acc_norm = jnp.linalg.norm(analytic_baseline_acc, axis=1)
        ab_pot_error = 100 * jnp.abs(
            (analytic_baseline_potential - true_pot) / true_pot
        )
        ab0_pot_error = 100 * jnp.abs((analytic_baseline_0 - true_pot) / true_pot)
        ab_acc_error = (
            100
            * jnp.linalg.norm(analytic_baseline_acc - true_acc, axis=1)
            / jnp.linalg.norm(true_acc, axis=1)
        )

        residual_pot = analytic_baseline_potential - predicted_pot
        average_residual_pot = jnp.mean(residual_pot)
        corrected_potential = predicted_pot + average_residual_pot
        corrected_pot_percent_error = 100 * jnp.abs(
            (corrected_potential - true_pot) / true_pot
        )
    else:
        analytic_baseline_potential = None
        ab_pot_error = None
        ab_acc_error = None
        residual_pot = None
        corrected_potential = None
        corrected_pot_percent_error = None
        analytic_baseline_acc_norm = None
        ab0_pot_error = None
        analytic_baseline_0 = None

    return {
        "r_eval": r_eval,
        "true_a": true_acc,
        "predicted_a": predicted_acc,
        "true_a_norm": true_acc_norm,
        "predicted_a_norm": predicted_acc_norm,
        "true_u": true_pot,
        "predicted_u": predicted_pot,
        "acc_percent_error": acc_percent_error,
        "pot_percent_error": pot_percent_error,
        "residual_pot": residual_pot,
        "corrected_pot_percent_error": corrected_pot_percent_error,
        "corrected_potential": corrected_potential,
        "analytic_baseline": analytic_baseline_potential,
        "ab_acc_norm": analytic_baseline_acc_norm,
        "ab_pot_error": ab_pot_error,
        "ab_acc_error": ab_acc_error,
        "ab0_pot_error": ab0_pot_error,
        "analytic_baseline_0": analytic_baseline_0,
    }


def bnn_performance(
    predictive: Callable[[Array, Array], Mapping[str, Array]],
    x_test: Array,
    config: Mapping[str, Any],
    rng_key: Array | None = None,
) -> dict[str, Any]:
    """Summarize Bayesian posterior predictive outputs for potential and acceleration.

    This function is designed for NumPyro-style `Predictive` callables that return
    posterior samples for:
      - "potential": shape (S, N, ...) in scaled space
      - "acceleration": shape (S, N, 3, ...) in scaled space
    where S is the number of posterior samples.

    It:
    - scales `x_test` using `config["x_transformer"]`,
    - runs the predictive sampler,
    - inverse-transforms outputs back to physical space,
    - returns posterior mean/std and the raw samples.

    Parameters
    ----------
    predictive : callable
        A callable like `numpyro.infer.Predictive(...)` with signature:
            predictive(rng_key, x_scaled) -> dict
        Expected to contain keys "potential" and "acceleration".
    x_test : array-like, shape (N, 3)
        Physical test positions (not scaled).
    config : dict
        Must contain:
        - "x_transformer": .transform
        - "u_transformer": .inverse_transform
        - "a_transformer": .inverse_transform
    rng_key : jax.random.PRNGKey, optional
        RNG key used for the predictive call. If None, uses `jr.PRNGKey(0)`.

    Returns
    -------
    summary : dict
        - "u_mean": posterior mean potential in physical units, shape (N,)
        - "a_mean": posterior mean acceleration in physical units, shape (N, 3)
        - "u_std": posterior std potential in physical units, shape (N,)
        - "a_std": posterior std acceleration in physical units, shape (N, 3)
        - "u_samples": posterior samples of potential in physical units,
          shape (S, N, ...)
        - "a_samples": posterior samples of acceleration in physical units,
          shape (S, N, 3, ...)

    """
    rng_key = jr.PRNGKey(0) if rng_key is None else rng_key

    x_test_scaled = config["x_transformer"].transform(x_test)
    pred = predictive(jr.PRNGKey(2), x_test_scaled)

    u_post_phys = config["u_transformer"].inverse_transform(pred["potential"])
    a_post_phys = config["a_transformer"].inverse_transform(pred["acceleration"])

    u_mean = u_post_phys.mean(axis=0)
    a_mean = a_post_phys.mean(axis=0)

    u_std = u_post_phys.std(axis=0)
    a_std = a_post_phys.std(axis=0)

    return {
        "u_mean": u_mean,
        "a_mean": a_mean,
        "u_std": u_std,
        "a_std": a_std,
        "a_samples": a_post_phys,
        "u_samples": u_post_phys,
    }
