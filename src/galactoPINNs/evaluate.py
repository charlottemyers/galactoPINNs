"""Model evaluation utilities."""

__all__ = (
    "bnn_performance",
    "evaluate_performance",
    "evaluate_performance_node",
)

from collections.abc import Callable, Mapping
from typing import Any, Literal

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from .inference import apply_model


def evaluate_performance(
    model: Any,
    raw_datadict: Mapping[str, Any],
    num_test: int,
    *,
    gauge_correct: Literal["reference", "median"] | None = None,
    r_ref: float | None = None,
    eps: float = 1e-10,
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
    gauge_correct : {"reference", "median"} or None, optional
        Gauge correction method for potential errors:
        - None: no gauge correction (default)
        - "reference": compute errors on potential differences relative to a
          reference radius `r_ref`.
        - "median": subtract median offset between predicted and true potential.
    r_ref : float, optional
        Reference radius for gauge correction when `gauge_correct="reference"`.
        Required if using reference-based correction.
    eps : float, optional
        Small constant for numerical stability in percent error calculations.
        Default is 1e-10.

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
          (gauge-corrected if `gauge_correct` is specified)
        - "acc_percent_error": percent acceleration error, shape (num_test,)
        If analytic baseline is enabled:
        - "analytic_baseline": analytic baseline potential at t=0
        - "ab_pot_error", "ab_acc_error": baseline percent errors

    Raises
    ------
    ValueError
        If `gauge_correct="reference"` but `r_ref` is not provided.

    """
    # --- Validate gauge correction arguments ---
    if gauge_correct == "reference" and r_ref is None:
        raise ValueError("r_ref must be provided when gauge_correct='reference'")

    # --- Extract validation data ---
    x_val = raw_datadict["x_val"][:num_test]
    true_pot = raw_datadict["u_val"][:num_test]
    true_acc = raw_datadict["a_val"][:num_test]

    config = model.config
    analytic_baseline = (
        model.ab_potential.value
        if model.ab_potential is not None and config.get("include_analytic", True)
        else None
    )

    # --- Compute radii ---
    r_eval = jnp.linalg.norm(x_val, axis=1)

    # --- Run model prediction ---
    scaled_x_val = config["x_transformer"].transform(x_val)
    output = apply_model(model, scaled_x_val)
    predicted_pot = config["u_transformer"].inverse_transform(output["u_pred"])
    predicted_acc = config["a_transformer"].inverse_transform(output["a_pred"])

    # --- Acceleration error (no gauge ambiguity) ---
    acc_percent_error = (
        100
        * jnp.linalg.norm(predicted_acc - true_acc, axis=1)
        / (jnp.linalg.norm(true_acc, axis=1) + eps)
    )

    # --- Potential error (with optional gauge correction) ---
    if gauge_correct is None:
        # No correction
        pot_percent_error = (
            100 * jnp.abs((true_pot - predicted_pot) / (jnp.abs(true_pot) + eps))
        )

    elif gauge_correct == "reference":
        # Gauge-invariant: compare potential differences from reference point
        i_ref = int(jnp.argmin(jnp.abs(r_eval - r_ref)))
        du_true = true_pot - true_pot[i_ref]
        du_pred = predicted_pot - predicted_pot[i_ref]
        pot_percent_error = 100.0 * jnp.abs((du_true - du_pred) / (jnp.abs(du_true) + eps))

    elif gauge_correct == "median":
        # Median offset correction
        offset = jnp.median(predicted_pot - true_pot)
        predicted_pot_corrected = predicted_pot - offset
        pot_percent_error = (
            100 * jnp.abs((true_pot - predicted_pot_corrected) / (jnp.abs(true_pot) + eps))
        )

    else:
        raise ValueError(f"Unknown gauge_correct='{gauge_correct}'")

    # --- Analytic baseline comparison ---
    if analytic_baseline is not None:
        analytic_baseline_potential = analytic_baseline.potential(x_val, t=0)
        analytic_baseline_acc = analytic_baseline.acceleration(x_val, t=0)

        ab_pot_error = 100 * jnp.abs(
            (analytic_baseline_potential - true_pot) / (jnp.abs(true_pot) + eps)
        )
        ab_acc_error = (
            100
            * jnp.linalg.norm(analytic_baseline_acc - true_acc, axis=1)
            / (jnp.linalg.norm(true_acc, axis=1) + eps)
        )
    else:
        analytic_baseline_potential = None
        ab_pot_error = None
        ab_acc_error = None

    return {
        "r_eval": r_eval,
        "x_val": x_val,
        "true_a": true_acc,
        "predicted_a": predicted_acc,
        "true_u": true_pot,
        "predicted_u": predicted_pot,
        "acc_percent_error": acc_percent_error,
        "pot_percent_error": pot_percent_error,
        "analytic_baseline": analytic_baseline_potential,
        "ab_pot_error": ab_pot_error,
        "ab_acc_error": ab_acc_error,
        "avg_percent_error": jnp.mean(acc_percent_error),
    }


def evaluate_performance_node(
    model: Any,
    t_eval: float | Any,
    raw_datadict: Mapping[str, Any],
    num_test: int,
    *,
    gauge_correct: Literal["reference", "median"] | None = None,
    r_ref: float | None = None,
    eps: float = 1e-10,
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
    gauge_correct : {"reference", "median"} or None, optional
        Gauge correction method for potential errors:
        - None: no gauge correction (default)
        - "reference": compute errors on potential differences relative to a
          reference radius `r_ref`.
        - "median": subtract median offset between predicted and true potential.
    r_ref : float, optional
        Reference radius for gauge correction when `gauge_correct="reference"`.
        Required if using reference-based correction.
    eps : float, optional
        Small constant for numerical stability in percent error calculations.
        Default is 1e-10.

    Returns
    -------
    results : dict
        A dictionary of evaluation metrics for the specified time slice.
        Includes predicted and true values, error metrics, and optional
        baseline comparison metrics.

    Raises
    ------
    ValueError
        If `gauge_correct="reference"` but `r_ref` is not provided.

    """
    # --- Validate gauge correction arguments ---
    if gauge_correct == "reference" and r_ref is None:
        raise ValueError("r_ref must be provided when gauge_correct='reference'")

    # --- Extract validation data ---
    val_data = raw_datadict["val"][t_eval]
    x_val = val_data["x"][:num_test]
    true_pot = val_data["u"][:num_test]
    true_acc = val_data["a"][:num_test]

    config = model.config

    # --- Compute radii ---
    r_eval = jnp.linalg.norm(x_val, axis=1)

    # --- Build scaled input [t, x, y, z] ---
    x_scaled = config["x_transformer"].transform(x_val)
    t_scaled = config["t_transformer"].transform(t_eval) * jnp.ones((x_val.shape[0], 1))
    tx_scaled = jnp.concatenate([t_scaled, x_scaled], axis=1)

    # --- Get analytic baseline (unwrap ExternalPytree if present) ---
    analytic_baseline = (
        model.ab_potential.value
        if model.ab_potential is not None and config.get("include_analytic", True)
        else None
    )

    # --- Run model prediction ---
    output = apply_model(model, tx_scaled)
    predicted_pot = config["u_transformer"].inverse_transform(output["u_pred"])
    predicted_acc = config["a_transformer"].inverse_transform(output["a_pred"])

    # --- Acceleration norms ---
    predicted_acc_norm = jnp.linalg.norm(predicted_acc, axis=1, keepdims=True)
    true_acc_norm = jnp.linalg.norm(true_acc, axis=1, keepdims=True)

    # --- Acceleration error (no gauge ambiguity) ---
    acc_percent_error = (
        100
        * jnp.linalg.norm(predicted_acc - true_acc, axis=1)
        / (jnp.linalg.norm(true_acc, axis=1) + eps)
    )

    # --- Potential error (with optional gauge correction) ---
    if gauge_correct is None:
        # No correction
        pot_percent_error = (
            100 * jnp.abs((true_pot - predicted_pot) / (jnp.abs(true_pot) + eps))
        )

    elif gauge_correct == "reference":
        # Gauge-invariant: compare potential differences from reference point
        i_ref = int(jnp.argmin(jnp.abs(r_eval - r_ref)))
        du_true = true_pot - true_pot[i_ref]
        du_pred = predicted_pot - predicted_pot[i_ref]
        pot_percent_error = 100.0 * jnp.abs((du_true - du_pred) / (jnp.abs(du_true) + eps))

    elif gauge_correct == "median":
        # Median offset correction
        offset = jnp.median(predicted_pot - true_pot)
        predicted_pot_corrected = predicted_pot - offset
        pot_percent_error = (
            100 * jnp.abs((true_pot - predicted_pot_corrected) / (jnp.abs(true_pot) + eps))
        )

    else:
        raise ValueError(f"Unknown gauge_correct='{gauge_correct}'")

    # --- Analytic baseline comparison ---
    if analytic_baseline is not None:
        analytic_baseline_potential = analytic_baseline.potential(x_val, t=t_eval)
        analytic_baseline_acc = analytic_baseline.acceleration(x_val, t=t_eval)
        analytic_baseline_0 = analytic_baseline.potential(x_val, t=0)
        analytic_baseline_acc_norm = jnp.linalg.norm(analytic_baseline_acc, axis=1)

        ab_pot_error = 100 * jnp.abs(
            (analytic_baseline_potential - true_pot) / (jnp.abs(true_pot) + eps)
        )
        ab0_pot_error = 100 * jnp.abs(
            (analytic_baseline_0 - true_pot) / (jnp.abs(true_pot) + eps)
        )
        ab_acc_error = (
            100
            * jnp.linalg.norm(analytic_baseline_acc - true_acc, axis=1)
            / (jnp.linalg.norm(true_acc, axis=1) + eps)
        )

        residual_pot = analytic_baseline_potential - predicted_pot
        average_residual_pot = jnp.mean(residual_pot)
        corrected_potential = predicted_pot + average_residual_pot
        corrected_pot_percent_error = 100 * jnp.abs(
            (corrected_potential - true_pot) / (jnp.abs(true_pot) + eps)
        )
    else:
        analytic_baseline_potential = None
        analytic_baseline_acc_norm = None
        analytic_baseline_0 = None
        ab_pot_error = None
        ab0_pot_error = None
        ab_acc_error = None
        corrected_potential = None
        corrected_pot_percent_error = None

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
