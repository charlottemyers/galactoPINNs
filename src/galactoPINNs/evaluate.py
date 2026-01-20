import jax.numpy as jnp
import numpy as np
import jax.random as jr

from .inference import apply_model, apply_model_time

__all__ = [
    "evaluate_performance",
    "evaluate_performance_node",
    "bnn_performance",
]


def evaluate_performance(
    model,
    trained_state_params,
    raw_datadict,
    num_test,
):
    """
    Evaluate a *static* model on a validation set and compute error metrics.

    This function:
    - transforms physical validation positions `x_val` to the model's scaled input space,
    - runs `apply_model(...)` to obtain predicted potential and acceleration in *scaled* space,
    - inverse-transforms predictions back to physical units,
    - computes percent errors against truth,
    - optionally compares against an analytic baseline potential/acceleration.

    Parameters
    ----------
    model : flax.linen.Module (or compatible)
        A trained (or partially trained) static model object with attribute `config`.
        The model is expected to support `model.apply({"params": params}, x_scaled)` via `apply_model`.
    trained_state_params : Any
        Parameters tree (e.g., `TrainState.params`) used for `model.apply`.
    raw_datadict : dict
        Dictionary containing (at minimum) the following keys:
        - "x_val": array-like, shape (N, 3), physical validation positions
        - "u_val": array-like, shape (N,) or (N, 1), physical true potential
        - "a_val": array-like, shape (N, 3), physical true acceleration
        Additional keys may exist and are ignored here.
    num_test : int
        Number of validation samples to evaluate (uses the first `num_test` rows).

    Required config keys (model.config)
    -----------------------------------
    - "x_transformer": must implement .transform(x_phys) -> x_scaled
    - "u_transformer": must implement .inverse_transform(u_scaled) -> u_phys
    - "a_transformer": must implement .inverse_transform(a_scaled) -> a_phys
    Optional:
    - "include_analytic" (bool): whether to compute analytic baseline errors (default True)
    - "lf_analytic_function": analytic potential object with .potential(x, t=...) and .acceleration(x, t=...)

    Returns
    -------
    results : dict
        Keys are designed for downstream plotting/analysis. Common entries include:
        - "r_eval": np.ndarray, shape (num_test,), radius of each evaluation point in physical space
        - "x_val": physical positions used, shape (num_test, 3)
        - "true_u", "predicted_u": physical potentials, shape (num_test,)
        - "true_a", "predicted_a": physical accelerations, shape (num_test, 3)
        - "pot_percent_error": percent potential error, shape (num_test,)
        - "acc_percent_error": percent acceleration error, shape (num_test,)
        If analytic baseline is enabled:
        - "lf_potential": analytic baseline potential at t=0
        - "lf_pot_error", "lf_acc_error": baseline percent errors
        - "residual_pot": lf_potential - predicted_u
        - "corrected_pot_percent_error": percent error after applying mean residual correction

    """
    true_pot = raw_datadict["u_val"][:num_test]
    true_acc = raw_datadict["a_val"][:num_test]

    config = model.config
    r_eval = np.linalg.norm(raw_datadict["x_val"][:num_test], axis=1)
    scaled_x_val = config["x_transformer"].transform(raw_datadict["x_val"][:num_test])
    output = apply_model(model, trained_state_params, scaled_x_val)
    predicted_pot = config["u_transformer"].inverse_transform(output["u_pred"])
    predicted_acc = config["a_transformer"].inverse_transform(output["a_pred"])
    acc_percent_error = (
        100
        * jnp.linalg.norm(predicted_acc - true_acc, axis=1)
        / jnp.linalg.norm(true_acc, axis=1)
    )
    pot_percent_error = 100 * np.abs((true_pot - predicted_pot) / true_pot)

    fiducial_acc = None
    fiducial_pot = None
    fiducial_acc_error = None
    fiducial_pot_error = None

    if config.get("include_analytic", True):
        lf_analytic = config["lf_analytic_function"]
        lf_analytic_potential = lf_analytic.potential(
            raw_datadict["x_val"][:num_test], t=0
        )
        lf_analytic_acc = lf_analytic.acceleration(
            raw_datadict["x_val"][:num_test], t=0
        )

        lf_pot_error = 100 * np.abs((lf_analytic_potential - true_pot) / true_pot)
        lf_acc_error = (
            100
            * jnp.linalg.norm(lf_analytic_acc - true_acc, axis=1)
            / jnp.linalg.norm(true_acc, axis=1)
        )

        residual_pot = lf_analytic_potential - predicted_pot
        average_residual_pot = jnp.mean(residual_pot)
        corrected_potential = predicted_pot + average_residual_pot
        corrected_pot_percent_error = 100 * jnp.abs(
            (corrected_potential - true_pot) / true_pot
        )

    else:
        lf_analytic_potential = None
        lf_pot_error = None
        lf_acc_error = None
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
        "lf_potential": lf_analytic_potential,
        "lf_pot_error": lf_pot_error,
        "lf_acc_error": lf_acc_error,
        "fiducial_acc": fiducial_acc,
        "fiducial_pot": fiducial_pot,
        "fiducial_pot_error": fiducial_pot_error,
        "fiducial_acc_error": fiducial_acc_error,
        "avg_percent_error": np.mean(acc_percent_error),
    }


def evaluate_performance_node(
    model, params, t_eval, raw_datadict, num_test
):
    """
    Evaluate a *time-dependent* model at a single evaluation time.

    This function expects the dataset to be organized by time keys:
        raw_datadict["val"][t_eval] -> dict with keys {"x", "u", "a"}

    It constructs a batched input array `tx_scaled` with columns [t_scaled, x_scaled]
    and uses `apply_model_time(...)` to generate predictions.

    Parameters
    ----------
    model : flax.linen.Module (or compatible)
        A time-dependent model with attribute `config`. The model is expected to accept
        a scaled input of shape (N, 1+3) = (N, 4): [t_scaled, x_scaled...].
    params : Any
        Parameters tree used for `model.apply`.
    t_eval : float or int (or time key type)
        The evaluation time. Used both to select `raw_datadict["val"][t_eval]` and to build
        the time input feature.
        Important: this must match the dictionary key exactly if keys are not floats.
    raw_datadict : dict
        Must contain `raw_datadict["val"]` mapping times -> per-time validation dict, where each
        per-time dict contains:
        - "x": shape (N, 3) physical positions
        - "u": shape (N,) physical true potential at that time
        - "a": shape (N, 3) physical true acceleration at that time
    num_test : int
        Number of validation samples to evaluate from that time slice.

    Required config keys (model.config)
    -----------------------------------
    - "x_transformer": .transform / .inverse_transform
    - "u_transformer": .transform / .inverse_transform
    - "a_transformer": .transform / .inverse_transform
    - "t_transformer": .transform / .inverse_transform
    Optional:
    - "include_analytic" (bool): whether to compute analytic baseline errors (default True)
    - "lf_analytic_function": analytic potential object with .potential(x, t=...) and .acceleration(x, t=...)

    Returns
    -------
    results : dict
        Includes:
        - "r_eval": radii of evaluation points, shape (num_test,)
        - "true_u", "predicted_u": physical potentials, shape (num_test,)
        - "true_a", "predicted_a": physical accelerations, shape (num_test, 3)
        - "pot_percent_error", "acc_percent_error": percent errors, shape (num_test,)
        - "true_a_norm", "predicted_a_norm": norms of acceleration vectors, shape (num_test, 1)
        If analytic baseline is enabled:
        - "lf_potential": analytic baseline potential at t_eval
        - "lf_analytic_0": analytic baseline potential at t=0 (same x)
        - "lf_pot_error": baseline potential percent error at t_eval
        - "lf0_pot_error": baseline potential percent error at t=0
        - "lf_acc_error": baseline acceleration percent error at t_eval
        - "lf_acc_norm": norm of baseline analytic acceleration, shape (num_test,)
        - "residual_pot", "corrected_potential", "corrected_pot_percent_error"

    """
    val_data = raw_datadict["val"][t_eval]
    x_val = val_data["x"][:num_test]

    config = model.config
    r_eval = np.linalg.norm(x_val, axis=1)

    true_pot = val_data["u"][:num_test]
    true_acc = val_data["a"][:num_test]

    x_scaled = config["x_transformer"].transform(x_val)
    t_scaled = config["t_transformer"].transform(t_eval) * jnp.ones(
        (x_val.shape[0], 1)
    )
    tx_scaled = jnp.concatenate([t_scaled, x_scaled], axis=1)

    output = apply_model_time(model, params, tx_scaled)

    predicted_pot = config["u_transformer"].inverse_transform(output["u_pred"])
    predicted_acc = config["a_transformer"].inverse_transform(output["a_pred"])

    predicted_acc_norm = jnp.linalg.norm(predicted_acc, axis=1, keepdims=True)
    true_acc_norm = jnp.linalg.norm(true_acc, axis=1, keepdims=True)

    acc_percent_error = (
        100
        * jnp.linalg.norm(predicted_acc - true_acc, axis=1)
        / jnp.linalg.norm(true_acc, axis=1)
    )
    pot_percent_error = 100 * np.abs((true_pot - predicted_pot) / true_pot)

    if config.get("include_analytic", True):
        lf_analytic = config["lf_analytic_function"]
        lf_analytic_potential = lf_analytic.potential(x_val, t=t_eval)
        lf_analytic_acc = lf_analytic.acceleration(x_val, t=t_eval)
        lf_analytic_0 = lf_analytic.potential(x_val, t=0)
        lf_analytic_acc_norm = jnp.linalg.norm(lf_analytic_acc, axis=1)
        lf_pot_error = 100 * np.abs((lf_analytic_potential - true_pot) / true_pot)
        lf0_pot_error = 100 * np.abs((lf_analytic_0 - true_pot) / true_pot)
        lf_acc_error = (
            100
            * jnp.linalg.norm(lf_analytic_acc - true_acc, axis=1)
            / jnp.linalg.norm(true_acc, axis=1)
        )

        residual_pot = lf_analytic_potential - predicted_pot
        average_residual_pot = jnp.mean(residual_pot)
        corrected_potential = predicted_pot + average_residual_pot
        corrected_pot_percent_error = 100 * jnp.abs(
            (corrected_potential - true_pot) / true_pot
        )
    else:
        lf_analytic_potential = None
        lf_pot_error = None
        lf_acc_error = None
        residual_pot = None
        corrected_potential = None
        corrected_pot_percent_error = None
        lf_analytic_acc_norm = None
        lf0_pot_error = None
        lf_analytic_0 = None

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
        "lf_potential": lf_analytic_potential,
        "lf_acc_norm": lf_analytic_acc_norm,
        "lf_pot_error": lf_pot_error,
        "lf_acc_error": lf_acc_error,
        "lf0_pot_error": lf0_pot_error,
        "lf_analytic_0": lf_analytic_0,
    }



def bnn_performance(predictive, x_test, config,  rng_key=None):
    """
    Summarize Bayesian posterior predictive outputs for potential and acceleration.
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
        - "u_samples": posterior samples of potential in physical units, shape (S, N, ...)
        - "a_samples": posterior samples of acceleration in physical units, shape (S, N, 3, ...)
    """
    if rng_key is None:
        rng_key = jr.PRNGKey(0)

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
