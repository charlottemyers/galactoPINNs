"""Model evaluation utilities."""

__all__ = (
    "bnn_performance",
    "evaluate_performance",
    "evaluate_performance_node",
)

from collections.abc import Callable, Mapping
from typing import Any, Literal
from galactoPINNs.models.static_model import StaticModel

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




def svi_performance(
    predictive: Callable,
    x_test: Array,
    config: Mapping[str, Any],
    net_template: StaticModel,
    analytic_param_dists: Mapping[str, Mapping[str, Any]] | None = None,
    composite_form: Callable | None = None,
) -> dict[str, Array]:
    """Evaluate posterior predictive statistics on a test set.

    Transforms ``x_test`` to scaled coordinates, draws posterior predictive
    samples for the potential and acceleration via ``predictive``, then
    maps the samples back to physical units and computes per-point means
    and standard deviations.

    Parameters
    ----------
    predictive
        A :class:`~numpyro.infer.Predictive` object wrapping :func:`model_svi`,
        called to draw posterior samples.
    x_test
        Test positions in physical units, shape ``(N, 3)`` in ``kpc``.
    config
        Model configuration dictionary. Must contain ``"x_transformer"``,
        ``"u_transformer"``, and ``"a_transformer"`` with ``.transform()``
        and ``.inverse_transform()`` methods.
    net_template
        Instantiated :class:`~galactoPINNs.models.static_model.StaticModel`
        forwarded to ``predictive``.
    analytic_param_dists
        Nested mapping of parameter name to prior keyword arguments,
        forwarded to ``predictive``. May be ``None`` when
        ``config["trainable"]`` is ``False``.
    composite_form
        Callable that constructs a trainable analytic layer from sampled
        parameters. May be ``None`` when ``config["trainable"]`` is ``False``.

    Returns
    -------
    metrics
        Dictionary containing:

        - ``"u_mean"``: posterior mean potential in physical units, shape ``(N,)``.
        - ``"a_mean"``: posterior mean acceleration in physical units, shape ``(N, 3)``.
        - ``"u_std"``: posterior std of potential in physical units, shape ``(N,)``.
        - ``"a_std"``: posterior std of acceleration in physical units, shape ``(N, 3)``.
        - ``"u_samples"``: full posterior potential samples in physical units, shape ``(S, N)``.
        - ``"a_samples"``: full posterior acceleration samples in physical units, shape ``(S, N, 3)``.

    """
    x_test_scaled = config["x_transformer"].transform(x_test)

    pred = predictive(
        jr.PRNGKey(2),
        x_test_scaled,
        a_obs=None,
        config=config,
        net_template=net_template,
        analytic_param_dists=analytic_param_dists,
        composite_form=composite_form,
    )

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


def gauge_invariant_rel_resid(
    true_u: Array,
    pred_u: Array,
    ny: int,
    nx: int,
    iy_ref: int | None = None,
    ix_ref: int | None = None,
    *,
    mode: str = "delta_ref",
    ref_point: str = "center",
    eps: float = 1e-12,
) -> tuple[Array, Array, Array]:
    """Compute a gauge-invariant relative residual map between true and predicted potentials.

    Gravitational potentials are defined only up to an additive constant, so
    naive relative residuals are gauge-dependent. This function removes the
    gauge ambiguity via one of two strategies before computing the residual,
    and returns the result as a percentage.

    Parameters
    ----------
    true_u
        True potential values, shape ``(N,)`` or ``(ny, nx)``, to be reshaped
        to ``(ny, nx)``.
    pred_u
        Predicted potential values, same shape convention as ``true_u``.
    ny
        Number of grid points along the y-axis.
    nx
        Number of grid points along the x-axis.
    iy_ref
        Row index of the reference point used in ``"delta_ref"`` mode. If
        ``None``, falls back to ``ref_point``.
    ix_ref
        Column index of the reference point used in ``"delta_ref"`` mode. If
        ``None``, falls back to ``ref_point``.
    mode
        Gauge-removal strategy. Options:

        - ``"delta_ref"``: subtract the value at a reference point from both
          fields before computing the residual.
        - ``"median_offset"`` (or ``"median"``): subtract the median of
          ``pred_u - true_u`` from the prediction before computing the
          residual.

        Default ``"delta_ref"``.
    ref_point
        Reference point selection when ``iy_ref``/``ix_ref`` are not provided
        and ``mode="delta_ref"``. Currently only ``"center"`` is supported,
        which uses ``(ny // 2, nx // 2)``. Default ``"center"``.
    eps
        Small value added to the denominator to avoid division by zero.
        Default ``1e-12``.

    Returns
    -------
    true_u_xy
        True potential reshaped to ``(ny, nx)``.
    pred_u_xy_gauged
        Gauge-corrected predicted potential, shape ``(ny, nx)``. Equal to
        ``pred_u_xy`` in ``"delta_ref"`` mode and ``pred_u_xy - C`` in
        ``"median_offset"`` mode.
    rel_resid
        Percentage relative residual map, shape ``(ny, nx)``.

    Raises
    ------
    ValueError
        If ``mode="delta_ref"`` and neither ``iy_ref``/``ix_ref`` nor
        ``ref_point="center"`` are provided, or if an unknown ``mode`` is
        given.

    """
    true_u_xy = jnp.asarray(true_u).reshape((ny, nx))
    pred_u_xy = jnp.asarray(pred_u).reshape((ny, nx))

    mode = (mode or "delta_ref").lower()

    if mode == "delta_ref":
        if iy_ref is None or ix_ref is None:
            if ref_point == "center":
                iy_ref, ix_ref = ny // 2, nx // 2
            else:
                raise ValueError("Provide iy_ref/ix_ref or set ref_point='center'.")

        true_ref = true_u_xy[iy_ref, ix_ref]
        pred_ref = pred_u_xy[iy_ref, ix_ref]

        dtrue = true_u_xy - true_ref
        dpred = pred_u_xy - pred_ref

        rel_resid = 100 * (dtrue - dpred) / (jnp.abs(dtrue) + eps)
        return true_u_xy, pred_u_xy, rel_resid

    elif mode in ("median_offset", "median"):
        C = jnp.median(pred_u_xy - true_u_xy)
        pred_gf = pred_u_xy - C

        rel_resid = 100 * (true_u_xy - pred_gf) / (jnp.abs(true_u_xy) + eps)
        return true_u_xy, pred_gf, rel_resid

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'delta_ref' or 'median_offset'.")
