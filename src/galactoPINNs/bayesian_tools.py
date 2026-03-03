"""Bayesian neural network model implementation."""

import copy
from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from flax import nnx
from jaxtyping import Array
from numpyro.contrib.module import random_nnx_module
from numpyro.infer import SVI, Trace_ELBO, init_to_feasible
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from galactoPINNs.models.static_model import StaticModel


def model_svi(
    x: Array,
    a_obs: Array | None = None,
    *,
    config: Mapping[str, Any],
    analytic_param_dists: Mapping[str, Mapping[str, Any]],
    net_template: StaticModel,
    sigma_theta: float = 0.05,
    sigma_lambda: float = 0.05,
    sigma_a: float = 2e-4,
    lambda_rel: float = 0.1,
    orbit_q: Array | None = None,
    orbit_p: Array | None = None,
    w_orbit: float = 1.0,
    composite_form: Callable | None = None,
) -> None:
    r"""NumPyro model for SVI-based inference of a gravitational potential BNN.

    Defines the probabilistic model used during stochastic variational
    inference. Optionally samples analytic potential parameters, places a
    normal prior over the BNN weights, and registers likelihood terms for
    acceleration residuals and orbital energy conservation.

    Parameters
    ----------
    x
        Scaled Cartesian input positions, shape ``(N, 3)``.
    a_obs
        Observed accelerations in scaled units, shape ``(N, 3)``. If ``None``,
        the data likelihood term is skipped.
    config
        Model configuration dictionary. The ``"trainable"`` key controls
        whether analytic potential parameters are sampled.
    analytic_param_dists
        Nested mapping of parameter name to distribution keyword arguments
        (``loc``, and optionally ``low``/``high``) passed to
        ``TruncatedNormal``. Expected keys when ``config["trainable"]`` is
        ``True``: ``"log_m_halo"``, ``"log_m_disk"``, ``"log_rs"``,
        ``"log_disk_a"``, ``"log_disk_b"``.
    net_template
        Instantiated :class:`~galactoPINNs.models.static_model.StaticModel`
        used as the template for :func:`random_nnx_module`.
    sigma_theta
        Standard deviation of the ``TruncatedNormal`` prior over log-space
        analytic parameters. Default ``0.05``.
    sigma_lambda
        Standard deviation of the normal prior over BNN weights. Default
        ``0.05``.
    sigma_a
        Observation noise scale for accelerations, used to set the data
        likelihood weight as ``1 / (2 * sigma_a^2)``. Default ``2e-4``.
    lambda_rel
        Relative weighting of the normalised acceleration residual term
        alongside the absolute residual. Default ``0.1``.
    orbit_q
        Scaled orbit positions, shape ``(B, T, 3)``. If provided together with
        ``orbit_p``, an energy-conservation loss is added.
    orbit_p
        Scaled orbit velocities, shape ``(B, T, 3)``.
    w_orbit
        Weight applied to the orbital energy conservation loss term. Default
        ``1.0``.
    composite_form
        Callable that constructs a trainable analytic layer from sampled
        parameters. Required when ``config["trainable"]`` is ``True``.

    Notes
    -----
    The acceleration likelihood uses a combined absolute and relative residual:

    .. math::

        \\mathcal{L}_{\\mathrm{acc}} = \\frac{1}{2\\sigma_a^2}
        \\operatorname{E}\\left[\\|\\Delta a\\| +
        \\lambda_{\\mathrm{rel}} \\frac{\\|\\Delta a\\|}{\\|a_{\\mathrm{obs}}\\|}\\right]

    The orbital energy loss penalises relative drift from the initial energy
    along each trajectory:

    .. math::

        \\mathcal{L}_{\\mathrm{orbit}} = \\operatorname{E}\\left[
        \\left(\\frac{E_t - E_0}{|E_0|}\\right)^2\\right]

    """
    if config.get("trainable", False):
        log_m_halo = numpyro.sample(
            "log_m_halo",
            dist.TruncatedNormal(scale=sigma_theta, **analytic_param_dists["log_m_halo"]),
        )
        log_m_disk = numpyro.sample(
            "log_m_disk",
            dist.TruncatedNormal(scale=sigma_theta, **analytic_param_dists["log_m_disk"]),
        )
        log_rs = numpyro.sample(
            "log_rs",
            dist.TruncatedNormal(scale=sigma_theta, **analytic_param_dists["log_rs"]),
        )
        log_disk_a = numpyro.sample(
            "log_disk_a",
            dist.TruncatedNormal(scale=sigma_theta, **analytic_param_dists["log_disk_a"]),
        )
        log_disk_b = numpyro.sample(
            "log_disk_b",
            dist.TruncatedNormal(scale=sigma_theta, **analytic_param_dists["log_disk_b"]),
        )

        halo_r_s  = jnp.exp(log_rs)
        halo_mass = jnp.exp(log_m_halo)
        disk_mass = jnp.exp(log_m_disk)
        disk_a    = jnp.exp(log_disk_a)
        disk_b    = jnp.exp(log_disk_b)

        trainable_analytic_layer = composite_form(
            init_disk_a=disk_a,
            init_disk_b=disk_b,
            init_halo_r_s=halo_r_s,
            init_halo_mass=halo_mass,
            init_disk_mass=disk_mass,
        )
    else:
        trainable_analytic_layer = None

    bnn = random_nnx_module(
        "full_model",
        net_template,
        prior=dist.Normal(0.0, sigma_lambda),
    )

    if config.get("trainable", False):
        out = bnn(x, trainable_analytic_layer=trainable_analytic_layer)
    else:
        out = bnn(x)

    a_pred = out["acceleration"]
    u_pred = out["potential"]
    numpyro.deterministic("acceleration", a_pred)
    numpyro.deterministic("potential", u_pred)

    # --- Acceleration data likelihood ---
    if a_obs is not None:
        data_weight = 1.0 / (2 * sigma_a**2)
        diff = a_pred - a_obs
        diff_norm = jnp.linalg.norm(diff, axis=1)
        a_true_norm = jnp.linalg.norm(a_obs, axis=1) + 1e-12
        per_point = diff_norm + lambda_rel * (diff_norm / a_true_norm)
        loss = jnp.mean(per_point)
        numpyro.factor("acc_loss", -data_weight * loss)

        # --- Orbital energy conservation loss ---
        if (orbit_q is not None) and (orbit_p is not None):
            n_sub = 20
            key = numpyro.prng_key()
            idx = jr.choice(key, orbit_q.shape[0], shape=(n_sub,), replace=False)
            orbit_q_sub = orbit_q[idx]  # (n_sub, T, 3)
            orbit_v_sub = orbit_p[idx]  # (n_sub, T, 3)

            B, T, _ = orbit_q_sub.shape
            T_ke  = 0.5 * jnp.sum(orbit_v_sub**2, axis=-1)  # (B, T)
            q_flat = orbit_q_sub.reshape(B * T, 3)

            if config.get("trainable", False):
                phi = bnn(
                    q_flat, trainable_analytic_layer=trainable_analytic_layer
                )["potential"].reshape(B, T)
            else:
                phi = bnn(q_flat)["potential"].reshape(B, T)

            E = T_ke + phi  # (B, T)

            E0 = E[:, 0:1]
            relative_drift = (E - E0) / (jnp.abs(E0) + 1e-8)
            L_orbit = jnp.mean(relative_drift**2)
            numpyro.factor("orbit_E_loss", -w_orbit * L_orbit)


def make_guide_for_config(
    config: Mapping[str, Any],
    analytic_param_dists: Mapping[str, Mapping[str, Any]],
    analytic_form: Callable | None,
    net_template: StaticModel,
) -> AutoNormal:
    """Construct an :class:`~numpyro.infer.autoguide.AutoNormal` variational guide.

    Wraps :func:`model_svi` in a closure that binds ``config``,
    ``analytic_param_dists``, and ``analytic_form``, then builds an
    ``AutoNormal`` guide over the resulting model. The guide is initialised
    with :func:`~numpyro.infer.init_to_feasible` to avoid invalid starting
    points.

    Parameters
    ----------
    config
        Model configuration dictionary forwarded to :func:`model_svi`.
    analytic_param_dists
        Nested mapping of parameter name to prior keyword arguments,
        forwarded to :func:`model_svi`.
    analytic_form
        Callable that constructs a trainable analytic layer from sampled
        parameters. May be ``None`` when ``config["trainable"]`` is ``False``.
    net_template
        Instantiated :class:`~galactoPINNs.models.static_model.StaticModel`
        forwarded to :func:`model_svi`.

    Returns
    -------
    guide
        An ``AutoNormal`` guide with mean-field normal approximations over
        all latent sites in :func:`model_svi`.

    Notes
    -----
    The inner ``_guided_model`` strips ``config``, ``analytic_param_dists``,
    and ``composite_form`` from ``**kw`` before forwarding to
    :func:`model_svi`, preventing duplicate-keyword errors when the guide is
    called with those keys present.

    """
    def _guided_model(x: Array, a_obs: Array | None = None, **kw: Any) -> None:
        kw.pop("config", None)
        kw.pop("analytic_param_dists", None)
        kw.pop("composite_form", None)
        return model_svi(
            x, a_obs,
            config=config,
            net_template=net_template,
            analytic_param_dists=analytic_param_dists,
            composite_form=analytic_form,
            **kw,
        )

    return AutoNormal(_guided_model, init_loc_fn=init_to_feasible)


def make_svi(
    *,
    guide: Callable[..., Any],
    sigma_lambda: float,
    sigma_theta: float,
    sigma_a: float,
    lambda_rel: float,
    config: Mapping[str, Any],
    analytic_param_dists: Mapping[str, Mapping[str, Any]],
    orbit_q: Array | None = None,
    orbit_p: Array | None = None,
    w_orbit: float = 1.0,
    lr: float = 5e-3,
    composite_form: Callable | None = None,
    net_template: StaticModel | None = None,
) -> SVI:
    """Construct an :class:`~numpyro.infer.SVI` object with :func:`model_svi` closed over config.

    Binds all hyperparameters and data into a ``_model`` closure, then wraps
    it with the provided guide and an :class:`~numpyro.optim.Adam` optimiser
    using the :class:`~numpyro.infer.Trace_ELBO` objective.

    Parameters
    ----------
    guide
        Variational guide, typically produced by :func:`make_guide_for_config`.
    sigma_lambda
        Standard deviation of the normal prior over BNN weights.
    sigma_theta
        Standard deviation of the ``TruncatedNormal`` prior over log-space
        analytic parameters.
    sigma_a
        Observation noise scale for accelerations, controlling the weight of
        the data likelihood term.
    lambda_rel
        Relative weighting of the normalised acceleration residual alongside
        the absolute residual in the data likelihood.
    config
        Model configuration dictionary forwarded to :func:`model_svi`.
    analytic_param_dists
        Nested mapping of parameter name to prior keyword arguments,
        forwarded to :func:`model_svi`.
    orbit_q
        Scaled orbit positions, shape ``(B, T, 3)``, forwarded to
        :func:`model_svi`. If ``None``, the orbit loss is disabled.
    orbit_p
        Scaled orbit velocities, shape ``(B, T, 3)``, forwarded to
        :func:`model_svi`. If ``None``, the orbit loss is disabled.
    w_orbit
        Weight applied to the orbital energy conservation loss term. Default
        ``1.0``.
    lr
        Learning rate for the :class:`~numpyro.optim.Adam` optimiser. Default
        ``2e-3``.
    composite_form
        Callable that constructs a trainable analytic layer from sampled
        parameters. May be ``None`` when ``config["trainable"]`` is ``False``.
    net_template
        Instantiated :class:`~galactoPINNs.models.static_model.StaticModel`
        forwarded to :func:`model_svi`.

    Returns
    -------
    svi
        A configured :class:`~numpyro.infer.SVI` instance ready for training
        via ``.run()`` or ``.update()``.

    """
    def _model(
        x: Array,
        a_obs: Array | None = None,
    ) -> None:
        return model_svi(
            x,
            a_obs,
            config=config,
            analytic_param_dists=analytic_param_dists,
            sigma_lambda=sigma_lambda,
            sigma_theta=sigma_theta,
            sigma_a=sigma_a,
            lambda_rel=lambda_rel,
            orbit_q=orbit_q,
            orbit_p=orbit_p,
            w_orbit=w_orbit,
            composite_form=composite_form,
            net_template=net_template,
        )

    return SVI(_model, guide, Adam(lr), Trace_ELBO())

def run_window(
    prev_result: Any,
    guide: Callable[..., Any],
    *,
    x_train: Array,
    a_train: Array,
    steps: int = 1000,
    lr: float = 5e-3,
    sigma_lambda: float = 0.05,
    sigma_theta: float = 0.05,
    sigma_a: float = 2e-4,
    lambda_rel: float = 0.1,
    config: Mapping[str, Any] | None = None,
    analytic_param_dists: Mapping[str, Mapping[str, float]] | None = None,
    composite_form: Callable | None = None,
    orbit_q: Array | None = None,
    orbit_p: Array | None = None,
    w_orbit: float = 1.0,
    net_template: StaticModel | None = None,
    rng_key: jr.PRNGKey,
    warm_params: dict | None = None,
) -> Any:
    """Run one SVI training window, optionally warm-starting from a previous result.

    Constructs an :class:`~numpyro.infer.SVI` object via :func:`make_svi` and
    runs it for ``steps`` gradient updates. Parameter initialisation follows a
    priority order: explicit ``warm_params`` take precedence, then parameters
    from ``prev_result``, and finally a fresh random initialisation if neither
    is available.

    Parameters
    ----------
    prev_result
        The result object returned by a previous :meth:`~numpyro.infer.SVI.run`
        call. Its ``.params`` attribute is used to warm-start the optimiser
        when ``warm_params`` is ``None``. Pass ``None`` for the first window.
    guide
        Variational guide, typically produced by :func:`make_guide_for_config`.
    x_train
        Scaled training positions, shape ``(N, 3)``.
    a_train
        Observed training accelerations in scaled units, shape ``(N, 3)``.
    steps
        Number of SVI gradient steps to run. Default ``1000``.
    lr
        Learning rate for the :class:`~numpyro.optim.Adam` optimiser. Default
        ``5e-3``.
    sigma_lambda
        Standard deviation of the normal prior over BNN weights. Default
        ``0.05``.
    sigma_theta
        Standard deviation of the ``TruncatedNormal`` prior over log-space
        analytic parameters. Default ``0.05``.
    sigma_a
        Observation noise scale for accelerations. Default ``2e-4``.
    lambda_rel
        Relative weighting of the normalised acceleration residual. Default
        ``0.1``.
    config
        Model configuration dictionary forwarded to :func:`make_svi`. Must
        not be ``None``.
    analytic_param_dists
        Nested mapping of parameter name to prior keyword arguments,
        forwarded to :func:`make_svi`.
    composite_form
        Callable that constructs a trainable analytic layer from sampled
        parameters. May be ``None`` when ``config["trainable"]`` is ``False``.
    orbit_q
        Scaled orbit positions, shape ``(B, T, 3)``. If ``None``, the orbit
        loss is disabled.
    orbit_p
        Scaled orbit velocities, shape ``(B, T, 3)``. If ``None``, the orbit
        loss is disabled.
    w_orbit
        Weight applied to the orbital energy conservation loss term. Default
        ``1.0``.
    net_template
        Instantiated :class:`~galactoPINNs.models.static_model.StaticModel`
        forwarded to :func:`make_svi`.
    rng_key
        JAX PRNG key passed to :meth:`~numpyro.infer.SVI.run`.
    warm_params
        Explicit parameter dictionary used to initialise the optimiser,
        overriding ``prev_result.params`` when provided.

    Returns
    -------
    result
        The :class:`~numpyro.infer.SVIRunResult` object returned by
        :meth:`~numpyro.infer.SVI.run`, containing ``.params`` and
        ``.losses``.

    Raises
    ------
    ValueError
        If ``config`` is ``None``.

    """
    if config is None:
        raise ValueError("run_window requires `config` to be provided.")

    if rng_key is None:
        rng_key = jr.PRNGKey(0)

    svi = make_svi(
        guide=guide,
        sigma_lambda=sigma_lambda,
        sigma_theta=sigma_theta,
        sigma_a=sigma_a,
        lambda_rel=lambda_rel,
        analytic_param_dists=analytic_param_dists,
        config=config,
        orbit_q=orbit_q,
        orbit_p=orbit_p,
        w_orbit=w_orbit,
        composite_form=composite_form,
        net_template=net_template,
        lr=lr,
    )

    if warm_params is not None:
        return svi.run(rng_key, steps, x_train, a_train, init_params=warm_params)
    elif prev_result is None:
        return svi.run(rng_key, steps, x_train, a_train)
    else:
        return svi.run(rng_key, steps, x_train, a_train, init_params=prev_result.params)


def draw_i(draws: dict[str, Array], i: int) -> dict[str, Array]:
    """Extract the ``i``-th posterior sample from a dictionary of SVI draws.

    Filters to only the latent sites of interest (log-space analytic parameters
    and BNN weights), then indexes into the sample dimension where applicable.

    Parameters
    ----------
    draws
        Dictionary mapping site names to stacked sample arrays, as returned
        by :meth:`~numpyro.infer.SVI.get_samples`. Scalar sites have shape
        ``()`` and batched sites have shape ``(S,)`` or ``(S, ...)``.
    i
        Index of the sample to extract along the leading sample dimension.

    Returns
    -------
    sample
        Dictionary containing only sites whose names begin with ``"log_"`` or
        ``"full_model/"``, with each value being the scalar or array at index
        ``i``.

    """
    out = {}
    for k, v in draws.items():
        if not (k.startswith("log_") or k.startswith("full_model/")):
            continue
        v = jnp.asarray(v)
        out[k] = v if v.ndim == 0 else v[i]
    return out


def theta_from_draw_halo_disk(d: dict[str, Array]) -> dict[str, Array]:
    """Convert a dictionary of log-space posterior samples to physical parameters.

    Exponentiates log-space samples for a halo-disk composite potential into
    their physical counterparts.

    Parameters
    ----------
    d
        Dictionary of log-space parameter samples, expected to contain keys
        ``"log_rs"``, ``"log_m_halo"``, ``"log_m_disk"``, ``"log_disk_a"``,
        and ``"log_disk_b"``.

    Returns
    -------
    theta
        Dictionary of physical-space parameters with keys ``"r_s"``,
        ``"halo_mass"``, ``"disk_mass"``, ``"disk_a"``, and ``"disk_b"``.

    """
    return {
        "r_s":       jnp.exp(d["log_rs"]),
        "halo_mass": jnp.exp(d["log_m_halo"]),
        "disk_mass": jnp.exp(d["log_m_disk"]),
        "disk_a":    jnp.exp(d["log_disk_a"]),
        "disk_b":    jnp.exp(d["log_disk_b"]),
    }



def normalize_kp(kp: tuple) -> tuple:
    """Normalise a key-path tuple, unwrapping single-element nesting if present.

    Parameters
    ----------
    kp
        A key-path tuple, potentially wrapped in an extra layer of nesting
        as ``((k1, k2, ...),)``.

    Returns
    -------
    kp
        The unwrapped key-path tuple ``(k1, k2, ...)``, or the original
        ``kp`` if no unwrapping was needed.

    """
    if isinstance(kp, tuple) and len(kp) == 1 and isinstance(kp[0], tuple):
        return kp[0]
    return kp


def parse_site_to_kp(site: str) -> tuple:
    """Parse a NumPyro site name into a key-path tuple.

    Strips the ``"full_model/"`` prefix and splits on ``"."`` to produce a
    tuple of path components, converting numeric segments to ``int``.

    Parameters
    ----------
    site
        A NumPyro site name beginning with ``"full_model/"``, e.g.
        ``"full_model/mlp.layers.0.weight"``.

    Returns
    -------
    kp
        Tuple of path components with integer indices where applicable, e.g.
        ``("mlp", "layers", 0, "weight")``.

    """
    s = site[len("full_model/"):]
    parts = []
    for part in s.split("."):
        parts.append(int(part) if part.isdigit() else part)
    return tuple(parts)


def _flatten_nested_dict(d: dict, prefix: tuple = ()) -> dict[tuple, Any]:
    """Recursively flatten a nested dict to ``{(key, path, ...): leaf}``.

    Parameters
    ----------
    d
        Arbitrarily nested dictionary to flatten.
    prefix
        Key-path accumulated by the recursive calls. Should be left as the
        default ``()`` at the top-level call.

    Returns
    -------
    flat
        Dictionary mapping each leaf's full key-path tuple to its value.

    """
    out = {}
    for k, v in d.items():
        path = prefix + (k,)
        if isinstance(v, dict):
            out.update(_flatten_nested_dict(v, path))
        else:
            out[path] = v
    return out


def _nested_from_flat(flat: dict[tuple, Any]) -> dict:
    """Convert a flat key-path dictionary back to a nested dict.

    Parameters
    ----------
    flat
        Dictionary mapping key-path tuples ``(k1, k2, ...)`` to leaf values,
        as produced by :func:`_flatten_nested_dict`.

    Returns
    -------
    nested
        Reconstructed nested dictionary.

    """
    nested: dict = {}
    for keys, val in flat.items():
        d = nested
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val
    return nested



def make_net_for_draw(net_template: nnx.Module, draw: dict[str, Array]) -> nnx.Module:
    """Instantiate a network with BNN weights from a single posterior draw.

    Deep-copies ``net_template``, then overwrites its ``Param`` state with
    the ``"full_model/"`` sites from ``draw``, leaving all other parameters
    (e.g. non-trainable buffers) unchanged.

    Parameters
    ----------
    net_template
        The base :class:`~flax.nnx.Module` whose architecture and
        non-parameter state are used as the template.
    draw
        Dictionary of posterior samples for a single draw, as produced by
        :func:`draw_i`. Only entries whose keys begin with ``"full_model/"``
        and whose key-paths match existing ``Param`` sites are applied.

    Returns
    -------
    net_i
        A deep copy of ``net_template`` with ``Param`` values replaced by
        those in ``draw``.

    """
    net_i = copy.deepcopy(net_template)

    param_state = nnx.state(net_i, nnx.Param)

    pure = nnx.to_pure_dict(param_state)
    flat_pure = _flatten_nested_dict(pure)
    valid = {normalize_kp(k) for k in flat_pure.keys()}

    updates_flat = {}
    for name, val in draw.items():
        if not name.startswith("full_model/"):
            continue
        kp = normalize_kp(parse_site_to_kp(name))
        if kp in valid:
            updates_flat[kp] = jnp.asarray(val)

    nnx.replace_by_pure_dict(param_state, _nested_from_flat(updates_flat))
    nnx.update(net_i, param_state)
    return net_i


def make_net_for_draw_with_analytic(
    d: dict[str, Array],
    *,
    config: Mapping[str, Any],
    composite_form: Callable,
) -> tuple[StaticModel, dict[str, Array]]:
    """Instantiate a full model with both BNN weights and analytic parameters from a posterior draw.

    Converts log-space samples to physical parameters via
    :func:`theta_from_draw_halo_disk`, constructs a trainable analytic layer,
    builds a fresh :class:`~galactoPINNs.models.static_model.StaticModel`, and
    then populates its BNN weights from ``d`` via :func:`make_net_for_draw`.

    Parameters
    ----------
    d
        Dictionary of posterior samples for a single draw, as produced by
        :func:`draw_i`. Must contain the log-space analytic parameter keys
        consumed by :func:`theta_from_draw_halo_disk` as well as
        ``"full_model/"`` BNN weight entries.
    config
        Model configuration dictionary forwarded to
        :class:`~galactoPINNs.models.static_model.StaticModel`.
    composite_form
        Callable that constructs a trainable analytic layer from physical
        parameter keyword arguments.

    Returns
    -------
    net_i
        A :class:`~galactoPINNs.models.static_model.StaticModel` with both
        analytic and BNN parameters set from ``d``.
    theta
        Dictionary of physical-space analytic parameters as returned by
        :func:`theta_from_draw_halo_disk`.

    """
    theta = theta_from_draw_halo_disk(d)

    train_layer = composite_form(
        init_halo_r_s=theta["r_s"],
        init_disk_a=theta["disk_a"],
        init_disk_b=theta["disk_b"],
        init_halo_mass=theta["halo_mass"],
        init_disk_mass=theta["disk_mass"],
    )
    net_i = StaticModel(
        config=config,
        trainable_analytic_layer=train_layer,
        rngs=nnx.Rngs(0),
    )
    net_i = make_net_for_draw(net_i, d)
    return net_i, theta
