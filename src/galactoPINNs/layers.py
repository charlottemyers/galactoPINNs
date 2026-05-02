"""Neural network layers for physics-informed models."""

__all__ = (
    "CartesianToModifiedSphericalLayer",
    "ExternalPytree",
    "FuseandBoundary",
    "ScaleNNPotentialLayer",
    "SmoothMLP",
    "TrainableGalaxPotential",
    "ZeroPotential",
)

import functools as ft
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import galax.potential as gp
import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, ScalarLike

_MASS_KEYS: frozenset[str] = frozenset({"m", "m_tot"})


class ExternalPytree(nnx.Variable):
    """Variable wrapper for external pytrees (like equinox modules)."""


class ActivationFn(Protocol):
    """Callable activation function interface."""

    def __call__(self, x: Array, /) -> Array: ...


class SmoothMLP(nnx.Module):
    """Multi-layer perceptron with a smooth activation.

    This module builds a stack of Linear layers with an activation applied after
    each hidden layer, followed by a final Linear layer that outputs a scalar
    per example.

    Parameters
    ----------
    in_features
        Input feature dimension.
    width
        Hidden layer width (number of units) for each Linear layer.
    depth
        Number of hidden layers.
    act
        Activation function applied after each hidden layer. Must be a callable
        that maps an array to an array of the same shape. Defaults to
        ``jax.nn.gelu``.
    rngs
        Random number generator state for parameter initialization.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from flax import nnx
    >>> mlp = SmoothMLP(
    ...     in_features=3, width=16, depth=2, act=jax.nn.tanh, rngs=nnx.Rngs(0)
    ... )
    >>> x = jnp.ones((4, 3))
    >>> y = mlp(x)
    >>> y.shape
    (4,)

    """

    network: nnx.Sequential

    def __init__(
        self,
        in_features: int,
        width: int = 128,
        depth: int = 3,
        act: ActivationFn = ft.partial(jax.nn.gelu, approximate=False),  # noqa: B008
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the MLP layers."""
        layers = [
            layer
            for i in range(depth)
            for layer in (
                nnx.Linear(in_features if i == 0 else width, width, rngs=rngs),
                # Repeat width -> width blocks for the remaining hidden layers.
                act,
            )
        ]
        output_in_features = width if depth > 0 else in_features
        layers.append(nnx.Linear(output_in_features, 1, rngs=rngs))
        self.network = nnx.Sequential(*layers)

    def __call__(self, x: Array, /) -> Array:
        # Degenerate case: no hidden layers, just a linear readout.
        """Forward pass.

        Parameters
        ----------
        x
            Input array, typically shape ``(N, D)``.

        Returns
        -------
        y
            Scalar output per example. Shape is typically ``(N,)``.

        """
        return jnp.squeeze(self.network(x), axis=-1)


class ZeroPotential(nnx.Module):
    """A zero-output potential module used when the NN component is disabled.

    Matches the ``SmoothMLP`` calling convention: takes input ``x`` of shape
    ``(N, D)`` and returns zeros of shape ``(N,)``.

    """

    def __call__(self, x: Array, /) -> Array:
        """Return zeros matching the batch size of ``x``."""
        return jnp.zeros(x.shape[:-1])


class CartesianToModifiedSphericalLayer(nnx.Module):
    """Converts Cartesian coordinates to modified spherical coordinates.

    This layer transforms 3D Cartesian coordinates (x, y, z) into a 5D
    representation consisting of a clipped radius, a clipped inverse radius,
    and the Cartesian unit vector.

    The output vector is `[r_i, r_e, s, t, u]`, where:
    - `r_i` is the radius clipped to a maximum value.
    - `r_e` is the inverse radius, also clipped.
    - `(s, t, u)` is the Cartesian unit vector (x/r, y/r, z/r).

    Parameters
    ----------
    clip
        The maximum value to which the radius and inverse radius are clipped.
        Helps stabilize the inputs to subsequent layers.

    """

    clip: float = 1.0

    def __init__(self, clip: float = 1.0) -> None:
        """Initialize the coordinate transformation layer."""
        self.clip = clip

    def __call__(self, X_cart: Array, /) -> Array:
        """Transform Cartesian coordinates to modified spherical representation.

        Parameters
        ----------
        X_cart
            Cartesian coordinates of shape ``(3,)`` or ``(N, 3)``.

        Returns
        -------
        Array
            Modified spherical representation ``[r_i, r_e, s, t, u]`` with
            shape ``(5,)`` or ``(N, 5)``.

        """
        X2 = jnp.atleast_2d(X_cart)  # (N, 3)
        r = jnp.linalg.norm(X2, axis=-1)  # (N,)
        r_safe = jnp.maximum(r, jnp.finfo(X2.dtype).tiny)  # (N,)
        r_i = jnp.clip(r, 0.0, self.clip)  # (N,)
        r_e = jnp.clip(1.0 / r_safe, 0.0, self.clip)  # (N,)
        stu = X2 / r_safe[:, None]  # (N, 3)
        Y2 = jnp.concatenate([r_i[:, None], r_e[:, None], stu], axis=-1)  # (N, 5)
        return Y2[(0 if X_cart.ndim == 1 else Ellipsis)]


# --- Scale function implementations (module-level, bound via ft.partial) ---
# Shared final signature after partial application: (x_cart, r, r_s, t) -> Array


def _scale_one(x_cart: Array, r: Array, r_s: float, t: Any) -> Array:  # noqa: ARG001
    return jnp.ones_like(r)


def _scale_power(x_cart: Array, r: Array, r_s: float, t: Any, *, power: float) -> Array:  # noqa: ARG001
    return jnp.power(1.0 / r, power)


def _scale_nfw(
    x_cart: Array,  # noqa: ARG001
    r: Array,
    r_s: float,
    t: Any,  # noqa: ARG001
    *,
    x_transformer: Any,
) -> Array:
    r_s_scaled = x_transformer.transform(r_s)
    return jnp.log(1.0 + r / r_s_scaled) / r


def _scale_precomputed(*_: Any, precomputed: Array) -> Array:
    return precomputed


def _scale_external(
    x_cart: Array,
    r: Array,
    r_s: float,  # noqa: ARG001
    t: Any,
    *,
    external_scale: ExternalPytree,
    r_ref: float | None,
    eps_frac: float,
    clip_min: float,
    clip_max: float,
    reciprocal: bool,
) -> Array:
    xB = jnp.atleast_2d(x_cart)
    ext_potential = external_scale.value

    u = ext_potential.potential(xB, t=t).squeeze()
    s_raw = jnp.abs(u)

    if r_ref is None:
        s_norm = s_raw
    else:
        r_safe = jnp.maximum(r, 1e-12)
        x_dir = xB / r_safe[:, None]
        u_ref = ext_potential.potential(x_dir * r_ref, t=t).squeeze()
        s_ref = jnp.abs(u_ref)
        eps = eps_frac * jnp.maximum(s_ref, 1e-12)
        s_norm = (s_raw + eps) / (s_ref + eps)

    s_out = 1.0 / jnp.maximum(s_norm, 1e-12) if reciprocal else s_norm
    return jnp.clip(s_out, clip_min, clip_max)


# Type alias for the fully-bound scale callable.
_ScaleFn = Callable[[Array, Array, float, Any], Array]


class ScaleNNPotentialLayer(nnx.Module):
    """Apply an analytic, radius-dependent prefactor to a proxy potential."""

    scale_fn: _ScaleFn

    def __init__(
        self,
        config: Mapping[str, Any],
        external_scale: ExternalPytree | None = None,
    ) -> None:
        """Initialize the scaling layer, resolving all config at construction time.

        Parameters
        ----------
        config
            Configuration dict. The ``"scale"`` key selects the mode:
            - ``str``: ``"one"``, ``"power"``, or ``"nfw"`` for built-in modes.
            - ``Array``: pre-computed scale values used directly.
            - Anything else: external potential mode; pass the wrapped object via
              ``external_scale``.
        external_scale
            An ``ExternalPytree``-wrapped galax potential for dynamic scaling.
            Required when ``config["scale"]`` is not a string or array.

        """
        scale_val = config.get("scale", "one")

        if isinstance(scale_val, str):
            mode = scale_val
        elif isinstance(scale_val, jax.Array):
            mode = "precomputed"
        else:
            mode = "external"

        # Keep as a named attribute so NNX can track the Variable.
        if mode == "external" and external_scale is None:
            raise ValueError(
                "scale mode is 'external' but no external_scale was provided."
            )

        self._default_r_s: float = float(config.get("r_s", 1.0))

        # Bind the correct implementation once — __call__ has zero branching.
        match mode:
            case "one":
                self.scale_fn: _ScaleFn = _scale_one
            case "power":
                self.scale_fn = ft.partial(
                    _scale_power, power=float(config.get("power", 1.0))
                )
            case "nfw":
                self.scale_fn = ft.partial(
                    _scale_nfw, x_transformer=config["x_transformer"]
                )
            case "precomputed":
                self.scale_fn = ft.partial(_scale_precomputed, precomputed=scale_val)
            case _:  # "external"
                self.scale_fn = ft.partial(
                    _scale_external,
                    external_scale=external_scale,
                    r_ref=config.get("scale_r_ref", None),
                    eps_frac=float(config.get("scale_eps_frac", 1e-6)),
                    clip_min=float(config.get("scale_clip_min", 1e-3)),
                    clip_max=float(config.get("scale_clip_max", 1e3)),
                    reciprocal=bool(config.get("scale_reciprocal", True)),
                )

    def __call__(
        self, x_cart: Array, u_nn: Array, /, *, r_s: float | None = None, t: Any = 0
    ) -> Array:
        """Apply scaling to the NN potential."""
        r = jnp.linalg.norm(x_cart, axis=-1)
        r_s_ = r_s if r_s is not None else self._default_r_s
        return self.scale_fn(x_cart, r, r_s_, t) * u_nn


class TrainableGalaxPotential(nnx.Module):
    """NNX module that wraps a Galax potential class with trainable parameters.

    Selected constructor arguments become trainable ``nnx.Param`` values; all
    others are stored as fixed arrays. Mass-like parameters (``"m"`` or
    ``"m_tot"``) are stored in log₁₀-space to enforce positivity and improve
    numerical conditioning.

    Parameters
    ----------
    pot_cls
        Galax potential constructor. Must return an object with a
        ``.potential(positions, t=...)`` method.
    init_kwargs
        Mapping from constructor-argument name to its initial value.
    trainable
        Keys from ``init_kwargs`` that should become trainable parameters.
    rngs
        Unused; kept for API consistency.

    """

    def __init__(
        self,
        pot_cls: type[gp.AbstractPotential],
        init_kwargs: Mapping[str, float],
        trainable: tuple[str, ...],
        *,
        rngs: nnx.Rngs | None = None,  # noqa: ARG002
    ) -> None:
        """Initialize trainable and fixed parameters from init_kwargs."""
        if "r_s" not in init_kwargs:
            raise KeyError("TrainableGalaxPotential requires 'r_s' in init_kwargs.")

        self.pot_cls = pot_cls
        self._trainable_keys = trainable

        params_dict: dict[str, nnx.Param | Array] = {}
        for name, val in init_kwargs.items():
            arr = jnp.asarray(val, dtype=jnp.float32)
            if name in trainable and name in _MASS_KEYS:
                params_dict[f"log10_{name}"] = nnx.Param(jnp.log10(arr))
            elif name in trainable:
                params_dict[name] = nnx.Param(arr)
            else:
                params_dict[name] = arr

        self._params = nnx.Dict(params_dict)

    def _build_kwargs(self) -> dict[str, Array]:
        """Reconstruct potential kwargs, back-transforming log₁₀-encoded params."""
        out: dict[str, Array] = {}
        for key, var in self._params.items():
            raw = var.value if isinstance(var, nnx.Variable) else var
            if key.startswith("log10_"):
                out[key[6:]] = jnp.power(10.0, raw)
            else:
                out[key] = raw
        return out

    def __call__(self, positions: Array, t: ScalarLike = 0) -> Array:
        """Evaluate the trainable potential at given positions.

        Parameters
        ----------
        positions
            Cartesian positions, shape ``(N, 3)``.
        t
            Time at which to evaluate the potential. Default 0.

        Returns
        -------
        Array
            Potential values at positions.

        """
        kwargs = self._build_kwargs()
        pot = self.pot_cls(**kwargs, units="galactic")
        return pot.potential(positions, t=t)

    @property
    def r_s(self) -> Array:
        """Convenience property to access the current r_s value."""
        return self._build_kwargs()["r_s"]

    def potential(self, positions: Array, *, t: Any = 0) -> Array:
        """Return the scalar potential at ``positions`` (drops the r_s output)."""
        phi, _ = self(positions, t=t)
        return phi

    def acceleration(self, positions: Array, *, t: Any = 0) -> Array:
        """Return acceleration at ``positions`` via the underlying Galax potential."""
        built_params = self._get_built_params()
        pot = self.PotClass(**built_params, units="galactic")
        return pot.acceleration(positions, t=t)


class FuseandBoundary(nnx.Module):
    """Fuse a neural-network potential with an analytic potential.

    Uses a smooth radial transition. The fused model is constructed as
        u_model(r) = g(r) * u_nn(r) + u_analytic(r),
    where g(r) = 1 - h(r) and h(r) transitions from ~0 at small radii to a
    configurable saturation value at large radii. This enforces a boundary
    condition in which the model asymptotes to the analytic potential at large
    radius while allowing the NN to represent residual structure at
    small/intermediate radii.

    The transition function h(r) can be configured as either:
    - Tanh blend:
        h(r) = 0.5 * saturation * (1 + tanh(k_smooth * (r - r_trans)))
    - Radial power law:
        h(r) = saturation * (r / (r + r_smooth))**radial_power

    Parameters
    ----------
    config
        Configuration mapping. Expected keys:

        Required:
        - x_transformer: Transformer that maps between scaled and physical
          coordinates. Must implement ``inverse_transform(positions)``.

        Optional (tanh blend):
        - r_trans: Transition radius (in physical units). Default 200.
        - k_smooth: Steepness of the tanh transition. Default 0.5.
        - train_k: If True, make k_smooth trainable. Default False.
        - min_k: Lower bound for k_smooth when trainable. Default 0.01.

        Optional (power-law blend):
        - radial_power: If provided, uses the power-law blend instead of tanh.
        - r_smooth: Smoothing radius for the power-law blend. Default 150.0.

        Optional (both):
        - saturation: Asymptotic value of h(r) as r -> infinity. Default 1.0.

    rngs
        Random number generator state (unused but kept for API consistency).

    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        rngs: nnx.Rngs | None = None,  # noqa: ARG002
    ) -> None:
        """Initialize blending parameters from config."""
        self.config = config
        self.log_k: float = float(config.get("k_smooth", 1.0))
        self.r_trans: float = float(config.get("r_trans", 200.0))

    def H(
        self, x: Array, r_smooth: float, k_smooth: Array, saturation: float = 1.0
    ) -> Array:
        """Tanh transition used as the default blend function h(r)."""
        return 0.5 * saturation * (1.0 + jnp.tanh(k_smooth * (x - r_smooth)))

    def radial_power_law(
        self, r: Array, r_smooth: float, exp: float, saturation: float
    ) -> Array:
        """Power-law alternative blend function h(r)."""
        return saturation * jnp.power(r / (r + r_smooth), exp)

    def __call__(
        self, positions: Array, u_nn: Array, u_analytic: Array
    ) -> dict[str, Array]:
        """Fuse NN and analytic potentials with a smooth radial transition.

        Parameters
        ----------
        positions
            Scaled Cartesian positions, shape ``(N, 3)`` or ``(3,)``.
        u_nn
            Neural network potential component (scaled).
        u_analytic
            Analytic potential component (scaled).

        Returns
        -------
        Array
            The fused potential (scaled).

        """
        x_transformer = self.config["x_transformer"]
        dimensional_positions: Array = x_transformer.inverse_transform(positions)

        saturation: float = float(self.config.get("saturation", 1.0))

        if self.config.get("train_k", False):
            min_k: float = float(self.config.get("min_k", 0.01))
            k_smooth: Array = jnp.maximum(min_k, jnp.exp(jnp.asarray(self.log_k)))
        else:
            k_smooth = jnp.asarray(float(self.config.get("k_smooth", 0.5)))

        r_trans: float = float(self.r_trans)

        r = jnp.linalg.norm(dimensional_positions, axis=-1)

        if self.config.get("radial_power", None) is not None:
            power: float = float(self.config["radial_power"])
            r_smooth: float = float(self.config.get("r_smooth", 150.0))
            h = self.radial_power_law(r, r_smooth, power, saturation=saturation)
        else:
            h = self.H(r, r_trans, k_smooth, saturation=saturation)

        g = 1.0 - h

        # Blend NN output and analytic function
        return g * u_nn + u_analytic
