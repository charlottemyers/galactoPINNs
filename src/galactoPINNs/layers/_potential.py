"""Potential-related layers: trainable potentials and boundary fusion."""

from collections.abc import Mapping
from typing import Any, Final, NotRequired, TypedDict, cast

import galax.potential as gp
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, ScalarLike

_MASS_KEYS: frozenset[str] = frozenset({"m", "m_tot"})


class ZeroPotential(nnx.Module):
    """A zero-output potential module used when the NN component is disabled.

    Matches the ``MLP`` calling convention: takes input ``x`` of shape
    ``(N, D)`` and returns zeros of shape ``(N,)``.

    """

    def __call__(self, x: Array, /) -> Array:
        """Return zeros matching the batch size of ``x``."""
        return jnp.zeros(x.shape[:-1])


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


class FuseandBoundaryConfig(TypedDict):
    """Configuration dictionary for FuseandBoundary.

    All fields except x_transformer are optional and have sensible defaults.
    """

    # Coordinate transformer (required)
    x_transformer: Any

    # Tanh blend parameters
    r_trans: NotRequired[float]
    k_smooth: NotRequired[float]
    train_k: NotRequired[bool]
    min_k: NotRequired[float]

    # Power-law blend parameters
    radial_power: NotRequired[float | None]
    r_smooth: NotRequired[float]

    # Shared parameters
    saturation: NotRequired[float]


DEFAULT_FUSE_AND_BOUNDARY_CONFIG: Final[FuseandBoundaryConfig] = {
    "x_transformer": None,
    "r_trans": 200.0,
    "k_smooth": 0.5,
    "train_k": False,
    "min_k": 0.01,
    "radial_power": None,
    "r_smooth": 150.0,
    "saturation": 1.0,
}


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
        config: FuseandBoundaryConfig,
        *,
        rngs: nnx.Rngs | None = None,  # noqa: ARG002
    ) -> None:
        """Initialize blending parameters from config."""
        # Merge provided config with defaults, filtering out any invalid keys.
        self.config = config = cast(
            "FuseandBoundaryConfig",
            DEFAULT_FUSE_AND_BOUNDARY_CONFIG
            | {k: config[k] for k in DEFAULT_FUSE_AND_BOUNDARY_CONFIG if k in config},
        )
        self.log_k: float = float(config.get("k_smooth", 0.5))
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

    def __call__(self, positions: Array, u_nn: Array, u_analytic: Array) -> Array:
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
