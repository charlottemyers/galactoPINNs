"""Neural network layers for physics-informed models."""

__all__ = (
    "AnalyticModelLayer",
    "CartesianToModifiedSphericalLayer",
    "FuseModelsLayer",
    "FuseandBoundary",
    "ScaleNNPotentialLayer",
    "SmoothMLP",
    "TrainableGalaxPotential",
)

import functools as ft
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol, TypeAlias, cast

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

ActFn = Callable[[Array], Array]


#######
# Protocols and types
#######

TrainableInitKwargs: TypeAlias = Mapping[str, float]
TrainableKeys: TypeAlias = tuple[str, ...]


class ActivationFn(Protocol):
    """Callable activation function interface."""

    def __call__(self, x: Array, /) -> Array: ...


class _HasPotential(Protocol):
    """Protocol for external analytic potential-like objects used for scaling."""

    def potential(self, x: Array, *, t: Any = ...) -> Array: ...


class _GalaxPotentialCtor(Protocol):
    """Protocol for a Galax potential class/constructor."""

    def __call__(self, *args: Any, **kwargs: Any) -> "_GalaxPotential": ...


class _GalaxPotential(Protocol):
    """Protocol for instantiated Galax potentials used here."""

    def potential(self, positions: Any, *, t: Any = ...) -> Any: ...


class _Transformer(Protocol):
    """Minimal protocol for (inverse_)transformers used in analytic layer."""

    def transform(self, x: Any) -> Any: ...
    def inverse_transform(self, x: Any) -> Any: ...


class _AnalyticPotential(Protocol):
    """Minimal protocol for an analytic potential used here."""

    def potential(self, positions: Any, *, t: Any = ...) -> Any: ...


ScaleSpec = str | _HasPotential

##############


#######
# Utilities
#######
def _radius(x: Array) -> Array:
    """Compute the Euclidean radius of Cartesian coordinates.

    Parameters
    ----------
    x
        Cartesian coordinates. Supported shapes:
        - ``(3,)`` for a single point
        - ``(N, 3)`` for a batch of points

    Returns
    -------
    r
        Euclidean radius:
        - scalar array if ``x`` has shape ``(3,)``
        - array of shape ``(N,)`` if ``x`` has shape ``(N, 3)``

    Raises
    ------
    ValueError
        If ``x`` does not have a supported shape.

    """
    x = jnp.asarray(x)

    if x.ndim > 2:
        msg = f"_radius expects x with ndim 1 or 2, got ndim={x.ndim}, shape={x.shape}."
        raise ValueError(msg)

    return jnp.linalg.norm(x, axis=-1)


def _as_batch(x: Array) -> Array:
    """Ensure a leading batch dimension.

    Parameters
    ----------
    x
        Input array with shape ``(D,)`` or ``(N, D)``.

    Returns
    -------
    x_batched
        If ``x`` has shape ``(D,)``, returns ``(1, D)``.
        If ``x`` already has shape ``(N, D)``, returns ``x`` unchanged.

    """
    x = jnp.asarray(x)
    return x[None, :] if x.ndim == 1 else x


##############


#######
# Layers
#######
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
        self.width = width
        self.depth = depth
        self.act = act

        # Build the hidden layers using nnx.List for proper pytree handling
        layers = []
        current_in = in_features
        for _ in range(depth):
            layers.append(nnx.Linear(current_in, width, rngs=rngs))
            current_in = width
        self.hidden_layers = nnx.List(layers)

        # Output layer: width -> 1
        self.output_layer = nnx.Linear(width, 1, rngs=rngs)

    def __call__(self, x: Array) -> Array:
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
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        x = self.output_layer(x)
        return jnp.squeeze(x, axis=-1)


@ft.partial(jax.jit, static_argnames=("clip",))
def _cart2sph_one(x3: Array, clip: float) -> Array:
    """Convert a single 3D Cartesian point to modified spherical coords."""
    r = jnp.linalg.norm(x3)
    r_safe = jnp.maximum(r, jnp.finfo(x3.dtype).tiny)

    r_inv = 1.0 / r_safe
    r_i = jnp.clip(r, 0.0, clip)
    r_e = jnp.clip(r_inv, 0.0, clip)

    stu = x3 / r_safe
    return jnp.concatenate([jnp.array([r_i, r_e], dtype=x3.dtype), stu], axis=0)  # (5,)


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

    def __init__(self, clip: float = 1.0) -> None:
        """Initialize the coordinate transformation layer."""
        self.clip = clip

    def __call__(self, X_cart: Array) -> Array:
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
        Y2 = jax.vmap(ft.partial(_cart2sph_one, clip=self.clip))(X2)  # (N, 5)
        return jnp.squeeze(Y2, axis=0) if (X_cart.ndim == 1) else Y2


class ScaleNNPotentialLayer(nnx.Module):
    """Apply an analytic, radius-dependent prefactor to a proxy potential.

    This layer is intended for architectures where the NN predicts a scaled proxy
    potential ``u_nn`` and an additional analytic factor is applied to enforce or
    approximate known radial behavior .

    The layer supports several modes controlled by ``config["scale"]``:
    - ``"one"``: no scaling (prefactor = 1)
    - ``"power"``: prefactor = (1/r)^power
    - ``"nfw"``: prefactor = log(1 + r/r_s) / r   (with r_s scaled via x_transformer)
    - external object: if ``config["scale"]`` is an object with a callable
      ``.potential(x, t=...)`` (i.e. a Galax potential), use |Phi(x)| to build a
      dimensionless scaling factor

    Parameters
    ----------
    config
        Configuration mapping controlling scaling behavior.

        Required keys:

        - x_transformer
            Object with ``.transform(...)`` used for scaling parameters such as ``r_s``.
        - r_s
            Physical scale radius (float).

        Optional keys:

        - scale
            One of:
            - string mode: {"one", "power", "nfw"}
            - an external potential-like object (i.e. Galax potential)
              with a callable ``.potential`` method

        - power
            Exponent for ``scale="power"``. Default 1.0.

        - scale_r_ref
            Reference radius used to normalize external potential scaling. If
            provided, scaling is normalized by |Phi(r_ref)| along each point's
            radial direction.

        - scale_eps_frac
            Fractional epsilon for stabilization when normalizing external
            potential.

        - scale_clip_min, scale_clip_max
            Clamps applied to the dimensionless scale factor. Defaults 1e-3,
            1e3.

        - scale_reciprocal
            If True, uses 1/|Phi| (after normalization), which tends to grow
            with r for |Phi| ~ 1/r. If False, uses |Phi| directly. Default True.

    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initialize the scaling layer with configuration."""
        self.config = nnx.data(config)
        self._scale_fn = self._build_scale_fn(config)

    def _build_scale_fn(
        self, cfg: Mapping[str, Any]
    ) -> Callable[[Array, Array, float, Any], Array]:
        """Build the scale function based on config['scale'].

        Returns a callable (x_cart, r, r_s, t) -> scale_external.
        """
        scale: ScaleSpec = cfg.get("scale", "one")
        x_transformer = cfg["x_transformer"]

        match scale:
            case "one":
                return lambda x_cart, r, r_s, t: jnp.ones_like(r)  # noqa: ARG005

            case "power":
                power = float(cfg.get("power", 1.0))

                def _power_scale(x_cart: Array, r: Array, r_s: float, t: Any) -> Array:
                    del r, r_s, t  # unused
                    r_here = jnp.linalg.norm(x_cart, axis=-1)
                    return jnp.power(1.0 / r_here, power)

                return _power_scale

            case "nfw":

                def _nfw_scale(x_cart: Array, r: Array, r_s: float, t: Any) -> Array:
                    del r, t  # unused
                    r_s_scaled = x_transformer.transform(r_s)
                    r_scaled = jnp.linalg.norm(x_cart, axis=-1)
                    return jnp.log(1.0 + r_scaled / r_s_scaled) / r_scaled

                return _nfw_scale

            case _ if hasattr(scale, "potential") and callable(scale.potential):
                # External potential scaling
                r_ref = cfg.get("scale_r_ref", None)
                eps_frac = float(cfg.get("scale_eps_frac", 1e-6))
                clip_min = float(cfg.get("scale_clip_min", 1e-3))
                clip_max = float(cfg.get("scale_clip_max", 1e3))
                reciprocal = bool(cfg.get("scale_reciprocal", True))
                ext_potential = cast("_HasPotential", scale)

                def _external_scale(
                    x_cart: Array, r: Array, r_s: float, t: Any
                ) -> Array:
                    del r_s  # unused
                    xB = _as_batch(x_cart)
                    r_safe = jnp.maximum(r, 1e-12)

                    u = ext_potential.potential(xB, t=t).squeeze()
                    s_raw = jnp.abs(u)

                    if r_ref is None:
                        s_norm = s_raw
                    else:
                        x_dir = xB / jnp.atleast_1d(r_safe)[:, None]
                        x_ref = x_dir * r_ref
                        u_ref = ext_potential.potential(x_ref, t=t).squeeze()
                        s_ref = jnp.abs(u_ref)
                        eps = eps_frac * jnp.maximum(s_ref, 1e-12)
                        s_norm = (s_raw + eps) / (s_ref + eps)

                    s_out = jnp.where(
                        reciprocal, 1.0 / jnp.maximum(s_norm, 1e-12), s_norm
                    )
                    return jnp.clip(s_out, clip_min, clip_max)

                return _external_scale

            case _:
                # Default / unknown: no scaling
                return lambda x_cart, r, r_s, t: jnp.ones_like(r)  # noqa: ARG005

    @staticmethod
    def modified_radial_coords(r: Array, clip: float) -> tuple[Array, Array]:
        """Compute clipped radius and clipped inverse-radius.

        Parameters
        ----------
        r
            Radius array.
        clip
            Maximum clip value applied to both r and 1/r.

        Returns
        -------
        r_i
            Clipped radius, ``clip(r, 0, clip)``.
        r_e
            Clipped inverse radius, ``clip(1/r, 0, clip)``.

        """
        r_inv = jnp.divide(1.0, r)
        r_i = jnp.clip(r, 0.0, clip)
        r_e = jnp.clip(r_inv, 0.0, clip)
        return r_i, r_e

    def __call__(
        self, x_cart: Array, u_nn: Array, r_s_learned: float | None = None, t: Any = 0.0
    ) -> Array:
        """Apply radius-dependent scaling to a proxy NN potential.

        Parameters
        ----------
        x_cart
            Cartesian positions, shape ``(3,)`` or ``(N, 3)``.
        u_nn
            Neural network proxy potential to be scaled.
        r_s_learned
            Optional learned scale radius; overrides ``config["r_s"]``.
        t
            Time for evaluating external potentials (if used).

        Returns
        -------
        Array
            Scaled potential values.

        """
        r_s: float = self.config["r_s"] if r_s_learned is None else r_s_learned
        r = _radius(x_cart)

        scale_external = self._scale_fn(x_cart, r, r_s, t)

        scaled = u_nn * scale_external
        return scaled[:, 0] if scaled.ndim > 1 else scaled


class TrainableGalaxPotential(nnx.Module):
    """NNX module that wraps a Galax potential class.

    Exposes selected constructor arguments as trainable NNX parameters. This
    layer takes a Galax potential constructor, a dictionary of initialization
    values, and a list/tuple of keys indicating which parameters should be
    trainable. On each call, it constructs a Galax potential instance using
    a mix of:
      - learned parameters for keys in `trainable, and
      - fixed values from `init_kwargs for all other keys.
      For mass-like parameters (`"m" or "m_tot"), the module trains the base-10
      logarithm and exponentiates during the forward pass to enforce positivity
      and improve numerical conditioning.

    Parameters
    ----------
    PotClass
        Galax potential constructor/class. Must be callable and return an object
        with a `.potential(positions, t=...) method.
    init_kwargs
        Mapping from constructor-argument name to its initial (float) value.
    trainable
        Tuple of keys from `init_kwargs that should become trainable parameters.
    rngs
        Random number generator state (unused but kept for API consistency).

    Returns
    -------
    phi, r_s
        `phi is the potential evaluated at positions with t=0. The second return
        value is `params["r_s"]`.

    """

    def __init__(
        self,
        PotClass: _GalaxPotentialCtor,
        init_kwargs: TrainableInitKwargs,
        trainable: TrainableKeys,
        *,
        rngs: nnx.Rngs | None = None,  # noqa: ARG002
    ) -> None:
        """Initialize trainable and fixed parameters from init_kwargs."""
        if "r_s" not in init_kwargs:
            raise KeyError("TrainableGalaxPotential requires 'r_s' in init_kwargs.")

        self.PotClass = PotClass
        self._trainable_keys = trainable

        # Build parameters dict, then wrap in nnx.Dict for proper pytree handling
        params_dict: dict[str, nnx.Param | Array] = {}
        for name, val in init_kwargs.items():
            arr = jnp.asarray(val, dtype=jnp.float32)
            match (name in trainable, name in ("m", "m_tot")):
                case (True, True):
                    # Train log10 of mass for positivity
                    params_dict[f"log10_{name}"] = nnx.Param(jnp.log10(arr))
                case (True, False):
                    params_dict[name] = nnx.Param(arr)
                case _:
                    # Fixed parameters stored as plain arrays
                    params_dict[name] = arr

        self._params = nnx.Dict(params_dict)

    def _get_built_params(self) -> dict[str, Array]:
        """Get the current parameter values, converting log-mass if needed."""
        params: dict[str, Array] = {}
        for name, val in self._params.items():
            if name.startswith("log10_"):
                # Convert back from log10
                actual_name = name[6:]  # Remove "log10_" prefix
                if isinstance(val, nnx.Param):
                    params[actual_name] = jnp.power(10.0, val.value)
                else:
                    params[actual_name] = jnp.power(10.0, val)
            elif isinstance(val, nnx.Param):
                params[name] = val.value
            else:
                params[name] = val
        return params

    def __call__(self, positions: Array) -> tuple[Array, Array]:
        """Evaluate the trainable potential at given positions.

        Parameters
        ----------
        positions
            Cartesian positions, shape ``(N, 3)``.

        Returns
        -------
        phi
            Potential values at positions.
        r_s
            The current scale radius parameter.

        """
        built_params = self._get_built_params()
        pot = self.PotClass(**built_params, units="galactic")
        phi = pot.potential(positions, t=0)
        r_s_out = jnp.asarray(built_params["r_s"])
        return phi, r_s_out


class AnalyticModelLayer(nnx.Module):
    """Evaluate an analytic (baseline) potential inside an NNX model.

    Operates in scaled space.

    Modes
    -----
    static
        Input is scaled Cartesian position(s) ``x_cart`` with shape ``(3,)``
        or ``(N, 3)``. Time is taken from ``config["time"]`` if present,
        otherwise ``0.0``.

    time
        Input is scaled concatenated ``tx_cart`` with shape ``(4,)`` or ``(N, 4)``,
        interpreted as ``[t_scaled, x_scaled, y_scaled, z_scaled]``.
        The physical time is obtained via ``config["t_transformer"].inverse_transform``.

    Parameters
    ----------
    config
        Configuration mapping. Required keys:

        - ``x_transformer``: object with ``inverse_transform`` for positions
        - ``u_transformer``: object with ``transform`` for potentials
        - ``ab_potential``: object with method ``potential(positions, t=...)``

        Additional keys by mode:
        - static: optional ``time`` (float-like), default 0.0
        - time: required ``t_transformer`` with ``inverse_transform``

    mode
        Either ``"static"`` or ``"time"``. Defaults to ``"static"``.

    """

    def __init__(
        self, config: Mapping[str, Any], mode: Literal["static", "time"] = "static"
    ) -> None:
        """Initialize the analytic layer with configuration and mode."""
        self.config = nnx.data(config)
        self.mode = mode

    def __call__(self, inp: Array) -> Any:
        """Evaluate the analytic potential in scaled coordinates.

        Parameters
        ----------
        inp
            Scaled inputs. Shape ``(3,)`` or ``(N, 3)`` for static mode;
            ``(4,)`` or ``(N, 4)`` for time mode.

        Returns
        -------
        u_scaled
            Analytic potential in scaled units, shape ``(N,)``.

        """
        # Ensure batched
        if inp.ndim == 1:
            inp = inp[None, :]

        x_transformer = cast("_Transformer", self.config["x_transformer"])
        u_transformer = cast("_Transformer", self.config["u_transformer"])
        analytic = cast("_AnalyticPotential", self.config["ab_potential"])

        if self.mode == "static":
            # inp: (N,3) in scaled coords
            x_cart: Array = inp
            positions = x_transformer.inverse_transform(x_cart)  # physical positions
            t = self.config.get("time", 0.0)

            # Evaluate potential (try batched; fallback to per-sample).
            def pot_one(pos: Any) -> Any:
                return analytic.potential(pos, t=t)

            try:
                dimensional_potential = analytic.potential(positions, t=t)
            except Exception:  # noqa: BLE001
                dimensional_potential = jax.vmap(pot_one)(positions)

            return u_transformer.transform(dimensional_potential)

        if self.mode == "time":
            # input: (N,4) = [t, x, y, z] in scaled coords
            t_scaled: Array = inp[:, 0]
            x_cart: Array = inp[:, 1:4]

            t_transformer = cast("_Transformer", self.config["t_transformer"])
            t_phys = t_transformer.inverse_transform(t_scaled)  # (N,)

            positions = x_transformer.inverse_transform(x_cart)  # (N,3)

            # Evaluate per sample: potential(pos_i, t_i)
            def pot_one(pos: Any, tt: Any) -> Any:
                return analytic.potential(pos, t=tt)

            dimensional_potential = jax.vmap(pot_one)(positions, t_phys)
            return u_transformer.transform(dimensional_potential)

        msg = f"Unknown mode={self.mode!r}. Use 'static' or 'time'."
        raise ValueError(msg)


class FuseModelsLayer(nnx.Module):
    """A simple layer that fuses an NN and an analytic potential by addition.

    This layer implements the most basic form of a hybrid model, where the total
    potential is the linear sum of a known analytic component and a flexible
    neural network component.
    """

    def __call__(self, nn_potential: Array, analytic_potential: Array) -> Array:
        """Fuse NN and analytic potentials by simple addition.

        Parameters
        ----------
        nn_potential
            Neural network potential component.
        analytic_potential
            Analytic (baseline) potential component.

        Returns
        -------
        Array
            Sum of both potentials.

        """
        return nn_potential + analytic_potential


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

        Required
        --------
        x_transformer
            Transformer that maps between scaled and physical coordinates.
            Must implement ``inverse_transform(positions)``.

        Optional (tanh blend)
        ---------------------
        r_trans
            Transition radius (in physical coordinate units).
            Default 200.
        k_smooth
            Steepness of the tanh transition. Default 0.5 (when not trainable).
        train_k
            If True, make k_smooth trainable via ``exp(log_k)``. Default False.
        min_k
            Lower bound for k_smooth when trainable. Default 0.01.

        Optional (power-law blend)
        --------------------------
        radial_power
            If provided (not None), uses the power-law blend instead of tanh.
        r_smooth
            Smoothing radius for the power-law blend. Default 150.0.

        Optional (both)
        ---------------
        saturation
            Asymptotic value of h(r) as r -> infinity. Default 1.0.

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
        self.config = nnx.data(config)
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
        """Fuse NN and analytic potentials with a smooth radial transition."""
        x_transformer = self.config["x_transformer"]
        dimensional_positions: Array = x_transformer.inverse_transform(positions)

        saturation: float = float(self.config.get("saturation", 1.0))

        if bool(self.config.get("train_k", False)):
            min_k: float = float(self.config.get("min_k", 0.01))
            k_smooth: Array = jnp.maximum(min_k, jnp.exp(jnp.asarray(self.log_k)))
        else:
            k_smooth = jnp.asarray(float(self.config.get("k_smooth", 0.5)))

        r_trans: float = float(self.r_trans)

        if positions.ndim == 1:
            r: Array = jnp.linalg.norm(dimensional_positions)
        else:
            r = jnp.linalg.norm(dimensional_positions, axis=1)

        if self.config.get("radial_power", None) is not None:
            power: float = float(self.config["radial_power"])
            r_smooth: float = float(self.config.get("r_smooth", 150.0))
            h = self.radial_power_law(r, r_smooth, power, saturation=saturation)
        else:
            h = self.H(r, r_trans, k_smooth, saturation=saturation)

        g = 1.0 - h

        # Blend NN output and analytic function
        u_model = g * u_nn + u_analytic
        return {"fused_potential": u_model, "h": h, "g": g}
