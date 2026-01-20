import jax
import jax.numpy as jnp
from flax import linen as nn
from collections.abc import Callable
from typing import Any, Dict, Tuple

__all__ = [
    "SmoothMLP",
    "CartesianToModifiedSphericalLayer",
    "ScaleNNPotentialLayer",
    "TrainableGalaxPotential",
    "AnalyticModelLayer",
    "AnalyticModelLayerTime",
    "FuseModelsLayer",
    "FuseandBoundary",
]


class SmoothMLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) with smooth activation functions.
    This module creates a stack of dense layers, each followed by a specified
    activation function. It is designed to approximate a smooth scalar field.

    Attributes:
        width (int): The number of neurons in each hidden layer.
        depth (int): The number of hidden layers.
        act (str | Callable): The activation function. Can be a string
            name ('softplus', 'gelu', 'tanh') or a callable function.
        gelu_approximate (bool): Whether to use the approximate form of the
            GELU activation function. Only used if `act` is 'gelu'.
    """
    width: int = 128
    depth: int = 4
    act: str | Callable[[jnp.ndarray], jnp.ndarray] = "softplus"
    gelu_approximate: bool = False

    def _resolve_activation(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        if callable(self.act):
            return self.act

        key = str(self.act).lower()
        if key == "softplus":
            return jax.nn.softplus
        elif key == "gelu":
            def _gelu(x):
                return jax.nn.gelu(x, approximate=self.gelu_approximate)

            return _gelu
        elif key == "tanh":
            return jnp.tanh
        else:
            raise ValueError(
                f"Unknown activation '{self.act}'. "
                "Use 'softplus', 'gelu', 'tanh', or pass a callable."
            )

    @nn.compact
    def __call__(self, x):
        act_fn = self._resolve_activation()
        for _ in range(self.depth):
            x = act_fn(nn.Dense(self.width)(x))
        x = nn.Dense(1)(x)  # scalar potential
        return x.squeeze(-1)


class CartesianToModifiedSphericalLayer(nn.Module):
    """Converts Cartesian coordinates to modified spherical coordinates.

    This layer transforms 3D Cartesian coordinates (x, y, z) into a 5D
    representation consisting of a clipped radius, a clipped inverse radius,
    and the Cartesian unit vector.

    The output vector is `[r_i, r_e, s, t, u]`, where:
    - `r_i` is the radius clipped to a maximum value.
    - `r_e` is the inverse radius, also clipped.
    - `(s, t, u)` is the Cartesian unit vector (x/r, y/r, z/r).

    Attributes:
        clip (float): The maximum value to which the radius and inverse
            radius are clipped. This helps stabilize the inputs to subsequent
            layers.
    """
    clip: float = 1.0

    @nn.compact
    def __call__(self, X_cart):
        def modified_radial_coords(r, clip):
            r_inv = jnp.divide(1.0, r)
            r_i = jnp.clip(r, 0.0, clip)
            r_e = jnp.clip(r_inv, 0.0, clip)
            return r_i, r_e

        def cart2sph(X_cart):
            if X_cart.ndim == 1:
                r = jnp.linalg.norm(X_cart, keepdims=True)
                r_i, r_e = modified_radial_coords(r, self.clip)
                stu = jnp.divide(X_cart, r)
                spheres = jnp.concatenate([r_i, r_e, stu])
                return spheres

            r = jnp.linalg.norm(X_cart, axis=1, keepdims=True)
            r_i, r_e = modified_radial_coords(r, self.clip)
            stu = jnp.divide(X_cart, r)
            spheres = jnp.concatenate([r_i, r_e, stu], axis=1)
            return spheres

        return cart2sph(X_cart)


class ScaleNNPotentialLayer(nn.Module):
    """Scale NN proxy potential using an analytic radial prefactor.

    Required config keys:
      - x_transformer: object with .transform(...)
      - r_s: float (physical scale radius)
      - scale: one of {"one","power","nfw","hernquist","disk"}

    Optional config keys (by scale):
      - power: float (for scale="power")
      - translation: float (for scale="nfw")
      - a_disk: float (for scale="disk")
    """
    config: dict

    def modified_radial_coords(r, clip):
        r_inv = jnp.divide(1.0, r)
        r_i = jnp.clip(r, 0.0, clip)
        r_e = jnp.clip(r_inv, 0.0, clip)
        return r_i, r_e

    @nn.compact
    def __call__(self, x_cart, u_nn, r_s_learned=None):
        x_transformer = self.config["x_transformer"]
        if r_s_learned is None:
            r_s = self.config["r_s"]
        else:
            r_s = r_s_learned
        scale = self.config.get("scale", "one")

        scale_external = 1.0

        if scale == "one":
            scale_external = 1

        if scale == "power":
            power = self.config.get("power", 1.0)
            if x_cart.ndim == 1:
                r = jnp.linalg.norm(x_cart)
            else:
                r = jnp.linalg.norm(x_cart, axis=1)
            r_inv = 1.0 / r
            scale_external = jnp.power(r_inv, power)

        if scale == "nfw":
            translation = self.config.get("translation", 0.0)
            r_s_scaled = x_transformer.transform(r_s)
            if x_cart.ndim == 1:
                r_scaled = jnp.linalg.norm(x_cart)
            else:
                r_scaled = jnp.linalg.norm(x_cart, axis=1)
            scale_external = jnp.log(1 + (r_scaled + translation) / r_s_scaled) / (
                r_scaled + translation
            )

        if scale == "hernquist":
            r_s_scaled = x_transformer.transform(jnp.array([[r_s, 0, 0]]))[:, 0]
            r_s_scaled = jnp.linalg.norm(r_s_scaled)
            if x_cart.ndim == 1:
                r_scaled = jnp.linalg.norm(x_cart)
            else:
                r_scaled = jnp.linalg.norm(x_cart, axis=1)
            r_hernquist = r_scaled + r_s_scaled
            scale_external = 1.0 / jnp.maximum(r_hernquist, 1e-8)

        if scale == "disk":
            a_disk = self.config["a_disk"]
            a_disk_scaled = x_transformer.transform(jnp.array([[a_disk, 0, 0]]))[:, 0]
            a_disk_scaled = jnp.linalg.norm(a_disk_scaled)  # Ensure scalar value
            if x_cart.ndim == 1:
                R_scaled = jnp.linalg.norm(
                    x_cart[:2]
                )
            else:
                R_scaled = jnp.linalg.norm(
                    x_cart[:, :2], axis=1
                )
            epsilon = 0.1
            scale_external = 1.0 / jnp.maximum(
                jnp.sqrt(R_scaled**2 + a_disk_scaled**2 + epsilon), 1e-8
            )

        scaled = u_nn * scale_external
        return scaled[:, 0] if scaled.ndim > 1 else scaled


class TrainableGalaxPotential(nn.Module):
    """A Flax module that wraps a `galax` potential to make its parameters trainable.

    This layer takes a `galax` potential class and allows a specified subset of
    its initialization parameters to be treated as trainable parameters within a
    Flax model. It handles the creation of these parameters and computes the
    potential at given positions.

    For mass-like parameters ('m', 'm_tot'), it trains their base-10 logarithm
    to ensure positivity and improve stability.

    Attributes:
        PotClass (Any): The `galax` potential class to be wrapped (e.g.,
            `galax.potential.MiyamotoNagaiPotential`).
        init_kwargs (Dict[str, float]): A dictionary of initial values for the
            potential's parameters.
        trainable (Tuple[str]): A tuple of strings specifying which keys from
            `init_kwargs` should be made trainable.
    """
    PotClass: Any
    init_kwargs: Dict[str, float]
    trainable: Tuple[str]

    @nn.compact
    def __call__(self, positions):
        params = {}
        for name, val in self.init_kwargs.items():
            if name in self.trainable:
                if name in ("m", "m_tot"):
                    log_m = self.param("log_m", lambda rng: jnp.log10(jnp.array(val)))
                    params[name] = 10 ** log_m
                else:
                    params[name] = self.param(name, lambda rng: jnp.array(val))
            else:
                params[name] = val
        pot = self.PotClass(**params, units="galactic")
        phi = pot.potential(positions, t=0)
        return phi, params["r_s"]


class AnalyticModelLayer(nn.Module):
    """
    Unified analytic potential wrapper.

    Modes:
      - mode="static": inputs are x_cart (3,) or (N,3). Time is taken from config["time"] (default 0).
      - mode="time":   inputs are tx_cart (4,) or (N,4) with [t, x, y, z]. Time is taken from input and
                       inverse-transformed with config["t_transformer"].
    """
    config: dict
    mode: str = "static"  # "static" or "time"

    def __call__(self, inp):
        # Ensure batched
        if inp.ndim == 1:
            inp = inp[None, :]

        x_transformer = self.config["x_transformer"]
        u_transformer = self.config["u_transformer"]
        analytic = self.config["lf_analytic_function"]

        if self.mode == "static":
            # inp: (N,3) in scaled coords
            x_cart = inp
            positions = x_transformer.inverse_transform(x_cart)  # physical positions
            t = self.config.get("time", 0.0)

            # Evaluate potential.
            def pot_one(pos):
                return analytic.potential(pos, t=t)
            try:
                dimensional_potential = analytic.potential(positions, t=t)
            except Exception:
                dimensional_potential = jax.vmap(pot_one)(positions)

            return u_transformer.transform(dimensional_potential)

        elif self.mode == "time":
            # inp: (N,4) = [t, x, y, z] in scaled coords
            t_scaled = inp[:, 0]
            x_cart   = inp[:, 1:4]

            t_transformer = self.config["t_transformer"]
            t_phys = t_transformer.inverse_transform(t_scaled)  # (N,)

            positions = x_transformer.inverse_transform(x_cart)  # (N,3)

            # Evaluate per sample: potential(pos_i, t_i)
            def pot_one(pos, t):
                return analytic.potential(pos, t=t)

            dimensional_potential = jax.vmap(pot_one)(positions, t_phys)
            return u_transformer.transform(dimensional_potential)

        else:
            raise ValueError(f"Unknown mode={self.mode!r}. Use 'static' or 'time'.")


class FuseModelsLayer(nn.Module):
    """A simple layer that fuses a neural network and an analytic potential by addition.

    This layer implements the most basic form of a hybrid model, where the total
    potential is the linear sum of a known analytic component and a flexible
    neural network component.
    """
    def __call__(self, nn_potential, analytic_potential):
        fused_output = nn_potential + analytic_potential
        return fused_output

class FuseandBoundary(nn.Module):
    """Fuses a neural network and an analytic potential with a smooth transition.
    This layer implements a sophisticated blending of a neural network potential (`u_nn`)
    and an analytic potential (`u_analytic`). It ensures that the model smoothly
    transitions to the analytic form at large radii, effectively enforcing a
    physical boundary condition (e.g., Keplerian fall-off).

    The fusion is controlled by a radial blending function, `h(r)`, which
    transitions from 0 at small radii to a saturation value  at
    large radii. A complementary function `g(r) = 1 - h(r)` is also defined.
    The blending function `h(r)` can be either a `tanh` sigmoid or a radial
    power law, controlled by the configuration.

    Attributes:
        config (dict): A dictionary containing configuration for the layer.
            - 'x_transformer': Coordinate transformer.
            - 'r_trans' (float, optional): Transition radius for the tanh blend.
            - 'k_smooth' (float, optional): Steepness of the tanh transition.
            - 'train_k' (bool, optional): Whether to make `k_smooth` a trainable parameter.
            - 'radial_power' (float, optional): If specified, uses a power-law blend instead of tanh.
            - 'r_smooth' (float, optional): Smoothing radius for the power-law blend.
            - 'saturation' (float, optional): The value `h(r)` approaches at large radii.
    """

    config: dict

    def setup(self):
        self.log_k = self.config.get(
            "k_smooth", 1.0
        )
        self.r_trans = self.config.get(
            "r_trans", 200
        )
    def H(self, x, r_smooth, k_smooth, saturation=1.0):
        """Smoothing function using tanh transition."""
        return 0.5 * saturation * (1 + jnp.tanh(k_smooth * (x - r_smooth)))

    def radial_power_law(self, r, r_smooth, pow, saturation):
        """Radial power law function."""
        return saturation * jnp.power(r / (r + r_smooth), pow)

    def __call__(self, positions, u_nn, u_analytic):
        x_transformer = self.config["x_transformer"]
        dimensional_positions = x_transformer.inverse_transform(positions)
        saturation = self.config.get("saturation", 1.0)

        if self.config.get("train_k", False):
            min_k = self.config.get("min_k", 0.01)
            k_smooth = jnp.maximum(min_k, jnp.exp(self.log_k))
        else:
            k_smooth = self.config.get("k_smooth", 0.5)

        r_trans = self.r_trans

        if positions.ndim == 1:
            r = jnp.linalg.norm(dimensional_positions)
        else:
            r = jnp.linalg.norm(dimensional_positions, axis=1)

        if self.config.get("radial_power", None) is not None:
            power = self.config["radial_power"]
            r_smooth = self.config.get("r_smooth", 150.0)
            h = self.radial_power_law(r, r_smooth, power, saturation=saturation)
        else:
            h = self.H(r, r_trans, k_smooth, saturation=saturation)

        g = 1.0 - h

        # Blend nn output and analytic function
        u_model = g * u_nn + u_analytic
        return {"fused_potential": u_model, "h": h, "g": g}
