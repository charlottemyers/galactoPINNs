import jax
import jax.numpy as jnp
from flax import linen as nn
from collections.abc import Callable


class SmoothMLP(nn.Module):
    width: int = 128
    depth: int = 3
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


class CartesianToModifiedSphericalLayer_alt(nn.Module):
    @nn.compact
    def __call__(self, X_cart):
        def modified_radial_coords_alt(r):
            r_i = jnp.where(r <= 1.0, r, 1.0)
            r_e = jnp.where(r <= 1.0, 1.0, 1.0 / r)
            return r_i, r_e

        def cart2sph(X_cart):
            if X_cart.ndim == 1:
                r = jnp.linalg.norm(X_cart, keepdims=True)
                r_i, r_e = modified_radial_coords_alt(r)
                stu = X_cart / r
                return jnp.concatenate([r_i, r_e, stu])
            else:
                r = jnp.linalg.norm(X_cart, axis=1, keepdims=True)
                r_i, r_e = modified_radial_coords_alt(r)
                stu = X_cart / r
                return jnp.concatenate([r_i, r_e, stu], axis=1)

        return cart2sph(X_cart)


class ScaleNNPotentialLayer(nn.Module):
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
            clip = self.config.get("clip", 1.0)
            power = self.config.get("power", 1.0)
            if x_cart.ndim == 1:
                r = jnp.linalg.norm(x_cart)
            else:
                r = jnp.linalg.norm(x_cart, axis=1)
            r_inv = 1.0 / r  # jnp.maximum(r, 1e-8)
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

        if scale == "bar":
            scale_external = 1 / r_scaled

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
                )  # Compute R for all points
            epsilon = 0.1
            scale_external = 1.0 / jnp.maximum(
                jnp.sqrt(R_scaled**2 + a_disk_scaled**2 + epsilon), 1e-8
            )

        scaled = u_nn * scale_external
        return scaled[:, 0] if scaled.ndim > 1 else scaled


################
# ANALYTIC LAYERS
################


class AnalyticModelLayer(nn.Module):
    config: dict

    def H(self, x, r_smooth, k_smooth):
        return 0.5 + 0.5 * jnp.tanh(k_smooth * (x - r_smooth))

    def __call__(self, x_phys):
        if x_phys.ndim == 1:
            x_phys = x_phys[None, :]  # Shape (1, N)

        x_transformer = self.config["x_transformer"]
        u_transformer = self.config["u_transformer"]

        positions = x_transformer.inverse_transform(x_phys)  # kpc

        lf_analytic_function = self.config["lf_analytic_function"]
        time = self.config.get("time", 0)
        dimensional_potential = lf_analytic_function.potential(positions, t=time)

        transformed_potential = u_transformer.transform(dimensional_potential)
        return transformed_potential


class AnalyticModelLayerTime(nn.Module):
    config: dict

    def __call__(self, tx_cart):
        if tx_cart.ndim == 1:
            tx_cart = tx_cart[None, :]  # Shape (1, 4)

        x_cart = tx_cart[:, 1:4]
        t_phys = self.config["t_transformer"].inverse_transform(tx_cart[:, 0])

        x_transformer = self.config["x_transformer"]
        u_transformer = self.config["u_transformer"]

        x_phys = x_transformer.inverse_transform(x_cart)
        analytic_function = self.config["lf_analytic_function"]

        # Call potential for each position at corresponding time using vmap
        def potential_fn(pos, t):
            return analytic_function.potential(pos, t)  # .ustrip("kpc2/Myr2")

        dimensional_potential = jax.vmap(potential_fn)(x_phys, t_phys)

        return u_transformer.transform(dimensional_potential)


###############


class FuseModelsLayer(nn.Module):
    def __call__(self, nn_potential, analytic_potential):
        fused_output = nn_potential + analytic_potential
        return fused_output


class FuseandBoundary(nn.Module):
    config: dict
    init_k: float = 0.01

    def setup(self):
        self.log_k = self.param("log_k", lambda rng: jnp.log(self.init_k))

    def H(self, x, r_smooth, k_smooth, saturation=1.0):
        """Smoothing function using tanh transition."""
        return 0.5 * saturation * (1 + jnp.tanh(k_smooth * (x - r_smooth)))

    def __call__(self, positions, u_nn, u_analytic, fuse=True):
        x_transformer = self.config["x_transformer"]
        dimensional_positions = x_transformer.inverse_transform(positions)
        r_smooth = self.config.get("r_trans")
        saturation = self.config.get("saturation", 1.0)

        if self.config.get("train_k", False):
            k_smooth = jnp.exp(self.log_k)
        else:
            k_smooth = self.config.get("k_smooth", 0.5)

        if positions.ndim == 1:
            r = jnp.linalg.norm(dimensional_positions)
        else:
            r = jnp.linalg.norm(dimensional_positions, axis=1)

        h = self.H(r, r_smooth, k_smooth, saturation=saturation)
        g = 1.0 - h

        if fuse:
            u_model = g * u_nn + u_analytic
        else:
            u_model = g * u_nn + h * u_analytic
        return {"potential": u_model, "h": h, "g": g}

class FuseandBoundary(nn.Module):
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

        #######

        if self.config.get("radial_power", None) is not None:
            power = self.config["radial_power"]
            r_smooth = self.config.get("r_smooth", 150.0)
            h = self.radial_power_law(r, r_smooth, power, saturation=saturation)
        else:
            h = self.H(r, r_trans, k_smooth, saturation=saturation)

        g = 1.0 - h
        u_model = g * u_nn + u_analytic
        return {"fused_potential": u_model, "h": h, "g": g}


class BoundaryConditionLayer(nn.Module):
    config: dict

    def setup(self):
        self.log_k = self.config.get(
            "k_smooth", 1.0
        )  # self.param("log_k", lambda rng: jnp.log(self.init_k))
        self.r_trans = self.config.get(
            "r_trans", 200
        )  # self.param("r_trans", lambda rng: float(self.init_r_trans))

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
        u_model = g * u_nn + h * u_analytic
        return u_model
