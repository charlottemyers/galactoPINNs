import jax.numpy as jnp
import numpy as np
import astropy.units as au
import coordinax as cx
from galax.potential import density
from unxt import Quantity
import galax.potential as gp
from flax.training.train_state import TrainState
import jax.random as jr
import jax


class UniformScaler:
    def __init__(self, feature_range=(-1, 1)):
        """Scale the variable by the min and max of the range or by a constant scaler"""
        self.feature_range = feature_range
        pass

    def fit(self, data, scaler=None):
        self.scaler = scaler

        data_max = np.max(data)
        data_min = np.min(data)

        data_range = data_max - data_min
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range

    def fit_transform(self, data, scaler=None):
        self.scaler = scaler
        data_max = np.max(data)
        data_min = np.min(data)

        data_range = data_max - data_min
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range

        if self.scaler is not None:
            X = data * self.scaler
            self.scale_ = self.scaler
            self.min_ = 0.0
        else:
            X = data * self.scale_ + self.min_
        return X

    def transform(self, data):
        if not hasattr(self, "scaler"):
            self.scaler = None

        if self.scaler is not None:
            X = data * self.scaler
        else:
            X = data * self.scale_ + self.min_
        return X

    def inverse_transform(self, data):
        if not hasattr(self, "scaler"):
            self.scaler = None
        if self.scaler is not None:
            return data / self.scaler
        else:
            return (data - self.min_) / self.scale_


class Transformer:
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, data):
        """
        Fit the transformer to the data, handling both 1D and 2D cases.
        """
        if data.ndim == 1:
            data = data[:, None]

        # Compute mean and scale for each feature
        self.mean = jnp.mean(data, axis=0)
        self.scale = jnp.max(jnp.abs(data - self.mean), axis=0)
        return self

    def transform(self, data):
        is_1d = False
        if data.ndim == 1:
            data = data[:, None]
            is_1d = True

        scaled_data = (data - self.mean) / self.scale
        return scaled_data.squeeze() if is_1d else scaled_data

    def inverse_transform(self, data):
        is_1d = False
        if data.ndim == 1:
            data = data[:, None]
            is_1d = True
        original_data = data * self.scale + self.mean
        return original_data.squeeze() if is_1d else original_data


def scale_by_non_dim_potential(data_dict, config):
    x_transformer = config.get("x_transformer", UniformScaler(feature_range=(-1, 1)))
    a_transformer = config.get("a_transformer", UniformScaler(feature_range=(-1, 1)))
    u_transformer = config.get("u_transformer", UniformScaler(feature_range=(-1, 1)))

    r_s = config["r_s"]  # Scale radius (kpc)

    if config["include_analytic"]:
        lf_potential = config["lf_analytic_function"]
        w = cx.CartesianPos3D(
            x=data_dict["x_train"][:, 0] * au.kpc,
            y=data_dict["x_train"][:, 1] * au.kpc,
            z=data_dict["x_train"][:, 2] * au.kpc,
        )
        u_analytic = lf_potential.potential(w, 0).ustrip("kpc2/Myr2")
        u_residual = data_dict["u_train"] - u_analytic  # Residual potential
        u_max = np.max(np.abs(u_residual))
    else:
        u_max = np.max(np.abs(data_dict["u_train"]))

    u_star = u_max
    t_star = np.sqrt(r_s**2 / u_star)
    a_star = r_s / t_star**2
    x_star = r_s

    x_transformer.fit(data_dict["x_train"], scaler=1 / x_star)
    a_transformer.fit(data_dict["a_train"], scaler=1 / a_star)
    u_transformer.fit(data_dict["u_train"], scaler=1 / u_star)

    x_train = x_transformer.transform(data_dict["x_train"])
    a_train = a_transformer.transform(data_dict["a_train"])
    u_train = u_transformer.transform(data_dict["u_train"])
    r_train = np.linalg.norm(x_train, axis=1)

    x_val = x_transformer.transform(data_dict["x_val"])
    a_val = a_transformer.transform(data_dict["a_val"])
    u_val = u_transformer.transform(data_dict["u_val"])
    r_val = np.linalg.norm(x_val, axis=1)

    data_dict = {
        "x_train": x_train,
        "a_train": a_train,
        "u_train": u_train,
        "r_train": r_train,
        "x_val": x_val,
        "a_val": a_val,
        "u_val": u_val,
        "r_val": r_val,
    }

    transformers = {"x": x_transformer, "a": a_transformer, "u": u_transformer}

    return data_dict, transformers


def scale_by_non_dim_potential_time(data_dict, config):
    x_transformer = config["x_transformer"]
    a_transformer = config["a_transformer"]
    u_transformer = config["u_transformer"]

    r_s = config["nfw_r_s"]
    M = config["nfw_M_200"]
    G = 4.514e-22  # kpc^3 / Myr^2 / Msun
    if config.get("include_analytic", False):
        x_train_spatial = data_dict["x_train"][:, :3]
        w = cx.CartesianPos3D(
            x=x_train_spatial[:, 0] * au.kpc,
            y=x_train_spatial[:, 1] * au.kpc,
            z=x_train_spatial[:, 2] * au.kpc,
        )
        gp_potential = config["lf_analytic_function"]
        u_nfw = gp_potential.potential(w, 0).ustrip("kpc2/Myr2")
        u_sans_nfw = data_dict["u_train"] - u_nfw
        u_max = np.max(np.abs(u_sans_nfw))
    else:
        u_max = np.max(np.abs(data_dict["u_train"]))

    u_star = u_max
    t_star = np.sqrt(r_s**2 / u_star)
    a_star = r_s / t_star**2
    x_star = r_s

    # --- Fit transformers only on spatial part (x, y, z) ---
    x_transformer.fit(data_dict["x_train"][:, :3], scaler=1 / x_star)
    a_transformer.fit(data_dict["a_train"], scaler=1 / a_star)
    u_transformer.fit(data_dict["u_train"], scaler=1 / u_star)

    def transform_spatial_with_time(x):
        x_scaled = x_transformer.transform(x[:, :3])
        t = x[:, 3:] 
        return np.concatenate([x_scaled, t], axis=1)

    x_train = transform_spatial_with_time(data_dict["x_train"])
    x_val = transform_spatial_with_time(data_dict["x_val"])

    a_train = a_transformer.transform(data_dict["a_train"])
    a_val = a_transformer.transform(data_dict["a_val"])

    u_train = u_transformer.transform(data_dict["u_train"])
    u_val = u_transformer.transform(data_dict["u_val"])

    config["ref_radius_s"] = r_s
    config["ref_velocity_s"] = np.sqrt(G * M / r_s)
    config["ref_potential_s"] = u_star
    config["ref_acceleration_s"] = a_star

    scaled_data_dict = {
        "x_train": x_train,
        "a_train": a_train,
        "u_train": u_train,
        "x_val": x_val,
        "a_val": a_val,
        "u_val": u_val,
    }

    transformers = {
        "x": x_transformer,
        "a": a_transformer,
        "u": u_transformer,
    }

    return scaled_data_dict, transformers




def scale_by_non_dim_potential_time_by_dict(data_dict, config):
    x_transformer = config.get("x_transformer", UniformScaler(feature_range=(-1, 1)))
    a_transformer = config.get("a_transformer", UniformScaler(feature_range=(-1, 1)))
    u_transformer = config.get("u_transformer", UniformScaler(feature_range=(-1, 1)))
    t_transformer = config.get("t_transformer", UniformScaler(feature_range=(-1, 1)))

    r_s = config["r_s"]
    train_data = data_dict["train"]
    val_data = data_dict["val"]

    # --- Flatten all training data to fit scalers ---
    all_x_train, all_a_train, all_u_train, all_t_train = [], [], [], []

    for t, d in train_data.items():
        all_x_train.append(d["x"])
        all_a_train.append(d["a"])
        all_u_train.append(d["u"])
        all_t_train.append(np.full(len(d["x"]), t))  # broadcast scalar time to array

    x_concat = np.vstack(all_x_train)
    a_concat = np.vstack(all_a_train)
    u_concat = np.hstack(all_u_train)
    t_concat = np.concatenate(all_t_train)


    if config.get("include_analytic"):
        lf_analytic_potential = config["lf_analytic_function"]
        pos = cx.CartesianPos3D(
            x=x_concat[:, 0] * au.kpc,
            y=x_concat[:, 1] * au.kpc,
            z=x_concat[:, 2] * au.kpc,
        )
        t_quant = Quantity(t_concat, au.Myr)
        u_analytic = lf_analytic_potential.potential(pos, t_quant).ustrip("kpc2/Myr2")
        u_sans_nfw = u_concat - u_analytic
        u_max = np.max(np.abs(u_sans_nfw))
    else:
        u_max = np.max(np.abs(u_concat))

    u_star = u_max
    t_star = np.sqrt(r_s**2 / u_star)
    a_star = r_s / t_star**2
    x_star = r_s

    x_transformer.fit(x_concat, scaler=1 / x_star)
    a_transformer.fit(a_concat, scaler=1 / a_star)
    u_transformer.fit(u_concat, scaler=1 / u_star)
    t_transformer.fit(t_concat, scaler=1 / t_star)

    def transform_block(x, a, u, t_val):
        x_scaled = x_transformer.transform(x)
        t_scaled = t_transformer.transform(np.full((len(x_scaled), 1), t_val))
        x_with_time = np.concatenate([t_scaled, x_scaled], axis=1)
        return {
            "x": x_with_time,
            "a": a_transformer.transform(a),
            "u": u_transformer.transform(u),
        }

    scaled_train = {}
    scaled_val = {}

    for t in train_data:
        d = train_data[t]
        scaled_train[t] = transform_block(d["x"], d["a"], d["u"], t)

    for t in val_data:
        d = val_data[t]
        scaled_val[t] = transform_block(d["x"], d["a"], d["u"], t)

    return {"train": scaled_train, "val": scaled_val}, {
        "x": x_transformer,
        "a": a_transformer,
        "u": u_transformer,
        "t": t_transformer,
    }


def flatten_time_dict_by_time(data_dict, split="train"):
    per_time_batches = []
    for t, d in data_dict[split].items():
        x_t = d["x"]
        a_t = d["a"]
        per_time_batches.append((x_t, a_t))
    return per_time_batches


def acc_cart_to_cyl(a):
    a_rho = np.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2)
    a_phi = np.arctan2(a[:, 1], a[:, 0])
    a_z = a[:, 2]
    a_mag = np.linalg.norm(a, axis=1)

    return np.stack([a_rho, a_phi, a_z, a_mag], axis=1)

