import jax.numpy as jnp
import numpy as np
import astropy.units as au
import coordinax as cx
from galax.potential import density
from unxt import Quantity
import galax.potential as gp


def biased_sphere_samples(N, r_min, r_max):
    r = np.exp(np.random.uniform(np.log(r_min), np.log(r_max), N))
    theta = np.random.uniform(0, 2 * np.pi, N)
    phi = np.arccos(np.random.uniform(-1, 1, N))  # isotropic

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def rejection_sample_sphere(
    galax_pot, N_samples, R_max, batch_size=10000, t=0, log_proposal_pts=False, R_min=0
):
    samples = []

    x_test = np.linspace(-R_max, R_max, 20)
    y_test = np.linspace(-R_max, R_max, 20)
    z_test = np.linspace(-R_max, R_max, 20)
    X, Y, Z = np.meshgrid(x_test, y_test, z_test, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    mask = R <= R_max

    pos_grid = cx.CartesianPos3D(
        x=X[mask].ravel() * au.kpc,
        y=Y[mask].ravel() * au.kpc,
        z=Z[mask].ravel() * au.kpc,
    )
    rho_grid = density(galax_pot, pos_grid, t=t)
    normalization = np.max(rho_grid).value * 1.1

    while len(samples) < N_samples:
        if log_proposal_pts:
            x, y, z = biased_sphere_samples(batch_size, 1.0, R_max)

        else:
            x = np.random.uniform(-R_max, R_max, batch_size)
            y = np.random.uniform(-R_max, R_max, batch_size)
            z = np.random.uniform(-R_max, R_max, batch_size)

        # enforce spherical region
        r = np.sqrt(x**2 + y**2 + z**2)
        inside = (r <= R_max) & (r >= R_min)
        x, y, z = x[inside], y[inside], z[inside]

        if len(x) == 0:
            continue

        pos = cx.CartesianPos3D(x=x * au.kpc, y=y * au.kpc, z=z * au.kpc)
        rho = density(galax_pot, pos, t=t).value

        threshold = np.random.uniform(0, 1, len(x))
        accepted = threshold < (rho / normalization)

        new_samples = np.stack([x[accepted], y[accepted], z[accepted]], axis=1)
        samples.append(new_samples)

        if sum(len(s) for s in samples) >= N_samples:
            break

    return np.vstack(samples)[:N_samples]


def stratified_rejection_sample(
    galax_pot,
    N_samples,
    R_bins,
    t=0,
    batch_size=10000,
):
    samples = []

    N_bins = len(R_bins) - 1
    N_per_bin = N_samples // N_bins

    for i in range(N_bins):
        R_min = R_bins[i]
        R_max = R_bins[i + 1]

        s = rejection_sample_sphere(
            galax_pot, N_per_bin, R_max, batch_size, t, R_min=R_min
        )
        samples.append(s)

    return np.vstack(samples)


def rejection_sample_time_sphere(galax_pot, t, N_samples, R_max, batch_size=10000):
    samples = []
    x_test = np.linspace(-R_max, R_max, 20)
    y_test = np.linspace(-R_max, R_max, 20)
    z_test = np.linspace(-R_max, R_max, 20)
    X, Y, Z = np.meshgrid(x_test, y_test, z_test, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    mask = R <= R_max

    pos_grid = cx.CartesianPos3D(
        x=X[mask].ravel() * au.kpc,
        y=Y[mask].ravel() * au.kpc,
        z=Z[mask].ravel() * au.kpc,
    )
    rho_grid = density(galax_pot, pos_grid, t=t)
    normalization = np.max(rho_grid).value * 1.1

    while sum(len(s) for s in samples) < N_samples:
        x = np.random.uniform(-R_max, R_max, batch_size)
        y = np.random.uniform(-R_max, R_max, batch_size)
        z = np.random.uniform(-R_max, R_max, batch_size)
        r = np.sqrt(x**2 + y**2 + z**2)

        inside = r <= R_max
        x, y, z = x[inside], y[inside], z[inside]

        if len(x) == 0:
            continue

        pos = cx.CartesianPos3D(x=x * au.kpc, y=y * au.kpc, z=z * au.kpc)
        rho = density(galax_pot, pos, t=t).value

        threshold = np.random.uniform(0, 1, len(x))
        accepted = threshold < (rho / normalization)

        new_samples = np.stack([x[accepted], y[accepted], z[accepted]], axis=1)
        samples.append(new_samples)

    return np.vstack(samples)[:N_samples]


def generate_static_datadict_sphere(
    galax_potential,
    N_samples_train,
    N_samples_test,
    r_max_train,
    r_max_test,
    eval_sample_mode="rejection",
    add_pts_train=None,
    add_pts_test=None,
    log_proposal_pts=False,
):

    def evaluate(samples, t=0):
        x, y, z = samples.T
        pos = cx.CartesianPos3D(x=x * au.kpc, y=y * au.kpc, z=z * au.kpc)

        acc = galax_potential.acceleration(pos, t=Quantity(t, au.Myr))
        pot = galax_potential.potential(pos, t=Quantity(t, au.Myr)).value

        a = np.stack([acc.x.value, acc.y.value, acc.z.value], axis=1)
        return samples, a, pot

    x_train = rejection_sample_sphere(
        galax_potential, N_samples_train, r_max_train, log_proposal_pts=log_proposal_pts
    )
    if add_pts_train is not None:
        if isinstance(add_pts_train, list):
            for pt in add_pts_train:
                x_train = np.vstack([x_train, pt])
        else:
            x_train = np.vstack([x_train, add_pts_train])

    if eval_sample_mode == "rejection":
        x_val = rejection_sample_sphere(galax_potential, N_samples_test, r_max_test)
    if eval_sample_mode == "uniform":
        x_val = np.random.uniform(-r_max_test, r_max_test, (N_samples_test, 3))

    if eval_sample_mode == "stratified":
        R_bins = [0, 10, 100, r_max_test]
        x_val = stratified_rejection_sample(
            galax_potential, N_samples_test, R_bins, t=0, batch_size=10000
        )

    if eval_sample_mode == "log_uniform":
        log_r_min = 1.0
        r_val = np.exp(
            np.random.uniform(np.log(log_r_min), np.log(r_max_test), N_samples_test)
        )

        theta = np.random.uniform(0, 2 * np.pi, N_samples_test)
        phi = np.random.uniform(0, np.pi, N_samples_test)

        x_val = np.stack(
            [
                r_val * np.sin(phi) * np.cos(theta),
                r_val * np.sin(phi) * np.sin(theta),
                r_val * np.cos(phi),
            ],
            axis=1,
        )

    if add_pts_test is not None:
        if isinstance(add_pts_test, list):
            for pt in add_pts_test:
                x_val = np.vstack([x_val, pt])
        else:
            x_val = np.vstack([x_val, add_pts_test])

    x_train, a_train, u_train = evaluate(x_train)
    x_val, a_val, u_val = evaluate(x_val)

    r_train = np.linalg.norm(x_train, axis=1)
    r_val = np.linalg.norm(x_val, axis=1)

    return {
        "x_train": x_train,
        "a_train": a_train,
        "u_train": u_train,
        "r_train": r_train,
        "x_val": x_val,
        "a_val": a_val,
        "u_val": u_val,
        "r_val": r_val,
    }


def generate_time_dep_datadict_sphere(
    galax_potential,
    times_train,
    times_test,
    N_samples_train,
    N_samples_test,
    r_max_train,
    r_max_test,
    N_train_pts_list=None,
):
    def get_data(t, N_samples, R_max):
        samples = rejection_sample_time_sphere(
            galax_potential, t, N_samples=N_samples, R_max=R_max
        )
        samples = np.array(samples)
        x, y, z = samples.T

        w = cx.CartesianPos3D(x=x * au.kpc, y=y * au.kpc, z=z * au.kpc)

        t_array = Quantity(np.full(len(w), t), au.Myr)

        acc = galax_potential.acceleration(w, t_array)
        pot = galax_potential.potential(w, t_array).ustrip("kpc2/Myr2")

        acc_x = acc.x.ustrip("kpc/Myr2")
        acc_y = acc.y.ustrip("kpc/Myr2")
        acc_z = acc.z.ustrip("kpc/Myr2")

        a_flat = np.stack([acc_x, acc_y, acc_z], axis=1)
        x_flat = np.stack([x, y, z], axis=1)

        return x_flat, a_flat, pot

    train_data = {}
    val_data = {}

    for i, t in enumerate(times_train):
        try:
            t = float(t.ustrip("Myr"))
        except:
            t = t
        if N_train_pts_list is None:
            x_flat, a_flat, u_flat = get_data(t, N_samples_train, r_max_train)
        else:
            x_flat, a_flat, u_flat = get_data(t, N_train_pts_list[i], r_max_train)

        train_data[t] = {
            "x": x_flat,
            "r": np.linalg.norm(x_flat, axis=-1),
            "a": a_flat,
            "u": u_flat,
        }

    for t in times_test:
        try:
            t = float(t.ustrip("Myr"))
        except:
            t = t
        x_flat, a_flat, u_flat = get_data(t, N_samples_test, r_max_test)
        val_data[t] = {
            "x": x_flat,
            "r": np.linalg.norm(x_flat, axis=-1),
            "a": a_flat,
            "u": u_flat,
        }

    return {"train": train_data, "val": val_data}


def generate_full_galaxy(masses="default", scales="default", alpha_bar="default", include_LMC = True):
    if masses == "default":
        masses = {"halo": 1e12, "disk": 1e11, "bulge": 2e10, "bar": 1e10}
    if scales == "default":
        scales = {
            "r_s": 10,
            "a_disk": 5,
            "b_disk": 0.5,
            "a_bar": 4,
            "b_bar": 1.5,
            "c_bar": 1.0,
        }
    if alpha_bar == "default":
        alpha_bar = 0
    base = gp.NFWPotential(m=masses["halo"], r_s=scales["r_s"], units="galactic")+ gp.HernquistPotential(
            m_tot=masses["bulge"], r_s=scales["r_s"], units="galactic"
        ) + gp.MiyamotoNagaiPotential(
            m_tot=masses["disk"],
            a=scales["a_disk"],
            b=scales["b_disk"],
            units="galactic") + gp.LongMuraliBarPotential(
            m_tot=masses["bar"],
            a=scales["a_bar"],
            b=scales["b_bar"],
            c=scales["c_bar"],
            alpha=alpha_bar,
            units="galactic")
    if include_LMC:
        lmc_pot = gp.NFWPotential(m=1e11, r_s=5, units="galactic")
        lmc_center = jnp.array([50.0, 0.0, 0.0])
        lmc_pot_centered = gp.TranslatedPotential(lmc_pot, translation=lmc_center)
        return base + lmc_pot_centered
    else:
        return base







###

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

        # mean and scale for each feature
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



def scale_by_non_dim_potential_time_by_dict(data_dict, config):
    x_transformer = config.get("x_transformer", UniformScaler(feature_range=(-1, 1)))
    a_transformer = config.get("a_transformer", UniformScaler(feature_range=(-1, 1)))
    u_transformer = config.get("u_transformer", UniformScaler(feature_range=(-1, 1)))
    t_transformer = config.get("t_transformer", UniformScaler(feature_range=(-1, 1)))

    r_s = config["r_s"]
    train_data = data_dict["train"]
    val_data = data_dict["val"]

    # --- Flatten all training data ---
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


def acc_cart_to_cyl(a):
    a_rho = np.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2)
    a_phi = np.arctan2(a[:, 1], a[:, 0])
    a_z = a[:, 2]
    a_mag = np.linalg.norm(a, axis=1)

    return np.stack([a_rho, a_phi, a_z, a_mag], axis=1)
