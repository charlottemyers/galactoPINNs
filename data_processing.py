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
    """
    Scale spatial positions (x, y, z), accelerations, and potential.
    Time (column 3 of x) is preserved and *not scaled*.
    """
    x_transformer = config["x_transformer"]
    a_transformer = config["a_transformer"]
    u_transformer = config["u_transformer"]

    r_s = config["nfw_r_s"]
    M = config["nfw_M_200"]
    G = 4.514e-22  # kpc^3 / Myr^2 / Msun

    # Compute u_star from residual potential if analytic included
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
        t = x[:, 3:]  # leave time untouched
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


def generate_full_galaxy(masses="default", scales="default", alpha_bar="default"):
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
    return (
        gp.NFWPotential(m=masses["halo"], r_s=scales["r_s"], units="galactic")
        + gp.HernquistPotential(
            m_tot=masses["bulge"], r_s=scales["r_s"], units="galactic"
        )
        + gp.MiyamotoNagaiPotential(
            m_tot=masses["disk"],
            a=scales["a_disk"],
            b=scales["b_disk"],
            units="galactic",
        )
        + gp.LongMuraliBarPotential(
            m_tot=masses["bar"],
            a=scales["a_bar"],
            b=scales["b_bar"],
            c=scales["c_bar"],
            alpha=alpha_bar,
            units="galactic",
        )
    )


def generate_data(
    galax_potential,
    galpy_potential,
    galpy_df,
    r_min,
    r_max_train,
    r_max_test,
    n_samples_train=100000,
    n_samples_test=2000,
):
    galpy_potential.turn_physical_on()
    samples_train = galpy_df.sample(n=n_samples_train)

    w_train = cx.CartesianPos3D(
        x=samples_train.x() * au.kpc,
        y=samples_train.y() * au.kpc,
        z=samples_train.z() * au.kpc,
    )

    acc_train = galax_potential.acceleration(w_train, 0)
    u_train = galax_potential.potential(w_train, 0).ustrip("kpc2/Myr2")

    acc_x_train = acc_train.x.ustrip("kpc/Myr2")
    acc_y_train = acc_train.y.ustrip("kpc/Myr2")
    acc_z_train = acc_train.z.ustrip("kpc/Myr2")

    x_train = samples_train.x()
    y_train = samples_train.y()
    z_train = samples_train.z()

    radii_train = np.sqrt(x_train**2 + y_train**2 + z_train**2)
    valid_indices_train = (radii_train > r_min) & (radii_train < r_max_train)

    x_train = np.array(
        [
            x_train[valid_indices_train],
            y_train[valid_indices_train],
            z_train[valid_indices_train],
        ]
    ).T
    u_train = u_train[valid_indices_train]
    a_train = np.array(
        [
            acc_x_train[valid_indices_train],
            acc_y_train[valid_indices_train],
            acc_z_train[valid_indices_train],
        ]
    ).T

    samples_test = galpy_df.sample(n=n_samples_test)

    w_test = cx.CartesianPos3D(
        x=samples_test.x() * au.kpc,
        y=samples_test.y() * au.kpc,
        z=samples_test.z() * au.kpc,
    )

    acc_test = galax_potential.acceleration(w_test, 0)
    u_test = galax_potential.potential(w_test, 0).ustrip("kpc2/Myr2")
    acc_x_test = acc_test.x.ustrip("kpc/Myr2")
    acc_y_test = acc_test.y.ustrip("kpc/Myr2")
    acc_z_test = acc_test.z.ustrip("kpc/Myr2")

    x_test = samples_test.x()
    y_test = samples_test.y()
    z_test = samples_test.z()
    radii_test = np.sqrt(x_test**2 + y_test**2 + z_test**2)
    valid_indices_test = (radii_test > r_min) & (radii_test < r_max_test)
    x_test = np.array(
        [
            x_test[valid_indices_test],
            y_test[valid_indices_test],
            z_test[valid_indices_test],
        ]
    ).T
    u_test = u_test[valid_indices_test]
    a_test = np.array(
        [
            acc_x_test[valid_indices_test],
            acc_y_test[valid_indices_test],
            acc_z_test[valid_indices_test],
        ]
    ).T

    return {
        "x_train": x_train,
        "a_train": a_train,
        "u_train": u_train,
        "x_val": x_test,
        "a_val": a_test,
        "u_val": u_test,
    }


def rejection_sample(
    galax_pot, N_samples, xmin, xmax, ymin, ymax, zmin, zmax, batch_size=10000
):
    samples = []

    # Estimate max density over a grid
    x_test = np.linspace(xmin, xmax, 20)
    y_test = np.linspace(ymin, ymax, 20)
    z_test = np.linspace(zmin, zmax, 20)

    X, Y, Z = np.meshgrid(x_test, y_test, z_test, indexing="ij")
    pos_grid = cx.CartesianPos3D(
        x=X.ravel() * au.kpc, y=Y.ravel() * au.kpc, z=Z.ravel() * au.kpc
    )
    rho_grid = density(galax_pot, pos_grid, t=0)
    normalization = np.max(rho_grid) * 1.1

    while len(samples) < N_samples:
        # Generate batch of random points
        x = np.random.uniform(xmin, xmax, batch_size)
        y = np.random.uniform(ymin, ymax, batch_size)
        z = np.random.uniform(zmin, zmax, batch_size)
        pos = cx.CartesianPos3D(x=x * au.kpc, y=y * au.kpc, z=z * au.kpc)

        rho = density(galax_pot, pos, t=0).value
        threshold = np.random.uniform(0, 1, batch_size)
        accepted = threshold < (rho / normalization)

        new_samples = np.stack([x[accepted], y[accepted], z[accepted]], axis=1)
        samples.append(new_samples)

        if sum(len(s) for s in samples) >= N_samples:
            break

    return np.vstack(samples)[:N_samples]


def biased_sphere_samples(N, r_min, r_max):
    # Sample radius log-uniformly
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
        # inside = r <= R_max
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


def rejection_sample_time(
    galax_pot, t, N_samples, xmin, xmax, ymin, ymax, zmin, zmax, batch_size=10000
):
    samples = []
    x_test = np.linspace(xmin, xmax, 20)
    y_test = np.linspace(ymin, ymax, 20)
    z_test = np.linspace(zmin, zmax, 20)
    X, Y, Z = np.meshgrid(x_test, y_test, z_test, indexing="ij")
    pos_grid = cx.CartesianPos3D(
        x=X.ravel() * au.kpc, y=Y.ravel() * au.kpc, z=Z.ravel() * au.kpc
    )
    rho_grid = density(galax_pot, pos_grid, t=t)
    normalization = np.max(rho_grid) * 1.1

    while len(samples) < N_samples:
        # Generate batch of random points
        x = np.random.uniform(xmin, xmax, batch_size)
        y = np.random.uniform(ymin, ymax, batch_size)
        z = np.random.uniform(zmin, zmax, batch_size)
        pos = cx.CartesianPos3D(x=x * au.kpc, y=y * au.kpc, z=z * au.kpc)

        rho = density(galax_pot, pos, t=t).value
        threshold = np.random.uniform(0, 1, batch_size)
        accepted = threshold < (rho / normalization)

        new_samples = np.stack([x[accepted], y[accepted], z[accepted]], axis=1)
        samples.append(new_samples)

        if sum(len(s) for s in samples) >= N_samples:
            break

    return np.vstack(samples)[:N_samples]


def rejection_sample_time_sphere(galax_pot, t, N_samples, R_max, batch_size=10000):
    """
    Rejection sampling at time t using a time-dependent potential.
    Samples are drawn uniformly within a sphere of radius R_max,
    weighted by density evaluated at time t.
    """
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

    # Fit t_transformer on all training time values
    # t_transformer.fit(t_concat.reshape(-1, 1))

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


def apply_model(model, params, x, return_analytic_weights=False):
    predictions = model.apply({"params": params}, x)
    u_pred = predictions["potential"]
    a_pred = predictions["acceleration"]

    if return_analytic_weights:
        analytic_weights = (predictions["outputs"]["h"], predictions["outputs"]["g"])
    else:
        analytic_weights = None
    if "outputs" in predictions:
        outputs = predictions["outputs"]
    else:
        outputs = None
    return {
        "u_pred": u_pred,
        "a_pred": a_pred,
        "analytic_weights": analytic_weights,
        "outputs": outputs,
    }


def evaluate_performance(
    model,
    trained_state_params,
    raw_datadict,
    num_test,
    return_analytic_weights=False,
    lf_trainable_model=None,
):
    true_pot = raw_datadict["u_val"][:num_test]
    true_acc = raw_datadict["a_val"][:num_test]

    print("here!", type(model))
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

    if config.get("include_analytic", False):
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

        if return_analytic_weights:
            analytic_weights = output["analytic_weights"]

        else:
            analytic_weights = None

    else:
        lf_analytic_potential = None
        lf_pot_error = None
        lf_acc_error = None
        residual_pot = None
        corrected_potential = None
        corrected_pot_percent_error = None
        analytic_weights = None

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
        "analytic_weights": analytic_weights,
        "fiducial_acc": fiducial_acc,
        "fiducial_pot": fiducial_pot,
        "fiducial_pot_error": fiducial_pot_error,
        "fiducial_acc_error": fiducial_acc_error,
        # benchmarks
        "avg_percent_error": np.mean(acc_percent_error),
    }


def evaluate_performance_time(
    model, params, t_eval, raw_datadict, num_test, return_analytic_weights=False
):
    val_data = raw_datadict["val"][t_eval]
    x_val = val_data["x"][:num_test]

    config = model.config
    r_eval = np.linalg.norm(x_val, axis=1)

    true_pot = val_data["u"][:num_test]
    true_acc = val_data["a"][:num_test]

    x_scaled = config["x_transformer"].transform(x_val)
    t_scaled = config["t_transformer"].transform(t_eval) * jnp.ones(
        (x_val.shape[0], 1)
    )  # shape (N, 1)
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

    if config.get("include_analytic", False):
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

        if return_analytic_weights:
            analytic_weights = output["analytic_weights"]

        else:
            analytic_weights = None

    else:
        lf_analytic_potential = None
        lf_pot_error = None
        lf_acc_error = None
        residual_pot = None
        corrected_potential = None
        corrected_pot_percent_error = None
        analytic_weights = None
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
        "analytic_weights": analytic_weights,
    }


def acc_cart_to_cyl(a):
    a_rho = np.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2)
    a_phi = np.arctan2(a[:, 1], a[:, 0])
    a_z = a[:, 2]
    a_mag = np.linalg.norm(a, axis=1)

    return np.stack([a_rho, a_phi, a_z, a_mag], axis=1)


def component_breakdown(data, performance):
    predicted_acc = performance["predicted_a"]
    true_acc = performance["true_a"]

    true_acc_cyl = acc_cart_to_cyl(true_acc)
    pred_acc_cyl = acc_cart_to_cyl(predicted_acc)

    acc_percent_error_cyl = 100 * np.abs((pred_acc_cyl - true_acc_cyl) / true_acc_cyl)

    x_val = performance["x_val"]
    z = x_val[:, 2]
    phi = np.arctan2(x_val[:, 1], x_val[:, 0])
    r = performance["r_eval"]
    rho = np.sqrt(x_val[:, 0] ** 2 + x_val[:, 1] ** 2)

    return {
        "true_acc_cyl": true_acc_cyl,
        "pred_acc_cyl": pred_acc_cyl,
        "phi": phi,
        "z": z,
        "r": r,
        "rho": rho,
        "acc_percent_error_rho": acc_percent_error_cyl[:, 0],
        "acc_percent_error_phi": acc_percent_error_cyl[:, 1],
        "acc_percent_error_z": acc_percent_error_cyl[:, 2],
        "acc_percent_error_mag": acc_percent_error_cyl[:, 3],
    }

def apply_model_time(model, params, tx_scaled):
    predictions = model.apply({"params": params}, tx_scaled)
    u_pred = predictions["potential"]
    a_pred = predictions["acceleration"]

    if "outputs" in predictions:
        return {"u_pred": u_pred, "a_pred": a_pred, "outputs": predictions["outputs"]}
    else:
        return {"u_pred": u_pred, "a_pred": a_pred}
