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
