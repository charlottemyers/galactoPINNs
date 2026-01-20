from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import astropy.units as au
import coordinax as cx
import galax.potential as gp
from galax.potential import density
from unxt import Quantity


__all__ = [
    # sampling
    "biased_sphere_samples",
    "rejection_sample_sphere",
    "rejection_sample_time_sphere",
    # datasets
    "generate_static_datadict_sphere",
    "generate_time_dep_datadict_sphere",
    # analytic potential factory
    # scaling utilities
    "UniformScaler",
    "Transformer",
    "scale_by_non_dim_potential",
    "scale_by_non_dim_potential_time_by_dict",
    # misc
    "acc_cart_to_cyl_like",
]


# -------------------------
# Sampling utilities
# -------------------------

def biased_sphere_samples(N: int, r_min: float, r_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample points in R^3 with radii distributed log-uniformly on [r_min, r_max]
    and isotropic angles.

    Parameters
    ----------
    N
        Number of samples.
    r_min, r_max
        Minimum and maximum radius in the same units tht x,y,z. are interpreted in.

    Returns
    -------
    x, y, z
        Arrays of shape (N,) giving Cartesian coordinates.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if r_min <= 0 or r_max <= 0 or r_max <= r_min:
        raise ValueError("Require 0 < r_min < r_max.")

    r = np.exp(np.random.uniform(np.log(r_min), np.log(r_max), N))
    theta = np.random.uniform(0.0, 2.0 * np.pi, N)
    # isotropic: cos(phi) uniform on [-1, 1]
    phi = np.arccos(np.random.uniform(-1.0, 1.0, N))

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def _estimate_density_upper_bound(
    galax_pot,
    R_max: float,
    t: float,
    grid_n: int = 20,
    safety_factor: float = 1.2,
) -> float:
    """
    Estimate an upper bound on density inside a ball of radius R_max by sampling
    a coarse 3D grid and taking max(density). This is used to define the rejection sampling envelope.
    """
    x_test = np.linspace(-R_max, R_max, grid_n)
    y_test = np.linspace(-R_max, R_max, grid_n)
    z_test = np.linspace(-R_max, R_max, grid_n)
    X, Y, Z = np.meshgrid(x_test, y_test, z_test, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    mask = R <= R_max

    pos_grid = cx.CartesianPos3D(
        x=X[mask].ravel() * au.kpc,
        y=Y[mask].ravel() * au.kpc,
        z=Z[mask].ravel() * au.kpc,
    )
    rho_grid = density(galax_pot, pos_grid, t=t).value
    rho_max = float(np.max(rho_grid))

    # Always inflate; prevents acceptance probability > 1 when grid misses maxima.
    return safety_factor * rho_max


def rejection_sample_sphere(
    galax_pot,
    N_samples: int,
    R_max: float,
    *,
    batch_size: int = 10_000,
    t: float = 0.0,
    R_min: float = 0.0,
    log_proposal_pts: bool = False,
    grid_n: int = 20,
    safety_factor: float = 1.2,
) -> np.ndarray:
    """
    Rejection sample points in a spherical shell R_min <= r <= R_max with acceptance
    probability proportional to the density at time t.

    Parameters
    ----------
    galax_pot
        A Galax potential with a defined density via `galax.potential.density`.
    N_samples
        Total number of accepted samples to return.
    R_max
        Outer radius (kpc).
    batch_size
        Proposal batch size per iteration. Larger is usually faster but higher memory.
    t
        Time passed to `density(...)`.
    R_min
        Inner radius (kpc). Use 0.0 for a full ball.
    log_proposal_pts
        If True, propose radii log-uniformly (biased toward small radii).
        If False, propose uniformly in a cube and then clip to the sphere.
    grid_n, safety_factor
        Controls the coarse upper-bound estimate of max density used in rejection envelope.

    Returns
    -------
    samples
        Array of shape (N_samples, 3) of accepted Cartesian positions in kpc units (unitless numbers).
    """
    if N_samples <= 0:
        raise ValueError("N_samples must be positive.")
    if R_max <= 0 or R_min < 0 or R_max <= R_min:
        raise ValueError("Require 0 <= R_min < R_max.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    normalization = _estimate_density_upper_bound(
        galax_pot, R_max=R_max, t=t, grid_n=grid_n, safety_factor=safety_factor
    )

    accepted_chunks: list[np.ndarray] = []
    accepted_total = 0

    while accepted_total < N_samples:
        # ---- propose points ----
        if log_proposal_pts:
            x, y, z = biased_sphere_samples(batch_size, max(R_min, 1e-6), R_max)
        else:
            x = np.random.uniform(-R_max, R_max, batch_size)
            y = np.random.uniform(-R_max, R_max, batch_size)
            z = np.random.uniform(-R_max, R_max, batch_size)

        r = np.sqrt(x**2 + y**2 + z**2)
        inside = (r <= R_max) & (r >= R_min)
        x, y, z = x[inside], y[inside], z[inside]

        if x.size == 0:
            continue

        pos = cx.CartesianPos3D(x=x * au.kpc, y=y * au.kpc, z=z * au.kpc)
        rho = density(galax_pot, pos, t=t).value

        # ---- accept/reject ----
        # Ensure probability never exceeds 1.0 even if normalization is imperfect.
        p = np.clip(rho / normalization, 0.0, 1.0)
        accepted = np.random.uniform(0.0, 1.0, size=x.size) < p

        new_samples = np.stack([x[accepted], y[accepted], z[accepted]], axis=1)
        accepted_chunks.append(new_samples)
        accepted_total += new_samples.shape[0]

    return np.vstack(accepted_chunks)[:N_samples]




def rejection_sample_time_sphere(
    galax_pot,
    t: float,
    N_samples: int,
    R_max: float,
    *,
    batch_size: int = 10_000,
) -> np.ndarray:
    """
    Backwards-compatible wrapper around `rejection_sample_sphere` with a time parameter.

    Parameters
    ----------
    galax_pot : Galax potential
    t : float
        Time passed to density.
    N_samples : int
    R_max : float
    batch_size : int

    Returns
    -------
    samples : (N_samples, 3)
    """
    return rejection_sample_sphere(
        galax_pot,
        N_samples=N_samples,
        R_max=R_max,
        batch_size=batch_size,
        t=t,
        R_min=0.0,
        log_proposal_pts=False,
    )


# -------------------------
# Dataset generation
# -------------------------

def generate_static_datadict_sphere(
    galax_potential,
    N_samples_train: int,
    N_samples_test: int,
    r_max_train: float,
    r_max_test: float,
    *,
    eval_sample_mode: Literal["rejection", "uniform", "log_uniform"] = "rejection",
    add_pts_train: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    add_pts_test: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    log_proposal_pts: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generate a static dataset at t=0: positions, accelerations, potentials.

    Returns a dict compatible with the training and evaluation features:
        x_train, a_train, u_train, r_train, x_val, a_val, u_val, r_val

    Notes
    -----
    - Positions are returned as unitless floats interpreted as kpc.
    - Accelerations/potentials are returned in Galax "galactic" units (typically kpc/Myr^2 and kpc^2/Myr^2)
      because we strip `.value` from the underlying Quantity objects.

    Parameters
    ----------
    galax_potential
        A Galax potential instance with .acceleration and .potential methods.
    N_samples_train, N_samples_test
        Number of train/val samples (not counting any added points).
    r_max_train, r_max_test
        Max radius for sampling train/val positions.
    eval_sample_mode
        How to sample validation points.
    add_pts_train, add_pts_test
        Optional additional points to append (single (3,) or (N,3), or list of such arrays).
    log_proposal_pts
        If True, train sampling uses log-radius proposal in rejection sampler.

    Returns
    -------
    data : dict[str, np.ndarray]
    """
    def _evaluate(samples: np.ndarray, t: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = samples.T
        pos = cx.CartesianPos3D(x=x * au.kpc, y=y * au.kpc, z=z * au.kpc)

        t_q = Quantity(t, au.Myr)
        acc = galax_potential.acceleration(pos, t=t_q)
        pot = galax_potential.potential(pos, t=t_q).value

        a = np.stack([acc.x.value, acc.y.value, acc.z.value], axis=1)
        return samples, a, pot

    def _append_points(base: np.ndarray, extra) -> np.ndarray:
        if extra is None:
            return base
        if isinstance(extra, (list, tuple)):
            out = base
            for pt in extra:
                out = np.vstack([out, np.atleast_2d(pt)])
            return out
        return np.vstack([base, np.atleast_2d(extra)])

    # --- train positions ---
    x_train = rejection_sample_sphere(
        galax_potential,
        N_samples_train,
        r_max_train,
        log_proposal_pts=log_proposal_pts,
    )
    x_train = _append_points(x_train, add_pts_train)

    # --- val positions ---
    if eval_sample_mode == "rejection":
        x_val = rejection_sample_sphere(galax_potential, N_samples_test, r_max_test)
    elif eval_sample_mode == "uniform":
        x_val = np.random.uniform(-r_max_test, r_max_test, size=(N_samples_test, 3))
    elif eval_sample_mode == "log_uniform":
        log_r_min = 1.0
        r_val = np.exp(np.random.uniform(np.log(log_r_min), np.log(r_max_test), N_samples_test))
        theta = np.random.uniform(0.0, 2.0 * np.pi, N_samples_test)
        phi = np.random.uniform(0.0, np.pi, N_samples_test)
        x_val = np.stack(
            [
                r_val * np.sin(phi) * np.cos(theta),
                r_val * np.sin(phi) * np.sin(theta),
                r_val * np.cos(phi),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unknown eval_sample_mode='{eval_sample_mode}'.")

    x_val = _append_points(x_val, add_pts_test)

    # --- evaluate ---
    x_train, a_train, u_train = _evaluate(x_train, t=0.0)
    x_val, a_val, u_val = _evaluate(x_val, t=0.0)

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
    times_train: Sequence,
    times_test: Sequence,
    N_samples_train: int,
    N_samples_test: int,
    r_max_train: float,
    r_max_test: float,
    *,
    N_train_pts_list: Optional[Sequence[int]] = None,
) -> Dict[str, Dict[float, Dict[str, np.ndarray]]]:
    """
    Generate a time-dependent dataset as nested dicts keyed by time.

    Output format:
        {
          "train": {t0: {"x": (N,3), "r": (N,), "a": (N,3), "u": (N,)}, ...},
          "val":   {t1: {...}, ...}
        }

    Parameters
    ----------
    galax_potential
    times_train, times_test
        Iterable of times (floats or Quantities).
    N_samples_train, N_samples_test
        Samples per time (unless N_train_pts_list provided for train).
    r_max_train, r_max_test
        Max radius for sampling positions.
    N_train_pts_list
        Optional list of per-time train sample counts (len must equal len(times_train)).

    Returns
    -------
    datadict : dict
        Nested time-keyed dicts as described above.
    """
    def _as_float_myr(t):
        try:
            return float(t.ustrip("Myr"))
        except Exception:
            return float(t)

    def _get_data(t_myr: float, N_samples: int, R_max: float):
        samples = rejection_sample_time_sphere(galax_potential, t_myr, N_samples=N_samples, R_max=R_max)
        x, y, z = samples.T
        pos = cx.CartesianPos3D(x=x * au.kpc, y=y * au.kpc, z=z * au.kpc)

        t_array = Quantity(np.full(len(samples), t_myr), au.Myr)

        acc = galax_potential.acceleration(pos, t_array)
        pot = galax_potential.potential(pos, t_array).ustrip("kpc2/Myr2")

        a_flat = np.stack(
            [
                acc.x.ustrip("kpc/Myr2"),
                acc.y.ustrip("kpc/Myr2"),
                acc.z.ustrip("kpc/Myr2"),
            ],
            axis=1,
        )
        x_flat = np.stack([x, y, z], axis=1)
        return x_flat, a_flat, pot

    if N_train_pts_list is not None and len(N_train_pts_list) != len(times_train):
        raise ValueError("If provided, N_train_pts_list must match len(times_train).")

    train_data: Dict[float, Dict[str, np.ndarray]] = {}
    val_data: Dict[float, Dict[str, np.ndarray]] = {}

    for i, t in enumerate(times_train):
        t_myr = _as_float_myr(t)
        N_here = N_samples_train if N_train_pts_list is None else int(N_train_pts_list[i])
        x_flat, a_flat, u_flat = _get_data(t_myr, N_here, r_max_train)

        train_data[t_myr] = {
            "x": x_flat,
            "r": np.linalg.norm(x_flat, axis=-1),
            "a": a_flat,
            "u": u_flat,
        }

    for t in times_test:
        t_myr = _as_float_myr(t)
        x_flat, a_flat, u_flat = _get_data(t_myr, N_samples_test, r_max_test)
        val_data[t_myr] = {
            "x": x_flat,
            "r": np.linalg.norm(x_flat, axis=-1),
            "a": a_flat,
            "u": u_flat,
        }

    return {"train": train_data, "val": val_data}

# -------------------------
# Scaling utilities
# -------------------------

class UniformScaler:
    """
    Simple affine scaler for NumPy arrays.

    Two modes:
    1) Min-max scaling to a feature_range (default [-1, 1]).
    2) Fixed scaling by a constant `scaler` (multiplicative), with zero offset.

    This is intentionally lightweight (no sklearn dependency).
    """

    def __init__(self, feature_range: Tuple[float, float] = (-1.0, 1.0)):
        self.feature_range = feature_range
        self.scaler: Optional[float] = None

        # Learned parameters (min-max mode)
        self.scale_: Optional[float] = None
        self.min_: Optional[float] = None

        # Diagnostics
        self.data_min_: Optional[float] = None
        self.data_max_: Optional[float] = None
        self.data_range_: Optional[float] = None

    def fit(self, data: np.ndarray, *, scaler: Optional[float] = None) -> "UniformScaler":
        """
        Fit scaler parameters from data.

        Parameters
        ----------
        data
            Data used to compute min/max if scaler is None.
        scaler
            If provided, activates constant scaling mode (x -> x * scaler).

        Returns
        -------
        self
        """
        data = np.asarray(data)

        self.scaler = scaler
        data_max = float(np.max(data))
        data_min = float(np.min(data))
        data_range = data_max - data_min

        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range

        if scaler is not None:
            # constant scaling mode
            self.scale_ = float(scaler)
            self.min_ = 0.0
            return self

        if data_range == 0:
            # Degenerate case: data constant. Map everything to midpoint of feature range.
            lo, hi = self.feature_range
            self.scale_ = 0.0
            self.min_ = 0.5 * (lo + hi)
            return self

        lo, hi = self.feature_range
        self.scale_ = (hi - lo) / data_range
        self.min_ = lo - data_min * self.scale_
        return self

    def fit_transform(self, data: np.ndarray, *, scaler: Optional[float] = None) -> np.ndarray:
        """Fit then transform."""
        self.fit(data, scaler=scaler)
        return self.transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted parameters.
        """
        if self.scale_ is None or self.min_ is None:
            raise RuntimeError("UniformScaler must be fit before calling transform().")

        data = np.asarray(data)
        if self.scaler is not None:
            return data * self.scaler
        if self.scale_ == 0.0:
            return np.zeros_like(data) + self.min_
        return data * self.scale_ + self.min_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Invert the transform.
        """
        if self.scale_ is None or self.min_ is None:
            raise RuntimeError("UniformScaler must be fit before calling inverse_transform().")

        data = np.asarray(data)
        if self.scaler is not None:
            return data / self.scaler
        if self.scale_ == 0.0:
            # all data mapped to constant; inverse is undefined -> return constant original min
            return np.zeros_like(data) + (self.data_min_ if self.data_min_ is not None else 0.0)
        return (data - self.min_) / self.scale_


class Transformer:
    """
    Robust per-feature scaler that centers by mean and scales by max absolute deviation.

    For a feature vector x:
        x_scaled = (x - mean) / scale
    where scale = max(|x - mean|) per feature.

    This is similar to sklearn's MaxAbsScaler on centered data.

    Notes
    -----
    - Uses NumPy arrays internally, but returns NumPy arrays that are compatible with JAX.
    - Adds epsilon when scale is zero to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-12):
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.eps = float(eps)

    def fit(self, data: np.ndarray) -> "Transformer":
        """Fit mean and scale from data."""
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[:, None]

        mean = np.mean(data, axis=0)
        scale = np.max(np.abs(data - mean), axis=0)
        scale = np.maximum(scale, self.eps)

        self.mean = mean
        self.scale = scale
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply scaling."""
        if self.mean is None or self.scale is None:
            raise RuntimeError("Transformer must be fit before calling transform().")

        data = np.asarray(data)
        is_1d = (data.ndim == 1)
        if is_1d:
            data = data[:, None]
        out = (data - self.mean) / self.scale
        return out.squeeze() if is_1d else out

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Invert scaling."""
        if self.mean is None or self.scale is None:
            raise RuntimeError("Transformer must be fit before calling inverse_transform().")

        data = np.asarray(data)
        is_1d = (data.ndim == 1)
        if is_1d:
            data = data[:, None]
        out = data * self.scale + self.mean
        return out.squeeze() if is_1d else out


def scale_by_non_dim_potential(
    data_dict: Dict[str, np.ndarray],
    config: dict,
) -> Tuple[Dict[str, np.ndarray], Dict[str, UniformScaler]]:
    """
    Non-dimensionalize position/acceleration/potential using scale radius r_s and
    an empirically chosen potential scale u_star.
    --------------------------------------
    Let u_star = max(|u_train|) or max(|u_train - u_analytic|) if include_analytic.
    Then:
        t_star = sqrt(r_s^2 / u_star)
        a_star = r_s / t_star^2
        x_star = r_s

    Scaling is implemented via UniformScaler in constant-scaling mode:
        x_scaled = x / x_star
        a_scaled = a / a_star
        u_scaled = u / u_star

    Returns
    -------
    scaled_data_dict, transformers
    """
    x_transformer = config.get("x_transformer", UniformScaler(feature_range=(-1, 1)))
    a_transformer = config.get("a_transformer", UniformScaler(feature_range=(-1, 1)))
    u_transformer = config.get("u_transformer", UniformScaler(feature_range=(-1, 1)))

    r_s = float(config["r_s"])  # kpc

    if config.get("include_analytic", False):
        lf_potential = config["lf_analytic_function"]
        pos = cx.CartesianPos3D(
            x=data_dict["x_train"][:, 0] * au.kpc,
            y=data_dict["x_train"][:, 1] * au.kpc,
            z=data_dict["x_train"][:, 2] * au.kpc,
        )
        u_analytic = lf_potential.potential(pos, 0).ustrip("kpc2/Myr2")
        u_residual = data_dict["u_train"] - u_analytic
        u_star = float(np.max(np.abs(u_residual)))
    else:
        u_star = float(np.max(np.abs(data_dict["u_train"])))

    if u_star <= 0:
        raise ValueError("Computed u_star must be positive.")

    t_star = np.sqrt(r_s**2 / u_star)
    a_star = r_s / (t_star**2)
    x_star = r_s

    # Fit constant scaling factors (1/x_star etc.)
    x_transformer.fit(data_dict["x_train"], scaler=1.0 / x_star)
    a_transformer.fit(data_dict["a_train"], scaler=1.0 / a_star)
    u_transformer.fit(data_dict["u_train"], scaler=1.0 / u_star)

    x_train = x_transformer.transform(data_dict["x_train"])
    a_train = a_transformer.transform(data_dict["a_train"])
    u_train = u_transformer.transform(data_dict["u_train"])
    r_train = np.linalg.norm(x_train, axis=1)

    x_val = x_transformer.transform(data_dict["x_val"])
    a_val = a_transformer.transform(data_dict["a_val"])
    u_val = u_transformer.transform(data_dict["u_val"])
    r_val = np.linalg.norm(x_val, axis=1)

    scaled = {
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
    return scaled, transformers


def scale_by_non_dim_potential_time_by_dict(
    data_dict: Dict[str, Dict[float, Dict[str, np.ndarray]]],
    config: dict,
) -> Tuple[Dict[str, Dict[float, Dict[str, np.ndarray]]], Dict[str, UniformScaler]]:
    """
    Non-dimensionalize a nested time-dependent dataset and attach time as an input column.

    Input format
    ------------
    data_dict = {"train": {t: {...}}, "val": {t: {...}}}
    Each block has:
        "x": (N, 3), "a": (N, 3), "u": (N,)

    Output format
    -------------
    Same structure, but each block's "x" becomes (N, 4) where:
        x_with_time = [t_scaled, x_scaled]

    Returns
    -------
    scaled_dict, transformers with keys {"x","a","u","t"}
    """
    x_transformer = config.get("x_transformer", UniformScaler(feature_range=(-1, 1)))
    a_transformer = config.get("a_transformer", UniformScaler(feature_range=(-1, 1)))
    u_transformer = config.get("u_transformer", UniformScaler(feature_range=(-1, 1)))
    t_transformer = config.get("t_transformer", UniformScaler(feature_range=(-1, 1)))

    r_s = float(config["r_s"])
    train_data = data_dict["train"]
    val_data = data_dict["val"]

    # Flatten training blocks for scale determination
    all_x, all_a, all_u, all_t = [], [], [], []
    for t, d in train_data.items():
        all_x.append(d["x"])
        all_a.append(d["a"])
        all_u.append(d["u"])
        all_t.append(np.full(len(d["x"]), float(t)))

    x_concat = np.vstack(all_x)
    a_concat = np.vstack(all_a)
    u_concat = np.hstack(all_u)
    t_concat = np.concatenate(all_t)

    if config.get("include_analytic", False):
        lf_analytic = config["lf_analytic_function"]
        pos = cx.CartesianPos3D(
            x=x_concat[:, 0] * au.kpc,
            y=x_concat[:, 1] * au.kpc,
            z=x_concat[:, 2] * au.kpc,
        )
        t_quant = Quantity(t_concat, au.Myr)
        u_analytic = lf_analytic.potential(pos, t_quant).ustrip("kpc2/Myr2")
        u_resid = u_concat - u_analytic
        u_star = float(np.max(np.abs(u_resid)))
    else:
        u_star = float(np.max(np.abs(u_concat)))

    if u_star <= 0:
        raise ValueError("Computed u_star must be positive.")

    t_star = np.sqrt(r_s**2 / u_star)
    a_star = r_s / (t_star**2)
    x_star = r_s

    x_transformer.fit(x_concat, scaler=1.0 / x_star)
    a_transformer.fit(a_concat, scaler=1.0 / a_star)
    u_transformer.fit(u_concat, scaler=1.0 / u_star)
    t_transformer.fit(t_concat, scaler=1.0 / t_star)

    def _transform_block(x: np.ndarray, a: np.ndarray, u: np.ndarray, t_val: float) -> Dict[str, np.ndarray]:
        x_scaled = x_transformer.transform(x)
        # force (N,1) then concat
        t_scaled = t_transformer.transform(np.full((len(x_scaled), 1), float(t_val)))
        x_with_time = np.concatenate([t_scaled, x_scaled], axis=1)
        return {"x": x_with_time, "a": a_transformer.transform(a), "u": u_transformer.transform(u)}

    scaled_train = {t: _transform_block(d["x"], d["a"], d["u"], t) for t, d in train_data.items()}
    scaled_val = {t: _transform_block(d["x"], d["a"], d["u"], t) for t, d in val_data.items()}

    transformers = {"x": x_transformer, "a": a_transformer, "u": u_transformer, "t": t_transformer}
    return {"train": scaled_train, "val": scaled_val}, transformers


# -------------------------
# Misc helpers
# -------------------------

def acc_cart_to_cyl_like(a: np.ndarray) -> np.ndarray:
    """
    Compute simple cylindrical-like diagnostics of an acceleration vector.
    Parameters
    ----------
    a
        Array of shape (N, 3) with Cartesian acceleration components.

    Returns
    -------
    out
        Array of shape (N, 4) with columns [a_rho, a_phi, a_z, a_mag].
    """
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("Expected a to have shape (N, 3).")

    a_rho = np.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2)
    a_phi = np.arctan2(a[:, 1], a[:, 0])
    a_z = a[:, 2]
    a_mag = np.linalg.norm(a, axis=1)
    return np.stack([a_rho, a_phi, a_z, a_mag], axis=1)
