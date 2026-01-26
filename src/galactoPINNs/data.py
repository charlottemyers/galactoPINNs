"""Data loading and preprocessing utilities."""

__all__ = (
    "Transformer",
    # scaling utilities
    "UniformScaler",
    # miscellaneous
    "acc_cart_to_cyl_like",
    # sampling
    "biased_sphere_samples",
    # datasets
    "generate_static_datadict",
    "generate_time_dep_datadict",
    "rejection_sample_sphere",
    "scale_data",
    "scale_data_time",
)

from collections.abc import Sequence
from typing import Any, Literal

import coordinax as cx
import jax.numpy as jnp
import jax.random as jr
import unxt as u
from galax.potential import density
from jaxtyping import Array, ArrayLike
from unxt.quantity import AllowValue

# -------------------------
# Sampling utilities
# -------------------------


def sample_log_uniform_r(
    key: jr.PRNGKey,
    r_min: float,
    r_max: float,
    N: int,
) -> Array:
    """Sample radii distributed log-uniformly on the interval [r_min, r_max].

    This draws samples according to:
        log(r) ~ Uniform(log(r_min), log(r_max))
    which corresponds to a probability density proportional to 1 / r.

    Parameters
    ----------
    key : jax.random.PRNGKey
        JAX random key used for sampling.
    r_min : float
        Minimum radius. Must be strictly positive.
    r_max : float
        Maximum radius. Must satisfy r_max > r_min.
    N : int
        Number of samples to draw.

    Returns
    -------
    r : Array, shape (N,)
        Log-uniformly distributed radii.

    Raises
    ------
    ValueError
        If ``r_min <= 0``, ``r_max <= r_min``, or ``N <= 0``.

    Examples
    --------
    >>> import jax.random as jr
    >>> key = jr.PRNGKey(0)
    >>> r = sample_log_uniform_r(key, 1.0, 100.0, 5)
    >>> r.shape
    (5,)
    >>> jnp.all((r >= 1.0) & (r <= 100.0))
    Array(True, dtype=bool)

    """
    if r_min <= 0:
        raise ValueError("r_min must be strictly positive.")
    if r_max <= r_min:
        raise ValueError("r_max must be greater than r_min.")
    if N <= 0:
        raise ValueError("N must be positive.")

    u = jr.uniform(
        key,
        shape=(N,),
        minval=jnp.log(r_min),
        maxval=jnp.log(r_max),
    )
    return jnp.exp(u)


def sample_angles(
    key: jr.PRNGKey,
    N: int,
) -> tuple[Array, Array]:
    """Sample angular coordinates corresponding to isotropic directions on the sphere.

    Sampling convention is:
        - theta ~ Uniform(0, 2π)
        - cos(phi) ~ Uniform(-1, 1)

    which produces an isotropic distribution of directions in 3D.

    Parameters
    ----------
    key : jax.random.PRNGKey
        JAX random key used for sampling.
    N : int
        Number of angular samples to draw.

    Returns
    -------
    theta : Array, shape (N,)
        Azimuthal angles in radians, in the range [0, 2π).
    phi : Array, shape (N,)
        Polar angles in radians, in the range [0, π].

    Raises
    ------
    ValueError
        If ``N <= 0``.

    Examples
    --------
    >>> import jax.random as jr
    >>> key = jr.PRNGKey(0)
    >>> theta, phi = sample_angles(key, 10)
    >>> theta.shape, phi.shape
    ((10,), (10,))
    >>> jnp.all((theta >= 0.0) & (theta <= 2.0 * jnp.pi))
    Array(True, dtype=bool)
    >>> jnp.all((phi >= 0.0) & (phi <= jnp.pi))
    Array(True, dtype=bool)

    """
    if N <= 0:
        raise ValueError("N must be positive.")

    key_theta, key_phi = jr.split(key, 2)

    theta = jr.uniform(
        key_theta,
        shape=(N,),
        minval=0.0,
        maxval=2.0 * jnp.pi,
    )

    # isotropic: cos(phi) uniform on [-1, 1]
    cos_phi = jr.uniform(
        key_phi,
        shape=(N,),
        minval=-1.0,
        maxval=1.0,
    )
    phi = jnp.arccos(cos_phi)

    return theta, phi


def biased_sphere_samples(
    N: int, r_min: float, r_max: float
) -> tuple[Array, Array, Array]:
    """Sample points in a sphere with radii distributed log-uniformly.

    Parameters
    ----------
    N
        Number of samples.
    r_min, r_max
        Minimum and maximum radius in the same units that x, y, z are interpreted in.

    Returns
    -------
    x, y, z
        Arrays of shape (N,) giving Cartesian coordinates.

    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if r_min <= 0 or r_max <= 0 or r_max <= r_min:
        raise ValueError("Require 0 < r_min < r_max.")

    key = jr.PRNGKey(0)
    r = sample_log_uniform_r(key, r_min, r_max, N)
    theta, phi = sample_angles(key, N)

    x = r * jnp.sin(phi) * jnp.cos(theta)
    y = r * jnp.sin(phi) * jnp.sin(theta)
    z = r * jnp.cos(phi)
    return x, y, z


def _estimate_density_upper_bound(
    galax_pot: Any,
    r_max: float,
    t: float,
    grid_n: int = 20,
    safety_factor: float = 1.2,
) -> float:
    """Estimate an upper bound on density inside a ball of radius r_max.

    Samples a coarse 3D grid and takes max(density). This is used to define
    the rejection sampling envelope.
    """
    x_test = jnp.linspace(-r_max, r_max, grid_n)
    y_test = jnp.linspace(-r_max, r_max, grid_n)
    z_test = jnp.linspace(-r_max, r_max, grid_n)
    X, Y, Z = jnp.meshgrid(x_test, y_test, z_test, indexing="ij")
    R = jnp.sqrt(X**2 + Y**2 + Z**2)
    mask = r_max >= R

    pos_grid = cx.CartesianPos3D(
        x=u.Quantity(X[mask].ravel(), "kpc"),
        y=u.Quantity(Y[mask].ravel(), "kpc"),
        z=u.Quantity(Z[mask].ravel(), "kpc"),
    )
    rho_grid = density(galax_pot, pos_grid, t=t).value
    rho_max = float(jnp.max(rho_grid))

    # Inflate; prevents acceptance probability > 1 when grid misses maxima.
    return safety_factor * rho_max


def rejection_sample_sphere(
    galax_pot: Any,
    n_samples: int,
    r_max: float,
    *,
    batch_size: int = 10_000,
    t: float = 0.0,
    r_min: float = 0.0,
    log_proposal_pts: bool = False,
    grid_n: int = 20,
    safety_factor: float = 1.2,
) -> Array:
    """Density-weighted rejection sample points in a spherical shell.

    Parameters
    ----------
    galax_pot
        A Galax potential with a defined density via `galax.potential.density`.
    n_samples
        Total number of accepted samples to return.
    r_max
        Outer radius (kpc).
    batch_size
        Proposal batch size per iteration.
    t
        Time passed to `density(...)`.
    r_min
        Inner radius (kpc). Use 0.0 for a full ball.
    log_proposal_pts
        If True, propose radii log-uniformly (biased toward small radii).
        If False, propose uniformly in a cube and then clip to the sphere.
    grid_n, safety_factor
        Controls the coarse upper-bound estimate of max density used in
        rejection envelope.

    Returns
    -------
    samples
        Array of shape (n_samples, 3) of accepted Cartesian positions in kpc units.

    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if r_max <= 0 or r_min < 0 or r_max <= r_min:
        raise ValueError("Require 0 <= r_min < r_max.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    normalization = _estimate_density_upper_bound(
        galax_pot, r_max=r_max, t=t, grid_n=grid_n, safety_factor=safety_factor
    )

    accepted_chunks: list[Array] = []
    accepted_total = 0

    key = jr.PRNGKey(0)
    while accepted_total < n_samples:
        # split key for this iteration
        key, key_prop = jr.split(key)

        # ---- propose points ----
        if log_proposal_pts:
            key_prop, key_bs = jr.split(key_prop)
            x, y, z = biased_sphere_samples(key_bs, batch_size, max(r_min, 1e-6), r_max)
        else:
            key_prop, kx, ky, kz = jr.split(key_prop, 4)

            x = jr.uniform(kx, (batch_size,), minval=-r_max, maxval=r_max)
            y = jr.uniform(ky, (batch_size,), minval=-r_max, maxval=r_max)
            z = jr.uniform(kz, (batch_size,), minval=-r_max, maxval=r_max)

        r = jnp.sqrt(x**2 + y**2 + z**2)
        inside = (r <= r_max) & (r >= r_min)

        x, y, z = x[inside], y[inside], z[inside]

        if x.shape[0] == 0:
            continue

        pos = cx.CartesianPos3D(
            x=u.Quantity(x, "kpc"), y=u.Quantity(y, "kpc"), z=u.Quantity(z, "kpc")
        )
        rho = density(galax_pot, pos, t=t).value

        # ---- accept/reject ----
        # Ensure probability never exceeds 1.0 even if normalization is imperfect.
        p = jnp.clip(rho / normalization, 0.0, 1.0)
        key, key_acc = jr.split(key)
        u_all = jr.uniform(key_acc, shape=p.shape, minval=0.0, maxval=1.0)
        accepted = u_all < p

        new_samples = jnp.stack([x[accepted], y[accepted], z[accepted]], axis=1)
        accepted_chunks.append(new_samples)
        accepted_total += new_samples.shape[0]

    return jnp.vstack(accepted_chunks)[:n_samples]


# -------------------------
# Dataset generation
# -------------------------


def generate_static_datadict(
    galax_potential: Any,
    n_samples_train: int,
    n_samples_test: int,
    r_max_train: float,
    r_max_test: float,
    *,
    eval_sample_mode: Literal["rejection", "uniform", "log_uniform"] = "rejection",
    add_pts_train: Array | Sequence[Array] | None = None,
    add_pts_test: Array | Sequence[Array] | None = None,
    log_proposal_pts: bool = False,
) -> dict[str, Array]:
    """Generate a static (time-independent) training and validation dataset at t = 0.

    This function samples 3D Cartesian positions within a spherical region,
    evaluates a Galax potential at those positions, and returns positions,
    accelerations, and potentials in a dictionary format compatible with
    the static training and evaluation pipelines.

    All returned arrays are unitless numeric arrays interpreted in physical
    units as follows:
      - positions: kpc
      - accelerations: kpc / Myr²
      - potential: kpc² / Myr²

    Parameters
    ----------
    galax_potential
        A Galax potential instance providing:
          - `acceleration(pos, t=...)`
          - `potential(pos, t=...)`
        The potential is evaluated at t = 0 for all samples.

    n_samples_train
        Number of training samples to generate.

    n_samples_test
        Number of validation (test) samples to generate.

    r_max_train
        Maximum radius (kpc) for training samples.

    r_max_test
        Maximum radius (kpc) for validation samples.

    eval_sample_mode
        Sampling strategy for validation points:
          - `"rejection"`: density-weighted rejection sampling using
            `rejection_sample_sphere` (default).
          - `"uniform"`: uniform sampling in a cube `[-r_max, r_max]^3`.
          - `"log_uniform"`: log-uniform sampling in radius with isotropic
            angular distribution, biased toward small radii.

    add_pts_train
        Optional additional training points to append *after* sampling.
        Allowed formats:
          - a single array of shape (3,)
          - an array of shape (N, 3)
          - a list/tuple of such arrays

    add_pts_test
        Same as `add_pts_train`, but appended to the validation set.

    log_proposal_pts
        If True, the training rejection sampler proposes points with a
        log-uniform radial distribution, improving coverage at small radii.

    Returns
    -------
    data : dict[str, jax.Array]
        Dictionary containing:

        Training data:
          - `"x_train"` : (N_train, 3)
              Cartesian positions.
          - `"a_train"` : (N_train, 3)
              Cartesian accelerations.
          - `"u_train"` : (N_train,)
              Scalar gravitational potential.
          - `"r_train"` : (N_train,)
              Radius `norm(x)`.

        Validation data:
          - `"x_val"`   : (N_val, 3)
          - `"a_val"`   : (N_val, 3)
          - `"u_val"`   : (N_val,)
          - `"r_val"`   : (N_val,)

    """

    def _evaluate(samples: Array, t: float = 0.0) -> tuple[Array, Array, Array]:
        x, y, z = samples.T
        pos = cx.CartesianPos3D(
            x=u.Quantity(x, "kpc"),
            y=u.Quantity(y, "kpc"),
            z=u.Quantity(z, "kpc"),
        )
        t_q = u.Quantity(t, "Myr")
        acc = galax_potential.acceleration(pos, t=t_q)
        pot = galax_potential.potential(pos, t=t_q).value  # keep as unitless array
        a = jnp.stack([acc.x.value, acc.y.value, acc.z.value], axis=1)
        return samples, a, pot

    def _append_points(base: Array, extra: ArrayLike | None) -> Array:
        if extra is None:
            return base
        if isinstance(extra, (list, tuple)):
            out = base
            for pt in extra:
                out = jnp.vstack([out, jnp.atleast_2d(pt)])
            return out
        return jnp.vstack([base, jnp.atleast_2d(extra)])

    # -----------------
    # Train positions
    # -----------------
    x_train = rejection_sample_sphere(
        galax_potential,
        n_samples_train,
        r_max_train,
        log_proposal_pts=log_proposal_pts,
    )
    x_train = _append_points(x_train, add_pts_train)

    # -----------------
    # Val positions
    # -----------------
    match eval_sample_mode:
        case "rejection":
            x_val = rejection_sample_sphere(galax_potential, n_samples_test, r_max_test)

        case "uniform":
            key = jr.PRNGKey(0)
            x_val = jr.uniform(
                key, shape=(n_samples_test, 3), minval=-r_max_test, maxval=r_max_test
            )

        case "log_uniform":
            # Log-uniform in radius, isotropic angles (cos(phi) uniform in [-1,1]).
            key = jr.PRNGKey(0)
            key_r, key_th, key_cphi = jr.split(key, 3)

            log_r_min = 1.0
            r_val = jnp.exp(
                jr.uniform(
                    key_r,
                    shape=(n_samples_test,),
                    minval=jnp.log(log_r_min),
                    maxval=jnp.log(r_max_test),
                )
            )
            theta = jr.uniform(
                key_th, shape=(n_samples_test,), minval=0.0, maxval=2.0 * jnp.pi
            )
            cos_phi = jr.uniform(
                key_cphi, shape=(n_samples_test,), minval=-1.0, maxval=1.0
            )
            phi = jnp.arccos(cos_phi)

            x_val = jnp.stack(
                [
                    r_val * jnp.sin(phi) * jnp.cos(theta),
                    r_val * jnp.sin(phi) * jnp.sin(theta),
                    r_val * jnp.cos(phi),
                ],
                axis=1,
            )

        case _:
            msg = f"Unknown eval_sample_mode='{eval_sample_mode}'."
            raise ValueError(msg)

    x_val = _append_points(x_val, add_pts_test)

    # -----------------
    # Evaluate targets
    # -----------------
    x_train, a_train, u_train = _evaluate(x_train, t=0.0)
    x_val, a_val, u_val = _evaluate(x_val, t=0.0)
    r_train = jnp.linalg.norm(x_train, axis=1)
    r_val = jnp.linalg.norm(x_val, axis=1)

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


def generate_time_dep_datadict(
    galax_potential: Any,
    times_train: Sequence[float],
    times_test: Sequence[float],
    n_samples_train: int,
    n_samples_test: int,
    r_max_train: float,
    r_max_test: float,
) -> dict[str, dict[float, dict[str, Array]]]:
    """Generate a time-dependent dataset as nested dicts keyed by time.

    Output format:
        {
          "train": {t0: {"x": (N,3), "r": (N,), "a": (N,3), "u": (N,)}, ...},
          "val":   {t1: {...}, ...}
        }

    Parameters
    ----------
    galax_potential
        A Galax potential representing the true potential model.
    times_train, times_test
        Iterable of times (floats or Quantities).
    n_samples_train, n_samples_test
        Samples per time.
    r_max_train, r_max_test
        Max radius for sampling positions.

    Returns
    -------
    datadict : dict
        Nested time-keyed dicts as described above.

    """

    def _as_float_myr(t: u.AbstractQuantity | ArrayLike) -> Array:
        return u.ustrip(AllowValue, "Myr", t)

    def _get_data(
        t_myr: float, n_samples: int, r_max: float
    ) -> tuple[Array, Array, Array]:
        samples = rejection_sample_sphere(
            galax_potential, t_myr, n_samples=n_samples, r_max=r_max
        )
        x, y, z = samples.T
        pos = cx.CartesianPos3D(
            x=u.Quantity(x, "kpc"), y=u.Quantity(y, "kpc"), z=u.Quantity(z, "kpc")
        )
        t_array = u.Quantity(jnp.full(len(samples), t_myr), "Myr")

        acc = galax_potential.acceleration(pos, t_array)
        pot = galax_potential.potential(pos, t_array).ustrip("kpc2/Myr2")

        a_flat = jnp.stack(
            [
                acc.x.ustrip("kpc/Myr2"),
                acc.y.ustrip("kpc/Myr2"),
                acc.z.ustrip("kpc/Myr2"),
            ],
            axis=1,
        )
        x_flat = jnp.stack([x, y, z], axis=1)
        return x_flat, a_flat, pot

    train_data: dict[float, dict[str, Array]] = {}
    val_data: dict[float, dict[str, Array]] = {}

    for t in times_train:
        t_myr = _as_float_myr(t)
        N_here = n_samples_train
        x_flat, a_flat, u_flat = _get_data(t_myr, N_here, r_max_train)

        train_data[t_myr] = {
            "x": x_flat,
            "r": jnp.linalg.norm(x_flat, axis=-1),
            "a": a_flat,
            "u": u_flat,
        }

    for t in times_test:
        t_myr = _as_float_myr(t)
        x_flat, a_flat, u_flat = _get_data(t_myr, n_samples_test, r_max_test)
        val_data[t_myr] = {
            "x": x_flat,
            "r": jnp.linalg.norm(x_flat, axis=-1),
            "a": a_flat,
            "u": u_flat,
        }

    return {"train": train_data, "val": val_data}


# -------------------------
# Scaling utilities
# -------------------------


class UniformScaler:
    """Affine scaler mapping data to a fixed range or applying a scaling factor.

    This class is designed for non-dimensionalization of physical quantities
    (e.g., positions, accelerations, potentials) where the scaling factor may be
    known a priori.

    Notes
    -----
    - Units are not tracked; inputs are assumed to be unitless numeric arrays

    Attributes
    ----------
    feature_range : tuple[float, float]
        Target range ``(min, max)`` for range-based scaling.
    scale_ : float
        Multiplicative scaling factor.
    min_ : float
        Additive offset applied after scaling.
    data_min_ : float
        Minimum of the data seen during fitting.
    data_max_ : float
        Maximum of the data seen during fitting.
    data_range_ : float
        Difference ``data_max_ - data_min_``.
    scaler : float or None
        Fixed scaling factor used in constant-scaling mode.

    Parameters
    ----------
    feature_range : tuple[float, float], optional
        Desired output range for range-based scaling.  Defaults to ``(-1, 1)``.

    """

    def __init__(self, feature_range: tuple[float, float] = (-1, 1)) -> None:
        """Initialize an unfitted UniformScaler.

        Parameters
        ----------
        feature_range : tuple[float, float], optional
            Desired output range for range-based scaling.

        """
        self.feature_range = feature_range

    def fit(self, data: ArrayLike, scaler: float | None = None) -> None:
        """Fit the scaler parameters from data.

        Parameters
        ----------
        data : array-like
            Input data used to determine scaling parameters.
        scaler : float or None, optional
            If provided, enables constant-scaling mode with
            ``x_scaled = x * scaler``.

        Returns
        -------
        None

        Notes
        -----
        - In constant-scaling mode, the data min/max are still recorded but are
          not used in the transformation.

        """
        self.scaler = scaler

        data_max = jnp.max(data)
        data_min = jnp.min(data)

        data_range = data_max - data_min
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range

    def fit_transform(self, data: ArrayLike, scaler: float | None = None) -> Array:
        """Fit the scaler parameters and apply the transformation in one step.

        Parameters
        ----------
        data : array-like
            Input data to be scaled.
        scaler : float or None, optional
            If provided, enables constant-scaling mode with
            ``x_scaled = x * scaler``.

        Returns
        -------
        scaled : Array
            Scaled data array with the same shape as the input.

        Notes
        -----
        - When ``scaler`` is provided, ``feature_range`` is ignored and the
          transformation reduces to multiplication by ``scaler``.

        """
        self.scaler = scaler
        data_max = jnp.max(data)
        data_min = jnp.min(data)

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

    def transform(self, data: ArrayLike) -> Array:
        """Apply the fitted scaling transformation to data.

        Parameters
        ----------
        data : array-like
            Input data to be scaled.

        Returns
        -------
        scaled : Array
            Scaled data array with the same shape as the input.

        Raises
        ------
        AttributeError
            If the scaler has not been fitted.

        """
        if not hasattr(self, "scaler"):
            self.scaler = None

        if self.scaler is not None:
            X = data * self.scaler
        else:
            X = data * self.scale_ + self.min_
        return X

    def inverse_transform(self, data: ArrayLike) -> Array:
        """Invert the scaling transformation.

        Parameters
        ----------
        data : array-like
            Scaled data to be transformed back to the original space.

        Returns
        -------
        unscaled : Array
            Data mapped back to the original scale.

        Notes
        -----
        - In constant-scaling mode, this divides by ``scaler``.
        - In range-based mode, this applies ``(x - min_) / scale_``.

        """
        if not hasattr(self, "scaler"):
            self.scaler = None

        if self.scaler is not None:
            return data / self.scaler
        return (data - self.min_) / self.scale_


class Transformer:
    """Per-feature affine scaler using mean and MAD.

    This transformer rescales each feature independently according to:

        x_scaled = (x - mean) / scale

    where:
        mean  = mean(x, axis=0)
        scale = max(|x - mean|, axis=0)

    This normalization is robust to outliers compared to standard deviation

    Notes
    -----
    - A small positive floor ``eps`` is applied to the scale to prevent division
      by zero for constant features.
    - Both 1D arrays (shape ``(N,)``) and 2D arrays (shape ``(N, D)``) are
      supported. One-dimensional inputs are treated as single-feature data and
      are returned with the same dimensionality.
    - This class does not track units; it assumes unitless numeric arrays

    Attributes
    ----------
    mean : Array or None
        Per-feature mean computed during :meth:`fit`. Shape ``(D,)``.
    scale : Array or None
        Per-feature scaling factor computed during :meth:`fit`. Shape ``(D,)``.

    Parameters
    ----------
    eps : float, optional
        Minimum allowed value for the scale to avoid division by zero.
        Defaults to ``1e-12``.

    """

    def __init__(self, eps: float = 1e-12) -> None:
        """Initialize an unfitted Transformer.

        Parameters
        ----------
        eps : float, optional
            Minimum allowed scale value used to regularize constant features.

        """
        self.mean: Array | None = None
        self.scale: Array | None = None
        self.eps = eps

    def fit(self, data: Array) -> "Transformer":
        """Compute per-feature mean and scale from data.

        Parameters
        ----------
        data : Array
            Input data of shape ``(N, D)`` or ``(N,)``.
            If one-dimensional, the data are treated as a single feature.

        Returns
        -------
        self : Transformer
            The fitted transformer instance.

        """
        data = jnp.asarray(data)
        if data.ndim == 1:
            data = data[:, None]

        mean = jnp.mean(data, axis=0)
        scale = jnp.max(jnp.abs(data - mean), axis=0)
        scale = jnp.maximum(scale, self.eps)

        self.mean = mean
        self.scale = scale
        return self

    def transform(self, data: Array) -> Array:
        """Apply the fitted scaling transformation to data.

        Parameters
        ----------
        data : Array
            Input data of shape ``(N, D)`` or ``(N,)``.

        Returns
        -------
        scaled : Array
            Scaled data with the same shape as the input.

        Raises
        ------
        RuntimeError
            If the transformer has not been fitted via :meth:`fit`.

        Notes
        -----
        - One-dimensional inputs are treated as single-feature data and returned
          as one-dimensional arrays.

        """
        if self.mean is None or self.scale is None:
            raise RuntimeError("Transformer must be fit before calling transform().")

        data = jnp.asarray(data)
        is_1d = data.ndim == 1
        if is_1d:
            data = data[:, None]

        out = (data - self.mean) / self.scale
        return out.squeeze() if is_1d else out

    def inverse_transform(self, data: Array) -> Array:
        """Invert the scaling transformation.

        Parameters
        ----------
        data : Array
            Scaled data of shape ``(N, D)`` or ``(N,)``.

        Returns
        -------
        unscaled : Array
            Data transformed back to the original feature space.

        Raises
        ------
        RuntimeError
            If the transformer has not been fitted via :meth:`fit`.

        """
        if self.mean is None or self.scale is None:
            raise RuntimeError(
                "Transformer must be fit before calling inverse_transform()."
            )

        data = jnp.asarray(data)
        is_1d = data.ndim == 1
        if is_1d:
            data = data[:, None]

        out = data * self.scale + self.mean
        return out.squeeze() if is_1d else out


def scale_data(
    data_dict: dict[str, Array],
    config: dict,
) -> tuple[dict[str, Array], dict[str, UniformScaler]]:
    """Non-dimensionalize input using scale information.

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
        lf_potential = config["ab_potential"]
        pos = cx.CartesianPos3D(
            x=u.Quantity(data_dict["x_train"][:, 0], "kpc"),
            y=u.Quantity(data_dict["x_train"][:, 1], "kpc"),
            z=u.Quantity(data_dict["x_train"][:, 2], "kpc"),
        )
        u_analytic = lf_potential.potential(pos, 0).ustrip("kpc2/Myr2")
        u_residual = data_dict["u_train"] - u_analytic
        u_star = float(jnp.max(jnp.abs(u_residual)))
    else:
        u_star = float(jnp.max(jnp.abs(data_dict["u_train"])))

    if u_star <= 0:
        raise ValueError("Computed u_star must be positive.")

    t_star = jnp.sqrt(r_s**2 / u_star)
    a_star = r_s / (t_star**2)
    x_star = r_s

    # Fit constant scaling factors (1/x_star etc.)
    x_transformer.fit(data_dict["x_train"], scaler=1.0 / x_star)
    a_transformer.fit(data_dict["a_train"], scaler=1.0 / a_star)
    u_transformer.fit(data_dict["u_train"], scaler=1.0 / u_star)

    x_train = x_transformer.transform(data_dict["x_train"])
    a_train = a_transformer.transform(data_dict["a_train"])
    u_train = u_transformer.transform(data_dict["u_train"])
    r_train = jnp.linalg.norm(x_train, axis=1)

    x_val = x_transformer.transform(data_dict["x_val"])
    a_val = a_transformer.transform(data_dict["a_val"])
    u_val = u_transformer.transform(data_dict["u_val"])
    r_val = jnp.linalg.norm(x_val, axis=1)

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


def scale_data_time(
    data_dict: dict[str, dict[float, dict[str, Array]]],
    config: dict,
) -> tuple[dict[str, dict[float, dict[str, Array]]], dict[str, UniformScaler]]:
    """Non-dimensionalize a nested time-dependent dataset.

    Attaches time as an input column.

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
        all_t.append(jnp.full(len(d["x"]), float(t)))

    x_concat = jnp.vstack(all_x)
    a_concat = jnp.vstack(all_a)
    u_concat = jnp.hstack(all_u)
    t_concat = jnp.concatenate(all_t)

    if config.get("include_analytic", False):
        analytic_baseline = config["ab_potential"]
        pos = cx.CartesianPos3D(
            x=u.Quantity(x_concat[:, 0], "kpc"),
            y=u.Quantity(x_concat[:, 1], "kpc"),
            z=u.Quantity(x_concat[:, 2], "kpc"),
        )
        t_quant = u.Quantity(t_concat, "Myr")
        u_analytic = analytic_baseline.potential(pos, t_quant).ustrip("kpc2/Myr2")
        u_resid = u_concat - u_analytic
        u_star = float(jnp.max(jnp.abs(u_resid)))
    else:
        u_star = float(jnp.max(jnp.abs(u_concat)))

    if u_star <= 0:
        raise ValueError("Computed u_star must be positive.")

    t_star = jnp.sqrt(r_s**2 / u_star)
    a_star = r_s / (t_star**2)
    x_star = r_s

    x_transformer.fit(x_concat, scaler=1.0 / x_star)
    a_transformer.fit(a_concat, scaler=1.0 / a_star)
    u_transformer.fit(u_concat, scaler=1.0 / u_star)
    t_transformer.fit(t_concat, scaler=1.0 / t_star)

    def _transform_block(
        x: Array, a: Array, u: Array, t_val: float
    ) -> dict[str, Array]:
        x_scaled = x_transformer.transform(x)
        # force (N,1) then concat
        t_scaled = t_transformer.transform(jnp.full((len(x_scaled), 1), float(t_val)))
        x_with_time = jnp.concatenate([t_scaled, x_scaled], axis=1)
        return {
            "x": x_with_time,
            "a": a_transformer.transform(a),
            "u": u_transformer.transform(u),
        }

    scaled_train = {
        t: _transform_block(d["x"], d["a"], d["u"], t) for t, d in train_data.items()
    }
    scaled_val = {
        t: _transform_block(d["x"], d["a"], d["u"], t) for t, d in val_data.items()
    }

    transformers = {
        "x": x_transformer,
        "a": a_transformer,
        "u": u_transformer,
        "t": t_transformer,
    }
    return {"train": scaled_train, "val": scaled_val}, transformers


# -------------------------
# Misc helpers
# -------------------------


def acc_cart_to_cyl_like(a: Array) -> Array:
    """Compute simple cylindrical-like diagnostics of an acceleration vector.

    Parameters
    ----------
    a
        Array of shape (N, 3) with Cartesian acceleration components.

    Returns
    -------
    out
        Array of shape (N, 4) with columns [a_rho, a_phi, a_z, a_mag].


    Examples
    --------
    A simple vector along +x:

    >>> import jax.numpy as jnp
    >>> acc_cart_to_cyl_like(jnp.array([[3.0, 0.0, 4.0]])).tolist()
    [[3.0, 0.0, 4.0, 5.0]]

    """
    a = jnp.asarray(a)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("Expected a to have shape (N, 3).")

    a_rho = jnp.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2)
    a_phi = jnp.arctan2(a[:, 1], a[:, 0])
    a_z = a[:, 2]
    a_mag = jnp.linalg.norm(a, axis=1)
    return jnp.stack([a_rho, a_phi, a_z, a_mag], axis=1)


def generate_xz_plane_grid(
    xmin: float,
    xmax: float,
    zmin: float,
    zmax: float,
    *,
    num_x: int = 50,
    num_z: int = 50,
    y: float = 0.0,
) -> Array:
    """Generate a Cartesian grid of points on the x-z plane at fixed y.

    Parameters
    ----------
    xmin, xmax
        Minimum and maximum x-coordinates.
    zmin, zmax
        Minimum and maximum z-coordinates.
    num_x
        Number of grid points along the x-direction.
    num_z
        Number of grid points along the z-direction.
    y
        Fixed y-coordinate for all points.

    Returns
    -------
    points : Array, shape (num_x * num_z, 3)
        Array of Cartesian coordinates `[x, y, z]` for each grid point.

    """
    x = jnp.linspace(xmin, xmax, num_x)
    z = jnp.linspace(zmin, zmax, num_z)
    xx, zz = jnp.meshgrid(x, z)
    yy = jnp.full_like(xx, y)
    return jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def generate_xy_plane_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    *,
    num_x: int = 50,
    num_y: int = 50,
    z: float = 0.0,
) -> Array:
    """Generate a Cartesian grid of points on the x-y plane at fixed z.

    Parameters
    ----------
    xmin, xmax
        Minimum and maximum x-coordinates.
    ymin, ymax
        Minimum and maximum y-coordinates.
    num_x
        Number of grid points along the x-direction.
    num_y
        Number of grid points along the y-direction.
    z
        Fixed z-coordinate for all points.

    Returns
    -------
    points : Array, shape (num_x * num_y, 3)
        Array of Cartesian coordinates `[x, y, z]` for each grid point.

    """
    x = jnp.linspace(xmin, xmax, num_x)
    y = jnp.linspace(ymin, ymax, num_y)
    xx, yy = jnp.meshgrid(x, y)
    zz = jnp.full_like(xx, z)
    return jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def generate_yz_plane_grid(
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    *,
    num_y: int = 50,
    num_z: int = 50,
    x: float = 0.0,
) -> Array:
    """Generate a Cartesian grid of points on the y-z plane at fixed x.

    Parameters
    ----------
    ymin, ymax
        Minimum and maximum y-coordinates.
    zmin, zmax
        Minimum and maximum z-coordinates.
    num_y
        Number of grid points along the y-direction.
    num_z
        Number of grid points along the z-direction.
    x
        Fixed x-coordinate for all points.

    Returns
    -------
    points : Array, shape (num_y * num_z, 3)
        Array of Cartesian coordinates `[x, y, z]` for each grid point.

    """
    y = jnp.linspace(ymin, ymax, num_y)
    z = jnp.linspace(zmin, zmax, num_z)
    yy, zz = jnp.meshgrid(y, z)
    xx = jnp.full_like(zz, x)
    return jnp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
