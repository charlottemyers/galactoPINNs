"""This module provides tools for orbit analysis and comparison in galactic dynamics."""
import galax.dynamics as gd
import galax.potential as gp
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import unxt as u
from galax.coordinates import PhaseSpaceCoordinate, PhaseSpacePosition
from jaxtyping import Array

__all__ = (
    "Euclidean_distance",
    "compare_orbits",
    "compare_orbits_analytic",
    "compose_velocity_bound_safe",
    "get_orbit_metrics",
    "get_orbit_metrics_analytic",
    "get_raw_orbit_coords",
    "get_w0s_from_data",
    "integrate_orbit_batch",
    "orbit_energy",
    "sample_outside_shell_bound",
    "scale_orbit_batch",
)


def get_raw_orbit_coords(orbit: gd.Orbit, c: str = "q") -> Array:
    """Extract raw coordinate arrays from a galax orbit object.

    Parameters
    ----------
    orbit
        A galax orbit object containing position and velocity data.
    c
        Which coordinates to extract. ``"q"`` for positions, ``"p"`` for
        velocities.

    Returns
    -------
    coords
        Stacked coordinate array of shape ``(T, 3)`` in units of ``kpc``
        (positions) or ``kpc/Myr`` (velocities).

    """
    if c == "q":
        qx = orbit.q.x.to_value("kpc")
        qy = orbit.q.y.to_value("kpc")
        qz = orbit.q.z.to_value("kpc")
        coords = jnp.stack([qx, qy, qz], axis=-1)
    if c == "p":
        px = orbit.p.x.to_value("kpc/Myr")
        py = orbit.p.y.to_value("kpc/Myr")
        pz = orbit.p.z.to_value("kpc/Myr")
        coords = jnp.stack([px, py, pz], axis=-1)
    return coords


def orbit_energy(pot: gp.AbstractPotential, orbit: gd.Orbit, ts: Array) -> Array:
    """Compute total orbital energy (kinetic + potential) along an orbit.

    Parameters
    ----------
    pot
        A galax potential object used to evaluate the potential energy.
    orbit
        A galax orbit object containing position and velocity data.
    ts
        Array of times at which the orbit was evaluated, in ``Myr``.

    Returns
    -------
    energy
        Total specific energy at each timestep, shape ``(T,)``, in units of
        ``kpc^2/Myr^2``.

    """
    unit_x = orbit.q.x.unit
    qx = orbit.q.x.to_value(unit_x)
    qy = orbit.q.y.to_value(unit_x)
    qz = orbit.q.z.to_value(unit_x)

    coords = jnp.stack([qx, qy, qz], axis=-1)
    if coords.ndim == 3:
        coords = coords.squeeze(0)
    x = u.Quantity(coords, unit_x)

    unit_v = orbit.p.x.unit
    vx = orbit.p.x.to_value(unit_v)
    vy = orbit.p.y.to_value(unit_v)
    vz = orbit.p.z.to_value(unit_v)

    vels = jnp.stack([vx, vy, vz], axis=-1)
    if vels.ndim == 3:
        vels = vels.squeeze(0)
    T = 0.5 * (vels**2).sum(axis=-1)
    Phi = u.ustrip("kpc2/Myr2", pot.potential(x, t=ts))
    return T + Phi


def Euclidean_distance(orbit_q1: Array, orbit_q2: Array) -> Array:
    """Compute the Euclidean distance between two orbits at each timestep.

    Parameters
    ----------
    orbit_q1
        Position array for the first orbit, shape ``(T, 3)``.
    orbit_q2
        Position array for the second orbit, shape ``(T, 3)``.

    Returns
    -------
    distance
        Scalar distance at each timestep, shape ``(T,)``, in ``kpc``.

    """
    return jnp.sqrt(jnp.sum((orbit_q1 - orbit_q2) ** 2, axis=-1)).squeeze()


def compare_orbits_analytic(
    true_potential: gp.AbstractPotential,
    analytic_potential: gp.AbstractPotential,
    w0: PhaseSpacePosition,
    ts: u.Quantity,
    true_orbit: gd.Orbit | None = None,
) -> dict:
    """Compare an orbit integrated under an analytic potential to a true orbit.

    Parameters
    ----------
    true_potential
        The ground-truth galax potential used to integrate the reference orbit.
    analytic_potential
        The analytic approximation potential to evaluate against the truth.
    w0
        Initial phase-space position for orbit integration.
    ts
        Array of integration times as a ``unxt.Quantity`` in ``Myr``.
    true_orbit
        Pre-computed true orbit. If ``None``, it is integrated from ``w0``
        using ``true_potential``.

    Returns
    -------
    outputs
        Dictionary containing:

        - ``"learned_orbit"``: orbit integrated under ``analytic_potential``.
        - ``"true_orbit"``: reference orbit.
        - ``"learned_energy"``: energy along the analytic orbit.
        - ``"true_energy"``: energy along the true orbit.
        - ``"true_energy_on_learned"``: true potential energy evaluated on the analytic orbit.
        - ``"distance_true_learned"``: per-timestep separation in ``kpc``.
        - ``"time_avg_error"``: time-averaged orbit separation in ``kpc``.

    """
    orbit = gd.evaluate_orbit(analytic_potential, w0, ts)
    if true_orbit is None:
        true_orbit = gd.evaluate_orbit(true_potential, w0, ts)

    true_coords = get_raw_orbit_coords(true_orbit)
    learned_coords = get_raw_orbit_coords(orbit)

    distance_true_learned = Euclidean_distance(learned_coords, true_coords)
    ts = ts.to_value("Myr")

    time_avg_error = jnp.trapezoid(distance_true_learned, x=ts) / (ts[-1] - ts[0])

    return {
        "learned_orbit": orbit,
        "true_orbit": true_orbit,
        "learned_energy": orbit_energy(analytic_potential, orbit, ts),
        "true_energy": orbit_energy(true_potential, true_orbit, ts),
        "true_energy_on_learned": orbit_energy(true_potential, orbit, ts),
        "distance_true_learned": distance_true_learned,
        "time_avg_error": time_avg_error,
    }


def compare_orbits(
    true_potential: gp.AbstractPotential,
    learned_galax_pot: gp.AbstractPotential,
    w0: PhaseSpacePosition,
    ts: u.Quantity,
    true_orbit: gd.Orbit | None = None,
) -> dict:
    """Compare an orbit integrated under a learned potential to a true orbit.

    Parameters
    ----------
    true_potential
        The ground-truth galax potential used to integrate the reference orbit.
    learned_galax_pot
        The learned potential to evaluate against the truth.
    w0
        Initial phase-space position for orbit integration.
    ts
        Array of integration times as a ``unxt.Quantity`` in ``Myr``.
    true_orbit
        Pre-computed true orbit. If ``None``, it is integrated from ``w0``
        using ``true_potential``.

    Returns
    -------
    outputs
        Dictionary containing:

        - ``"learned_orbit"``: orbit integrated under ``learned_galax_pot``.
        - ``"true_orbit"``: reference orbit.
        - ``"learned_energy"``: energy along the learned orbit.
        - ``"true_energy"``: energy along the true orbit.
        - ``"true_energy_on_learned"``: true potential energy evaluated on the learned orbit.
        - ``"distance_true_learned"``: per-timestep separation in ``kpc``.
        - ``"time_avg_error"``: time-integrated average orbit separation in ``kpc``.
        - ``"mod"``: mean orbit separation in ``kpc``.

    """
    orbit = gd.evaluate_orbit(learned_galax_pot, w0, ts)
    if true_orbit is None:
        true_orbit = gd.evaluate_orbit(true_potential, w0, ts)

    true_coords = get_raw_orbit_coords(true_orbit)
    learned_coords = get_raw_orbit_coords(orbit)

    distance_true_learned = Euclidean_distance(learned_coords, true_coords)
    ts = ts.to_value("Myr")

    time_avg_error = jnp.trapezoid(distance_true_learned, x=ts) / (ts[-1] - ts[0])
    mod = jnp.mean(distance_true_learned)

    return {
        "learned_orbit": orbit,
        "true_orbit": true_orbit,
        "learned_energy": orbit_energy(learned_galax_pot, orbit, ts),
        "true_energy": orbit_energy(true_potential, true_orbit, ts),
        "true_energy_on_learned": orbit_energy(true_potential, orbit, ts),
        "distance_true_learned": distance_true_learned,
        "time_avg_error": time_avg_error,
        "mod": mod,
    }


def get_orbit_metrics_analytic(
    true_potential: gp.AbstractPotential,
    analytic_potential: gp.AbstractPotential,
    ts: u.Quantity,
    w0s: list[PhaseSpacePosition],
    true_orbits: list[gd.Orbit] | None = None,
) -> list[dict]:
    """Compute orbit comparison metrics for a batch of initial conditions using an analytic potential.

    Parameters
    ----------
    true_potential
        The ground-truth galax potential.
    analytic_potential
        The analytic approximation potential to evaluate.
    ts
        Integration times as a ``unxt.Quantity`` in ``Myr``.
    w0s
        List of initial phase-space positions.
    true_orbits
        Optional list of pre-computed true orbits, one per entry in ``w0s``.
        If ``None``, each orbit is integrated on the fly.

    Returns
    -------
    outputs
        List of metric dictionaries, one per initial condition. Each dict
        matches the output format of :func:`compare_orbits_analytic`.

    """
    outputs = []
    for i, w0 in enumerate(w0s):
        true_orbit = true_orbits[i] if true_orbits is not None else None
        orbit_outputs = compare_orbits_analytic(
            true_potential, analytic_potential, w0, ts, true_orbit=true_orbit
        )
        outputs.append(orbit_outputs)
    return outputs


def get_w0s_from_data(
    raw_datadict: dict,
    true_potential: gp.AbstractPotential,
    n_draws: int = 10,
) -> list[PhaseSpacePosition]:
    """Sample circular orbit initial conditions from training data positions.

    Draws random positions from the dataset and assigns circular velocities
    in the plane perpendicular to the radial direction.

    Parameters
    ----------
    raw_datadict
        Dictionary containing at least a ``"x_train"`` key with positions
        of shape ``(N, 3)`` in ``kpc``.
    true_potential
        Galax potential used to compute the local circular velocity at each
        sampled position.
    n_draws
        Number of initial conditions to sample.

    Returns
    -------
    w0s
        List of ``PhaseSpacePosition`` objects with positions in ``kpc``
        and circular velocities in ``kpc/Myr``.

    """
    w0s = []
    x_test = raw_datadict["x_train"]
    rng = np.random.default_rng()
    for i in range(n_draws):
        idx = rng.integers(0, x_test.shape[0])
        x_random = x_test[idx]
        w_dummy = PhaseSpaceCoordinate(
            q=u.Quantity([x_random], "kpc"),
            p=u.Quantity([0, 0, 0], "kpc/Myr"),
            t=u.Quantity([0], "Myr"),
        )
        vc = true_potential.local_circular_velocity(w_dummy)
        r = np.linalg.norm(x_random)
        v_vec = vc * np.array([-x_random[1], x_random[0], 0.0]) / r
        w0 = PhaseSpacePosition(q=u.Quantity(x_random, "kpc"), p=v_vec)
        w0s.append(w0)
    return w0s


def get_orbit_metrics(
    true_potential: gp.AbstractPotential,
    learned_galax_pot: gp.AbstractPotential,
    ts: u.Quantity,
    w0s: list[PhaseSpacePosition],
    true_orbits: list[gd.Orbit] | None = None,
) -> list[dict]:
    """Compute orbit comparison metrics for a batch of initial conditions using a learned potential.

    Parameters
    ----------
    true_potential
        The ground-truth galax potential.
    learned_galax_pot
        The learned potential to evaluate against the truth.
    ts
        Integration times as a ``unxt.Quantity`` in ``Myr``.
    w0s
        List of initial phase-space positions.
    true_orbits
        Optional list of pre-computed true orbits, one per entry in ``w0s``.
        If ``None``, each orbit is integrated on the fly.

    Returns
    -------
    outputs
        List of metric dictionaries, one per initial condition. Each dict
        matches the output format of :func:`compare_orbits`.

    """
    outputs = []
    for i, w0 in enumerate(w0s):
        true_orbit = true_orbits[i] if true_orbits is not None else None
        orbit_outputs = compare_orbits(
            true_potential, learned_galax_pot, w0, ts, true_orbit=true_orbit
        )
        outputs.append(orbit_outputs)
    return outputs


def _rand_unit_vectors(N: int, key: Array, eps: float = 1e-12) -> Array:
    """Sample N random unit vectors uniformly on the 2-sphere.

    Parameters
    ----------
    N
        Number of unit vectors to generate.
    key
        JAX PRNG key.
    eps
        Small value added to norms before division to avoid division by zero.

    Returns
    -------
    unit_vectors
        Array of shape ``(N, 3)`` with unit-norm rows.

    """
    v = jr.normal(key, shape=(N, 3))
    return v / (jnp.linalg.norm(v, axis=1, keepdims=True) + eps)


def _tangent_frame(e_r: Array) -> tuple[Array, Array]:
    """Construct two orthonormal tangent vectors perpendicular to a radial direction.

    Parameters
    ----------
    e_r
        Radial unit vectors, shape ``(N, 3)``.

    Returns
    -------
    e_t1, e_t2
        Two orthonormal tangent vector arrays each of shape ``(N, 3)``,
        forming a right-handed frame with ``e_r``.

    """
    z = jnp.array([0.0, 0.0, 1.0])
    x = jnp.array([1.0, 0.0, 0.0])
    ref = jnp.where((e_r @ z)[:, None] < 0.9, z, x)
    e_t1 = jnp.cross(ref, e_r)
    e_t1 /= jnp.linalg.norm(e_t1, axis=1, keepdims=True) + 1e-12
    e_t2 = jnp.cross(e_r, e_t1)
    return e_t1, e_t2


def compose_velocity_bound_safe(
    pot_for_vcirc: gp.AbstractPotential,
    pot_for_escape: gp.AbstractPotential,
    q_phys: Array,
    tangential_scatter: float = 0.10,
    radial_frac_sigma: float = 0.20,
    cap_vs_vcirc: float = 1.2,
    seed: int = 0,
) -> Array:
    """Compose velocities near circular speed, capped below the local escape velocity.

    Generates random velocities with a dominant tangential component close to
    the circular speed plus a small radial perturbation.

    Parameters
    ----------
    pot_for_vcirc
        Galax potential used to compute the local circular velocity.
    pot_for_escape
        Galax potential used to compute the local escape velocity.
    q_phys
        Cartesian positions in physical units, shape ``(N, 3)`` in ``kpc``.
    tangential_scatter
        Fractional scatter applied to the tangential component as a normal
        deviate around 1. Default ``0.10``.
    radial_frac_sigma
        Standard deviation of the radial velocity component as a fraction of
        the circular speed. Default ``0.20``.
    cap_vs_vcirc
        Hard cap on speed as a multiple of the circular velocity before escape
        velocity capping. Default ``1.2``.
    seed
        Integer seed for the JAX PRNG. Default ``0``.

    Returns
    -------
    velocities
        Velocity array of shape ``(N, 3)`` in ``kpc/Myr``.

    """
    key = jr.PRNGKey(seed)
    N = q_phys.shape[0]

    e_r = q_phys / (jnp.linalg.norm(q_phys, axis=1, keepdims=True) + 1e-12)
    e_t1, e_t2 = _tangent_frame(e_r)

    key, subkey = jr.split(key)
    ang = jr.uniform(subkey, shape=(N, 1), minval=0.0, maxval=2.0 * jnp.pi)
    e_t = jnp.cos(ang) * e_t1 + jnp.sin(ang) * e_t2

    qQ = u.Quantity(q_phys, "kpc")
    v_circ = gp.local_circular_velocity(
        pot_for_vcirc, qQ, t=u.Quantity(0.0, "Gyr")
    ).to_value("kpc/Myr").reshape(N, 1)

    key, k_ft = jr.split(key)
    ft = 1.0 + tangential_scatter * jr.normal(k_ft, shape=(N, 1))

    key, k_fr = jr.split(key)
    fr = radial_frac_sigma * jr.normal(k_fr, shape=(N, 1))

    v = (ft * v_circ) * e_t + (fr * v_circ) * e_r

    speed = jnp.linalg.norm(v, axis=1, keepdims=True)
    v = v * jnp.minimum(1.0, cap_vs_vcirc * v_circ / (speed + 1e-12))

    phi = pot_for_escape.potential(
        qQ, t=u.Quantity(0.0, "Gyr")
    ).to_value("kpc2/Myr2").reshape(N, 1)

    v_esc = jnp.sqrt(jnp.maximum(0.0, -2.0 * phi))
    scale = jnp.minimum(1.0, 0.98 * v_esc / (jnp.linalg.norm(v, axis=1, keepdims=True) + 1e-12))
    v = v * scale

    return jnp.asarray(v)


def sample_outside_shell_bound(
    true_pot: gp.AbstractPotential,
    r_min: float = 200.0,
    r_max: float = 600.0,
    n_orbits: int = 24,
    log_r: bool = True,
    seed: int = 456,
    **vel_kwargs,
) -> tuple[Array, Array]:
    """Sample bound initial conditions from a spherical shell.

    Draws positions uniformly (or log-uniformly) in radius between ``r_min``
    and ``r_max`` with random directions, then assigns velocities via
    :func:`compose_velocity_bound_safe`.

    Parameters
    ----------
    true_pot
        Galax potential used for circular and escape velocity computation.
    r_min
        Minimum galactocentric radius in ``kpc``. Default ``200.0``.
    r_max
        Maximum galactocentric radius in ``kpc``. Default ``600.0``.
    n_orbits
        Number of initial conditions to sample. Default ``24``.
    log_r
        If ``True``, sample radii log-uniformly between ``r_min`` and
        ``r_max``. Otherwise sample uniformly. Default ``True``.
    seed
        Integer seed for the JAX PRNG. Default ``456``.
    **vel_kwargs
        Additional keyword arguments forwarded to
        :func:`compose_velocity_bound_safe`.

    Returns
    -------
    q0
        Initial positions, shape ``(N, 3)`` in ``kpc``.
    v0
        Initial velocities, shape ``(N, 3)`` in ``kpc/Myr``.

    """
    key = jr.PRNGKey(seed)

    key, k_r = jr.split(key)
    if log_r:
        u_r = jr.uniform(k_r, shape=(n_orbits, 1), minval=0.0, maxval=1.0)
        r = jnp.exp(jnp.log(r_min) + u_r * (jnp.log(r_max) - jnp.log(r_min)))
    else:
        r = jr.uniform(k_r, shape=(n_orbits, 1), minval=r_min, maxval=r_max)

    key, k_dir = jr.split(key)
    e_r = _rand_unit_vectors(n_orbits, k_dir)
    q0 = e_r * r

    v0 = compose_velocity_bound_safe(
        pot_for_vcirc=true_pot,
        pot_for_escape=true_pot,
        q_phys=q0,
        **vel_kwargs,
    )

    return jnp.asarray(q0), jnp.asarray(v0)


def scale_orbit_batch(
    config: dict,
    q_seq_phys: Array,
    p_seq_phys: Array,
) -> tuple[Array, Array]:
    """Transform a batch of orbit sequences from physical to scaled coordinates.

    Parameters
    ----------
    config
        Configuration dictionary containing ``"x_transformer"`` and
        ``"v_transformer"`` objects with a ``.transform()`` method.
    q_seq_phys
        Positions in physical units, shape ``(B, T, 3)`` in ``kpc``.
    p_seq_phys
        Velocities in physical units, shape ``(B, T, 3)`` in ``kpc/Myr``.

    Returns
    -------
    q_scaled
        Scaled positions, shape ``(B, T, 3)``.
    p_scaled
        Scaled velocities, shape ``(B, T, 3)``.

    """
    B, T, _ = q_seq_phys.shape
    x_tr = config["x_transformer"]
    v_tr = config["v_transformer"]
    q_scaled = x_tr.transform(q_seq_phys.reshape(-1, 3)).reshape(B, T, 3)
    p_scaled = v_tr.transform(p_seq_phys.reshape(-1, 3)).reshape(B, T, 3)
    return q_scaled, p_scaled


def integrate_orbit_batch(
    true_pot: gp.AbstractPotential,
    q0_phys: Array,
    v0_phys: Array,
    ts_myr: Array,
) -> tuple[Array, Array]:
    """Integrate a batch of orbits under a given potential.

    Parameters
    ----------
    true_pot
        Galax potential used for orbit integration.
    q0_phys
        Initial positions, shape ``(B, 3)`` in ``kpc``.
    v0_phys
        Initial velocities, shape ``(B, 3)`` in ``kpc/Myr``.
    ts_myr
        Integration times in ``Myr``, shape ``(T,)``.

    Returns
    -------
    q_seq_phys
        Integrated positions, shape ``(B, T, 3)`` in ``kpc``.
    p_seq_phys
        Integrated velocities, shape ``(B, T, 3)`` in ``kpc/Myr``.

    """
    B = q0_phys.shape[0]
    ts_Q = u.Quantity(jnp.asarray(ts_myr), "Myr")
    q_list, p_list = [], []
    for b in range(B):
        w0 = PhaseSpacePosition(
            q=u.Quantity(q0_phys[b][None, :], "kpc"),
            p=u.Quantity(v0_phys[b][None, :], "kpc/Myr"),
        )
        orb = gd.evaluate_orbit(true_pot, w0, ts_Q)
        q_list.append(get_raw_orbit_coords(orb, c="q")[0])
        p_list.append(get_raw_orbit_coords(orb, c="p")[0])
    q_seq_phys = jnp.array(jnp.stack(q_list, axis=0))
    p_seq_phys = jnp.array(jnp.stack(p_list, axis=0))
    return q_seq_phys, p_seq_phys
