"""Coordinate transformation layers."""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


class CartesianLayer(nnx.Module):
    """Identity layer for Cartesian coordinates (pass-through)."""

    def __call__(self, x_cart: Array, /) -> Array:
        """Return the input Cartesian coordinates unchanged."""
        return x_cart


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

    clip: float = 1.0

    def __init__(self, clip: float = 1.0) -> None:
        """Initialize the coordinate transformation layer."""
        self.clip = clip

    def __call__(self, x_cart: Array, /) -> Array:
        """Transform Cartesian coordinates to modified spherical representation.

        Parameters
        ----------
        x_cart
            Cartesian coordinates of shape ``(3,)`` or ``(N, 3)``.

        Returns
        -------
        Array
            Modified spherical representation ``[r_i, r_e, s, t, u]`` with
            shape ``(5,)`` or ``(N, 5)``.

        """
        is_1d = x_cart.ndim == 1
        x = jnp.atleast_2d(x_cart)  # (N, 3)

        # Compute radius with keepdims=True to avoid reshape operations
        r = jnp.linalg.norm(x, axis=-1, keepdims=True)  # (N, 1)
        r_safe = jnp.maximum(r, jnp.finfo(x.dtype).tiny)  # (N, 1)

        # Clip radii
        r_i = jnp.clip(r, 0.0, self.clip)  # (N, 1)
        r_e = jnp.clip(1.0 / r_safe, 0.0, self.clip)  # (N, 1)

        # Unit vector (no explicit reshape needed due to keepdims)
        stu = x / r_safe  # (N, 3)

        # Stack along feature axis
        Y = jnp.concatenate([r_i, r_e, stu], axis=-1)  # (N, 5)

        return Y[0] if is_1d else Y
