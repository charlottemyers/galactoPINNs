"""Multi-layer perceptron implementation."""

import functools as ft

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

from ._base import ActivationFn


class MLP(nnx.Module):
    """Multi-layer perceptron with a smooth activation for physics-informed models.

    This module wraps ``nnx.Sequential`` to provide a convenient constructor for
    building MLPs with smooth activation functions, which are essential for
    physics-informed neural networks (PINNs). Since NNX does not package a
    dedicated MLP class, this wrapper handles the layer construction pattern
    needed for PINN applications.

    Smooth activations (e.g., GELU, tanh) are critical for PINNs because:
    - They enable accurate computation of derivatives through the network, which
      is required for physics-based loss terms involving differential equations.
    - Piecewise activations (e.g., ReLU) have discontinuous or undefined gradients
      at specific points, introducing numerical instabilities in derivative
      calculations.
    - The default smooth activation (GELU with exact mode) ensures well-behaved
      gradients throughout the network for better physical accuracy.

    This module builds a stack of Linear layers with an activation applied after
    each hidden layer, followed by a final Linear layer that outputs a scalar
    per example.

    Parameters
    ----------
    in_features
        Input feature dimension.
    width
        Hidden layer width (number of units) for each Linear layer.
    depth
        Number of hidden layers.
    act
        Activation function applied after each hidden layer. Must be a callable
        that maps an array to an array of the same shape. Defaults to
        ``jax.nn.gelu`` (exact mode) for optimal derivative flow in physics
        computations.
    rngs
        Random number generator state for parameter initialization.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from flax import nnx
    >>> mlp = MLP(in_features=3, width=16, depth=2, act=jax.nn.tanh, rngs=nnx.Rngs(0))
    >>> x = jnp.ones((4, 3))
    >>> y = mlp(x)
    >>> y.shape
    (4,)

    """

    network: nnx.Sequential

    def __init__(
        self,
        in_features: int,
        width: int = 128,
        depth: int = 3,
        act: ActivationFn = ft.partial(jax.nn.gelu, approximate=False),  # noqa: B008
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the MLP layers."""
        layers = [
            layer
            for i in range(depth)
            for layer in (
                nnx.Linear(in_features if i == 0 else width, width, rngs=rngs),
                # Repeat width -> width blocks for the remaining hidden layers.
                act,
            )
        ]
        output_in_features = width if depth > 0 else in_features
        layers.append(nnx.Linear(output_in_features, 1, rngs=rngs))
        self.network = nnx.Sequential(*layers)

    def __call__(self, x: Array, /) -> Array:
        # Degenerate case: no hidden layers, just a linear readout.
        """Forward pass.

        Parameters
        ----------
        x
            Input array, typically shape ``(N, D)``.

        Returns
        -------
        y
            Scalar output per example. Shape is typically ``(N,)``.

        """
        return jnp.squeeze(self.network(x), axis=-1)
