"""Tests for NNX layer implementations."""

import jax
import jax.numpy as jnp
from flax import nnx

from galactoPINNs.layers import (
    CartesianToModifiedSphericalLayer,
    FuseModelsLayer,
    ScaleNNPotentialLayer,
    SmoothMLP,
)


class TestSmoothMLP:
    """Tests for SmoothMLP layer."""

    def test_init_creates_correct_layers(self):
        """Test that SmoothMLP initializes with the correct number of layers."""
        mlp = SmoothMLP(in_features=3, width=16, depth=2, rngs=nnx.Rngs(0))

        assert len(mlp.hidden_layers) == 2
        assert mlp.output_layer is not None
        assert mlp.width == 16
        assert mlp.depth == 2

    def test_forward_single_sample(self):
        """Test forward pass with a single sample."""
        mlp = SmoothMLP(in_features=3, width=16, depth=2, rngs=nnx.Rngs(0))
        x = jnp.ones((1, 3))
        y = mlp(x)

        assert y.shape == (1,)
        assert jnp.isfinite(y).all()

    def test_forward_batch(self):
        """Test forward pass with a batch of samples."""
        mlp = SmoothMLP(in_features=3, width=32, depth=3, rngs=nnx.Rngs(42))
        x = jnp.ones((10, 3))
        y = mlp(x)

        assert y.shape == (10,)
        assert jnp.isfinite(y).all()

    def test_different_activations(self):
        """Test that different activation functions work."""
        for act in [jax.nn.tanh, jax.nn.relu, jax.nn.silu]:
            mlp = SmoothMLP(in_features=3, width=8, depth=1, act=act, rngs=nnx.Rngs(0))
            x = jnp.ones((5, 3))
            y = mlp(x)

            assert y.shape == (5,)
            assert jnp.isfinite(y).all()

    def test_jit_compatible(self):
        """Test that SmoothMLP works with JAX JIT compilation."""
        mlp = SmoothMLP(in_features=3, width=16, depth=2, rngs=nnx.Rngs(0))

        @jax.jit
        def forward(model, x):
            return model(x)

        x = jnp.ones((5, 3))
        y = forward(mlp, x)

        assert y.shape == (5,)
        assert jnp.isfinite(y).all()

    def test_vmap_compatible(self):
        """Test that SmoothMLP works with JAX vmap."""
        mlp = SmoothMLP(in_features=3, width=16, depth=2, rngs=nnx.Rngs(0))

        # vmap over a batch dimension
        x_batch = jnp.ones((4, 1, 3))  # (batch, 1, features)

        @jax.vmap
        def batched_forward(x):
            return mlp(x)

        y = batched_forward(x_batch)
        assert y.shape == (4, 1)

    def test_grad_compatible(self):
        """Test that gradients can be computed through SmoothMLP."""
        mlp = SmoothMLP(in_features=3, width=16, depth=2, rngs=nnx.Rngs(0))

        def loss_fn(model):
            x = jnp.ones((5, 3))
            return jnp.sum(model(x) ** 2)

        grads = nnx.grad(loss_fn)(mlp)
        # Check that we get gradients (grads is the model with gradient values)
        assert grads is not None


class TestCartesianToModifiedSphericalLayer:
    """Tests for CartesianToModifiedSphericalLayer."""

    def test_init(self):
        """Test layer initialization."""
        layer = CartesianToModifiedSphericalLayer(clip=1.0)
        assert layer.clip == 1.0

    def test_single_point(self):
        """Test transformation of a single 3D point."""
        layer = CartesianToModifiedSphericalLayer(clip=10.0)
        x = jnp.array([1.0, 0.0, 0.0])
        y = layer(x)

        # Output should be [r_i, r_e, s, t, u] = [1, 1, 1, 0, 0]
        assert y.shape == (5,)
        assert jnp.isclose(y[0], 1.0)  # r_i = r = 1
        assert jnp.isclose(y[1], 1.0)  # r_e = 1/r = 1
        assert jnp.isclose(y[2], 1.0)  # s = x/r = 1
        assert jnp.isclose(y[3], 0.0)  # t = y/r = 0
        assert jnp.isclose(y[4], 0.0)  # u = z/r = 0

    def test_batch_points(self):
        """Test transformation of a batch of points."""
        layer = CartesianToModifiedSphericalLayer(clip=10.0)
        x = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        y = layer(x)

        assert y.shape == (3, 5)
        # First point: r=1, unit vector = (1, 0, 0)
        assert jnp.isclose(y[0, 0], 1.0)  # r_i = 1
        # Second point: r=2, unit vector = (0, 1, 0)
        assert jnp.isclose(y[1, 0], 2.0)  # r_i = 2
        # Third point: r=3, unit vector = (0, 0, 1)
        assert jnp.isclose(y[2, 0], 3.0)  # r_i = 3

    def test_clipping(self):
        """Test that clipping is applied correctly."""
        layer = CartesianToModifiedSphericalLayer(clip=2.0)
        # Point with r=5, should be clipped to r_i=2
        x = jnp.array([5.0, 0.0, 0.0])
        y = layer(x)

        assert jnp.isclose(y[0], 2.0)  # r_i clipped to 2

    def test_jit_compatible(self):
        """Test JIT compatibility."""
        layer = CartesianToModifiedSphericalLayer(clip=1.0)

        @jax.jit
        def transform(x):
            return layer(x)

        x = jnp.array([[1.0, 2.0, 3.0]])
        y = transform(x)
        assert y.shape == (1, 5)


class TestScaleNNPotentialLayer:
    """Tests for ScaleNNPotentialLayer."""

    def test_scale_one(self):
        """Test 'one' scaling mode (no scaling)."""
        config = {
            "scale": "one",
            "x_transformer": MockTransformer(),
            "r_s": 1.0,
        }
        layer = ScaleNNPotentialLayer(config=config)

        x_cart = jnp.array([[1.0, 0.0, 0.0]])
        u_nn = jnp.array([1.0])
        result = layer(x_cart, u_nn)

        # With scale="one", output should equal input
        assert jnp.isclose(result, u_nn).all()

    def test_scale_power(self):
        """Test 'power' scaling mode."""
        config = {
            "scale": "power",
            "power": 2.0,
            "x_transformer": MockTransformer(),
            "r_s": 1.0,
        }
        layer = ScaleNNPotentialLayer(config=config)

        # Point at r=2
        x_cart = jnp.array([[2.0, 0.0, 0.0]])
        u_nn = jnp.array([1.0])
        result = layer(x_cart, u_nn)

        # With power=2: prefactor = (1/r)^2 = (1/2)^2 = 0.25
        expected = u_nn * 0.25
        assert jnp.isclose(result, expected, atol=1e-5).all()

    def test_jit_compatible(self):
        """Test JIT compatibility."""
        config = {
            "scale": "one",
            "x_transformer": MockTransformer(),
            "r_s": 1.0,
        }
        layer = ScaleNNPotentialLayer(config=config)

        @nnx.jit
        def apply_scale(layer, x, u):
            return layer(x, u)

        x_cart = jnp.array([[1.0, 2.0, 3.0]])
        u_nn = jnp.array([0.5])
        result = apply_scale(layer, x_cart, u_nn)
        assert jnp.isfinite(result).all()


class TestFuseModelsLayer:
    """Tests for FuseModelsLayer."""

    def test_init(self):
        """Test layer initialization."""
        layer = FuseModelsLayer()
        assert layer is not None

    def test_fusion(self):
        """Test that fusion adds two potential components."""
        layer = FuseModelsLayer()

        nn_potential = jnp.array([1.0, 2.0, 3.0])
        analytic_potential = jnp.array([0.5, 0.5, 0.5])

        result = layer(nn_potential, analytic_potential)

        expected = nn_potential + analytic_potential
        assert jnp.allclose(result, expected)

    def test_jit_compatible(self):
        """Test JIT compatibility."""
        layer = FuseModelsLayer()

        @jax.jit
        def fuse(u_nn, u_analytic):
            return layer(u_nn, u_analytic)

        u_nn = jnp.array([1.0, 2.0])
        u_analytic = jnp.array([0.1, 0.2])
        result = fuse(u_nn, u_analytic)

        assert jnp.allclose(result, u_nn + u_analytic)


# Helper classes for testing


class MockTransformer:
    """Mock transformer for testing."""

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


class MockAnalyticPotential:
    """Mock analytic potential for testing."""

    def potential(self, x, *, t=0):
        # Simple radial potential
        r = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return -1.0 / (r + 0.1)

    def acceleration(self, x, *, t=0):
        # Negative gradient of potential (simplified)
        return -x / (jnp.linalg.norm(x, axis=-1, keepdims=True) ** 3 + 0.01)
