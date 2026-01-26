"""Tests for NNX model implementations."""

import jax.numpy as jnp
from flax import nnx

from galactoPINNs.models.static_model import StaticModel


class MockTransformer:
    """Mock transformer for testing."""

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


class MockAnalyticPotential:
    """Mock analytic potential for testing."""

    def potential(self, x, *, t=0):
        # Simple radial potential: -1/r
        r = jnp.linalg.norm(x, axis=-1)
        return -1.0 / (r + 0.1)

    def acceleration(self, x, *, t=0):
        # Simplified acceleration
        r = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return -x / (r**3 + 0.01)


def make_minimal_config(*, include_analytic: bool = False) -> dict:
    """Create a minimal configuration for testing."""
    return {
        "x_transformer": MockTransformer(),
        "u_transformer": MockTransformer(),
        "a_transformer": MockTransformer(),
        "r_s": 1.0,
        "clip": 1.0,
        "scale": "one",
        "include_analytic": include_analytic,
        "ab_potential": MockAnalyticPotential(),
        "convert_to_spherical": True,
        "depth": 2,
        "width": 16,
        "nn_off": False,
    }


class TestStaticModel:
    """Tests for StaticModel."""

    def test_init(self):
        """Test that StaticModel initializes correctly."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        assert model.config is config
        assert model.cart_to_sph_layer is not None
        assert model.scale_layer is not None
        assert model.fuse_layer is not None
        assert model.mlp is not None

    def test_init_nn_off(self):
        """Test initialization with NN disabled."""
        config = make_minimal_config()
        config["nn_off"] = True
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        assert model.mlp is None
        assert model.nn_off is True

    def test_forward_potential_mode(self):
        """Test forward pass in potential-only mode."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        x = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        outputs = model(x, mode="potential")

        assert "potential" in outputs
        assert outputs["potential"].shape == (2,)
        assert jnp.isfinite(outputs["potential"]).all()

    def test_forward_full_mode(self):
        """Test forward pass in full mode (potential + acceleration)."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        x = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        outputs = model(x, mode="full")

        assert "potential" in outputs
        assert "acceleration" in outputs
        assert outputs["potential"].shape == (2,)
        assert outputs["acceleration"].shape == (2, 3)
        assert jnp.isfinite(outputs["potential"]).all()
        assert jnp.isfinite(outputs["acceleration"]).all()

    def test_compute_potential(self):
        """Test compute_potential method."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        x = jnp.array([[1.0, 2.0, 3.0]])
        potential = model.compute_potential(x)

        assert potential.shape == (1,)
        assert jnp.isfinite(potential).all()

    def test_compute_acceleration(self):
        """Test compute_acceleration method."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        acceleration = model.compute_acceleration(x)

        assert acceleration.shape == (2, 3)
        assert jnp.isfinite(acceleration).all()

    def test_compute_laplacian(self):
        """Test compute_laplacian method."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        x = jnp.array([[1.0, 2.0, 3.0]])
        laplacian = model.compute_laplacian(x)

        assert laplacian.shape == (1,)
        assert jnp.isfinite(laplacian).all()

    def test_jit_compatible(self):
        """Test that StaticModel works with JIT compilation."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        @nnx.jit
        def forward(model, x):
            return model(x, mode="potential")

        x = jnp.array([[1.0, 0.0, 0.0]])
        outputs = forward(model, x)

        assert "potential" in outputs
        assert jnp.isfinite(outputs["potential"]).all()

    def test_grad_compatible(self):
        """Test that gradients can be computed through the model."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        def loss_fn(model):
            x = jnp.array([[1.0, 2.0, 3.0]])
            outputs = model(x, mode="potential")
            return jnp.sum(outputs["potential"] ** 2)

        grads = nnx.grad(loss_fn)(model)
        assert grads is not None

    def test_with_analytic(self):
        """Test model with analytic potential included."""
        config = make_minimal_config(include_analytic=True)
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        x = jnp.array([[1.0, 0.0, 0.0]])
        outputs = model(x, mode="full")

        assert "potential" in outputs
        assert "acceleration" in outputs
        assert jnp.isfinite(outputs["potential"]).all()
        assert jnp.isfinite(outputs["acceleration"]).all()

    def test_reproducible_with_same_seed(self):
        """Test that same seed produces same model outputs."""
        config = make_minimal_config()

        model1 = StaticModel(config, in_features=5, rngs=nnx.Rngs(42))
        model2 = StaticModel(config, in_features=5, rngs=nnx.Rngs(42))

        x = jnp.array([[1.0, 2.0, 3.0]])

        out1 = model1(x, mode="potential")["potential"]
        out2 = model2(x, mode="potential")["potential"]

        assert jnp.allclose(out1, out2)

    def test_different_seeds_different_outputs(self):
        """Test that different seeds produce different model outputs."""
        config = make_minimal_config()

        model1 = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))
        model2 = StaticModel(config, in_features=5, rngs=nnx.Rngs(1))

        x = jnp.array([[1.0, 2.0, 3.0]])

        out1 = model1(x, mode="potential")["potential"]
        out2 = model2(x, mode="potential")["potential"]

        # Different seeds should produce different weights -> different outputs
        assert not jnp.allclose(out1, out2)


class TestStaticModelIntegration:
    """Integration tests for StaticModel with JAX transformations."""

    def test_vmap_over_batch(self):
        """Test vmapping the model over inputs."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        # Process single points through vmap
        x_batch = jnp.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]])

        @nnx.vmap(in_axes=(0,))
        def forward_single(x):
            return model(x, mode="potential")["potential"]

        potentials = forward_single(x_batch)
        assert potentials.shape == (3, 1)

    def test_jit_grad_combination(self):
        """Test JIT-compiled gradient computation."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        @nnx.jit
        def compute_loss_and_grad(model, x):
            def loss_fn(m):
                return jnp.mean(m(x, mode="potential")["potential"] ** 2)

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            return loss, grads

        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        loss, grads = compute_loss_and_grad(model, x)

        assert jnp.isfinite(loss)
        assert grads is not None
