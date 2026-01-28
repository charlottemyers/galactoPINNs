"""Tests for NNX training utilities."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from galactoPINNs.models.static_model import StaticModel
from galactoPINNs.train import create_optimizer, get_model_params, train_step_static


class MockTransformer:
    """Mock transformer for testing."""

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


class MockAnalyticPotential:
    """Mock analytic potential for testing."""

    def potential(self, x, *, t=0):
        r = jnp.linalg.norm(x, axis=-1)
        return -1.0 / (r + 0.1)

    def acceleration(self, x, *, t=0):
        r = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return -x / (r**3 + 0.01)


def make_minimal_config() -> dict:
    """Create a minimal configuration for testing."""
    return {
        "x_transformer": MockTransformer(),
        "u_transformer": MockTransformer(),
        "a_transformer": MockTransformer(),
        "r_s": 1.0,
        "clip": 1.0,
        "scale": "one",
        "include_analytic": False,
        "ab_potential": MockAnalyticPotential(),
        "convert_to_spherical": True,
        "depth": 2,
        "width": 16,
        "nn_off": False,
    }


class TestCreateOptimizer:
    """Tests for create_optimizer function."""

    def test_create_optimizer(self):
        """Test creating an optimizer from model and optax transform."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))
        tx = optax.adam(1e-3)

        optimizer = create_optimizer(model, tx)

        assert optimizer is not None
        assert isinstance(optimizer, nnx.Optimizer)

    def test_get_model_params(self):
        """Test accessing parameters through get_model_params."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        params = get_model_params(model)

        # params should be a nested dict structure
        assert isinstance(params, dict)


class TestTrainStepStatic:
    """Tests for train_step_static function."""

    def test_train_step_acceleration_target(self):
        """Test a single training step with acceleration target."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))
        tx = optax.adam(1e-3)
        optimizer = create_optimizer(model, tx)

        # Create fake training data
        x = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        a_true = jnp.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])

        # Run one training step
        loss = train_step_static(model, optimizer, x, a_true, target="acceleration")

        # Check that loss is computed
        assert jnp.isfinite(loss)

    def test_train_step_reduces_loss(self):
        """Test that multiple training steps reduce loss."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))
        tx = optax.adam(1e-2)  # Higher learning rate for faster convergence
        optimizer = create_optimizer(model, tx)

        # Create fake training data
        x = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        a_true = jnp.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])

        # Get initial loss
        initial_loss = train_step_static(
            model, optimizer, x, a_true, target="acceleration"
        )

        # Run more training steps
        for _ in range(10):
            loss = train_step_static(model, optimizer, x, a_true, target="acceleration")

        # Loss should decrease with training
        assert jnp.isfinite(loss)
        assert loss < initial_loss, f"Loss did not decrease: {loss} >= {initial_loss}"

    def test_train_step_jit_compiled(self):
        """Test that train_step_static works (it's nnx.jit decorated)."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))
        tx = optax.adam(1e-3)
        optimizer = create_optimizer(model, tx)

        x = jnp.array([[1.0, 2.0, 3.0]])
        a_true = jnp.array([[0.1, 0.2, 0.3]])

        # This should work without issues
        loss = train_step_static(model, optimizer, x, a_true, target="acceleration")
        assert jnp.isfinite(loss)


class TestTrainingIntegration:
    """Integration tests for the full training workflow."""

    def test_multiple_epochs(self):
        """Test running multiple epochs of training."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))
        tx = optax.adam(1e-3)
        optimizer = create_optimizer(model, tx)

        # Create a small dataset
        n_samples = 20
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (n_samples, 3))
        # Create target accelerations (simplified: pointing toward origin)
        a_true = -x / (jnp.linalg.norm(x, axis=-1, keepdims=True) ** 2 + 0.1)

        losses = []
        for _ in range(5):
            loss = train_step_static(model, optimizer, x, a_true, target="acceleration")
            losses.append(float(loss))

        # All losses should be finite
        assert all(map(jnp.isfinite, losses))

    def test_optimizer_state_updates(self):
        """Test that optimizer state is properly maintained."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))
        tx = optax.adam(1e-3)
        optimizer = create_optimizer(model, tx)

        x = jnp.array([[1.0, 0.0, 0.0]])
        a_true = jnp.array([[0.1, 0.0, 0.0]])

        # Run a few steps
        for _ in range(3):
            train_step_static(model, optimizer, x, a_true, target="acceleration")

        # Optimizer should still be functional
        loss = train_step_static(model, optimizer, x, a_true, target="acceleration")
        assert jnp.isfinite(loss)
