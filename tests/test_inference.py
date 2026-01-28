"""Tests for NNX inference utilities."""

import jax.numpy as jnp
from flax import nnx

from galactoPINNs.inference import apply_model
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


class TestApplyModel:
    """Tests for apply_model function."""

    def test_apply_model_returns_predictions(self):
        """Test that apply_model returns potential and acceleration predictions."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        x_scaled = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        result = apply_model(model, x_scaled)

        assert "u_pred" in result
        assert "a_pred" in result
        assert result["u_pred"].shape == (2,)
        assert result["a_pred"].shape == (2, 3)

    def test_apply_model_finite_outputs(self):
        """Test that apply_model produces finite outputs."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        x_scaled = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.1, 0.2, 0.3]])

        result = apply_model(model, x_scaled)

        assert jnp.isfinite(result["u_pred"]).all()
        assert jnp.isfinite(result["a_pred"]).all()

    def test_apply_model_single_point(self):
        """Test apply_model with a single point (2D array with 1 row)."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        x_scaled = jnp.array([[1.0, 0.0, 0.0]])

        result = apply_model(model, x_scaled)

        assert result["u_pred"].shape == (1,)
        assert result["a_pred"].shape == (1, 3)

    def test_apply_model_batch_consistency(self):
        """Test that batching gives same results as individual calls."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        x1 = jnp.array([[1.0, 0.0, 0.0]])
        x2 = jnp.array([[0.0, 1.0, 0.0]])
        x_batch = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        result1 = apply_model(model, x1)
        result2 = apply_model(model, x2)
        result_batch = apply_model(model, x_batch)

        # First row of batch should match individual result
        assert jnp.allclose(result_batch["u_pred"][0], result1["u_pred"][0])
        assert jnp.allclose(result_batch["u_pred"][1], result2["u_pred"][0])

        assert jnp.allclose(result_batch["a_pred"][0], result1["a_pred"][0])
        assert jnp.allclose(result_batch["a_pred"][1], result2["a_pred"][0])


class TestNNXModelProtocol:
    """Tests to verify NNX models satisfy the expected protocol."""

    def test_model_is_callable(self):
        """Test that NNX model is directly callable."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        # Model should be callable
        assert callable(model)

        x = jnp.array([[1.0, 2.0, 3.0]])
        result = model(x, mode="full")

        assert "potential" in result
        assert "acceleration" in result

    def test_model_has_config(self):
        """Test that model has config attribute."""
        config = make_minimal_config()
        model = StaticModel(config, in_features=5, rngs=nnx.Rngs(0))

        assert hasattr(model, "config")
        assert model.config is config
