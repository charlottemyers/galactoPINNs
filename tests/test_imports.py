"""Tests for package imports."""

import galactoPINNs
from galactoPINNs import evaluate, inference, layers, train
from galactoPINNs.models import static_model


def test_import_package():
    """Test that the main package can be imported."""
    assert galactoPINNs is not None


def test_import_layers():
    """Test that layers module can be imported."""
    assert layers is not None
    assert hasattr(layers, "SmoothMLP")
    assert hasattr(layers, "CartesianToModifiedSphericalLayer")
    assert hasattr(layers, "ScaleNNPotentialLayer")
    assert hasattr(layers, "FuseModelsLayer")


def test_import_models():
    """Test that model modules can be imported."""
    assert static_model is not None
    assert hasattr(static_model, "StaticModel")


def test_import_train():
    """Test that training module can be imported."""
    assert train is not None
    assert hasattr(train, "train_step_static")
    assert hasattr(train, "train_model_static")


def test_import_inference():
    """Test that inference module can be imported."""
    assert inference is not None
    assert hasattr(inference, "apply_model")


def test_import_evaluate():
    """Test that evaluate module can be imported."""
    assert evaluate is not None
    assert hasattr(evaluate, "evaluate_performance")
