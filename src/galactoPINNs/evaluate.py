
import jax.numpy as jnp
import numpy as np
from unxt import unitsystems
import jax.random as jr

from dataclasses import KW_ONLY
from typing import Any, final
from jaxtyping import Array, Float
from galax.potential._src.base import default_constants
from galax.potential._src.base_single import AbstractPotential
import equinox as eqx
from xmmutablemap import ImmutableMap
import unxt as u
from unxt.quantity import AbstractQuantity

def apply_model(model, params, x, return_analytic_weights=False):
    predictions = model.apply({"params": params}, x)
    u_pred = predictions["potential"]
    a_pred = predictions["acceleration"]

    if return_analytic_weights:
        analytic_weights = (predictions["outputs"]["h"], predictions["outputs"]["g"])
    else:
        analytic_weights = None
    if "outputs" in predictions:
        outputs = predictions["outputs"]
    else:
        outputs = None
    return {
        "u_pred": u_pred,
        "a_pred": a_pred,
        "analytic_weights": analytic_weights,
        "outputs": outputs,
    }


def apply_model_time(model, params, tx_scaled):
    predictions = model.apply({"params": params}, tx_scaled)
    u_pred = predictions["potential"]
    a_pred = predictions["acceleration"]

    if "outputs" in predictions:
        return {"u_pred": u_pred, "a_pred": a_pred, "outputs": predictions["outputs"]}
    else:
        return {"u_pred": u_pred, "a_pred": a_pred}


def evaluate_performance(
    model,
    trained_state_params,
    raw_datadict,
    num_test,
    return_analytic_weights=False,
):
    true_pot = raw_datadict["u_val"][:num_test]
    true_acc = raw_datadict["a_val"][:num_test]

    print("here!", type(model))
    config = model.config
    r_eval = np.linalg.norm(raw_datadict["x_val"][:num_test], axis=1)
    scaled_x_val = config["x_transformer"].transform(raw_datadict["x_val"][:num_test])
    output = apply_model(model, trained_state_params, scaled_x_val)
    predicted_pot = config["u_transformer"].inverse_transform(output["u_pred"])
    predicted_acc = config["a_transformer"].inverse_transform(output["a_pred"])
    acc_percent_error = (
        100
        * jnp.linalg.norm(predicted_acc - true_acc, axis=1)
        / jnp.linalg.norm(true_acc, axis=1)
    )
    pot_percent_error = 100 * np.abs((true_pot - predicted_pot) / true_pot)

    fiducial_acc = None
    fiducial_pot = None
    fiducial_acc_error = None
    fiducial_pot_error = None

    if config.get("include_analytic", True):
        lf_analytic = config["lf_analytic_function"]
        lf_analytic_potential = lf_analytic.potential(
            raw_datadict["x_val"][:num_test], t=0
        )
        lf_analytic_acc = lf_analytic.acceleration(
            raw_datadict["x_val"][:num_test], t=0
        )

        lf_pot_error = 100 * np.abs((lf_analytic_potential - true_pot) / true_pot)
        lf_acc_error = (
            100
            * jnp.linalg.norm(lf_analytic_acc - true_acc, axis=1)
            / jnp.linalg.norm(true_acc, axis=1)
        )

        residual_pot = lf_analytic_potential - predicted_pot
        average_residual_pot = jnp.mean(residual_pot)
        corrected_potential = predicted_pot + average_residual_pot
        corrected_pot_percent_error = 100 * jnp.abs(
            (corrected_potential - true_pot) / true_pot
        )

        if return_analytic_weights:
            analytic_weights = output["analytic_weights"]

        else:
            analytic_weights = None

    else:
        lf_analytic_potential = None
        lf_pot_error = None
        lf_acc_error = None
        residual_pot = None
        corrected_potential = None
        corrected_pot_percent_error = None
        analytic_weights = None

    return {
        "r_eval": r_eval,
        "x_val": raw_datadict["x_val"][:num_test],
        "true_a": true_acc,
        "predicted_a": predicted_acc,
        "true_u": true_pot,
        "predicted_u": predicted_pot,
        "acc_percent_error": acc_percent_error,
        "pot_percent_error": pot_percent_error,
        "residual_pot": residual_pot,
        "corrected_pot_percent_error": corrected_pot_percent_error,
        "lf_potential": lf_analytic_potential,
        "lf_pot_error": lf_pot_error,
        "lf_acc_error": lf_acc_error,
        "analytic_weights": analytic_weights,
        "fiducial_acc": fiducial_acc,
        "fiducial_pot": fiducial_pot,
        "fiducial_pot_error": fiducial_pot_error,
        "fiducial_acc_error": fiducial_acc_error,
        "avg_percent_error": np.mean(acc_percent_error),
    }


def evaluate_performance_node(
    model, params, t_eval, raw_datadict, num_test, return_analytic_weights=False
):
    val_data = raw_datadict["val"][t_eval]
    x_val = val_data["x"][:num_test]

    config = model.config
    r_eval = np.linalg.norm(x_val, axis=1)

    true_pot = val_data["u"][:num_test]
    true_acc = val_data["a"][:num_test]

    x_scaled = config["x_transformer"].transform(x_val)
    t_scaled = config["t_transformer"].transform(t_eval) * jnp.ones(
        (x_val.shape[0], 1)
    )
    tx_scaled = jnp.concatenate([t_scaled, x_scaled], axis=1)

    output = apply_model_time(model, params, tx_scaled)

    predicted_pot = config["u_transformer"].inverse_transform(output["u_pred"])
    predicted_acc = config["a_transformer"].inverse_transform(output["a_pred"])

    predicted_acc_norm = jnp.linalg.norm(predicted_acc, axis=1, keepdims=True)
    true_acc_norm = jnp.linalg.norm(true_acc, axis=1, keepdims=True)

    acc_percent_error = (
        100
        * jnp.linalg.norm(predicted_acc - true_acc, axis=1)
        / jnp.linalg.norm(true_acc, axis=1)
    )
    pot_percent_error = 100 * np.abs((true_pot - predicted_pot) / true_pot)

    if config.get("include_analytic", True):
        lf_analytic = config["lf_analytic_function"]
        lf_analytic_potential = lf_analytic.potential(x_val, t=t_eval)
        lf_analytic_acc = lf_analytic.acceleration(x_val, t=t_eval)
        lf_analytic_0 = lf_analytic.potential(x_val, t=0)
        lf_analytic_acc_norm = jnp.linalg.norm(lf_analytic_acc, axis=1)
        lf_pot_error = 100 * np.abs((lf_analytic_potential - true_pot) / true_pot)
        lf0_pot_error = 100 * np.abs((lf_analytic_0 - true_pot) / true_pot)
        lf_acc_error = (
            100
            * jnp.linalg.norm(lf_analytic_acc - true_acc, axis=1)
            / jnp.linalg.norm(true_acc, axis=1)
        )

        residual_pot = lf_analytic_potential - predicted_pot
        average_residual_pot = jnp.mean(residual_pot)
        corrected_potential = predicted_pot + average_residual_pot
        corrected_pot_percent_error = 100 * jnp.abs(
            (corrected_potential - true_pot) / true_pot
        )

        if return_analytic_weights:
            analytic_weights = output["analytic_weights"]

        else:
            analytic_weights = None

    else:
        lf_analytic_potential = None
        lf_pot_error = None
        lf_acc_error = None
        residual_pot = None
        corrected_potential = None
        corrected_pot_percent_error = None
        analytic_weights = None
        lf_analytic_acc_norm = None
        lf0_pot_error = None
        lf_analytic_0 = None

    return {
        "r_eval": r_eval,
        "true_a": true_acc,
        "predicted_a": predicted_acc,
        "true_a_norm": true_acc_norm,
        "predicted_a_norm": predicted_acc_norm,
        "true_u": true_pot,
        "predicted_u": predicted_pot,
        "acc_percent_error": acc_percent_error,
        "pot_percent_error": pot_percent_error,
        "residual_pot": residual_pot,
        "corrected_pot_percent_error": corrected_pot_percent_error,
        "corrected_potential": corrected_potential,
        "lf_potential": lf_analytic_potential,
        "lf_acc_norm": lf_analytic_acc_norm,
        "lf_pot_error": lf_pot_error,
        "lf_acc_error": lf_acc_error,
        "lf0_pot_error": lf0_pot_error,
        "lf_analytic_0": lf_analytic_0,
        "analytic_weights": analytic_weights,
    }



@final
class ModelPotential(AbstractPotential):
    """A potential whose gradient is learned by a neural network."""

    #model: Any = eqx.field(static=True)
    apply_fn: Any = eqx.field(static=True)
    params: Any = eqx.field(static=True)
    config: dict = eqx.field(static=True)

    _: KW_ONLY
    units: u.AbstractUnitSystem = eqx.field(converter=u.unitsystem, static=True)
    constants: ImmutableMap[str, AbstractQuantity] = eqx.field(
        default=default_constants, converter=ImmutableMap
    )


    def _acceleration(self, q, t):
        x = jnp.asarray(getattr(q, "value", q)).astype(jnp.float32)
        batched = x.ndim == 2
        x_batched = jnp.reshape(x, (1, 3)) if not batched else x

        x_trans = self.config["x_transformer"].transform(x_batched)
        a_pred = self.apply_fn({"params": self.params}, x_trans)["acceleration"]
        a_out = self.config["a_transformer"].inverse_transform(a_pred)

        return jnp.squeeze(a_out, axis=0) if not batched else a_out

    def _potential(self, q, t):
        x = jnp.asarray(getattr(q, "value", q)).astype(jnp.float32)
        batched = x.ndim == 2
        x_batched = jnp.reshape(x, (1, 3)) if not batched else x

        x_trans = self.config["x_transformer"].transform(x_batched)
        u_pred = self.apply_fn({"params": self.params}, x_trans)["potential"]
        u_out = self.config["u_transformer"].inverse_transform(u_pred)

        # Return shape (N,) if batched, or () if single
        return jnp.squeeze(u_out, axis=0) if not batched else jnp.ravel(u_out)

    def _gradient(self, q, t):
        return -self._acceleration(q, t)


def make_galax_potential(model, params):
    learned_potential = ModelPotential(
    apply_fn = model,
    params   = params,
    config   = model.config,
    units    = unitsystems.galactic)
    return learned_potential


def bnn_performance(predictive, x_test, config):
    x_test_scaled = config["x_transformer"].transform(x_test)
    pred = predictive(jr.PRNGKey(2), x_test_scaled)

    u_post_phys = config["u_transformer"].inverse_transform(pred["potential"])
    a_post_phys = config["a_transformer"].inverse_transform(pred["acceleration"])

    u_mean = u_post_phys.mean(axis=0)
    a_mean = a_post_phys.mean(axis=0)

    u_std = u_post_phys.std(axis=0)
    a_std = a_post_phys.std(axis=0)

    return {
        "u_mean": u_mean,
        "a_mean": a_mean,
        "u_std": u_std,
        "a_std": a_std,
        "a_samples": a_post_phys,
        "u_samples": u_post_phys,
    }
