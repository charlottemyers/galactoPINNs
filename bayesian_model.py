import astropy.units as u
import numpy as np
import quaxed.numpy as jnp
import galax.dynamics as gd
import galax.potential as gp
import jax
import jax.random as jr
import jax.numpy as jnp
import flax.linen as nn
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
from numpyro.contrib.module import random_flax_module
from galax.coordinates import PhaseSpacePosition
from unxt import unitsystems
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer import SVI, Trace_ELBO, init_to_feasible
from numpyro.optim import Adam
from static_model import Model_with_analytic


def model_svi(
    x, a_obs=None, *, sigma_theta=0.05, sigma_lambda=0.05, analytic_only=False, sigma_a = 2e-4, parameter_dict, analytic_form,
    config
):
    parameter_distributions = {}
    for param_name, param_info in parameter_dict.items():
        parameter_distributions[param_name] = dist.TruncatedNormal(*param_info)

    analytic = analytic_form(parameter_distributions)

    net = Model_with_analytic(config=config, trainable_analytic_layer=analytic)

    bnn = random_flax_module(
        "full_model",
        net,
        prior=dist.Normal(0, sigma_lambda),
        input_shape=(x.shape[0], x.shape[1]),
    )

    sigma_a = 2e-4

    # exact deterministic loss
    data_weight = 1.0 / (2 * sigma_a**2)
    lambda_rel = 0.1
    out = bnn(x)
    a_pred = out["acceleration"]
    u_pred = out["potential"]
    numpyro.deterministic("acceleration", a_pred)
    numpyro.deterministic("potential", u_pred)

    if a_obs is not None:
        diff = a_pred - a_obs
        diff_norm = jnp.linalg.norm(diff, axis=1)
        a_true_norm = jnp.linalg.norm(a_obs, axis=1)
        per_point = diff_norm + lambda_rel * (diff_norm / a_true_norm)
        loss = jnp.mean(per_point)

        numpyro.factor("acc_loss", -data_weight * loss)


def run_window(prev_result, guide, sigma_theta, sigma_lambda,  x_train, a_train, steps=1000, lr0 = 5e-3):
    svi = SVI(
        lambda x, a_obs: model_svi(
            x, a_obs, sigma_theta=sigma_theta, sigma_lambda=sigma_lambda
        ),
        guide,
        Adam(lr0),
        Trace_ELBO(),
    )
    if prev_result is None:
        result = svi.run(jr.PRNGKey(0), steps, x_train, a_train)
    else:
        result = svi.run(
            jr.PRNGKey(0), steps, x_train, a_train, init_params=prev_result.params
        )
    return result
