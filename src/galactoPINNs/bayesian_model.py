import quaxed.numpy as jnp
import jax
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module
from numpyro.optim import Adam
from numpyro.infer import SVI, Trace_ELBO


from .static_model import StaticModel


def model_svi(
    x, a_obs=None, *, sigma_theta=0.05, sigma_lambda=0.05, sigma_a = 2e-4, parameter_dict, analytic_form,
    config, orbit_q=None, orbit_p=None, w_orbit=1.0, std_weight=1.0
):
    parameter_distributions = {}
    for param_name, param_info in parameter_dict.items():
        parameter_distributions[param_name] = dist.TruncatedNormal(*param_info)

    analytic = analytic_form(parameter_distributions)
    net = StaticModel(config=config, trainable_analytic_layer=analytic)
    bnn = random_flax_module(
        "full_model",
        net,
        prior=dist.Normal(0, sigma_lambda),
        input_shape=(x.shape[0], x.shape[1]),
    )

    lambda_rel = 0.1
    sigma_a = 2e-4
    data_weight = 1.0 / (2 * sigma_a**2)
    out = bnn(x)
    a_pred = out["acceleration"]
    u_pred = out["potential"]

    numpyro.deterministic("acceleration", a_pred)
    numpyro.deterministic("potential", u_pred)

    if a_obs is not None:
            diff = a_pred - a_obs
            diff_norm = jnp.linalg.norm(diff, axis=1)
            a_true_norm = jnp.linalg.norm(a_obs, axis=1) + 1e-12
            per_point = diff_norm + lambda_rel * (diff_norm / a_true_norm)
            loss = jnp.mean(per_point)
            numpyro.factor("acc_loss", -data_weight * loss)

            # ---- Orbit energy loss factor ----
            if (orbit_q is not None) and (orbit_p is not None):
                B, T, _ = orbit_q.shape
                T_ke = 0.5 * jnp.sum(orbit_p**2, axis=-1)

                q_flat = orbit_q.reshape(B*T, 3)
                phi_flat = bnn(q_flat)["potential"]
                phi = phi_flat.reshape(B, T)

                E = T_ke + phi
                dE = E[:, -1] - E[:, 0]
                E_cent = E - jnp.mean(E, axis=1, keepdims=True)
                std_E = jnp.std(E_cent, axis=1)

                L_orbit = jnp.mean(dE**2 + std_weight * std_E**2)
                numpyro.factor("orbit_E_loss", -w_orbit * L_orbit)



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
