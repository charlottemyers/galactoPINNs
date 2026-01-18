import jax
from functools import partial
from flax.training.train_state import TrainState
import jax.numpy as jnp
import numpy as np
import jax.random as jr
from typing import Literal


@partial(jax.jit, static_argnames=("target"))
def train_step_static(
    state: TrainState,
    x,
    a_true,
    target: Literal["acceleration", "orbit_energy", "mixed"] = "acceleration",
    lambda_rel: float = 1.0,
    orbit_q = None,
    orbit_p = None,
    lambda_E: float = 5.0,
    std_weight: float = 10.0,
):
    def acc_loss_fn(params, lambda_rel):
        a_pred = state.apply_fn({"params": params}, x)["acceleration"]
        diff = a_pred - a_true
        diff_norm = jnp.linalg.norm(diff, axis=1, keepdims=True)
        a_true_norm = jnp.linalg.norm(a_true, axis=1, keepdims=True)
        return jnp.mean(diff_norm + lambda_rel * (diff_norm / a_true_norm))

    def orbit_energy(params, orbit_q_scaled, orbit_p_scaled):
        """
        orbit_q_scaled: (B, T, 3) in model *scaled* coordinates
        orbit_p_scaled: (B, T, 3) in model *scaled* velocities
        Returns E_pred: (B, T) in model's scaled energy units
        """
        B, T, _  = orbit_q_scaled.shape
        T_ke = 0.5 * jnp.sum(orbit_p_scaled**2, axis=-1)  # (B, T)

        q_flat = orbit_q_scaled.reshape(B*T, 3)
        Phi_flat = state.apply_fn({"params": params}, q_flat, mode="potential")["potential"]  # (BT,)
        Phi = Phi_flat.reshape(B, T)
        Phi = Phi.reshape(B, T)  # ensure (B, T)
        return T_ke + Phi  # (B, T)

    def orbit_E_loss(params, orbit_q_scaled, orbit_p_scaled, std_weight=20.0):
        """
        Per-trajectory energy consistency:
        - end-start drift (ΔE)^2
        - within-trajectory variance (std_E)^2
        Aggregated across B orbits by mean.
        """
        E = orbit_energy(params, orbit_q_scaled, orbit_p_scaled)  # (B, T)

        # end-start drift per trajectory
        delta_E = E[:, -1] - E[:, 0]                   # (B,)

        # std across time per trajectory
        E_centered = E - jnp.mean(E, axis=1, keepdims=True)
        std_E = jnp.std(E_centered, axis=1)               # (B,)

        return jnp.mean(delta_E**2 + std_weight * std_E**2)  # scalar

    def total_loss(params):
        if target == "mixed":
            loss = lambda_E*orbit_E_loss(params, orbit_q, orbit_p, std_weight=std_weight) + acc_loss_fn(params, lambda_rel)
        elif target == "orbit_energy":
            loss = orbit_E_loss(params, orbit_q, orbit_p)
        else:
            loss = acc_loss_fn(params, lambda_rel)
        return loss

    # Compute loss and gradients
    loss, grads = jax.value_and_grad(total_loss)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss




@jax.jit
def train_step_node(state, tx_cart, a_true, lambda_rel=1.0):
    def loss_fn(params):
        outputs = state.apply_fn({"params": params}, tx_cart)
        a_pred = outputs["acceleration"]

        diff = a_pred - a_true
        diff_norm = jnp.linalg.norm(diff, axis=1, keepdims=True)
        a_true_norm = jnp.linalg.norm(a_true, axis=1, keepdims=True)
        return jnp.mean(diff_norm + lambda_rel * (diff_norm / a_true_norm))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    state = state.apply_gradients(grads=grads)
    return state, loss



def train_model_static(
    model,
    optimizer,
    x_train,
    a_train,
    num_epochs,
    mode="full",
    target="acceleration",
    lap_true=None,
    log_every=100,
    lambda_rel=1.0,
    init_state=None,
    orbit_q = None,
    orbit_p = None,
    lambda_E = 5.0,
    std_weight = 10.0,
    train_dict = None
):
    epochs = []
    losses = []
    if init_state is None:
        state = TrainState.create(
            apply_fn=model.apply,
            params=model.init(jax.random.PRNGKey(0), x_train)["params"],
            tx=optimizer,
        )
    else:
        state = init_state
    if train_dict is None:
        train_dict = {target: num_epochs}

    for target, n_epochs in train_dict.items():
        print(f"Training for {n_epochs} epochs on target: {target}")
        for epoch in range(n_epochs):
            state, loss = train_step_static(
                state,
                x_train,
                a_train,
                mode=mode,
                target=target,
                lap_true=lap_true,
                lambda_rel=lambda_rel,
                orbit_q=orbit_q,
                orbit_p=orbit_p,
                lambda_E=lambda_E,
                std_weight=std_weight,
            )
            if epoch % log_every == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss:.6f}")
        epochs.append(epoch)
        losses.append(loss)
    return {"state": state, "epochs": epochs, "losses": losses}



def train_model_state_node(
    initial_state, x_train, a_train, num_epochs, log_every=1000
):
    epochs = []
    losses = []

    state = initial_state
    for epoch in range(num_epochs):
        state, loss = train_step_node(state, x_train, a_train)
        if epoch % log_every == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
        epochs.append(epoch)
        losses.append(loss)
    return {"state": state, "epochs": epochs, "losses": losses}




def train_model_state_trainable_analytic(
    init_state, train_step, x_train, a_train, num_epochs, radial_weight=False, log_every=1000
):
    ######
    ### train model with a trainable analytic layer
    ######

    epochs = []
    losses = []
    analytic_params_history = []
    fusing_params_history = []

    state = init_state
    for epoch in range(num_epochs):
        state, loss = train_step(state, x_train, a_train, radial_weight=radial_weight)
        if epoch % log_every == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.6f}")
        analytic_params = state.params["trainable_analytic_layer"]
        fusing_params = state.params["FuseandBoundary_0"]
        analytic_params_history.append(analytic_params)
        fusing_params_history.append(fusing_params)

        epochs.append(epoch)
        losses.append(loss)
    return {
        "state": state,
        "epochs": epochs,
        "losses": losses,
        "fusing_params_history": fusing_params_history,
        "analytic_params_history": analytic_params_history,
    }


def alternate_training(
    train_step,
    x_train,
    a_train,
    model,
    num_epochs_stage1,
    num_epochs_stage2,
    cycles,
    param_list,
    tx_1,
    tx_2,
    burn_in=0,
):
    #####
    ## function for alternating training between analytic and NN parts of the model
    #####

    history = {
        "losses": [],
        "epochs": np.arange(cycles * (num_epochs_stage1 + num_epochs_stage2)),
    }
    for param in param_list:
        history[f"{param}"] = []

    for cycle in range(cycles):
        if cycle == 0:
            params = model.init(jr.PRNGKey(0), x_train[:1])["params"]
        else:
            params = stage2_output_state.params

        stage1_input_state = TrainState.create(
            apply_fn=model.apply, params=params, tx=tx_1
        )

        if cycle == 0:
            output1 = train_model_state_trainable_analytic(
                stage1_input_state,
                train_step,
                x_train,
                a_train,
                num_epochs_stage1 + burn_in,
            )
        else:
            output1 = train_model_state_trainable_analytic(
                stage1_input_state,
                train_step,
                x_train,
                a_train,
                num_epochs_stage1,
            )

        stage1_output_state = output1["state"]

        history["losses"] += output1["losses"]
        for param in param_list:
            if param in stage1_output_state.params["trainable_analytic_layer"]:
                history[f"{param}"] += [
                    d[param] for d in output1["analytic_params_history"]
                ]
                print(f"Learned {param} in cycle {cycle}: {history[f'{param}'][-1]}")

        # === Stage 2: train NN (analytic frozen) ===
        stage2_inputstate = TrainState.create(
            apply_fn=model.apply, params=stage1_output_state.params, tx=tx_2
        )

        output2 = train_model_state_trainable_analytic(
            stage2_inputstate,
            train_step,
            x_train,
            a_train,
            num_epochs_stage2
            )

        stage2_output_state = output2["state"]

        for param in param_list:
            if param in stage1_output_state.params["trainable_analytic_layer"]:
                history[f"{param}"] += [
                    d[param] for d in output2["analytic_params_history"]
                ]

        history["losses"] += output2["losses"]
    return {"state": stage2_output_state, "history": history, "model": model}
