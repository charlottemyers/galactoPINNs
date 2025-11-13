import jax.numpy as jnp
import numpy as np
import astropy.units as au
import coordinax as cx
from galax.potential import density
from unxt import Quantity
import galax.potential as gp
from flax.training.train_state import TrainState
import jax.random as jr
import jax


def train_model_state(
    init_state, train_step, x_train, a_train, num_epochs, radial_weight=False
):
    epochs = []
    losses = []
    analytic_params_history = []
    fusing_params_history = []

    state = init_state
    for epoch in range(num_epochs):
        state, loss = train_step(state, x_train, a_train, radial_weight=radial_weight)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.6f}")
        analytic_params = state.params["analytic_layer"]
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



def train_model(
    model,
    train_step,
    optimizer,
    x_train,
    a_train,
    num_epochs,
    i_large=None,
    mode="full",
    target="acceleration",
    lap_true=None,
    log_every=100,
    lambda_rel=1.0,
    init_state=None,
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

    for epoch in range(num_epochs):
        state, loss = train_step(
            state,
            x_train,
            a_train,
            i_large=i_large,
            mode=mode,
            target=target,
            lap_true=lap_true,
            lambda_rel=lambda_rel,
        )
        if epoch % log_every == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.6f}")
        epochs.append(epoch)
        losses.append(loss)
    return {"state": state, "epochs": epochs, "losses": losses}



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
    train_target="acceleration",
    radial_weight=False,
    burn_in=0,
):
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
            output1 = train_model_state(
                stage1_input_state,
                train_step,
                x_train,
                a_train,
                num_epochs_stage1 + burn_in,
                radial_weight=radial_weight,
            )
        else:
            output1 = train_model_state(
                stage1_input_state,
                train_step,
                x_train,
                a_train,
                num_epochs_stage1,
                radial_weight=radial_weight,
            )

        stage1_output_state = output1["state"]

        history["losses"] += output1["losses"]
        for param in param_list:
            if param in stage1_output_state.params["analytic_layer"]:
                history[f"{param}"] += [
                    d[param] for d in output1["analytic_params_history"]
                ]
                print(f"Learned {param} in cycle {cycle}: {history[f'{param}'][-1]}")
            if param in stage1_output_state.params["FuseandBoundary_0"]:
                history[f"{param}"] += [
                    d[param] for d in output1["fusing_params_history"]
                ]
                print(f"Learned {param} in cycle {cycle}: {history[f'{param}'][-1]}")

        # === Stage 2: train NN (analytic frozen) ===
        stage2_inputstate = TrainState.create(
            apply_fn=model.apply, params=stage1_output_state.params, tx=tx_2
        )

        output2 = train_model_state(
            stage2_inputstate,
            train_step,
            x_train,
            a_train,
            num_epochs_stage2,
            radial_weight=radial_weight,
        )
        stage2_output_state = output2["state"]

        for param in param_list:
            if param in stage1_output_state.params["analytic_layer"]:
                history[f"{param}"] += [
                    d[param] for d in output2["analytic_params_history"]
                ]
            if param in stage1_output_state.params["FuseandBoundary_0"]:
                history[f"{param}"] += [
                    d[param] for d in output2["fusing_params_history"]
                ]
        history["losses"] += output2["losses"]
    return {"state": stage2_output_state, "history": history, "model": model}
