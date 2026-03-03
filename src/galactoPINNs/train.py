"""Training utilities for galactoPINNs models."""

__all__ = (
    "alternate_training",
    "create_optimizer",
    "train_model_node",
    "train_model_static",
    "train_model_with_trainable_analytic_layer",
    "train_step_node",
    "train_step_static",
)

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxtyping import Array

log = logging.getLogger(__name__)

StaticTarget = Literal["acceleration", "orbit_energy", "mixed"]
TrainStepFn = Callable[..., Array]


def create_optimizer(
    model: nnx.Module, tx: optax.GradientTransformation
) -> nnx.Optimizer:
    """Create an NNX optimizer from a model and optax transformation.

    Parameters
    ----------
    model
        An NNX module instance (already initialized with parameters).
    tx
        An optax gradient transformation.

    Returns
    -------
    nnx.Optimizer
        An optimizer that tracks the model parameters.

    """
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def get_model_params(model: nnx.Module) -> dict[str, Any]:
    """Get the current model parameters as a nested dict.

    Parameters
    ----------
    model
        An NNX module instance.

    Returns
    -------
    dict
        The model parameters in dict form.

    """
    _, state = nnx.split(model)
    return nnx.to_pure_dict(state)


#############


@nnx.jit(static_argnames=("target", "ramp_kind", "balance_grads"))
def train_step_static(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x: Array,
    a_true: Array,
    *,
    lambda_rel: float = 1.0,
    orbit_q: Array | None = None,
    orbit_p: Array | None = None,
    lambda_E: float = 5.0,
    target: Literal[
        "acceleration",
        "orbit_energy",
        "mixed",
        "mixed_dev_from_initial",
    ] = "acceleration",
    # --- ramp / balancing  ---
    step: int = 0,
    total_steps: int = 1,
    ramp_kind: Literal["linear", "cosine", "sigmoid"] = "cosine",
    ramp_sharp: float = 10.0,       # only used when ramp_kind="sigmoid"
    balance_grads: bool = True,
) -> Array:
    """Perform one optimizer step for a static potential model.

    This function computes a scalar training loss, differentiates it with
    respect to the model parameters, and applies the resulting gradients
    via the optimizer. The loss target is controlled by ``target``:

    - ``"acceleration"``: matches predicted accelerations to ``a_true`` at input
      positions ``x``.
    - ``"orbit_energy"``: penalizes non-conservation of total specific energy
      along pre-computed orbit trajectories (std of energy only).
    - ``"mixed"``: weighted sum of the orbit-energy loss and the acceleration
      loss, with a warm-up ramp on ``lambda_E`` and optional gradient-norm
      balancing.
    - ``"mixed_dev_from_initial"``: same as ``"mixed"`` but uses a fractional
      energy-drift loss (relative to the initial energy) instead of the
      std-based loss.

    Notes
    -----
    - This function is JIT-compiled using ``nnx.jit``.
    - The model and optimizer are mutated in place (NNX pattern).
    - ``lambda_E`` is ramped from 0 → ``lambda_E`` over training using the
      schedule selected by ``ramp_kind``.
    - When ``balance_grads=True`` the energy-loss gradient tree is rescaled so
      that its L2 norm matches that of the acceleration-loss gradient tree
      before combining, preventing one term from dominating purely due to
      magnitude differences.
    - The implementation assumes the model returns a dict containing:
        - ``"acceleration"`` when called as ``model(x)``
        - ``"potential"``    when called as ``model(q_flat, mode="potential")``
    - Orbit arrays ``orbit_q`` and ``orbit_p`` are expected to already be in
      the model's scaled / nondimensional coordinates/velocities.

    Parameters
    ----------
    model
        NNX module implementing the static potential model.
    optimizer
        NNX optimizer wrapping the model parameters.
    x
        Batch of input positions, shape ``(N, 3)`` (scaled coordinates).
    a_true
        True accelerations at ``x``, shape ``(N, 3)`` (scaled accelerations).
    lambda_rel
        Weight for the relative-error term in the acceleration loss.
    orbit_q
        Orbit positions over time, shape ``(B, T, 3)``.
        Required when ``target`` is not ``"acceleration"``.
    orbit_p
        Orbit momenta (unit test mass) over time, shape ``(B, T, 3)``.
        Required when ``target`` is not ``"acceleration"``.
    lambda_E
        Maximum weight applied to the orbit-energy loss (reached at the end of
        the ramp).
    target
        Loss configuration; one of ``"acceleration"``, ``"orbit_energy"``,
        ``"mixed"``, ``"mixed_dev_from_initial"``.
    step
        Current training step (0-indexed). Used to compute ramp progress.
    total_steps
        Total number of training steps. Used to compute ramp progress.
    ramp_kind
        Schedule shape for ramping ``lambda_E``:
        ``"linear"``, ``"cosine"``, or ``"sigmoid"``.
    ramp_sharp
        Sharpness parameter for the sigmoid ramp (ignored otherwise).
    balance_grads
        If ``True``, rescale the energy-loss gradient tree so its L2 norm
        matches that of the acceleration-loss gradient tree before summing.

    Returns
    -------
    loss
        Scalar JAX array (shape ``()``) containing the combined loss for this
        step.

    Raises
    ------
    ValueError
        If ``target`` requires orbit inputs but ``orbit_q`` or ``orbit_p``
        is ``None``.
    """
    if target != "acceleration" and (orbit_q is None or orbit_p is None):
        raise ValueError(f"target='{target}' requires orbit_q and orbit_p (got None).")

    # ------------------------------------------------------------------
    # Training ramp helper
    # ------------------------------------------------------------------
    def _ramp(progress: Array) -> Array:
        p = jnp.clip(progress, 0.0, 1.0)
        if ramp_kind == "linear":
            return p
        elif ramp_kind == "cosine":
            return 0.5 * (1.0 - jnp.cos(jnp.pi * p))
        else:  # "sigmoid"
            return jax.nn.sigmoid(ramp_sharp * (p - 0.5))

    progress = jnp.asarray(step, dtype=jnp.float32) / jnp.maximum(
        1.0, jnp.asarray(total_steps - 1, dtype=jnp.float32)
    )
    wE = lambda_E * _ramp(progress)   # effective energy weight this step

    # ------------------------------------------------------------------
    # Split model into trainable vs frozen state (NNX pattern)
    # ------------------------------------------------------------------
    graphdef, train_state, frozen_state = nnx.split(model, optimizer.wrt, ...)

    # ------------------------------------------------------------------
    # Per-objective loss functions (take train_state, return scalar)
    # ------------------------------------------------------------------
    def _acc_loss(ts: nnx.State) -> Array:
        m = nnx.merge(graphdef, ts, frozen_state)
        a_pred = m(x)["acceleration"]          # (N, 3)
        diff      = a_pred - a_true
        diff_norm = jnp.linalg.norm(diff,   axis=1)   # (N,)
        true_norm = jnp.linalg.norm(a_true, axis=1)   # (N,)
        eps = 1e-10
        return jnp.mean(diff_norm + lambda_rel * (diff_norm / (true_norm + eps)))

    def _orbit_energy(ts: nnx.State, oq: Array, op: Array) -> Array:
        """Return total specific energy E(t), shape ``(B, T)``."""
        m = nnx.merge(graphdef, ts, frozen_state)
        B, T, _ = oq.shape
        T_ke    = 0.5 * jnp.sum(op ** 2, axis=-1)          # (B, T)
        q_flat  = oq.reshape(B * T, 3)
        Phi     = m(q_flat, mode="potential")["potential"].reshape(B, T)
        return T_ke + Phi

    def _E_loss_std(ts: nnx.State) -> Array:
        """Std-only energy-conservation loss (Linen `orbit_E_loss`)."""
        assert orbit_q is not None and orbit_p is not None
        E = _orbit_energy(ts, orbit_q, orbit_p)             # (B, T)
        std_E = jnp.std(E - jnp.mean(E, axis=1, keepdims=True), axis=1)
        return jnp.mean(std_E ** 2)

    def _E_loss_dev_from_initial(ts: nnx.State) -> Array:
        """Fractional energy drift relative to E(t=0)."""
        assert orbit_q is not None and orbit_p is not None
        E  = _orbit_energy(ts, orbit_q, orbit_p)            # (B, T)
        E0 = E[:, 0:1]                                      # (B, 1)
        return jnp.mean(((E - E0) / (jnp.abs(E0) + 1e-8)) ** 2)

    # ------------------------------------------------------------------
    # Gradient-norm helper
    # ------------------------------------------------------------------
    def _tree_l2(tree: nnx.State) -> Array:
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.sqrt(sum(jnp.sum(l * l) for l in leaves))

    # ------------------------------------------------------------------
    # Compute loss + gradients
    # ------------------------------------------------------------------
    if target == "acceleration":
        loss, grads = nnx.value_and_grad(_acc_loss)(train_state)

    elif target == "orbit_energy":
        # Pure energy loss (no acceleration term)
        loss, grads = nnx.value_and_grad(_E_loss_std)(train_state)

    else:
        # "mixed" or "mixed_dev_from_initial" — compute both grad trees
        loss_a, grad_a = nnx.value_and_grad(_acc_loss)(train_state)

        if target == "mixed_dev_from_initial":
            loss_e, grad_e = nnx.value_and_grad(_E_loss_dev_from_initial)(train_state)
        else:  # "mixed"
            loss_e, grad_e = nnx.value_and_grad(_E_loss_std)(train_state)

        # Optional gradient-norm balancing
        if balance_grads:
            na    = _tree_l2(grad_a) + 1e-12
            ne    = _tree_l2(grad_e) + 1e-12
            scale = jax.lax.stop_gradient(na / ne)
        else:
            scale = 1.0

        loss  = loss_a + wE * loss_e
        grads = jax.tree_util.tree_map(
            lambda ga, ge: ga + wE * scale * ge,
            grad_a, grad_e,
        )

    optimizer.update(model, grads)
    return loss


@nnx.jit
def train_step_node(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    tx_cart: Array,
    a_true: Array,
    *,
    lambda_rel: float = 1.0,
) -> Array:
    """Optimization step for NODE with acceleration training objective.

    Perform one optimizer step for a time-dependent (Neural ODE / NODE-style)
    model using an acceleration-only training objective.

    The loss is the mean of an absolute error term plus a relative-error term:

        L = mean( ||a_pred - a_true|| + lambda_rel * ||a_pred - a_true|| /
        (||a_true|| + eps) )

    Notes
    -----
    - This function is JIT-compiled using ``nnx.jit``.
    - The model and optimizer are mutated in place (NNX pattern).
    - The implementation assumes the model returns a dict containing
      the key ``"acceleration"`` when called as ``model(tx_cart)``.
    - ``tx_cart`` is assumed to be concatenated time + Cartesian position,
      typically shaped ``(N, 4)`` with columns ``[t, x, y, z]``.

    Parameters
    ----------
    model
        NNX module implementing the time-dependent potential model.
    optimizer
        NNX optimizer wrapping the model parameters.
    tx_cart
        Batch of model inputs containing time and Cartesian coordinates.
        Recommended shape ``(N, 4)`` with columns ``[t, x, y, z]``.
    a_true
        True accelerations corresponding to the positions (and times) in
        ``tx_cart``. Shape ``(N, 3)``.
    lambda_rel
        Weight applied to the relative-error component of the loss. Larger
        values emphasize fractional error in regions where ``||a_true||`` is
        small.

    Returns
    -------
    loss
        A scalar JAX array (shape ``()``) with the loss value for this step.

    """

    def loss_fn(model: nnx.Module) -> Array:
        """Acceleration-only objective.

        Parameters
        ----------
        model
            The NNX model.

        Returns
        -------
        loss : Array
            Scalar loss (shape ``()``).

        """
        outputs: dict[str, Array] = model(tx_cart)
        a_pred = outputs["acceleration"]  # (N, 3)

        diff = a_pred - a_true
        diff_norm = jnp.linalg.norm(diff, axis=1)  # (N,)
        a_true_norm = jnp.linalg.norm(a_true, axis=1)  # (N,)

        eps = 1e-10
        return jnp.mean(diff_norm + lambda_rel * (diff_norm / (a_true_norm + eps)))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def train_model_static(
    model: nnx.Module,
    tx: optax.GradientTransformation,
    x_train: Array,
    a_train: Array,
    num_epochs: int,
    *,
    target: StaticTarget = "acceleration",
    log_every: int = 100,
    lambda_rel: float = 1.0,
    optimizer: nnx.Optimizer | None = None,
    orbit_q: Array | None = None,
    orbit_p: Array | None = None,
    lambda_E: float = 5.0,
    train_dict: Mapping[StaticTarget, int] | None = None,
    ramp_kind: Literal["linear", "cosine", "sigmoid"] = "cosine",
    ramp_sharp: float = 10.0,
    balance_grads: bool = True,
) -> dict[str, Any]:
    """Train a static potential model for a specified number of epochs.

    This function is a simple training driver that repeatedly calls
    :func:`train_step_static` on the full training arrays.
    It supports either:
      - a single training objective for ``num_epochs`` (via ``target``), or
      - a staged schedule of objectives (via ``train_dict``).

    The ramp and gradient-balancing parameters are forwarded directly to
    :func:`train_step_static`. The ramp progress is computed globally across
    all stages so that ``lambda_E`` reaches its full value only at the very
    last step of the entire training run, regardless of how many stages are
    used.

    Parameters
    ----------
    model
        An NNX ``Module`` implementing the static model (already initialized).
    tx
        Optax optimizer transformation used to update parameters.
    x_train
        Training positions, shape ``(N, 3)`` in scaled coordinates.
    a_train
        True accelerations at ``x_train``, shape ``(N, 3)`` in scaled units.
    num_epochs
        Total number of epochs to train if ``train_dict`` is not provided.
    target
        Default loss target when ``train_dict`` is not provided. One of
        ``"acceleration"``, ``"orbit_energy"``, ``"mixed"``, or
        ``"mixed_dev_from_initial"``.
    log_every
        Print a progress line every ``log_every`` epochs.
    lambda_rel
        Weight for the relative-error term in the acceleration loss.
    optimizer
        Optional pre-initialized ``nnx.Optimizer``. If not provided, a new
        optimizer is created from the model and ``tx``.
    orbit_q
        Scaled orbit positions for orbit-energy loss, shape ``(B, T, 3)``.
        Required when any stage uses a non-acceleration target.
    orbit_p
        Scaled orbit momenta for orbit-energy loss, shape ``(B, T, 3)``.
        Required when any stage uses a non-acceleration target.
    lambda_E
        Maximum weight for the orbit-energy loss component (reached at end of
        the ramp).
    train_dict
        Optional staged training schedule mapping targets to epoch counts, e.g.
        ``{"acceleration": 2000, "mixed": 1000}``.
    ramp_kind
        Schedule shape for ramping ``lambda_E`` from 0 to its full value:
        ``"linear"``, ``"cosine"``, or ``"sigmoid"``.
    ramp_sharp
        Sharpness parameter for the sigmoid ramp (ignored for other kinds).
    balance_grads
        If ``True``, rescale the energy-loss gradient tree so its L2 norm
        matches the acceleration-loss gradient tree before summing.

    Returns
    -------
    out : dict
        Dictionary with:
        - ``"model"``:     the trained model (mutated in place).
        - ``"optimizer"``: the final optimizer state.
        - ``"epochs"``:    list of final epoch indices completed per stage.
        - ``"losses"``:    list of final loss values per stage (JAX scalars).
    """
    epochs: list[int] = []
    losses: list[Array] = []

    if optimizer is None:
        optimizer = create_optimizer(model, tx)

    schedule: Mapping[StaticTarget, int] = (
        train_dict if train_dict is not None else {target: num_epochs}
    )

    # Total steps across all stages
    total_steps: int = int(sum(schedule.values()))
    global_step: int = 0

    for stage_target, n_epochs in schedule.items():
        log.info("Training for %d epochs on target: %s", n_epochs, stage_target)

        for epoch in range(n_epochs):
            loss = train_step_static(
                model,
                optimizer,
                x_train,
                a_train,
                target=stage_target,
                lambda_rel=lambda_rel,
                orbit_q=orbit_q,
                orbit_p=orbit_p,
                lambda_E=lambda_E,
                # ramp / balancing
                step=global_step,
                total_steps=total_steps,
                ramp_kind=ramp_kind,
                ramp_sharp=ramp_sharp,
                balance_grads=balance_grads,
            )
            global_step += 1

            if log_every > 0 and (epoch % log_every == 0):
                log.info("Epoch %d, Loss: %.6f", epoch + 1, float(loss))

        epochs.append(epoch)
        losses.append(loss)

    return {"model": model, "optimizer": optimizer, "epochs": epochs, "losses": losses}


def train_model_node(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x_train: Array,
    a_train: Array,
    num_epochs: int,
    *,
    lambda_rel: float = 1.0,
    log_every: int = 1000,
) -> dict[str, Any]:
    """Train a time-dependent model.

    This function controls the training of a time-dependent model by
    repeatedly calling `train_step_node`.

    Parameters
    ----------
    model
        The NNX model to train.
    optimizer
        The NNX optimizer wrapping the model parameters.
    x_train
        Training data (concatenated time and position).
    a_train
        Training accelerations.
    num_epochs
        The number of epochs to train for.
    lambda_rel
        Weight for the relative-error term in the loss function.
    log_every
        The interval at which to log training progress.

    Returns
    -------
    dict
        A dictionary containing the trained model, optimizer, epochs, and losses.

    """
    epochs = []
    losses = []

    for epoch in range(num_epochs):
        loss = train_step_node(model, optimizer, x_train, a_train, lambda_rel = lambda_rel)
        if epoch % log_every == 0:
            log.info("Epoch %d, Loss: %.6f", epoch, loss)
        epochs.append(epoch)
        losses.append(loss)
    return {"model": model, "optimizer": optimizer, "epochs": epochs, "losses": losses}


def train_model_with_trainable_analytic_layer(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    train_step: TrainStepFn,
    x_train: Array,
    a_train: Array,
    num_epochs: int,
    *,
    log_every: int = 1000,
    lambda_rel: float = 1.0,
) -> dict[str, Any]:
    """Train a model with a trainable analytic component.

    This function drives a training loop over ``num_epochs`` iterations using
    the provided ``train_step`` callable. In addition to tracking the loss and
    epoch index, it records the parameters associated with the
    ``"trainable_analytic_layer"`` subtree of the model parameters at every
    step.

    The primary use case is monitoring how an analytic component evolves during
    training when combined with a learned correction.

    Parameters
    ----------
    model
        NNX module containing the model (with trainable_analytic_layer).
    optimizer
        NNX optimizer wrapping the model parameters.
    train_step
        Callable implementing a single training step. It must accept:

            ``(model, optimizer, x_train, a_train, lambda_rel=...)``
        and return ``loss``, a scalar JAX array.
    x_train
        Training input positions. Shape ``(N, 3)`` (scaled coordinates).
    a_train
        True accelerations corresponding to ``x_train``.  Shape ``(N, 3)``
        (scaled accelerations).
    num_epochs
        Number of training epochs to run.
    log_every
        Interval (in epochs) at which to print progress information.
    lambda_rel
        Weight for relative error term in the loss function.

    Returns
    -------
    out : dict
        Dictionary containing:

        - ``"model"`` : nnx.Module
          The trained model (mutated in place).

        - ``"optimizer"`` : nnx.Optimizer
          The optimizer state.

        - ``"final_loss"`` : float
          Final loss value after training.

        - ``"final_analytic_params"`` : dict | None
          Final parameter values from trainable_analytic_layer (if present).

    Raises
    ------
    AttributeError
        If ``trainable_analytic_layer`` is not present in the model.

    """
    if log_every > 0:
        log.info("Training for %d epochs...", num_epochs)

    # Training loop
    loss = jnp.array(0.0)  # Initialize for type checker
    for epoch in range(num_epochs):
        loss = train_step(model, optimizer, x_train, a_train, lambda_rel=lambda_rel)
        if log_every > 0 and (epoch % log_every == 0):
            log.info("Epoch %d, Loss: %.6f", epoch + 1, float(loss))

    # Extract final analytic params
    final_analytic_params = None
    analytic_layer = model.trainable_analytic_layer
    if analytic_layer is not None:
        _, layer_state = nnx.split(analytic_layer)
        final_analytic_params = layer_state.to_pure_dict()

    return {
        "model": model,
        "optimizer": optimizer,
        "final_loss": float(loss),
        "final_analytic_params": final_analytic_params,
    }


def alternate_training(
    train_step: TrainStepFn,
    x_train: Array,
    a_train: Array,
    model: nnx.Module,
    num_epochs_stage1: int,
    num_epochs_stage2: int,
    cycles: int,
    param_list: Sequence[str],
    optimizer1: nnx.Optimizer,
    optimizer2: nnx.Optimizer,
    *,
    log_every: int = 1000,
    lambda_rel: float = 1.0,
) -> dict[str, Any]:
    """Cycle optimization between two parameter groups (analytic vs neural).

    This function implements a two-stage training loop used when a model
    contains (i) a trainable analytic component (e.g., a baseline potential /
    fusing layer) and (ii) a learned neural component. The training proceeds in
    cycles:

      Stage 1 (analytic focus):
          - Create an optimizer using ``tx_1``.
          - Train for ``num_epochs_stage1`` epochs.
      Stage 2 (NN focus):
          - Create an optimizer using ``tx_2``.
          - Train for ``num_epochs_stage2`` epochs.

    During training, this function tracks:
      - the loss history across all stages and cycles, and
      - selected parameter values from the trainable_analytic_layer
        over train time

    Parameters
    ----------
    train_step
        Callable implementing one training step. It must accept:
        ``(model, optimizer, x_train, a_train, lambda_rel=...)``
        and return ``loss``.
    x_train
        Training positions, typically shape ``(N, 3)`` (scaled).
    a_train
        True accelerations at ``x_train``, shape ``(N, 3)`` (scaled).
    model
        NNX module to train (already initialized with parameters).
    num_epochs_stage1
        Number of epochs to run in Stage 1 for each cycle (excluding burn-in).
    num_epochs_stage2
        Number of epochs to run in Stage 2 for each cycle.
    cycles
        Number of times to repeat the Stage1 / Stage2 cycle.
    param_list
        Names of parameters inside the trainable_analytic_layer to
        record into history.  Each entry is treated as a key into the params dict.
    optimizer1
        NNX optimizer for Stage 1 (trainable_analytic_layer only).
    optimizer2
        NNX optimizer for Stage 2 (all parameters).
    log_every
        Interval (in epochs) at which to log training progress.
    lambda_rel
        Weight for relative error term in the loss function.


    Returns
    -------
    out : dict
        Dictionary containing:
        - ``"model"``: nnx.Module
            The trained model (mutated in place).
        - ``"history"``: dict
            Contains:
              - ``"final_losses"``: list[float] - final loss from each stage
              - ``"total_epochs"``: int - total epochs trained
              - For each name in ``param_list``: final learned value.

    Notes
    -----
    - This function does not freeze parameter subsets; the actual freezing
      behavior must be implemented inside ``train_step`` (by controlling which
      params are updated by the optimizer).
    - The NNX model should already be initialized with parameters.
    - For performance, per-epoch loss/param tracking is removed. Only final
      values per stage are recorded.

    Raises
    ------
    ValueError
        If epoch counts or cycles are non-positive.
    AttributeError
        If ``trainable_analytic_layer`` is missing when tracking analytic
        parameters.

    """
    if cycles <= 0:
        raise ValueError("cycles must be positive.")
    if num_epochs_stage1 <= 0 or num_epochs_stage2 <= 0:
        raise ValueError("Stage epoch counts must be positive.")

    total_epochs = cycles * (num_epochs_stage1 + num_epochs_stage2)
    final_losses: list[float] = []
    final_params: dict[str, Any] = dict.fromkeys(param_list)

    for cycle in range(cycles):
        # === Stage 1 ===
        log.info("=== Starting Cycle %d / %d: Stage 1 ===", cycle + 1, cycles)
        output1 = train_model_with_trainable_analytic_layer(
            model,
            optimizer1,
            train_step,
            x_train,
            a_train,
            num_epochs_stage1,
            log_every=log_every,
            lambda_rel=lambda_rel,
        )
        final_losses.append(output1["final_loss"])

        # Log learned analytic params at end of stage
        if output1["final_analytic_params"] is not None:
            for param in param_list:
                val = output1["final_analytic_params"].get(param)
                if val is not None:
                    final_params[param] = val
                    log.info(
                        "Learned %s in cycle %d stage 1: %s", param, cycle + 1, val
                    )

        # === Stage 2 ===
        log.info("=== Starting Cycle %d / %d: Stage 2 ===", cycle + 1, cycles)

        output2 = train_model_with_trainable_analytic_layer(
            model,
            optimizer2,
            train_step,
            x_train,
            a_train,
            num_epochs_stage2,
            log_every=log_every,
            lambda_rel=lambda_rel,
        )
        final_losses.append(output2["final_loss"])

        # Update final params from stage 2
        if output2["final_analytic_params"] is not None:
            for param in param_list:
                val = output2["final_analytic_params"].get(param)
                if val is not None:
                    final_params[param] = val

    history = {
        "final_losses": final_losses,
        "total_epochs": total_epochs,
        **final_params,
    }

    return {"model": model, "history": history}
