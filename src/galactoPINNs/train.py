"""Training utilities for galactoPINNs models."""

__all__ = (
    "alternate_training",
    "create_optimizer",
    "train_model_state_node",
    "train_model_static",
    "train_model_with_trainable_analytic_layer",
    "train_step_node",
    "train_step_static",
)

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

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


@nnx.jit(static_argnames=("target",))
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
    std_weight: float = 10.0,
    target: Literal["acceleration", "orbit_energy", "mixed"] = "acceleration",
) -> Array:
    """Perform one optimizer step for a static potential model.

    This function computes a scalar training loss, differentiates it with
    respect to the model parameters, and applies the resulting gradients
    via the optimizer. The loss target is controlled by ``target``:

    - ``"acceleration"``: matches predicted accelerations to ``a_true`` at input
      positions ``x``.
    - ``"orbit_energy"``: penalizes non-conservation of total specific energy
      along pre-computed orbit trajectories.
    - ``"mixed"``: uses a weighted sum of the orbit-energy loss and the
      acceleration loss.

    Notes
    -----
    - This function is JIT-compiled using ``nnx.jit``.
    - The model and optimizer are mutated in place (NNX pattern).
    - The implementation assumes the model returns a dict containing:
        - ``"acceleration"`` when called as ``model(x)``
        - ``"potential"`` when called as ``model(q_flat, mode="potential")``
    - Orbit arrays ``orbit_q`` and ``orbit_p`` are expected to already be in the
      model's scaled / nondimensional coordinates/velocities

    Parameters
    ----------
    model
        NNX module implementing the static potential model.
    optimizer
        NNX optimizer wrapping the model parameters.
    x
        Batch of input positions. Shape ``(N, 3)`` (scaled coordinates).
    a_true
        True accelerations at ``x``. Shape ``(N, 3)`` (scaled accelerations).
    lambda_rel
        Weight applied to a relative-error term in the acceleration loss.
        Larger values emphasize fractional error in regions where |a_true| is small.
    orbit_q
        Orbit positions over time for orbit-energy training.
        Required if ``target`` is ``"orbit_energy"`` or ``"mixed"``.
        Shape ``(B, T, 3)``.
    orbit_p
        Orbit momenta (assuming unit test mass) over time for orbit-energy training.
        Required if ``target`` is ``"orbit_energy"`` or ``"mixed"``.
        Shape ``(B, T, 3)``.
    lambda_E
        Weight applied to the orbit-energy loss when ``target="mixed"``.
    std_weight
        Weight applied to the within-orbit energy standard deviation term.
    target
        Loss configuration. One of:
        ``"acceleration"``, ``"orbit_energy"``, or ``"mixed"``.

    Returns
    -------
    loss
        A scalar JAX array (shape ``()``) containing the loss value for this step.

    Raises
    ------
    ValueError
        If ``target`` requires orbit inputs but ``orbit_q`` or ``orbit_p`` is None.

    """
    if target in ("orbit_energy", "mixed") and (orbit_q is None or orbit_p is None):
        msg = f"target='{target}' requires orbit_q and orbit_p (got None)."
        raise ValueError(msg)

    def acc_loss_fn(model: nnx.Module, lambda_rel_: float) -> Array:
        """Acceleration loss with an absolute + relative magnitude term.

        Computes::

            mean( ||a_pred - a_true||
                  + lambda_rel * ||a_pred - a_true|| / (||a_true|| + eps) )

        Returns
        -------
        loss : Array
            Scalar loss (shape ``()``).

        """
        out: dict[str, Array] = model(x)
        a_pred = out["acceleration"]  # (N, 3)

        diff = a_pred - a_true
        diff_norm = jnp.linalg.norm(diff, axis=1)  # (N,)
        a_true_norm = jnp.linalg.norm(a_true, axis=1)  # (N,)

        eps = 1e-10
        return jnp.mean(diff_norm + lambda_rel_ * (diff_norm / (a_true_norm + eps)))

    def orbit_energy(
        model: nnx.Module, orbit_q_scaled: Array, orbit_p_scaled: Array
    ) -> Array:
        """Compute predicted total specific energy along orbits.

        Energy is defined as:
            E(t) = T(t) + Phi(q(t))
        with:
            T(t) = 0.5 * ||v(t)||^2

        Parameters
        ----------
        model
            The NNX model.
        orbit_q_scaled
            Orbit positions (scaled). Shape ``(B, T, 3)``.
        orbit_p_scaled
            Orbit velocities/momenta (scaled). Shape ``(B, T, 3)``.

        Returns
        -------
        E : Array
            Total specific energy along each orbit. Shape ``(B, T)``.

        """
        B, T, _ = orbit_q_scaled.shape
        T_ke = 0.5 * jnp.sum(orbit_p_scaled**2, axis=-1)  # (B, T)

        q_flat = orbit_q_scaled.reshape(B * T, 3)  # (B*T, 3)
        out: dict[str, Array] = model(q_flat, mode="potential")
        Phi_flat = out["potential"]  # (B*T,)
        Phi = Phi_flat.reshape(B, T)  # (B, T)

        return T_ke + Phi

    def orbit_E_loss(
        model: nnx.Module,
        orbit_q_scaled: Array,
        orbit_p_scaled: Array,
        std_weight_: float,
    ) -> Array:
        """Penalize energy non-conservation along each trajectory.

        For each orbit b:
            drift_b = E_b(T_end) - E_b(T_start)
            std_b   = std_t (E_b(t) - mean_t E_b(t))

        Loss:
            mean_b [ drift_b^2 + std_weight * std_b^2 ]

        Parameters
        ----------
        model
            The NNX model.
        orbit_q_scaled
            Orbit positions. Shape ``(B, T, 3)``.
        orbit_p_scaled
            Orbit velocities/momenta. Shape ``(B, T, 3)``.
        std_weight_
            Weight on the within-orbit variance proxy term.

        Returns
        -------
        loss : Array
            Scalar loss (shape ``()``).

        """
        E = orbit_energy(model, orbit_q_scaled, orbit_p_scaled)  # (B, T)
        delta_E = E[:, -1] - E[:, 0]  # (B,)

        # Center each orbit's energy time series before computing std
        E_centered = E - jnp.mean(E, axis=1, keepdims=True)
        std_E = jnp.std(E_centered, axis=1)  # (B,)

        return jnp.mean(delta_E**2 + std_weight_ * std_E**2)

    def total_loss(model: nnx.Module) -> Array:
        """Dispatch the total loss based on ``target``.

        Returns
        -------
        loss : Array
            Scalar loss (shape ``()``).

        """
        if target == "mixed":
            assert orbit_q is not None
            assert orbit_p is not None
            return lambda_E * orbit_E_loss(
                model, orbit_q, orbit_p, std_weight
            ) + acc_loss_fn(model, lambda_rel)

        if target == "orbit_energy":
            assert orbit_q is not None
            assert orbit_p is not None
            return orbit_E_loss(model, orbit_q, orbit_p, std_weight)

        # default: acceleration-only
        return acc_loss_fn(model, lambda_rel)

    loss, grads = nnx.value_and_grad(total_loss)(model)
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
    std_weight: float = 10.0,
    train_dict: Mapping[StaticTarget, int] | None = None,
) -> dict[str, Any]:
    """Train a static potential model for a specified number of epochs.

    This function is a simple training driver that repeatedly calls
    :func:`train_step_static` on the full training arrays
    It supports either:
      - a single training objective for ``num_epochs`` (via ``target``), or
      - a staged schedule of objectives (via ``train_dict``).

    Parameters
    ----------
    model
        An NNX ``Module`` implementing the static model (already initialized).
    tx
        Optax optimizer transformation used to update parameters.
    x_train
        Training positions, typically shape ``(N, 3)`` in scaled coordinates.
    a_train
        True accelerations at ``x_train``, shape ``(N, 3)`` in scaled units.
    num_epochs
        Total number of epochs to train if ``train_dict`` is not provided.
    target
        Default loss target when ``train_dict`` is not provided. One of
        ``"acceleration"``, ``"orbit_energy"``, or ``"mixed"``.
    log_every
        Print a progress line every ``log_every`` epochs.
    lambda_rel
        Weight for the relative-error term in the acceleration loss.
    optimizer
        Optional pre-initialized ``nnx.Optimizer``. If not provided, a new
        optimizer is created from the model and ``tx``.
    orbit_q
        Scaled orbit positions used for orbit-energy loss. Required when any
        stage uses ``"orbit_energy"`` or ``"mixed"``. Expected shape ``(B, T, 3)``.
    orbit_p
        Scaled orbit velocities/momenta used for orbit-energy loss. Required when any
        stage uses ``"orbit_energy"`` or ``"mixed"``. Expected shape ``(B, T, 3)``.
    lambda_E
        Weight for the orbit-energy loss component when target is ``"mixed"``.
    std_weight
        Weight for the within-orbit energy standard deviation term.
    train_dict
        Optional staged training schedule mapping targets to epoch counts, e.g.:
            ``{"acceleration": 2000, "mixed": 1000}``

    Returns
    -------
    out : dict
        Dictionary with:
        - ``"model"``: the trained model
        - ``"optimizer"``: the final optimizer state
        - ``"epochs"``: list of epoch indices completed per stage (ints)
        - ``"losses"``: list of final loss values per stage (JAX scalars)

    """
    epochs: list[int] = []
    losses: list[Array] = []

    if optimizer is None:
        optimizer = create_optimizer(model, tx)

    schedule: Mapping[StaticTarget, int] = (
        train_dict if train_dict is not None else {target: num_epochs}
    )

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
                std_weight=std_weight,
            )
            if log_every > 0 and (epoch % log_every == 0):
                log.info("Epoch %d, Loss: %.6f", epoch + 1, float(loss))

        epochs.append(epoch)
        losses.append(loss)

    return {"model": model, "optimizer": optimizer, "epochs": epochs, "losses": losses}


def train_model_state_node(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x_train: Array,
    a_train: Array,
    num_epochs: int,
    *,
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
        loss = train_step_node(model, optimizer, x_train, a_train)
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
    # Log at start
    if log_every > 0:
        log.info("Training for %d epochs...", num_epochs)

    # Training loop - train_step is already JIT-compiled
    loss = jnp.array(0.0)  # Initialize for type checker
    for epoch in range(num_epochs):
        loss = train_step(model, optimizer, x_train, a_train, lambda_rel=lambda_rel)
        if log_every > 0 and (epoch % log_every == 0):
            log.info("Epoch %d, Loss: %.6f", epoch + 1, float(loss))

    # Extract final analytic params (only once at the end, not every epoch)
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


def initialize_staged_optimizers(
    model: nnx.Module,
    optimizer: optax.GradientTransformation,
) -> tuple[optax.GradientTransformation, optax.GradientTransformation]:
    """Create two Optax `multi_transform` optimizers for staged training.

    This utility builds parameter partitions from a model's parameter
    pytree and returns two optimizer transformations:

    - `tx1` ("AB-only"): trains only parameters under the subtree whose path
      contains `"trainable_analytic_layer"` and freezes all other parameters by
      applying `optax.set_to_zero()` to their updates.

    - `tx2` ("joint training"): trains all parameters (i.e., no freezing).

    Parameters
    ----------
    model
        NNX module (already initialized with parameters).
    optimizer
        The Optax optimizer to use for trainable parameters (e.g.,
        `optax.adam(...)`).  Frozen parameters receive `optax.set_to_zero()`
        regardless of this choice.

    Returns
    -------
    tx1, tx2
        A pair of Optax gradient transformations:

        - `tx1`: multi-transform optimizer that trains only
          `"trainable_analytic_layer"` params
        - `tx2`: multi-transform optimizer that trains all params

    Examples
    --------
    .. skip: start

    >>> tx1, tx2 = initialize_staged_optimizers(model, optax.adam(1e-3))
    >>> optimizer1 = create_optimizer(model, tx1)  # trains analytic only
    >>> optimizer2 = create_optimizer(model, tx2)  # trains all

    .. skip: end

    """
    # Get the parameter state and extract pure arrays for partition labels.
    # nnx.Optimizer uses nnx.pure() internally, which extracts raw arrays from
    # the State. The partition labels must match this pure array structure,
    # not the State structure with nnx.Param wrappers.
    param_state = nnx.state(model, nnx.Param)
    # Extract the pure array structure that optax will see
    pure_params = nnx.pure(param_state)

    partition_optimizers: Mapping[str, optax.GradientTransformation] = {
        "trainable": optimizer,
        "frozen": optax.set_to_zero(),
    }

    # Build partition labels using tree_map_with_path to match pure array structure
    def make_label_fn(train_path_substr: str | None) -> Callable[[Any, Any], str]:
        """Create a labeling function for tree_map_with_path."""

        def label_fn(path: Any, _: Any) -> str:
            # Convert path elements to string for matching
            path_str = "/".join(
                str(p.key) if hasattr(p, "key") else str(p) for p in path
            )
            if train_path_substr is None or train_path_substr in path_str:
                return "trainable"
            return "frozen"

        return label_fn

    # Stage 1: train only trainable_analytic_layer (use pure_params for structure)
    param_partitions_AB_only = jax.tree_util.tree_map_with_path(
        make_label_fn("trainable_analytic_layer"), pure_params
    )

    # Stage 2: train all parameters
    param_partitions_joint = jax.tree_util.tree_map_with_path(
        make_label_fn(None), pure_params
    )

    tx1 = optax.multi_transform(partition_optimizers, param_partitions_AB_only)
    tx2 = optax.multi_transform(partition_optimizers, param_partitions_joint)
    return tx1, tx2


def alternate_training(
    train_step: TrainStepFn,
    x_train: Array,
    a_train: Array,
    model: nnx.Module,
    num_epochs_stage1: int,
    num_epochs_stage2: int,
    cycles: int,
    param_list: Sequence[str],
    tx_1: optax.GradientTransformation,
    tx_2: optax.GradientTransformation,
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
    tx_1
        Optax optimizer used in Stage 1 (analytic-focused phase).
    tx_2
        Optax optimizer used in Stage 2 (NN-focused phase).
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

        optimizer1 = create_optimizer(model, tx_1)
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

        optimizer2 = create_optimizer(model, tx_2)
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
