import jax
from functools import partial
from flax.training.train_state import TrainState
import jax.numpy as jnp
import jax.random as jr
from flax.linen import Module
import optax
from typing import Any, Dict, Literal, Optional, Tuple, Mapping, List, Callable, Sequence, TypeAlias, Protocol
from flax import traverse_util

##### Type aliases and protocols
Array = jax.Array
StaticTarget = Literal["acceleration", "orbit_energy", "mixed"]
TrainStepFn = Callable[..., Tuple[TrainState, Array]]
PyTree: TypeAlias = Any

class _FlaxModuleWithInit(Protocol):
    """Minimal protocol for Flax modules used here."""
    def init(self, rngs: Any, *args: Any, **kwargs: Any) -> Mapping[str, Any]: ...

#############

__all__ = (
    "train_step_static",
    "train_step_node",
    "train_model_static",
    "train_model_state_node",
    "train_model_with_trainable_analytic_layer",
    "alternate_training",
)



@partial(jax.jit, static_argnames=("target",))
def train_step_static(
    state: TrainState,
    x: Array,
    a_true: Array,
    *,
    target: Literal["acceleration", "orbit_energy", "mixed"] = "acceleration",
    lambda_rel: float = 1.0,
    orbit_q: Optional[Array] = None,
    orbit_p: Optional[Array] = None,
    lambda_E: float = 5.0,
    std_weight: float = 10.0,
) -> Tuple[TrainState, Array]:
    """
    Perform one optimizer step for a static potential model.

    This function computes a scalar training loss, differentiates it with respect
    to the model parameters stored in ``state.params``, and applies the resulting
    gradients via ``state.apply_gradients``. The loss target is controlled by ``target``:

    - ``"acceleration"``: matches predicted accelerations to ``a_true`` at input
      positions ``x``.
    - ``"orbit_energy"``: penalizes non-conservation of total specific energy
      along pre-computed orbit trajectories.
    - ``"mixed"``: uses a weighted sum of the orbit-energy loss and the
      acceleration loss.

    Notes
    -----
    - This function is JIT-compiled. The ``target`` argument is marked static
      via ``static_argnames``.
    - The implementation assumes ``state.apply_fn`` returns a dict containing:
        - ``"acceleration"`` when called as ``apply_fn({"params": params}, x)``
        - ``"potential"`` when called as
          ``apply_fn({"params": params}, q_flat, mode="potential")``
    - Orbit arrays ``orbit_q`` and ``orbit_p`` are expected to already be in the
      model’s scaled / nondimensional coordinates/velocities

    Parameters
    ----------
    state
        Flax ``TrainState`` holding:
        - ``params``: parameter PyTree
        - ``tx``: optimizer transformation
        - ``apply_fn``: model forward function
    x
        Batch of input positions. Shape ``(N, 3)`` (scaled coordinates).
    a_true
        True accelerations at ``x``. Shape ``(N, 3)`` (scaled accelerations).
    target
        Loss configuration. One of:
        ``"acceleration"``, ``"orbit_energy"``, or ``"mixed"``.
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

    Returns
    -------
    new_state, loss
        ``new_state`` is the updated TrainState after applying gradients.
        ``loss`` is a scalar JAX array (shape ``()``) containing the loss value
        for this step.

    Raises
    ------
    ValueError
        If ``target`` requires orbit inputs but ``orbit_q`` or ``orbit_p`` is None.

    """

    if target in ("orbit_energy", "mixed"):
        if orbit_q is None or orbit_p is None:
            raise ValueError(f"target='{target}' requires orbit_q and orbit_p (got None).")

    def acc_loss_fn(params: PyTree, lambda_rel_: float) -> Array:
        """
        Acceleration loss with an absolute + relative magnitude term.

        Computes:
            mean( ||a_pred - a_true|| + lambda_rel * ||a_pred - a_true|| / (||a_true|| + eps) )

        Returns
        -------
        loss : Array
            Scalar loss (shape ``()``).
        """
        out: Dict[str, Array] = state.apply_fn({"params": params}, x)
        a_pred = out["acceleration"]  # (N, 3)

        diff = a_pred - a_true
        diff_norm = jnp.linalg.norm(diff, axis=1)          # (N,)
        a_true_norm = jnp.linalg.norm(a_true, axis=1)      # (N,)

        eps = 1e-10
        return jnp.mean(diff_norm + lambda_rel_ * (diff_norm / (a_true_norm + eps)))

    def orbit_energy(params: PyTree, orbit_q_scaled: Array, orbit_p_scaled: Array) -> Array:
        """
        Compute predicted total specific energy along orbits.

        Energy is defined as:
            E(t) = T(t) + Phi(q(t))
        with:
            T(t) = 0.5 * ||v(t)||^2

        Parameters
        ----------
        params
            Model parameters PyTree.
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
        out: Dict[str, Array] = state.apply_fn({"params": params}, q_flat, mode="potential")
        Phi_flat = out["potential"]  # (B*T,)
        Phi = Phi_flat.reshape(B, T)  # (B, T)

        return T_ke + Phi

    def orbit_E_loss(params: PyTree, orbit_q_scaled: Array, orbit_p_scaled: Array, std_weight_: float) -> Array:
        """
        Penalize energy non-conservation along each trajectory.

        For each orbit b:
            drift_b = E_b(T_end) - E_b(T_start)
            std_b   = std_t (E_b(t) - mean_t E_b(t))

        Loss:
            mean_b [ drift_b^2 + std_weight * std_b^2 ]

        Parameters
        ----------
        params
            Model parameters PyTree.
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
        E = orbit_energy(params, orbit_q_scaled, orbit_p_scaled)  # (B, T)
        delta_E = E[:, -1] - E[:, 0]                              # (B,)

        # Center each orbit’s energy time series before computing std
        E_centered = E - jnp.mean(E, axis=1, keepdims=True)
        std_E = jnp.std(E_centered, axis=1)                       # (B,)

        return jnp.mean(delta_E**2 + std_weight_ * std_E**2)

    def total_loss(params: PyTree) -> Array:
        """
        Dispatch the total loss based on ``target``.

        Returns
        -------
        loss : Array
            Scalar loss (shape ``()``).
        """
        if target == "mixed":
            assert orbit_q is not None and orbit_p is not None
            return lambda_E * orbit_E_loss(params, orbit_q, orbit_p, std_weight) + acc_loss_fn(params, lambda_rel)

        if target == "orbit_energy":
            assert orbit_q is not None and orbit_p is not None
            return orbit_E_loss(params, orbit_q, orbit_p, std_weight)

        # default: acceleration-only
        return acc_loss_fn(params, lambda_rel)

    loss, grads = jax.value_and_grad(total_loss)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss



@jax.jit
def train_step_node(
    state: TrainState,
    tx_cart: Array,
    a_true: Array,
    *,
    lambda_rel: float = 1.0,
) -> Tuple[TrainState, Array]:
    """
    Perform one optimizer step for a time-dependent (Neural ODE / NODE-style) model
    using an acceleration-only training objective.

    The loss is the mean of an absolute error term plus a relative-error term:

        L = mean( ||a_pred - a_true|| + lambda_rel * ||a_pred - a_true|| / (||a_true|| + eps) )

    Notes
    -----
    - The implementation assumes ``state.apply_fn`` returns a dict containing
      the key ``"acceleration"`` when called as:
          ``state.apply_fn({"params": params}, tx_cart)``.
    - ``tx_cart`` is assumed to be concatenated time + Cartesian position,
      typically shaped ``(N, 4)`` with columns ``[t, x, y, z]``.

    Parameters
    ----------
    state
        Flax ``TrainState`` holding parameters, optimizer state, and the model
        apply function.
    tx_cart
        Batch of model inputs containing time and Cartesian coordinates.
        Recommended shape ``(N, 4)`` with columns ``[t, x, y, z]``.
    a_true
        True accelerations corresponding to the positions (and times) in
        ``tx_cart``. Shape ``(N, 3)``.
    lambda_rel
        Weight applied to the relative-error component of the loss. Larger
        values emphasize fractional error in regions where ``||a_true||`` is small.

    Returns
    -------
    new_state, loss
        ``new_state`` is the updated TrainState after applying gradients.
        ``loss`` is a scalar JAX array (shape ``()``) with the loss value for
        this step.
    """

    def loss_fn(params: PyTree) -> Array:
        """
        Acceleration-only objective.

        Parameters
        ----------
        params
            Model parameters PyTree.

        Returns
        -------
        loss : Array
            Scalar loss (shape ``()``).
        """
        outputs: Dict[str, Array] = state.apply_fn({"params": params}, tx_cart)
        a_pred = outputs["acceleration"]  # (N, 3)

        diff = a_pred - a_true
        diff_norm = jnp.linalg.norm(diff, axis=1)      # (N,)
        a_true_norm = jnp.linalg.norm(a_true, axis=1)  # (N,)

        eps = 1e-10
        return jnp.mean(diff_norm + lambda_rel * (diff_norm / (a_true_norm + eps)))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


def train_model_static(
    model: Module,
    optimizer: optax.GradientTransformation,
    x_train: Array,
    a_train: Array,
    num_epochs: int,
    *,
    target: StaticTarget = "acceleration",
    log_every: int = 100,
    lambda_rel: float = 1.0,
    init_state: Optional[TrainState] = None,
    orbit_q: Optional[Array] = None,
    orbit_p: Optional[Array] = None,
    lambda_E: float = 5.0,
    std_weight: float = 10.0,
    train_dict: Optional[Mapping[StaticTarget, int]] = None,
) -> Dict[str, Any]:
    """
    Train a static potential model for a specified number of epochs.

    This function is a simple training driver that repeatedly calls
    :func:`train_step_static` on the full training arrays
    It supports either:
      - a single training objective for ``num_epochs`` (via ``target``), or
      - a staged schedule of objectives (via ``train_dict``).

    Parameters
    ----------
    model
        A Flax ``Module`` implementing the static model. Its ``apply`` function is
        used as the TrainState apply_fn.
    optimizer
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
    init_state
        Optional pre-initialized ``TrainState``. If not provided, a new TrainState
        is created using ``model.init`` with a fixed PRNG key (0).
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
        - ``"state"``: final TrainState
        - ``"epochs"``: list of epoch indices completed per stage (ints)
        - ``"losses"``: list of final loss values per stage (JAX scalars)
    """
    epochs: List[int] = []
    losses: List[Array] = []

    if init_state is None:
        state = TrainState.create(
            apply_fn=model.apply,
            params=model.init(jax.random.PRNGKey(0), x_train)["params"],
            tx=optimizer,
        )
    else:
        state = init_state

    schedule: Mapping[StaticTarget, int] = train_dict if train_dict is not None else {target: num_epochs}

    for stage_target, n_epochs in schedule.items():
        print(f"Training for {n_epochs} epochs on target: {stage_target}")

        for epoch in range(n_epochs):
            state, loss = train_step_static(
                state,
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
                print(f"Epoch {epoch + 1}, Loss: {float(loss):.6f}")

        epochs.append(epoch)
        losses.append(loss)

    return {"state": state, "epochs": epochs, "losses": losses}



def train_model_state_node(
    initial_state, x_train, a_train, num_epochs, log_every=1000
):
    """
    Trains a time-dependent model from an initial state.
    This function controls the training of a time-dependent model by
    repeatedly calling `train_step_node`.

    Parameters
    ----------
        initial_state (TrainState): The initial training state to start from.
        x_train (jax.numpy.ndarray): Training data (concatenated time and position).
        a_train (jax.numpy.ndarray): Training accelerations.
        num_epochs (int): The number of epochs to train for.
        log_every (int): The interval at which to log training progress.

    Returns
    -------
        Dict[str, Any]: A dictionary containing the final `TrainState`, a list of
                        epochs, and a list of losses over time.
    """
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




def train_model_with_trainable_analytic_layer(
    init_state: TrainState,
    train_step: Callable[
        [TrainState, Array, Array],
        Tuple[TrainState, Array],
    ],
    x_train: Array,
    a_train: Array,
    num_epochs: int,
    *,
    log_every: int = 1000,
    lambda_rel: float = 1.0,
) -> Dict[str, Any]:
    """
    Train a model with a trainable analytic component and record its parameter evolution.

    This function drives a training loop over ``num_epochs`` iterations using the
    provided ``train_step`` callable. In addition to tracking the loss and epoch
    index, it records the parameters associated with the
    ``"trainable_analytic_layer"`` subtree of the model parameters at every step.

    The primary use case is monitoring how an analytic component evolves during training
    when combined with a learned correction.

    Parameters
    ----------
    init_state
        Initial Flax ``TrainState`` containing parameters, optimizer state, and
        apply function.
    train_step
        Callable implementing a single training step. It must accept:

            ``(state, x_train, a_train, radial_weight=...)``
        and return ``(new_state, loss)``, where ``loss`` is a scalar JAX array.
    x_train
        Training input positions. Shape ``(N, 3)`` (scaled coordinates).
    a_train
        True accelerations corresponding to ``x_train``.
        Shape ``(N, 3)`` (scaled accelerations).
    num_epochs
        Number of training epochs to run.
    log_every
        Interval (in epochs) at which to print progress information.

    Returns
    -------
    out : dict
        Dictionary containing:

        - ``"state"`` : TrainState
          Final training state after ``num_epochs`` updates.

        - ``"epochs"`` : list[int]
          Epoch indices completed (0-based).

        - ``"losses"`` : list[float]
          Scalar loss values per epoch (converted to Python floats).

        - ``"analytic_params_history"`` : list[PyTree]
          History of the parameter subtree
          ``state.params["trainable_analytic_layer"]`` recorded at each epoch.

    Raises
    ------
    KeyError
        If ``"trainable_analytic_layer"`` is not present in ``state.params``.
    """

    epochs: List[int] = []
    losses: List[float] = []
    analytic_params_history: List[PyTree] = []

    state = init_state

    for epoch in range(num_epochs):
        state, loss = train_step(
            state,
            x_train,
            a_train,
            lambda_rel = lambda_rel,

        )

        if log_every > 0 and (epoch % log_every == 0):
            print(f"Epoch {epoch + 1}, Loss: {float(loss):.6f}")

        # Track analytic parameter subtree
        analytic_params = state.params["trainable_analytic_layer"]
        analytic_params_history.append(analytic_params)

        epochs.append(epoch)
        losses.append(float(loss))

    return {
        "state": state,
        "epochs": epochs,
        "losses": losses,
        "analytic_params_history": analytic_params_history,
    }



def initialize_staged_optimizers(
    net: _FlaxModuleWithInit,
    optimizer: optax.GradientTransformation,
    x_train: Array,
) -> Tuple[optax.GradientTransformation, optax.GradientTransformation]:
    """
    Create two Optax `multi_transform` optimizers for staged training.

    This utility builds parameter partitions from a model's initialized parameter
    pytree and returns two optimizer transformations:

    - `tx1` ("AB-only"): trains only parameters under the subtree whose path contains
      `"trainable_analytic_layer"` and freezes all other parameters by applying
      `optax.set_to_zero()` to their updates.

    - `tx2` ("joint training"): trains all parameters (i.e., no freezing).

    Parameters
    ----------
    net
        Flax module (or compatible object) providing `init(rng, x_example)` and returning
        a dict containing the `"params"` pytree.
    optimizer
        The Optax optimizer to use for trainable parameters (e.g., `optax.adam(...)`).
        Frozen parameters receive `optax.set_to_zero()` regardless of this choice.
    x_train
        Training inputs. Only the first example (`x_train[:1]`) is used to initialize
        model parameters via `net.init(...)`. Must be compatible with the model's
        input signature.

    Returns
    -------
    tx1, tx2
        A pair of Optax gradient transformations:
        - `tx1`: multi-transform optimizer that trains only `"trainable_analytic_layer"` params
        - `tx2`: multi-transform optimizer that trains all params

    Examples
    --------
    >>> # tx1, tx2 = initialize_staged_optimizers(net, optax.adam(1e-3), x_train)
    >>> # state1 = TrainState.create(apply_fn=net.apply, params=params, tx=tx1)  # trains analytic only
    >>> # state2 = TrainState.create(apply_fn=net.apply, params=params, tx=tx2)  # trains all
    """
    rng = jr.PRNGKey(0)
    init_params: PyTree = net.init(rng, x_train[:1])["params"]

    partition_optimizers: Mapping[str, optax.GradientTransformation] = {
        "trainable": optimizer,
        "frozen": optax.set_to_zero(),
    }

    param_partitions_AB_only: PyTree = traverse_util.path_aware_map(
        lambda path, _: "trainable" if ("trainable_analytic_layer" in path) else "frozen",
        init_params,
    )
    param_partitions_joint: PyTree = traverse_util.path_aware_map(
        lambda path, _: "trainable",
        init_params,
    )

    tx1 = optax.multi_transform(partition_optimizers, param_partitions_AB_only)
    tx2 = optax.multi_transform(partition_optimizers, param_partitions_joint)
    return tx1, tx2




def alternate_training(
    train_step: TrainStepFn,
    x_train: Array,
    a_train: Array,
    model: Module,
    num_epochs_stage1: int,
    num_epochs_stage2: int,
    cycles: int,
    param_list: Sequence[str],
    tx_1: optax.GradientTransformation,
    tx_2: optax.GradientTransformation,
    *,
    log_every: int = 1000,
    lambda_rel: float = 1.0,
) -> Dict[str, Any]:
    """
    Alternate optimization between two parameter groups (analytic vs neural) in repeated cycles.

    This function implements a two-stage training loop used when a model contains
    (i) a trainable analytic component (e.g., a baseline potential / fusing layer)
    and (ii) a learned neural component. The training proceeds in cycles:

      Stage 1 (analytic focus):
          - Create a TrainState using optimizer ``tx_1``.
          - Train for ``num_epochs_stage1`` epochs.
      Stage 2 (NN focus):
          - Create a TrainState using optimizer ``tx_2`` initialized from Stage 1 parameters.
          - Train for ``num_epochs_stage2`` epochs.

    During training, this function tracks:
      - the loss history across all stages and cycles, and
      - selected parameter values from ``params["trainable_analytic_layer"]`` over train time

    Parameters
    ----------
    train_step
        Callable implementing one training step. It is passed through to the trainer
        and should return ``(new_state, loss)``.
    x_train
        Training positions, typically shape ``(N, 3)`` (scaled).
    a_train
        True accelerations at ``x_train``, shape ``(N, 3)`` (scaled).
    model
        Flax module to train. Must support:
          - ``model.init(key, x_example)["params"]`` for initialization, and
          - ``model.apply`` for TrainState apply_fn.
    num_epochs_stage1
        Number of epochs to run in Stage 1 for each cycle (excluding burn-in).
    num_epochs_stage2
        Number of epochs to run in Stage 2 for each cycle.
    cycles
        Number of times to repeat the Stage1 / Stage2 cycle.
    param_list
        Names of parameters inside ``params["trainable_analytic_layer"]`` to record into history.
        Each entry is treated as a key into that dict.
    tx_1
        Optax optimizer used in Stage 1 (analytic-focused phase).
    tx_2
        Optax optimizer used in Stage 2 (NN-focused phase).


    Returns
    -------
    out : dict
        Dictionary containing:
        - ``"state"``: TrainState
            Final TrainState after the last Stage 2 of the last cycle.
        - ``"history"``: dict
            Contains:
              - ``"losses"``: list[float]
              - ``"epochs"``: jnp.ndarray of shape (total_steps,) containing 0..total_steps-1
              - For each name in ``param_list``: a list of tracked values over time.
        - ``"model"``: Module
            The model instance

    Notes
    -----
    - This function does not freeze parameter subsets; the actual freezing behavior
      must be implemented inside ``train_step`` (by controlling which params are
      updated by the optimizer).
    - The initialization uses ``model.init(jr.PRNGKey(0), x_train[:1])`` in the first cycle.

    Raises
    ------
    ValueError
        If epoch counts or cycles are non-positive.
    KeyError
        If ``"trainable_analytic_layer"`` is missing when tracking analytic parameters.
    """
    if cycles <= 0:
        raise ValueError("cycles must be positive.")
    if num_epochs_stage1 <= 0 or num_epochs_stage2 <= 0:
        raise ValueError("Stage epoch counts must be positive.")

    total_steps = cycles * (num_epochs_stage1 + num_epochs_stage2)
    history: Dict[str, Any] = {
        "losses": [],  # will extend with stage outputs
        "epochs": jnp.arange(total_steps),
    }
    for param in param_list:
        history[param] = []

    stage2_output_state: Optional[TrainState] = None

    for cycle in range(cycles):
        # Initialize params in first cycle; otherwise continue from previous cycle.
        if cycle == 0:
            params: PyTree = model.init(jr.PRNGKey(0), x_train[:1])["params"]
        else:
            assert stage2_output_state is not None
            params = stage2_output_state.params

        # === Stage 1 ===
        print(f"=== Starting Cycle {cycle + 1} / {cycles}: Stage 1 ===")
        stage1_input_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx_1)

        epochs_s1 = num_epochs_stage1
        output1: Dict[str, Any] = train_model_with_trainable_analytic_layer(
            stage1_input_state,
            train_step,
            x_train,
            a_train,
            epochs_s1,
            log_every = log_every,
            lambda_rel = lambda_rel
        )
        stage1_output_state: TrainState = output1["state"]

        history["losses"] += list(output1["losses"])


        analytic_tree = stage1_output_state.params["trainable_analytic_layer"]
        for param in param_list:
            if isinstance(analytic_tree, Mapping) and (param in analytic_tree):
                # output1["analytic_params_history"] entries are expected to be dict-like
                history[param] += [d[param] for d in output1["analytic_params_history"]]
                print(f"Learned {param} in cycle {cycle}: {history[param][-1]}")

        # === Stage 2 ===
        print(f"=== Starting Cycle {cycle + 1} / {cycles}: Stage 2 ===")
        stage2_input_state = TrainState.create(
            apply_fn=model.apply,
            params=stage1_output_state.params,
            tx=tx_2,
        )

        output2: Dict[str, Any] = train_model_with_trainable_analytic_layer(
            stage2_input_state,
            train_step,
            x_train,
            a_train,
            num_epochs_stage2,
            log_every = log_every,
            lambda_rel = lambda_rel
        )
        stage2_output_state = output2["state"]

        analytic_tree2 = stage1_output_state.params["trainable_analytic_layer"]
        for param in param_list:
            if isinstance(analytic_tree2, Mapping) and (param in analytic_tree2):
                history[param] += [d[param] for d in output2["analytic_params_history"]]

        history["losses"] += list(output2["losses"])

    assert stage2_output_state is not None
    return {"state": stage2_output_state, "history": history, "model": model}
