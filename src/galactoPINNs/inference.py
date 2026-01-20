__all__ = [
    "apply_model",
    "apply_model_time",
]


def apply_model(model, params, x_scaled):
    """
    Apply a *static* model to scaled Cartesian inputs and return standardized outputs.

    Parameters
    ----------
    model : flax.linen.Module
        Model object with `.apply(...)`.
    params : Any
        Parameter pytree to pass to `model.apply`.
    x_scaled : array-like
        Scaled positions. Expected shape (N, 3) or (3,).
    Returns
    -------
    out : dict
        - "u_pred": scaled predicted potential
        - "a_pred": scaled predicted acceleration
        - "outputs": auxiliary outputs dict if present, else None
    """
    predictions = model.apply({"params": params}, x_scaled)

    u_pred = predictions["potential"]
    a_pred = predictions["acceleration"]

    outputs = predictions.get("outputs", None)

    return {
        "u_pred": u_pred,
        "a_pred": a_pred,
        "outputs": outputs,
    }


def apply_model_time(model, params, tx_scaled):
    """
    Apply a *time-dependent* model to scaled (t, x) inputs and return standardized outputs.

    Parameters
    ----------
    model : flax.linen.Module (or compatible)
        Time-dependent model object with `.apply(...)`.
    params : Any
        Parameter pytree to pass to `model.apply`.
    tx_scaled : array-like
        Scaled inputs with time in the first column.
        Expected shape (N, 4) = [t_scaled, x_scaled...], or (4,) for a single point.
    Returns
    -------
    out : dict
        - "u_pred": scaled predicted potential
        - "a_pred": scaled predicted acceleration
        - "outputs": auxiliary outputs dict if present, else None
    """
    predictions = model.apply({"params": params}, tx_scaled)

    u_pred = predictions["potential"]
    a_pred = predictions["acceleration"]

    outputs = predictions.get("outputs", None)

    return {
        "u_pred": u_pred,
        "a_pred": a_pred,
        "outputs": outputs,
    }
