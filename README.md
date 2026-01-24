# galactoPINNs

<!-- [![PyPI version](https://badge.fury.io/py/galactoPINNs.svg)](https://badge.fury.io/py/galactoPINNs) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

Physics-informed neural networks for modeling galactic gravitational potentials.

This repository contains models and utilities for learning static and time-dependent gravitational potentials. The package is built using JAX, Flax, and Equinox, and includes optional Bayesian extensions with `numpyro`.

## Key Features

- **Hybrid Models**: Combine known analytic potentials with flexible neural networks to capture complex gravitational features.
- **Static and Time-Dependent Potentials**: Tools for modeling both static galactic potentials and dynamic, time-evolving systems.
- **Bayesian Inference**: Includes `numpyro` models for Bayesian parameter inference and uncertainty quantification.
- **JAX-based**: Fully implemented in JAX for high-performance, automatic differentiation, and execution on CPUs/GPUs.

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/charlottemyers/galactoPINNs.git
```

## Usage
To start, simply provide a set of training positions and accelerations. These can be provided from an arbitrary source, or generated using a Galax potential:

```python
from galactoPINNs.data import generate_static_datadict
import galax.potential as gp

true_potential = gp.MilkyWayPotential()
raw_datadict = generate_static_datadict(
    galax_potential=true_potential,
    N_samples_train = 2048,
    N_samples_test = 4096,
    r_max_train = 100,
    r_max_test = 150,
)

```

Next, simply specify the training configuration based on the physics-informed priors you wish to include in the model. This includes features to radially scale the output, and include. Based on these choices, non-dimensionalize the training data:

```python
from galactoPINNs.data import scale_data

halo_rs = 15.62
analytic_baseline_potential  = gp.NFWPotential(m= 5.4e11, r_s= halo_rs, units="galactic")

## define a configuration for the data non-dimensionalization
scale_config = {
    "r_s": halo_rs,
    "include_analytic": True,
    "lf_analytic_function": analytic_baseline_potential
}

scaled_data, transformers = scale_data(
    raw_datadict, scale_config
)

train_config = {
    "x_transformer": transformers["x"],
    "a_transformer": transformers["a"],
    "u_transformer": transformers["u"],
    "r_s": halo_rs,
    "lf_analytic_function": analytic_baseline_potential,
    "include_analytic": True,
    "scale": "nfw",
    "depth": 6,
}

```


Next, initialize the model based on your specified configuration, and train!


```python
from galactoPINNs.models.static_model import StaticModel
from galactoPINNs.train import train_model_static
import optax

net = StaticModel(train_config)
x_train = scaled_data["x_train"]
a_train = scaled_data["a_train"]


optimizer = optax.adam(1e-3)
rng = jax.random.PRNGKey(0)
train_output = train_model_static(
        net, optimizer, x_train, a_train, num_epochs=10)

```
You can use the evaluation features to evaluate the acceleration and potential predictions. You can also instantiate a Galax potential to represent the learned potential. Simply provide the trained parameters, and use the potential to generate acceleration and potential predictions, and integrate orbits!

```python
from galactoPINNs.model_potential import make_galax_potential
import unxt as u
import galax.dynamics as gd

learned_potential = make_galax_potential(net, train_output["state"].params)

# compute potential and accelerations
test_points = raw_datadict["x_val"]
learned_potential = learned_potential.potential(test_points, t=0)
learned_acceleration = learned_potential.acceleration(test_points, t=0)

#integrate orbits in the learned potential
w0 = PhaseSpacePosition(
   q=u.Quantity(jnp.array([[10.0, 16.0, 0.0]]), "kpc") ,
   p=u.Quantity([[1,  0.0, 0.0]], "kpc/Myr"))

ts = u.Quantity(jnp.linspace(0,  500.0, 500), "Myr")

learned_orbit = gd.evaluate_orbit(learned_potential, w0, ts)


```
