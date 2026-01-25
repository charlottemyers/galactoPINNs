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

## Basic Usage
Training data consists of positions and accelerations, which may be provided externally or generated from a known analytic potential (e.g. using galax):

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

Specify the training configuration based on the physics-informed priors you wish to include in the model. This includes features to radially scale the output, and fuse with an analytic baseline potential. Based on these choices, non-dimensionalize the training data into scaled model space:

```python
from galactoPINNs.data import scale_data

halo_rs = 15.62
analytic_baseline_potential  = gp.NFWPotential(m= 5.4e11, r_s= halo_rs, units="galactic")

## define a configuration for the data non-dimensionalization
scale_config = {
    "r_s": halo_rs,
    "include_analytic": True,
    "ab_potential": analytic_baseline_potential
}

## non-dimensionalize the data
scaled_data, transformers = scale_data(
    raw_datadict, scale_config
)

train_config = {
    "x_transformer": transformers["x"],
    "a_transformer": transformers["a"],
    "u_transformer": transformers["u"],
    "r_s": halo_rs,
    "ab_potential": analytic_baseline_potential,
    "include_analytic": True,
    "scale": "nfw",
    "depth": 6,
}

```


Next, initialize the model based on your specified configuration, and train:


```python
from galactoPINNs.models.static_model import StaticModel
from galactoPINNs.train import train_model_static
import optax
import jax.random as jr

net = StaticModel(train_config)
x_train = scaled_data["x_train"]
a_train = scaled_data["a_train"]


optimizer = optax.adam(1e-3)
rng = jr.PRNGKey(0)
train_output = train_model_static(
        net, optimizer, x_train, a_train, num_epochs=10)

```
Use the provided evaluation features to assess the acceleration and potential predictions.
You can also instantiate a Galax potential backed by the learned potential. Provide the trained parameters, and then use the model to generate acceleration/potential predictions and integrate orbits!

```python
from galactoPINNs.model_potential import make_galax_potential
import unxt as u
import galax.dynamics as gd
from galax.coordinates import PhaseSpacePosition
import jax.numpy as np

learned_potential = make_galax_potential(net, train_output["state"].params)

# compute the predicted potential and acceleration
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
