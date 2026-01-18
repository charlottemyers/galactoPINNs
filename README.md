# galactoPINNs

[![PyPI version](https://badge.fury.io/py/galactoPINNs.svg)](https://badge.fury.io/py/galactoPINNs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
