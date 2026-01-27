"""Physics-informed neural networks for modeling galactic gravitational potentials."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("galactoPINNs")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
