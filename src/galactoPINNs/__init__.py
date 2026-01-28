"""Physics-informed neural networks for modeling galactic gravitational potentials."""
<<<<<<< HEAD

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("galactoPINNs")
except PackageNotFoundError:
    __version__ = "0+unknown"
=======
>>>>>>> fa64b126e2f9c5a01438d99892523c5f1bb39ae1

__all__ = ["__version__"]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("galactoPINNs")
except PackageNotFoundError:
    __version__ = "0+unknown"
