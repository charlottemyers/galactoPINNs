#!/usr/bin/env -S uv run
"""Nox setup."""
# /// script
# requires-python = ">=3.11"
# dependencies = ["nox"]
# ///

import shutil
from pathlib import Path

import nox

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"

DIR = Path(__file__).parent.resolve()


# =============================================================================
# Comprehensive sessions


@nox.session(default=True)
def all(s: nox.Session, /) -> None:  # noqa: A001
    """Run all default sessions."""
    s.notify("lint")
    s.notify("test")


# =============================================================================
# Linting


@nox.session(reuse_venv=True)
def lint(s: nox.Session, /) -> None:
    """Run the linter."""
    s.install("pre-commit")
    s.run("pre-commit", "run", "--all-files", *s.posargs)


# =============================================================================
# Testing


@nox.session(reuse_venv=True)
def test(s: nox.Session, /) -> None:
    """Run the unit and regular tests."""
    s.install("-e", ".[dev]")
    s.run("pytest", *s.posargs)


# =============================================================================
# Build


@nox.session
def build(s: nox.Session, /) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    s.install("build")
    s.run("python", "-m", "build")


# =============================================================================

if __name__ == "__main__":
    nox.main()
