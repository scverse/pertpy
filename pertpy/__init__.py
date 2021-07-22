"""Top-level package for pertpy."""

__author__ = "Lukas Heumos"
__email__ = "lukas.heumos@posteo.net"
__version__ = "0.1.0"

from pypi_latest import PypiLatest

from pertpy.api import data, plot, preprocessing, tools

pertpy_pypi_latest = PypiLatest("pertpy", __version__)
pertpy_pypi_latest.check_latest()
