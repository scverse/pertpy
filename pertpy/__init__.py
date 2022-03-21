"""Top-level package for pertpy."""

__author__ = "Lukas Heumos"
__email__ = "lukas.heumos@posteo.net"
__version__ = "0.1.0"


from pypi_latest import PypiLatest

pertpy_pypi_latest = PypiLatest("pertpy", __version__)
pertpy_pypi_latest.check_latest()

from . import data as dt
from . import plot as pl
from . import tools as tl
