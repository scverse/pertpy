"""Top-level package for pertpy."""

__author__ = "Lukas Heumos"
__email__ = "lukas.heumos@posteo.net"
__version__ = "0.5.0"

from numba import NumbaDeprecationWarning
from pypi_latest import PypiLatest

pertpy_pypi_latest = PypiLatest("pertpy", __version__)
pertpy_pypi_latest.check_latest()

import warnings

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

from . import data as dt
from . import plot as pl
from . import preprocessing as pp
from . import tools as tl
