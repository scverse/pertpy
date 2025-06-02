"""Top-level package for pertpy."""

__author__ = "Lukas Heumos"
__email__ = "lukas.heumos@posteo.net"
__version__ = "1.0.0"

import warnings

from anndata._core.aligned_df import ImplicitModificationWarning
from matplotlib import MatplotlibDeprecationWarning
from numba import NumbaDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="scvi._settings")
warnings.filterwarnings("ignore", message="Environment variable.*redefined by R")
warnings.filterwarnings("ignore", message="Transforming to str index.", category=ImplicitModificationWarning)

import mudata

mudata.set_options(pull_on_update=False)

from . import data as dt
from . import metadata as md
from . import plot as pl
from . import preprocessing as pp
from . import tools as tl
