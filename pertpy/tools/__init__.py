from rich import print

from pertpy.tools._augur import Augur
from pertpy.tools._cinemaot import Cinemaot
from pertpy.tools._dialogue import Dialogue
from pertpy.tools._differential_gene_expression import DifferentialGeneExpression
from pertpy.tools._distances._distance_tests import DistanceTest
from pertpy.tools._distances._distances import Distance
from pertpy.tools._metadata._cell_line import CellLineMetaData
from pertpy.tools._milo import Milo
from pertpy.tools._mixscape import Mixscape
from pertpy.tools._perturbation_space._clustering import ClusteringSpace
from pertpy.tools._perturbation_space._discriminator_classifier import DiscriminatorClassifierSpace
from pertpy.tools._perturbation_space._simple import CentroidSpace, DBSCANSpace, KMeansSpace, PseudobulkSpace
from pertpy.tools._scgen import SCGEN

try:
    from pertpy.tools._coda._sccoda import Sccoda
    from pertpy.tools._coda._tasccoda import Tasccoda
except ImportError as e:
    if "ete3" in str(e):
        print("[bold yellow]To use sccoda or tasccoda please install ete3 with [green]pip install ete3")
    else:
        raise e
