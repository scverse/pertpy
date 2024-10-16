from pathlib import Path
from typing import Literal

import anndata as ad
import pandas as pd
from django.core.exceptions import ImproperlyConfigured
from lamin_utils import logger


class _PerturbationValidatorUnavailable:
    def __init__(self):
        raise RuntimeError("PerturbationValidator can only be instantiated if connected to a lamindb instance.")


try:
    import bionty as bt
    from cellxgene_lamin import CellxGeneFields, Curate
    from lamindb_setup.core.types import UPathStr
    from lnschema_core import Record
    from lnschema_core.types import FieldAttr

    class PerturbationValidator(Curate):
        def __init__(
            self,
            adata: ad.AnnData | str | Path,
            var_index: FieldAttr = bt.Gene.ensembl_gene_id,
            categoricals: dict[str, FieldAttr] = CellxGeneFields.OBS_FIELDS,
            organism: Literal["human", "mouse"] = "human",
            *,
            defaults: dict[str, str] = None,
            sources: dict[str, Record] = None,
            using_key: str = "laminlabs/pertpy-datasets",
            verbosity: str = "hint",
            schema_version: Literal["5.0.0", "5.1.0"] = "5.1.0",
        ):
            """Defines the Curator flow for Perturbation data."""
            self.organism = organism

            super().__init__(
                adata=adata,
                var_index=var_index,
                categoricals=categoricals,
                using_key=using_key,
                defaults=defaults,
                verbosity=verbosity,
                organism=self.organism,
                extra_sources=sources,
                schema_version=schema_version,
            )

        def validate(self) -> bool:
            """Validates the AnnData object against cellxgene and pertpy's requirements."""
            return super().validate()


except ImproperlyConfigured:
    PerturbationValidator = _PerturbationValidatorUnavailable  # type: ignore
