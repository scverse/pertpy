from typing import Literal

import anndata as ad
import pandas as pd


class _PerturbationValidatorUnavailable:
    def __init__(self):
        raise RuntimeError("PerturbationValidator can only be instantiated if connected to a lamindb instance.")


try:
    import bionty as bt
    import wetlab as wl
    from cellxgene_lamin import CellxGeneFields, Curate
    from django.core.exceptions import ImproperlyConfigured
    from lamin_utils import logger
    from lamindb_setup.core.types import UPathStr
    from lnschema_core import Record
    from lnschema_core.types import FieldAttr

    pt_defaults = CellxGeneFields.OBS_FIELD_DEFAULTS | {
        "cell_line": "unknown",
        "genetic_treatments": "",
        "compound_treatments": "",
        "environmental_treatments": "",
        "combination_treatments": "",
    }

    pt_categoricals = CellxGeneFields.OBS_FIELDS | {
        "cell_line": bt.CellLine.name,
        "genetic_treatments": wl.GeneticTreatment.name,
        "compound_treatments": wl.CompoundTreatment.name,
        "environmental_treatments": wl.EnvironmentalTreatment.name,
        "combination_treatments": wl.CombinationTreatment.name,
    }

    pt_sources: dict[str, Record] = {
        "depmap_id": bt.Source.filter(name="depmap").one(),
        "cell_line": bt.Source.filter(name="depmap").one(),
        # "compound_treatments": bt.Source.filter(entity="Drug", name="chebi").first()
    }

    class PerturbationCurator(Curate):
        def __init__(
            self,
            adata: ad.AnnData | UPathStr,
            var_index: FieldAttr = bt.Gene.ensembl_gene_id,
            categoricals: dict[str, FieldAttr] = pt_categoricals,
            organism: Literal["human", "mouse"] = "human",
            *,
            defaults: dict[str, str] = pt_defaults,
            extra_sources: dict[str, Record] = pt_sources,
            verbosity: str = "hint",
            schema_version: Literal["5.0.0", "5.1.0"] = "5.1.0",
            using_key: str = "laminlabs/pertpy-datasets",
        ):
            """Curator flow for Perturbation data.

            Args:
                adata: Path to or AnnData object to curate against the CELLxGENE schema.
                var_index: The registry field for mapping the ``.var`` index.
                categoricals: A dictionary mapping ``.obs.columns`` to a registry field.
                    The PerturbationCurator maps against the required CELLxGENE fields and perturbation fields by default.
                organism: The organism name. CELLxGENE restricts it to 'human' and 'mouse' and therefore so do we.
                defaults: Default values that are set if columns or column values are missing.
                extra_sources: A dictionary mapping ``.obs.columns`` to Source records.
                verbosity: The verbosity level.
                schema_version: The CELLxGENE schema version to curate against.
                using_key: A reference LaminDB instance.
            """
            self.organism = organism

            # Set the Compound source to chebi; we don't want output if the source has already been set
            with logger.mute():
                chebi_source = bt.Source.filter(entity="Drug", name="chebi").first()
                wl.Compound.add_source(chebi_source)

            super().__init__(
                adata=adata,
                var_index=var_index,
                categoricals=categoricals,
                using_key=using_key,
                defaults=defaults,
                verbosity=verbosity,
                organism=self.organism,
                extra_sources=extra_sources,
                schema_version=schema_version,
            )

        def validate(self) -> bool:
            """Validates the AnnData object against cellxgene and pertpy's requirements."""
            return super().validate()


except ImproperlyConfigured:
    PerturbationCurator = _PerturbationValidatorUnavailable  # type: ignore
