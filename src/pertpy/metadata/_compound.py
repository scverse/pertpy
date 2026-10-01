from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Literal

import pandas as pd

from pertpy._types import cast_frame

from ._look_up import LookUp
from ._metadata import MetaData

if TYPE_CHECKING:
    from anndata import AnnData

_PUBCHEM_REQUESTS_PER_SECOND = 5


class _Throttle:
    """Space out the calls made from any thread to at most ``rate`` per second."""

    def __init__(self, rate: float):
        self._interval = 1 / rate
        self._lock = threading.Lock()
        self._next_call = time.monotonic()

    def __call__(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next_call - now
            self._next_call = max(now, self._next_call) + self._interval
        if delay > 0:
            time.sleep(delay)


def _fetch_compound(
    compound: str, query_id_type: Literal["name", "cid"], throttle: _Throttle
) -> list[str | int | None] | None:
    """Fetch the PubChem name, CID and SMILES of a compound, or None if PubChem does not know it."""
    import pubchempy as pcp

    throttle()
    if query_id_type == "name":
        cids = pcp.get_compounds(compound, "name")
        if len(cids) == 0:
            return None
        throttle()
        # If the name matches the first synonym offered by PubChem (outside of capitalization),
        # it is not changed (outside of capitalization). Otherwise, it is replaced with the first synonym.
        return [cids[0].synonyms[0], cids[0].cid, cids[0].canonical_smiles]
    try:
        cid = pcp.Compound.from_cid(compound)  # type: ignore[arg-type]
    except pcp.BadRequestError:
        # pubchempy throws badrequest if a cid is not found
        return None
    throttle()
    return [cid.synonyms[0], compound, cid.canonical_smiles]


class Compound(MetaData):
    """Utilities to fetch metadata for compounds."""

    def __init__(self):
        super().__init__()

    def annotate_compounds(
        self,
        adata: AnnData,
        query_id: str = "perturbation",
        query_id_type: Literal["name", "cid"] = "name",
        verbosity: int | str = 5,
        copy: bool = False,
    ) -> AnnData:
        """Fetch compound annotation from pubchempy.

        Args:
            adata: The data object to annotate.
            query_id: The column of `.obs` with compound identifiers.
            query_id_type: The type of compound identifiers, 'name' or 'cid'.
            verbosity: The number of unmatched identifiers to print, can be either non-negative values or "all".
            copy: Determines whether a copy of the `adata` is returned.

        Returns:
            Returns an AnnData object with compound annotation.
        """
        if copy:
            adata = adata.copy()

        if query_id not in adata.obs.columns:
            raise ValueError(f"The requested query_id {query_id} is not in `adata.obs`.\n Please check again.")

        compounds = cast_frame(adata.obs)[query_id].dropna().astype(str).unique()
        fetch = partial(_fetch_compound, query_id_type=query_id_type, throttle=_Throttle(_PUBCHEM_REQUESTS_PER_SECOND))
        with ThreadPoolExecutor(max_workers=_PUBCHEM_REQUESTS_PER_SECOND) as pool:
            fetched = list(pool.map(fetch, compounds))
        query_dict = {compound: info for compound, info in zip(compounds, fetched, strict=True) if info is not None}
        not_matched_identifiers = [compound for compound, info in zip(compounds, fetched, strict=True) if info is None]

        identifier_num_all = len(adata.obs[query_id].unique())
        self._warn_unmatch(
            total_identifiers=identifier_num_all,
            unmatched_identifiers=not_matched_identifiers,
            query_id=query_id,
            reference_id=query_id_type,
            metadata_type="compound",
            verbosity=verbosity,
        )

        query_df = pd.DataFrame.from_dict(query_dict, orient="index", columns=["pubchem_name", "pubchem_ID", "smiles"])
        # Merge and remove duplicate columns
        # Column is converted to float after merging due to unmatches
        # Convert back to integers afterwards
        if query_id_type == "cid":
            query_df["pubchem_ID"] = query_df["pubchem_ID"].astype("Int64")
            adata.obs = (
                cast_frame(adata.obs)
                .merge(
                    query_df,
                    left_on=query_id,
                    right_on="pubchem_ID",
                    how="left",
                    suffixes=("", "_fromMeta"),
                )
                .filter(regex="^(?!.*_fromMeta)")
                .set_index(adata.obs.index)
            )
        else:
            adata.obs = (
                cast_frame(adata.obs)
                .merge(
                    query_df,
                    left_on=query_id,
                    right_index=True,
                    how="left",
                    suffixes=("", "_fromMeta"),
                )
                .filter(regex="^(?!.*_fromMeta)")
                .set_index(adata.obs.index)
            )
            adata.obs["pubchem_ID"] = cast_frame(adata.obs)["pubchem_ID"].astype("Int64")

        return adata

    def lookup(self) -> LookUp:
        """Generate LookUp object for CompoundMetaData.

        The LookUp object provides an overview of the metadata to annotate.
        Each annotate_{metadata} function has a corresponding lookup function in the LookUp object,
        where users can search the reference_id in the metadata and compare with the query_id in their own data.

        Returns:
            Returns a LookUp object specific for compound annotation.
        """
        return LookUp(type="compound")
