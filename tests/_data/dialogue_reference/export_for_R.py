"""Export dialogue_example() into CSVs an R session can ingest to build `cell.type` objects.

Output (one set per cell type) under /tmp/claude/dialogue_fixture/:
    <celltype>/X.csv         # cells x PCA components
    <celltype>/tpm.csv       # genes x cells expression (log-normalized for the toy)
    <celltype>/samples.csv   # one column "sample" indexed by cell
    <celltype>/cellQ.csv     # one column "cellQ"
    <celltype>/metadata.csv  # remaining obs columns indexed by cell
    celltype_order.txt
    sample_list.txt

The preprocessing here mirrors the existing pertpy test:
    - PCA on adata.X
    - drop cells of cell type "CD8+ IL17+" (matches test_dialogue.py)
    - keep only samples where every remaining cell type has at least 3 cells.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/claude/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/claude/mpl")

import numpy as np
import pandas as pd
import scanpy as sc

import pertpy as pt

CELLTYPE_KEY = "cell.subtypes"
SAMPLE_KEY = "sample"
N_COUNTS_KEY = "nCount_RNA"
N_COMPONENTS = 30
OUT = Path("/tmp/claude/dialogue_fixture")
OUT.mkdir(parents=True, exist_ok=True)


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s))


def main() -> None:
    adata = pt.dt.dialogue_example()
    sc.pp.pca(adata, n_comps=N_COMPONENTS, random_state=0)

    # Mirror test_dialogue.py
    isecs = pd.crosstab(adata.obs[CELLTYPE_KEY], adata.obs[SAMPLE_KEY])
    adata = adata[adata.obs[CELLTYPE_KEY] != "CD8+ IL17+"].copy()
    isecs = pd.crosstab(adata.obs[CELLTYPE_KEY], adata.obs[SAMPLE_KEY])
    keep_pts = list(isecs.loc[:, (isecs > 3).sum(axis=0) == isecs.shape[0]].columns.values)
    adata = adata[adata.obs[SAMPLE_KEY].isin(keep_pts), :].copy()
    adata.obs[CELLTYPE_KEY] = adata.obs[CELLTYPE_KEY].astype("category")
    adata.obs[CELLTYPE_KEY] = adata.obs[CELLTYPE_KEY].cat.remove_unused_categories()

    print(f"After filtering: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"Cell types: {list(adata.obs[CELLTYPE_KEY].cat.categories)}")
    print(f"Samples retained: {len(keep_pts)}")

    celltypes = list(adata.obs[CELLTYPE_KEY].cat.categories)
    (OUT / "celltype_order.txt").write_text("\n".join(celltypes) + "\n")
    (OUT / "sample_list.txt").write_text("\n".join(keep_pts) + "\n")

    pca = adata.obsm["X_pca"][:, :N_COMPONENTS]
    pca_cols = [f"PC{i + 1}" for i in range(N_COMPONENTS)]
    X_full = pd.DataFrame(pca, index=adata.obs_names, columns=pca_cols)

    for ct in celltypes:
        ct_dir = OUT / safe_name(ct)
        ct_dir.mkdir(exist_ok=True)
        mask = (adata.obs[CELLTYPE_KEY] == ct).to_numpy()
        cells = adata.obs_names[mask]

        # X (cells x PCs)
        X_full.loc[cells].to_csv(ct_dir / "X.csv")

        # tpm (genes x cells) — log1p-normalize the raw counts for this comparison
        sub = adata[mask].copy()
        # convert sparse to dense for CSV writing
        Xg = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
        # log-normalize (mirrors a sensible default; DIALOGUE expects tpm-like log-scale)
        sf = Xg.sum(axis=1, keepdims=True) + 1e-8
        Xnorm = np.log1p(1e4 * Xg / sf)
        df_tpm = pd.DataFrame(Xnorm.T, index=adata.var_names, columns=cells)
        df_tpm.to_csv(ct_dir / "tpm.csv")

        pd.DataFrame({"sample": adata.obs.loc[cells, SAMPLE_KEY].astype(str).to_numpy()}, index=cells).to_csv(
            ct_dir / "samples.csv"
        )
        pd.DataFrame({"cellQ": adata.obs.loc[cells, "cellQ"].to_numpy()}, index=cells).to_csv(ct_dir / "cellQ.csv")

        keep_meta = [c for c in adata.obs.columns if c not in (CELLTYPE_KEY,)]
        adata.obs.loc[cells, keep_meta].to_csv(ct_dir / "metadata.csv")

    print(f"Wrote fixture into {OUT}")


if __name__ == "__main__":
    main()
