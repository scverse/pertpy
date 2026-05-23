# DIALOGUE R-reference fixture

Regression-test fixture produced by running the R implementation of
[DIALOGUE](https://github.com/livnatje/DIALOGUE) on the same data the Python
port operates on (`pertpy.data.dialogue_example`).

## Contents

| File pattern | Source | Shape |
|--------------|--------|-------|
| `weights_<celltype>.csv` | `R$cca$ws[[ct]]` | features × programs |
| `cca_scores_<celltype>.parquet` | `R$cca.scores[[ct]]` after DIALOGUE1 | cells × programs |
| `final_scores_<celltype>.parquet` | `R$scores[[ct]]` after DIALOGUE3 | cells × (programs + obs cols) |
| `gene_pval_<celltype>.parquet` | `R$gene.pval[[ct]]` after DIALOGUE3 | genes × pair-z + Fisher-combined |
| `mcp_<MCP>.csv` | `R$MCPs[[MCP]]` | gene signatures per (program, celltype) |
| `empirical_pvalues.csv` | `R$emp.p` | programs × cell-type pairs |
| `cca_cor_R.csv` / `cca_cor_P.csv` | `R$cca.cor$R` / `R$cca.cor$P` | programs × cell-type pairs |
| `pref_<pair>.csv` | `R$pref[[pair]]` | programs × (R, hlm_p) |
| `program_celltypes.csv` | `R$MCP.cell.types` | program → semicolon-joined celltypes |
| `celltype_order.txt` | cell-type ordering used by R | text list |

## Regenerating

The fixture is small (~800 KB) and committed directly so CI does not depend on R.
To regenerate after upstream DIALOGUE changes:

```bash
# 1. From the pertpy env, export the dialogue_example dataset for R
python tests/_data/dialogue_reference/export_for_R.py

# 2. From an env with R + DIALOGUE, run
Rscript tests/_data/dialogue_reference/generate_reference.R

# 3. Convert the bulky CSVs to parquet and copy into this directory
#    (see the helper at the end of the same R script for the layout).
```

The R DIALOGUE package needs:
`lme4 lmerTest PMA plyr matrixStats psych ppcor stringi reshape2 ggplot2 nnls assertthat devtools RColorBrewer Hmisc unikn beanplot UpSetR`

Then install DIALOGUE itself with `R CMD INSTALL <path-to-cloned-repo>`.

## Parameters used

```r
DLG.get.param(
  k = 3,
  conf = "cellQ",
  covar = c("cellQ", "tme.qc"),
  abn.c = 5,
  n.genes = 100,
  bypass.emp = FALSE  # 100 PMD permutations for empirical p
)
```

Preprocessing on the Python side mirrors the original pertpy test:
- 30-component PCA on `adata.X`.
- Drop cells of cell type `"CD8+ IL17+"`.
- Keep only samples where every remaining cell type has more than 3 cells.

This leaves 5156 cells, 29 samples, and 4 cell types (CD8+ IELs, CD8+ LP,
Macrophages, TA2).
