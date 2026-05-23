suppressPackageStartupMessages({
  library(DIALOGUE)
  library(matrixStats)
})

set.seed(1234)

fixture_dir <- "/tmp/claude/dialogue_fixture"
out_dir <- "/tmp/claude/dialogue_reference"
results_dir <- "/tmp/claude/dialogue_results"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

celltypes <- readLines(file.path(fixture_dir, "celltype_order.txt"))
celltypes <- celltypes[nzchar(celltypes)]
cat("Cell types:", paste(celltypes, collapse=", "), "\n")

read_celltype <- function(ct) {
  safe <- gsub("[^A-Za-z0-9]", "_", ct)
  ct_dir <- file.path(fixture_dir, safe)
  X <- as.matrix(read.csv(file.path(ct_dir, "X.csv"), row.names = 1, check.names = FALSE))
  tpm <- as.matrix(read.csv(file.path(ct_dir, "tpm.csv"), row.names = 1, check.names = FALSE))
  samples <- read.csv(file.path(ct_dir, "samples.csv"), row.names = 1, check.names = FALSE)$sample
  cellQ <- read.csv(file.path(ct_dir, "cellQ.csv"), row.names = 1, check.names = FALSE)$cellQ
  metadata <- read.csv(file.path(ct_dir, "metadata.csv"), row.names = 1, check.names = FALSE)
  stopifnot(identical(rownames(X), colnames(tpm)))
  stopifnot(length(samples) == nrow(X))
  list(name = ct, X = X, tpm = tpm, samples = samples, cellQ = cellQ, metadata = metadata)
}

dat <- lapply(celltypes, read_celltype)
names(dat) <- celltypes

rA <- lapply(dat, function(d) {
  make.cell.type(
    name = d$name,
    tpm = d$tpm,
    samples = d$samples,
    X = d$X,
    metadata = d$metadata,
    cellQ = d$cellQ
  )
})
names(rA) <- vapply(rA, function(r) r@name, character(1))

param <- DLG.get.param(
  k = 3,
  conf = "cellQ",
  covar = c("cellQ", "tme.qc"),
  results.dir = paste0(results_dir, "/"),
  abn.c = 5,
  n.genes = 100,
  bypass.emp = FALSE
)

cat("Running DIALOGUE...\n")
R <- DIALOGUE.run(rA = rA, main = "reference", param = param, plot.flag = FALSE)
cat("Done.\n")

# DIALOGUE3 strips out cca/ws/cca.scores from R. Reload the DIALOGUE1 intermediate to recover them.
R1 <- readRDS(file.path(results_dir, "DIALOGUE1_reference.rds"))
R$cca <- R1$cca
R$cca.scores <- R1$cca.scores

# Persist whatever R produces in flat formats.
ct_names <- names(rA)

safe_write <- function(obj, path) {
  if (is.null(obj)) {
    cat("  skip (NULL): ", path, "\n")
    return(invisible(NULL))
  }
  write.csv(as.matrix(obj), file = path, row.names = TRUE)
}

# 1) PMD weights (one CSV per cell type): features x programs
for (ct in ct_names) {
  safe_write(R$cca$ws[[ct]], file.path(out_dir, paste0("weights_", make.names(ct), ".csv")))
}
writeLines(ct_names, file.path(out_dir, "celltype_order.txt"))

# 2) CCA per-cell scores (cells x programs)
for (ct in ct_names) {
  safe_write(R$cca.scores[[ct]], file.path(out_dir, paste0("cca_scores_", make.names(ct), ".csv")))
}

# 3) Final scores after DIALOGUE3 refinement
if (!is.null(R$scores)) {
  for (ct in ct_names) {
    sc <- R$scores[[ct]]
    if (!is.null(sc)) {
      write.csv(sc, file = file.path(out_dir, paste0("final_scores_", make.names(ct), ".csv")), row.names = TRUE)
    } else {
      cat("  skip (NULL): final_scores for", ct, "\n")
    }
  }
}

# 4) Empirical p-values
safe_write(R$emp.p, file.path(out_dir, "empirical_pvalues.csv"))

# 5) Per-pair CCA correlations
safe_write(R$cca.cor$R, file.path(out_dir, "cca_cor_R.csv"))
safe_write(R$cca.cor$P, file.path(out_dir, "cca_cor_P.csv"))

# 6) gene.pval per cell type
if (!is.null(R$gene.pval)) {
  for (ct in ct_names) {
    gp <- R$gene.pval[[ct]]
    if (!is.null(gp)) {
      write.csv(gp, file = file.path(out_dir, paste0("gene_pval_", make.names(ct), ".csv")), row.names = TRUE)
    } else {
      cat("  skip (NULL): gene_pval for", ct, "\n")
    }
  }
}

# 7) MCP signatures (up/down genes per program per cell type)
if (!is.null(R$MCPs)) {
  for (mcp_name in names(R$MCPs)) {
    mcp <- R$MCPs[[mcp_name]]
    if (is.null(mcp)) next
    rows <- list()
    for (nm in names(mcp)) {
      genes <- mcp[[nm]]
      if (length(genes) == 0) next
      rows[[length(rows)+1]] <- data.frame(slot = nm, gene = genes, stringsAsFactors = FALSE)
    }
    if (length(rows) > 0) {
      df <- do.call(rbind, rows)
      write.csv(df, file = file.path(out_dir, paste0("mcp_", mcp_name, ".csv")), row.names = FALSE)
    }
  }
}

# 8) program -> cell types it spans
if (!is.null(R$MCP.cell.types)) {
  mct <- lapply(R$MCP.cell.types, function(x) if (is.null(x)) "" else paste(x, collapse=";"))
  df <- data.frame(program = names(mct), celltypes = unlist(mct), stringsAsFactors = FALSE)
  write.csv(df, file.path(out_dir, "program_celltypes.csv"), row.names = FALSE)
}

# 9) pref (per-pair correlation R and HLM p)
if (!is.null(R$pref)) {
  for (pair_name in names(R$pref)) {
    pr <- R$pref[[pair_name]]
    if (!is.null(pr)) {
      write.csv(pr, file = file.path(out_dir, paste0("pref_", make.names(pair_name), ".csv")), row.names = TRUE)
    } else {
      cat("  skip (NULL): pref for", pair_name, "\n")
    }
  }
}

cat("Reference outputs written to:", out_dir, "\n")
sessionInfo()
