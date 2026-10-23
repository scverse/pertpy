# Glossary

```{glossary}
:sorted:

Perturbation
  An intervention on cells, such as a genetic knockout, knockdown or activation, a drug, a cytokine or a disease state.
  Pertpy treats all of them alike: a perturbation is a label in `.obs` that groups cells for comparison.

Control cells
  Cells that received no perturbation, a non-targeting guide RNA or only the drug vehicle.
  Most tools compare perturbed cells to them, so they are passed as the `control` or `contrast` group.

Pooled CRISPR screen
  An experiment that delivers many guide RNAs to one population of cells and reads out both the guide RNAs and the transcriptome of every cell, as in Perturb-seq or CROP-seq.

Guide RNA
  The RNA that directs a CRISPR enzyme to its target gene.
  Its counts per cell show which perturbation a cell received; {class}`~pertpy.preprocessing.GuideAssignment` turns them into assignments.

Multiplicity of infection
  The average number of viral particles, and hence guide RNAs, per cell.
  At a high multiplicity of infection, many cells carry several perturbations.

Perturbation efficacy
  How strongly a perturbation changed a cell.
  Some cells carry a guide RNA but escape the perturbation; {class}`~pertpy.tools.Mixscape` classifies cells as perturbed or not, and {class}`~pertpy.tools.Mixscale` scores the effect continuously.

Perturbation signature
  A cell's expression minus that of its most similar control cells, which removes variation unrelated to the perturbation, such as cell cycle.
  {class}`~pertpy.tools.Mixscape` and {class}`~pertpy.tools.Mixscale` work on it.

Pseudobulk
  The summed counts of all cells of one sample, or one sample and cell type.
  Count-based differential expression with {class}`~pertpy.tools.PyDESeq2` and {class}`~pertpy.tools.EdgeR` is usually run on pseudobulks, so that samples rather than cells are the replicates.

Design formula
  A formula such as `~condition + batch` that names the covariates of a model, as used by the differential expression tools, {class}`~pertpy.tools.Milo` and {class}`~pertpy.tools.Sccoda`.

Contrast
  The comparison a model tests, such as treated versus control within the same batch.

Random intercept
  A per-group offset, written `(1 | donor)` in a design formula, that accounts for cells or neighborhoods from the same donor being correlated without estimating a fixed effect per donor.
  {class}`~pertpy.tools.Milo` fits it in a mixed model.

Compositional analysis
  Testing for changes in the proportions of cell types.
  Proportions sum to one, so an increase in one cell type decreases the others; {class}`~pertpy.tools.Sccoda` and {class}`~pertpy.tools.Tasccoda` model this.

Reference cell type
  The cell type that {class}`~pertpy.tools.Sccoda` assumes to be unchanged, against which the changes of all other cell types are expressed.

Differential abundance
  A change in how many cells occupy a region of the cell state space between conditions, tested without cell type labels.
  {class}`~pertpy.tools.Milo` tests it on neighborhoods.

Neighborhood
  A group of cells around an index cell in the k-nearest neighbor graph, the unit that {class}`~pertpy.tools.Milo` counts and tests.

Cell type prioritization
  Ranking cell types by how strongly they respond to a perturbation.
  {class}`~pertpy.tools.Augur` scores each cell type by how well a classifier separates the conditions within it.

Multicellular program
  A set of gene programs in different cell types that vary together across samples, as found by {class}`~pertpy.tools.Dialogue`.

Gene set enrichment
  Testing whether a gene set, such as a pathway or the targets of a drug, is overrepresented among the genes a perturbation changed.
  {class}`~pertpy.tools.Enrichment` scores pathways and drug targets.

Perturbation space
  A representation with one observation per perturbation, such as its pseudobulk or centroid, that makes perturbations comparable as a whole.
  See {class}`~pertpy.tools.PseudobulkSpace` and {class}`~pertpy.tools.CentroidSpace`.

E-distance
  The energy distance between two groups of cells: twice the mean distance between the groups minus the mean distances within each group.
  It is the default metric of {class}`~pertpy.tools.Distance`.

Maximum mean discrepancy
  A kernel-based distance between two distributions of cells, available as `"mmd"` in {class}`~pertpy.tools.Distance`.

Wasserstein distance
  The minimal cost of transporting one group of cells onto another, available as `"wasserstein"` in {class}`~pertpy.tools.Distance`.

Permutation test
  A test that shuffles the group labels to build the null distribution of a statistic.
  {class}`~pertpy.tools.DistanceTest` uses it to test whether perturbations differ from the control cells.

Counterfactual
  The state a perturbed cell would have had without the perturbation.
  {class}`~pertpy.tools.Cinemaot` matches each perturbed cell to its counterfactual after removing confounding variation.

Response prediction
  Predicting the expression of perturbed cells that were not measured, such as the response of an unseen cell type.
  {class}`~pertpy.tools.Scgen` predicts it from a latent space learned on the measured cells.

Mechanism of action
  The biochemical target or process through which a drug acts, which {class}`~pertpy.metadata.Moa` annotates.
```
