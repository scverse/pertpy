from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType

import pandas as pd

from pertpy.tools._differential_gene_expression._checks import check_is_numeric_matrix
from pertpy.tools._differential_gene_expression._formulaic import (
    AmbiguousAttributeError,
    Factor,
    get_factor_storage_and_materializer,
    resolve_ambiguous,
)


@dataclass
class Contrast:
    """Simple contrast for comparison between groups"""

    column: str
    baseline: str
    group_to_compare: str


ContrastType = Contrast | tuple[str, str, str]


class MethodBase(ABC):
    def __init__(self, adata, *, mask=None, layer=None, **kwargs):
        """
        Initialize the method.

        Args:
            adata: AnnData object, usually pseudobulked.
            mask: A column in adata.var that contains a boolean mask with selected features.
            layer: Layer to use in fit(). If None, use the X array.
            **kwargs: Keyword arguments specific to the method implementation.
        """
        self.adata = adata
        if mask is not None:
            self.adata = self.adata[:, self.adata.var[mask]]

        self.layer = layer
        check_is_numeric_matrix(self.data)

    @property
    def data(self):
        """Get the data matrix from anndata this object was initalized with (X or layer)."""
        if self.layer is None:
            return self.adata.X
        else:
            return self.adata.layer[self.layer]

    @classmethod
    @abstractmethod
    def compare_groups(
        cls,
        adata,
        column,
        baseline,
        groups_to_compare,
        *,
        paired_by=None,
        mask=None,
        layer=None,
        fit_kwargs=MappingProxyType({}),
        test_kwargs=MappingProxyType({}),
    ):
        """
        Compare between groups in a specified column.

        Args:
            adata: AnnData object.
            column: column in obs that contains the grouping information.
            baseline: baseline value (one category from variable).
            groups_to_compare: One or multiple categories from variable to compare against baseline.
            paired_by: Column from `obs` that contains information about paired sample (e.g. subject_id).
            mask: Subset anndata by a boolean mask stored in this column in `.obs` before making any tests.
            layer: Use this layer instead of `.X`.
            fit_kwargs: Additional fit options.
            test_kwargs: Additional test options.

        Returns:
            Pandas dataframe with results ordered by significance. If multiple comparisons were performed this is indicated in an additional column.
        """
        ...


class LinearModelBase(MethodBase):
    def __init__(self, adata, design, *, mask=None, layer=None, **kwargs):
        """
        Initialize the method.

        Args:
            adata: AnnData object, usually pseudobulked.
            design: Model design. Can be either a design matrix, a formulaic formula.Formulaic formula in the format 'x + z' or '~x+z'.
            mask: A column in adata.var that contains a boolean mask with selected features.
            layer: Layer to use in fit(). If None, use the X array.
            **kwargs: Keyword arguments specific to the method implementation.
        """
        super().__init__(adata, mask=mask, layer=layer)
        self._check_counts()

        self.factor_storage = None
        self.variable_to_factors = None

        if isinstance(design, str):
            self.factor_storage, self.variable_to_factors, materializer_class = get_factor_storage_and_materializer()
            self.design = materializer_class(adata.obs, record_factor_metadata=True).get_model_matrix(design)
        else:
            self.design = design

    @classmethod
    def compare_groups(
        cls,
        adata,
        column,
        baseline,
        groups_to_compare,
        *,
        paired_by=None,
        mask=None,
        layer=None,
        fit_kwargs=MappingProxyType({}),
        test_kwargs=MappingProxyType({}),
    ):
        design = f"~{column}"
        if paired_by is not None:
            design += f"+{paired_by}"
        if isinstance(groups_to_compare, str):
            groups_to_compare = [groups_to_compare]
        model = cls(adata, design=design, mask=mask, layer=layer)

        model.fit(**fit_kwargs)

        de_res = model.test_contrasts(
            {
                group_to_compare: model.contrast(column=column, baseline=baseline, group_to_compare=group_to_compare)
                for group_to_compare in groups_to_compare
            },
            **test_kwargs,
        )

        return de_res

    @property
    def variables(self):
        """Get the names of the variables used in the model definition."""
        try:
            return self.design.model_spec.variables_by_source["data"]
        except AttributeError:
            raise ValueError(
                "Retrieving variables is only possible if the model was initialized using a formula."
            ) from None

    @abstractmethod
    def _check_counts(self):
        """
        Check that counts are valid for the specific method.

        Raises:
            ValueError: if the data matrix does not comply with the expectations.
        """
        ...

    @abstractmethod
    def fit(self, **kwargs):
        """
        Fit the model.

        Args:
            **kwargs: Additional arguments for fitting the specific method.
        """
        ...

    @abstractmethod
    def _test_single_contrast(self, contrast, **kwargs): ...

    def test_contrasts(self, contrasts, **kwargs):
        """
        Perform a comparison as specified in a contrast vector.

        Args:
            contrasts: Either a numeric contrast vector, or a dictionary of numeric contrast vectors.
            **kwargs: passed to the respective implementation.

        Returns:
            A dataframe with the results.
        """
        if not isinstance(contrasts, dict):
            contrasts = {None: contrasts}
        results = []
        for name, contrast in contrasts.items():
            results.append(self._test_single_contrast(contrast, **kwargs).assign(contrast=name))

        results_df = pd.concat(results)
        return results_df

    def test_reduced(self, modelB):
        """
        Test against a reduced model.

        Args:
            modelB: the reduced model against which to test.

        Example:
            modelA = Model().fit()
            modelB = Model().fit()
            modelA.test_reduced(modelB)
        """
        raise NotImplementedError

    def cond(self, **kwargs):
        """
        Get a contrast vector representing a specific condition.

        Args:
            **kwargs: column/value pairs.

        Returns:
            A contrast vector that aligns to the columns of the design matrix.
        """
        if self.factor_storage is None:
            raise RuntimeError(
                "Building contrasts with `cond` only works if you specified the model using a formulaic formula. Please manually provide a contrast vector."
            )
        cond_dict = kwargs
        if not set(cond_dict.keys()).issubset(self.variables):
            raise ValueError(
                "You specified a variable that is not part of the model. Available variables: "
                + ",".join(self.variables)
            )
        for var in self.variables:
            if var in cond_dict:
                self._check_category(var, cond_dict[var])
            else:
                cond_dict[var] = self._get_default_value(var)
        df = pd.DataFrame([kwargs])
        return self.design.model_spec.get_model_matrix(df).iloc[0]

    def _get_factor_metadata_for_variable(self, var):
        factors = self.variable_to_factors[var]
        return list(chain.from_iterable(self.factor_storage[f] for f in factors))

    def _get_default_value(self, var):
        factor_metadata = self._get_factor_metadata_for_variable(var)
        if resolve_ambiguous(factor_metadata, "kind") == Factor.Kind.CATEGORICAL:
            try:
                tmp_base = resolve_ambiguous(factor_metadata, "base")
            except AmbiguousAttributeError as e:
                raise ValueError(
                    f"Could not automatically resolve base category for variable {var}. Please specify it explicity in `model.cond`."
                ) from e
            return tmp_base if tmp_base is not None else "\0"
        else:
            return 0

    def _check_category(self, var, value):
        factor_metadata = self._get_factor_metadata_for_variable(var)
        tmp_categories = resolve_ambiguous(factor_metadata, "categories")
        if resolve_ambiguous(factor_metadata, "kind") == Factor.Kind.CATEGORICAL and value not in tmp_categories:
            raise ValueError(
                f"You specified a non-existant category for {var}. Possible categories: {', '.join(tmp_categories)}"
            )

    def contrast(self, column, baseline, group_to_compare):
        """
        Build a simple contrast for pairwise comparisons.

        Args:
            column: column in adata.obs to test on.
            baseline: baseline category (denominator).
            group_to_compare: category to compare against baseline (nominator).

        Returns:
            Numeric contrast vector.
        """
        return self.cond(**{column: group_to_compare}) - self.cond(**{column: baseline})
