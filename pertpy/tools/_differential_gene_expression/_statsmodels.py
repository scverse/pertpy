import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels
import statsmodels.api as sm
from tqdm.auto import tqdm

from ._base import LinearModelBase
from ._checks import check_is_numeric_matrix


class Statsmodels(LinearModelBase):
    """Differential expression test using a statsmodels linear regression."""

    def _check_counts(self):
        check_is_numeric_matrix(self.data)

    def fit(
        self,
        regression_model: type[sm.OLS] | type[sm.GLM] = sm.OLS,
        **kwargs,
    ) -> None:
        """Fit the specified regression model.

        Args:
            regression_model: A statsmodels regression model class, either OLS or GLM.
            **kwargs: Additional arguments for fitting the specific method. In particular, this
                is where you can specify the family for GLM.

        Examples:
            >>> import statsmodels.api as sm
            >>> import pertpy as pt
            >>> model = pt.tl.Statsmodels(adata, design="~condition")
            >>> model.fit(sm.GLM, family=sm.families.NegativeBinomial(link=sm.families.links.Log()))
            >>> results = model.test_contrasts(np.array([0, 1]))
        """
        self.models = []
        for var in tqdm(self.adata.var_names):
            mod = regression_model(
                sc.get.obs_df(self.adata, keys=[var], layer=self.layer)[var],
                self.design,
                **kwargs,
            )
            mod = mod.fit()
            self.models.append(mod)

    def _test_single_contrast(self, contrast, **kwargs) -> pd.DataFrame:
        res = []
        for var, mod in zip(tqdm(self.adata.var_names), self.models, strict=False):
            t_test = mod.t_test(contrast)
            res.append(
                {
                    "variable": var,
                    "p_value": t_test.pvalue,
                    "t_value": t_test.tvalue.item(),
                    "sd": t_test.sd.item(),
                    "log_fc": t_test.effect.item(),
                }
            )
        return (
            pd.DataFrame(res)
            .sort_values("p_value")
            .assign(adj_p_value=lambda x: statsmodels.stats.multitest.fdrcorrection(x["p_value"])[1])
        )
