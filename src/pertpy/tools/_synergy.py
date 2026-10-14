from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

from pertpy._types import cast_frame

if TYPE_CHECKING:
    from anndata import AnnData


class Synergy:
    """Score drug combinations against reference models of joint inhibition."""

    def score(
        self,
        adata: AnnData,
        response: str = "inhibition",
        *,
        dose_cols: tuple[str, str] = ("dose_a", "dose_b"),
        method: Literal["bliss"] = "bliss",
        key_added: str | None = None,
    ) -> None:
        """Score two-drug combinations using Bliss independence.

        Supply one assay measurement per observation for a fixed drug pair under matched experimental conditions.
        Inhibition must already be control-normalized to fractions between 0 (no inhibition) and 1 (complete inhibition).
        A zero dose denotes the absence of that drug; no expression matrix is required.

        Single-agent replicates are averaged at each dose, ignoring missing responses.
        Doses are matched exactly, without interpolation; use consistent dose values and units.
        Expected inhibition is ``1 - (1 - inhibition_a) * (1 - inhibition_b)``; the score is observed minus expected inhibition.
        Positive scores indicate excess inhibition over the Bliss expectation, not statistical significance.
        The reference model follows `Bliss (1939) <https://doi.org/10.1111/j.1744-7348.1939.tb06990.x>`_.

        Args:
            adata: AnnData for one drug pair and experimental condition, with measurements in `.obs`.
            response: `.obs` column containing fractional inhibition.
            dose_cols: `.obs` columns containing non-negative doses of the first and second drug.
            method: Reference model. Currently only ``"bliss"`` is supported.
            key_added: Prefix of the result columns in `.obs`. Defaults to ``{response}_{method}``.

        Returns:
            Adds ``{prefix}_expected`` and ``{prefix}_score`` to `.obs` for combination observations.
            Controls and single agents get NaN results.
            Missing responses or single-agent references yield NaN scores and one summary warning.

        Examples:
            >>> import pandas as pd
            >>> import pertpy as pt
            >>> from anndata import AnnData
            >>> adata = AnnData(
            ...     obs=pd.DataFrame({"dose_a": [10, 0, 10], "dose_b": [0, 20, 20], "inhibition": [0.2, 0.3, 0.6]})
            ... )
            >>> pt.tl.Synergy().score(adata, key_added="bliss")
            >>> print(f"{adata.obs['bliss_score'].iloc[-1]:.2f}")
            0.16
        """
        if method != "bliss":
            raise ValueError("Only method='bliss' is currently supported.")
        if len(dose_cols) != 2 or len({*dose_cols, response}) != 3:
            raise ValueError("Provide two distinct dose columns and a separate response column.")

        prefix = f"{response}_{method}" if key_added is None else key_added
        if {f"{prefix}_expected", f"{prefix}_score"} & {*dose_cols, response}:
            raise ValueError("Result columns must not overwrite dose or response columns.")

        data = cast_frame(adata.obs)[[*dose_cols, response]].astype(float)
        values = data.to_numpy()
        doses, inhibition = values[:, :2], values[:, 2]
        if not np.isfinite(doses).all() or (doses < 0).any():
            raise ValueError("Dose values must be finite and non-negative.")
        if ((inhibition < 0) | (inhibition > 1)).any():
            raise ValueError("Inhibition must be control-normalized to fractions between 0 and 1, or NaN if missing.")

        combination = (doses[:, 0] > 0) & (doses[:, 1] > 0)
        if not combination.any():
            raise ValueError("No combination measurements with both doses greater than zero were found.")

        dose_a, dose_b = dose_cols
        single_a = data.loc[(doses[:, 0] > 0) & (doses[:, 1] == 0)].groupby(dose_a)[response].mean()
        single_b = data.loc[(doses[:, 0] == 0) & (doses[:, 1] > 0)].groupby(dose_b)[response].mean()
        combinations = data.loc[combination]
        inhibition_a = combinations[dose_a].map(single_a).to_numpy()
        inhibition_b = combinations[dose_b].map(single_b).to_numpy()

        expected = np.full(adata.n_obs, np.nan)
        expected[combination] = 1 - (1 - inhibition_a) * (1 - inhibition_b)
        scores = np.full(adata.n_obs, np.nan)
        scores[combination] = inhibition[combination] - expected[combination]
        missing = np.isnan(scores[combination])
        if missing.any():
            warnings.warn(
                f"Cannot score {missing.sum()} combination observations: inhibition or matching single-agent "
                "responses are missing; their scores are NaN.",
                UserWarning,
                stacklevel=2,
            )

        adata.obs[f"{prefix}_expected"] = expected
        adata.obs[f"{prefix}_score"] = scores
