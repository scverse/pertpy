"""Helpers to interact with Formulaic Formulas

Some helpful definitions for working with formulaic formulas (e.g. `~ 0 + C(donor):treatment + np.log1p(continuous)`):
 * A *term* refers to an expression in the formula, separated by `+`, e.g. `C(donor):treatment`, or `np.log1p(continuous)`.
 * A *variable* refers to a column of the data frame passed to formulaic, e.g. `donor`.
 * A *factor* is the specification of how a certain variable is represented in the design matrix, e.g. treatment coding with base level "A" and reduced rank.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from formulaic import FactorValues, ModelSpec
from formulaic.materializers import PandasMaterializer
from formulaic.materializers.types import EvaluatedFactor
from formulaic.parser.types import Factor
from interface_meta import override


@dataclass
class FactorMetadata:
    """Store (relevant) metadata for a factor of a formula."""

    name: str
    """The unambiguous factor name as specified in the formula. E.g. `donor`, or `C(donor, contr.treatment(base="A"))`"""

    reduced_rank: bool
    """Whether a column will be dropped because it is redundant"""

    custom_encoder: bool
    """Whether or not a custom encoder (e.g. `C(...)`) was used."""

    categories: Sequence[str]
    """The unique categories in this factor (after applying `drop_rows`)"""

    kind: Factor.Kind
    """Type of the factor"""

    drop_field: str = None
    """The category that is dropped.

    Note that
      * this may also be populated if `reduced_rank = False`
      * this is only populated when no encoder was used (e.g. `~ donor` but NOT `~ C(donor)`.
    """

    column_names: Sequence[str] = None
    """The column names for this factor included in the design matrix.

    This may be the same as `categories` if the default encoder is used, or
    categories without the base level if a custom encoder (e.g. `C(...)`) is used.
    """

    colname_format: str = None
    """A formattable string that can be used to generate the column name in the design matrix, e.g. `{name}[T.{field}]`"""

    @property
    def base(self) -> str | None:
        """
        The base category for this categorical factor.

        This is derived from `drop_field` (for default encoding) or by comparing the column names in
        the design matrix with all categories (for custom encoding, e.g. `C(...)`).
        """
        if not self.reduced_rank:
            return None
        else:
            if self.custom_encoder:
                tmp_base = set(self.categories) - set(self.column_names)
                assert len(tmp_base) == 1
                return tmp_base.pop()
            else:
                assert self.drop_field is not None
                return self.drop_field


def get_factor_storage_and_materializer() -> tuple[dict[str, list[FactorMetadata]], dict[str, set[str]], type]:
    """Keeps track of categorical factors used in a model specification by generating a custom materializer.

    This materializer reports back metadata upon materialization of the model matrix.

    Returns:
        - A dictionary storing metadata for each factor processed by the custom materializer, named `factor_storage`.
        - A dictionary mapping variables to factor names, which works similarly to model_spec.variable_terms
            but maps to factors rather than terms, named `variable_to_factors`.
        - A materializer class tied to the specific instance of `factor_storage`.
    """
    # There can be multiple FactorMetadata entries per sample, for instance when formulaic generates an interaction
    # term, it generates the factor with both full rank and reduced rank.
    factor_storage: dict[str, list[FactorMetadata]] = defaultdict(list)
    variable_to_factors: dict[str, set[str]] = defaultdict(set)

    class CustomPandasMaterializer(PandasMaterializer):
        """An extension of the PandasMaterializer that records all categorical variables and their (base) categories."""

        REGISTER_NAME = "custom_pandas"
        REGISTER_INPUTS = ("pandas.core.frame.DataFrame",)
        REGISTER_OUTPUTS = ("pandas", "numpy", "sparse")

        def __init__(
            self,
            data: Any,
            context: Mapping[str, Any] | None = None,
            record_factor_metadata: bool = False,
            **params: Any,
        ):
            """Initialize the Materializer.

            Args:
                data: Passed to PandasMaterializer.
                context: Passed to PandasMaterializer
                record_factor_metadata: Flag that tells whether this particular instance of the custom materializer class
                    is supposed to record factor metadata. Only the instance that is used for building the design
                    matrix should record the metadata. All other instances (e.g. used to generate contrast vectors)
                    should not record metadata to not overwrite the specifications from the design matrix.
                **params: Passed to PandasMaterializer
            """
            self.factor_metadata_storage = factor_storage if record_factor_metadata else None
            self.variable_to_factors = variable_to_factors if record_factor_metadata else None
            # temporary pointer to metadata of factor that is currently evaluated
            self._current_factor: FactorMetadata = None
            super().__init__(data, context, **params)

        @override
        def _encode_evaled_factor(
            self, factor: EvaluatedFactor, spec: ModelSpec, drop_rows: Sequence[int], reduced_rank: bool = False
        ) -> dict[str, Any]:
            """Function is called just before the factor is evaluated.

            We can record some metadata, before we call the original function.
            """
            assert (
                self._current_factor is None
            ), "_current_factor should always be None when we start recording metadata"
            if self.factor_metadata_storage is not None:
                # Don't store if the factor is cached (then we should already have recorded it)
                if factor.expr in self.encoded_cache or (factor.expr, reduced_rank) in self.encoded_cache:
                    assert factor.expr in self.factor_metadata_storage, "Factor should be there since it's cached"
                else:
                    for var in factor.variables:
                        self.variable_to_factors[var].add(factor.expr)
                    self._current_factor = FactorMetadata(
                        name=factor.expr,
                        reduced_rank=reduced_rank,
                        categories=tuple(sorted(factor.values.drop(index=factor.values.index[drop_rows]).unique())),
                        custom_encoder=factor.metadata.encoder is not None,
                        kind=factor.metadata.kind,
                    )
            return super()._encode_evaled_factor(factor, spec, drop_rows, reduced_rank)

        @override
        def _flatten_encoded_evaled_factor(self, name: str, values: FactorValues[dict]) -> dict[str, Any]:
            """
            Function is called at the end, before the design matrix gets materialized.

            Here we have access to additional metadata, such as `drop_field`.
            """
            if self._current_factor is not None:
                assert self._current_factor.name == name
                self._current_factor.drop_field = values.__formulaic_metadata__.drop_field
                self._current_factor.column_names = values.__formulaic_metadata__.column_names
                self._current_factor.colname_format = values.__formulaic_metadata__.format
                self.factor_metadata_storage[name].append(self._current_factor)
                self._current_factor = None

            return super()._flatten_encoded_evaled_factor(name, values)

    return factor_storage, variable_to_factors, CustomPandasMaterializer


class AmbiguousAttributeError(ValueError):
    pass


def resolve_ambiguous(objs: Sequence[Any], attr: str) -> Any:
    """Given a list of objects, return an attribute if it is the same between all object. Otherwise, raise an error."""
    if not objs:
        raise ValueError("Collection is empty")

    first_obj_attr = getattr(objs[0], attr)

    # Check if the attribute is the same for all objects
    for obj in objs[1:]:
        if getattr(obj, attr) != first_obj_attr:
            raise AmbiguousAttributeError(f"Ambiguous attribute '{attr}': values differ between objects")

    # If attribute is the same for all objects, return it
    return first_obj_attr
