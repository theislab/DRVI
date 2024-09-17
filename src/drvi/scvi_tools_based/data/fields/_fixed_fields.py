from anndata import AnnData
from pandas.api.types import CategoricalDtype
from scvi.data._utils import _make_column_categorical
from scvi.data.fields import CategoricalJointObsField


class FixedCategoricalJointObsField(CategoricalJointObsField):
    """An AnnDataField for a collection of categorical .obs fields in the AnnData data structure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _make_array_categorical(self, adata: AnnData, category_dict: dict[str, list[str]] | None = None) -> dict:
        """Make the .obsm categorical."""
        if self.attr_keys != getattr(adata, self.attr_name)[self.attr_key].columns.tolist():
            raise ValueError(
                f"Original .{self.source_attr_name} keys do not match the columns in the ",
                f"generated .{self.attr_name} field.",
            )

        categories = {}
        df = getattr(adata, self.attr_name)[self.attr_key]
        for key in self.attr_keys:
            categorical_dtype = (
                # TODO: make a PR for this fix
                # Only the following line (ordered=True) is changed in the whole function
                CategoricalDtype(categories=category_dict[key], ordered=True) if category_dict is not None else None
            )
            mapping = _make_column_categorical(df, key, key, categorical_dtype=categorical_dtype)
            categories[key] = mapping

        store_cats = categories if category_dict is None else category_dict

        mappings_dict = self._default_mappings_dict()
        mappings_dict[self.MAPPINGS_KEY] = store_cats
        mappings_dict[self.FIELD_KEYS_KEY] = self.attr_keys
        for k in self.attr_keys:
            mappings_dict[self.N_CATS_PER_KEY].append(len(store_cats[k]))
        return mappings_dict
