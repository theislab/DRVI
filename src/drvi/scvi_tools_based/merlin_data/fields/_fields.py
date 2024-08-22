from scvi.data.fields import CategoricalObsField, LayerField, NumericalJointObsField

from drvi.scvi_tools_based.data.fields import FixedCategoricalJointObsField
from drvi.scvi_tools_based.merlin_data._data import MerlinData


class MerlinLayerField(LayerField):
    def validate_field(self, adata: MerlinData) -> None:
        return

    def register_field(self, adata: MerlinData) -> dict:
        return {
            # self.N_OBS_KEY: adata.n_obs, ?
            self.N_VARS_KEY: adata.n_vars,
            # self.COLUMN_NAMES_KEY: np.asarray(adata.var_names), ?
        }

    def get_summary_stats(self, state_registry: dict) -> dict:
        summary_stats = {self.count_stat_key: state_registry[self.N_VARS_KEY]}
        # if self.registry_key == REGISTRY_KEYS.X_KEY:
        #     summary_stats[self.N_CELLS_KEY] = state_registry[self.N_OBS_KEY]
        return summary_stats

    def transfer_field(self, state_registry: dict, adata_target: MerlinData, **kwargs) -> dict:
        return super().transfer_field(state_registry, adata_target, **kwargs)


class MerlinCategoricalObsField(CategoricalObsField):
    def _setup_default_attr(self, adata: MerlinData) -> None:
        return

    def validate_field(self, adata: MerlinData) -> None:
        if self.is_default:
            return
        merlin_data = adata
        assert merlin_data.has_col(self.attr_key)

    def register_field(self, adata: MerlinData) -> dict:
        if self.is_default:
            return {
                self.CATEGORICAL_MAPPING_KEY: {},
            }
        merlin_data = adata
        categorical_mapping = merlin_data.get_categorical_mapping(self.attr_key)
        return {
            self.CATEGORICAL_MAPPING_KEY: categorical_mapping,
            self.ORIGINAL_ATTR_KEY: self._original_attr_key,
        }

    def get_summary_stats(self, state_registry: dict) -> dict:
        if self.is_default:
            return {}
        categorical_mapping = state_registry[self.CATEGORICAL_MAPPING_KEY]
        n_categories = len(categorical_mapping)
        return {self.count_stat_key: n_categories}

    def transfer_field(
        self,
        state_registry: dict,
        adata_target: MerlinData,
        extend_categories: bool = False,
        **kwargs,
    ) -> dict:
        """Transfer field from registry to target AnnData."""
        if self.is_default:
            self._setup_default_attr(adata_target)

        self.validate_field(adata_target)

        mapping = state_registry[self.CATEGORICAL_MAPPING_KEY].copy()

        assert not extend_categories

        return {
            self.CATEGORICAL_MAPPING_KEY: mapping,
            self.ORIGINAL_ATTR_KEY: self._original_attr_key,
        }


class MerlinCategoricalJointObsField(FixedCategoricalJointObsField):
    def validate_field(self, adata: MerlinData) -> None:
        merlin_data = adata
        for key in self.attr_keys:
            assert merlin_data.has_col(key)

    def register_field(self, adata: MerlinData) -> dict:
        merlin_data = adata
        categories = {}
        for key in self.attr_keys:
            mapping = merlin_data.get_categorical_mapping(key)
            categories[key] = mapping

        store_cats = categories

        mappings_dict = self._default_mappings_dict()
        mappings_dict[self.MAPPINGS_KEY] = store_cats
        mappings_dict[self.FIELD_KEYS_KEY] = self.attr_keys
        for k in self.attr_keys:
            mappings_dict[self.N_CATS_PER_KEY].append(len(store_cats[k]))
        return mappings_dict

    def get_summary_stats(self, state_registry: dict) -> dict:
        return super().get_summary_stats(state_registry)

    def transfer_field(
        self,
        state_registry: dict,
        adata_target: MerlinData,
        extend_categories: bool = False,
        **kwargs,
    ) -> dict:
        """Transfer the field."""
        if self.is_empty:
            return

        source_cat_dict = state_registry[self.MAPPINGS_KEY].copy()
        assert not extend_categories

        return source_cat_dict


class MerlinNumericalJointObsField(NumericalJointObsField):
    def validate_field(self, adata: MerlinData) -> None:
        merlin_data = adata
        for key in self.attr_keys:
            assert merlin_data.has_col(key)

    def register_field(self, adata: MerlinData) -> dict:
        if self.attr_keys:
            raise NotImplementedError()
        return {}

    def get_summary_stats(self, state_registry: dict) -> dict:
        if self.attr_keys:
            raise NotImplementedError()
        return {}

    def transfer_field(
        self,
        state_registry: dict,
        adata_target: MerlinData,
        **kwargs,
    ) -> dict:
        """Transfer the field."""
        super().transfer_field(state_registry, adata_target, **kwargs)
        return self.register_field(adata_target)


MerlinDataField = (
    MerlinLayerField | MerlinCategoricalObsField | MerlinCategoricalJointObsField | MerlinNumericalJointObsField
)
