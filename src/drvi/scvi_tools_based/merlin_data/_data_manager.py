from collections import defaultdict
from uuid import uuid4

import scvi
from scvi._types import AnnOrMuData
from scvi.data import AnnDataManager, AnnDataManagerValidationCheck, _constants

from drvi.scvi_tools_based.merlin_data._data import MerlinData
from drvi.scvi_tools_based.merlin_data.fields import (
    MerlinCategoricalJointObsField,
    MerlinCategoricalObsField,
    MerlinDataField,
    MerlinLayerField,
    MerlinNumericalJointObsField,
)


class MerlinDataManager(AnnDataManager):
    """
    Provides an interface to validate and process a MerlinData object for use in scvi-tools.

    Parameters
    ----------
    fields : list[MerlinDataField] | None, optional
        List of MerlinDataField objects to register.
    setup_method_args : dict | None, optional
        Dictionary describing the model and arguments passed in by the user
        to setup this AnnDataManager.
    validation_checks : AnnDataManagerValidationCheck | None, optional
        DataClass specifying which global validation checks to run on the data object.
    """

    def __init__(
        self,
        fields: list[MerlinDataField] | None = None,
        setup_method_args: dict | None = None,
        validation_checks: AnnDataManagerValidationCheck | None = None,
    ) -> None:
        self.id = str(uuid4())
        self.adata = None
        self.fields = fields or []
        self.validation_checks = validation_checks or AnnDataManagerValidationCheck()
        self._registry = {
            _constants._SCVI_VERSION_KEY: scvi.__version__,
            _constants._MODEL_NAME_KEY: None,
            _constants._SETUP_ARGS_KEY: None,
            _constants._FIELD_REGISTRIES_KEY: defaultdict(dict),
        }
        if setup_method_args is not None:
            self._registry.update(setup_method_args)

    def _validate_anndata_object(self, adata: AnnOrMuData | MerlinData):
        """For a given AnnData object, runs general scvi-tools compatibility checks."""
        if isinstance(adata, MerlinData):
            return True
        return super()._validate_anndata_object(adata)

    def get_fields_schema_mapping(self):
        mapping = []
        for field in self.fields:
            if isinstance(field, MerlinLayerField):
                assert field.attr_name == _constants._ADATA_ATTRS.LAYERS
                mapping.append((field.registry_key, field.attr_key))
            elif isinstance(field, MerlinCategoricalObsField):
                if field.is_default:
                    continue
                assert field.attr_name == _constants._ADATA_ATTRS.OBS
                mapping.append((field.registry_key, field._original_attr_key))
            elif isinstance(field, MerlinCategoricalJointObsField):
                assert field.attr_name == _constants._ADATA_ATTRS.OBSM
                if len(field.attr_keys) > 0:
                    mapping.append((field.registry_key, field.attr_keys))
            elif isinstance(field, MerlinNumericalJointObsField):
                assert field.attr_name == _constants._ADATA_ATTRS.OBSM
                if len(field.attr_keys) > 0:
                    mapping.append((field.registry_key, field.attr_keys))
            else:
                raise NotImplementedError()
        return mapping

    def get_dataset(self, split, **dataset_kwargs):
        """
        Get the dataset for a given split.

        Parameters
        ----------
            split (str): The split to retrieve the dataset for.
            **dataset_kwargs: Additional keyword arguments to pass to the dataset.

        Returns
        -------
            The dataset for the given split.
        """
        merlin_data = self.adata
        columns = []
        for _, col in self.get_fields_schema_mapping():
            if isinstance(col, list):
                columns.extend(col)
            else:
                columns.append(col)
        return merlin_data.get_dataset(split, columns, **dataset_kwargs)
