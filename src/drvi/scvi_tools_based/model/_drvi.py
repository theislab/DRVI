import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from anndata import AnnData
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField, NumericalJointObsField
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.utils import setup_anndata_dsp

import drvi
from drvi.nn_modules.feature_interface import FeatureInfoList
from drvi.scvi_tools_based.data.fields import FixedCategoricalJointObsField
from drvi.scvi_tools_based.merlin_data import (
    MerlinData,
    MerlinDataManager,
    MerlinDataSplitter,
    MerlinTransformedDataLoader,
)
from drvi.scvi_tools_based.merlin_data import (
    fields as melin_fields,
)
from drvi.scvi_tools_based.model.base import DRVIArchesMixin, GenerativeMixin
from drvi.scvi_tools_based.module import DRVIModule

logger = logging.getLogger(__name__)


class DRVI(VAEMixin, DRVIArchesMixin, UnsupervisedTrainingMixin, BaseModelClass, GenerativeMixin):
    """DRVI model based on scvi-tools framework for disentangled representation learning.

    Parameters
    ----------
    adata
        AnnData object or MerlinData object that has been registered via :meth:`~drvi.model.DRVI.setup_anndata`.
    n_latent
        Dimensionality of the latent space.
    encoder_dims
        Number of nodes in hidden layers of the encoder.
    decoder_dims
        Number of nodes in hidden layers of the decoder.
    prior
        Prior model type.
    prior_init_obs
        When using "gmm_x" or "vamp_x" priors, these observations are used to initialize the prior parameters.
        Number of observations must match the x value in the prior name.
    categorical_covariates
        List of categorical covariates to condition on. Each covariate can specify its embedding dimension
        by appending @dim (e.g. "batch@32"). Default embedding dimension is 10.
    **model_kwargs
        Additional keyword arguments passed to :class:`~drvi.model.DRVIModule`.

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> drvi.model.DRVI.setup_anndata(adata, categorical_covariate_keys=["batch"])
    >>> vae = drvi.model.DRVI(adata)
    >>> vae.train()
    >>> adata.obsm["latent"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData | MerlinData | None = None,  # TODO: align with all scvi changes: registry, etc.
        registry: dict | None = None,  # TODO: align with all scvi changes: registry, etc.
        n_latent: int = 32,
        encoder_dims: Sequence[int] = (128, 128),
        decoder_dims: Sequence[int] = (128, 128),
        prior: Literal["normal", "gmm_x", "vamp_x"] = "normal",
        prior_init_obs: np.ndarray | None = None,
        categorical_covariates: list[str] = (),
        **model_kwargs,
    ) -> None:
        super().__init__(adata, registry)

        # TODO: Remove later. Currently used to detect autoreload problems sooner.
        if isinstance(adata, AnnData):
            pass
        elif MerlinData is not None and isinstance(adata, MerlinData):
            self._data_splitter_cls = MerlinDataSplitter
        else:
            raise ValueError(
                "Only AnnData and MerlinData are supported. "
                "If you have passed an instance of MerlinData and still get this error, "
                "make sure merlin is installed as a dependency."
            )

        categorical_covariates_info = FeatureInfoList(categorical_covariates, axis="obs", default_dim=10)
        if REGISTRY_KEYS.CAT_COVS_KEY in self.registry["field_registries"]:
            cat_cov_stats = self.registry["field_registries"][REGISTRY_KEYS.CAT_COVS_KEY]["state_registry"]
            print(cat_cov_stats)
            n_cats_per_cov = cat_cov_stats.get("n_cats_per_key", [])
            assert tuple(categorical_covariates_info.names) == tuple(cat_cov_stats.get("field_keys", []))
        else:
            n_cats_per_cov = []
            assert len(categorical_covariates_info) == 0
        n_continuous_cov = self.summary_stats.get("n_extra_continuous_covs", 0)

        prior_init_dataloader = None
        if prior_init_obs is not None:
            assert "_" in prior
            assert len(prior_init_obs) == int(prior.split("_")[-1])
            prior_init_dataloader = self._make_data_loader(
                adata=adata[prior_init_obs], batch_size=len(prior_init_obs), shuffle=False
            )

        self.module = DRVIModule(
            n_input=self.summary_stats["n_vars"],
            n_latent=n_latent,
            encoder_dims=encoder_dims,
            decoder_dims=decoder_dims,
            n_cats_per_cov=n_cats_per_cov,
            n_continuous_cov=n_continuous_cov,
            prior=prior,
            prior_init_dataloader=prior_init_dataloader,
            categorical_covariate_dims=categorical_covariates_info.dims,
            **model_kwargs,
        )

        self._model_summary_string = (
            "DRVI \n"
            + (f"Covariates: {categorical_covariates_info.names}, \n" if len(categorical_covariates_info) > 0 else "")
            + f"Latent size: {self.module.n_latent}, "
            f"splits: {self.module.n_split_latent}, "
            f"pooling of splits: '{self.module.split_aggregation}', \n"
            f"Encoder dims: {encoder_dims}, \n"
            f"Decoder dims: {decoder_dims}, \n"
            f"Gene likelihood: {self.module.gene_likelihood}, \n"
        )
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: str | None = None,
        layer: str | None = None,
        is_count_data: bool = True,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ) -> None:
        """
        %(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_labels_key)s
        %(param_layer)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s

        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        setup_method_args["drvi_version"] = drvi.__version__

        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=is_count_data),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            FixedCategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @classmethod
    def setup_merlin_data(
        cls,
        merlin_data: MerlinData,
        labels_key: str | None = None,
        layer: str = "X",
        is_count_data: bool = True,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Setup MerlinData for use with DRVI.

        Parameters
        ----------
        merlin_data
            MerlinData object to register.
        labels_key
            Key in `merlin_data` for labels.
        layer
            key in `merlin_data` to use as input.
        is_count_data
            Whether the data is count data.
        categorical_covariate_keys
            List of categorical covariate keys in `merlin_data`.
        continuous_covariate_keys
            List of continuous covariate keys in `merlin_data`.
        **kwargs
            Additional keyword arguments passed to the MerlinDataManager registration.

        Returns
        -------
        None
            This method sets up the class for use with MerlinData and does not return anything.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        setup_method_args["drvi_version"] = drvi.__version__

        fields = [
            melin_fields.MerlinLayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=is_count_data),
            melin_fields.MerlinCategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            melin_fields.MerlinCategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            melin_fields.MerlinNumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        merlin_manager = MerlinDataManager(fields, setup_method_args=setup_method_args)
        merlin_manager.register_fields(merlin_data, **kwargs)
        cls.register_manager(merlin_manager)

    def _make_data_loader(
        self,
        adata: AnnData | MerlinData,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        shuffle: bool = False,
        data_loader_class: type | None = None,
        **data_loader_kwargs,
    ) -> Any:
        """Create a AnnDataLoader object for data iteration.

        Parameters
        ----------
        adata
            AnnData or MerlinData object with equivalent structure to initial AnnData.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model.
        shuffle
            Whether observations are shuffled each iteration though.
        data_loader_class
            Class to use for data loader.
        data_loader_kwargs
            Kwargs to the class-specific data loader class.

        Returns
        -------
        Any
            Data loader object for iteration.
        """
        if isinstance(adata, AnnData):
            return super()._make_data_loader(
                adata, indices, batch_size, shuffle, data_loader_class, **data_loader_kwargs
            )
        elif MerlinData is not None and isinstance(adata, MerlinData):
            adata_manager = self.get_anndata_manager(adata)
            if adata_manager is None:
                raise AssertionError(
                    "AnnDataManager not found. Call `self._validate_anndata` prior to calling this function."
                )
            if batch_size is None:
                batch_size = settings.batch_size
            return MerlinTransformedDataLoader(
                self.adata_manager.get_dataset("default"),
                mapping=self.adata_manager.get_fields_schema_mapping(),
                batch_size=batch_size,
                shuffle=shuffle,
                parts_per_chunk=1,
                **data_loader_kwargs,
            )
        else:
            raise ValueError(
                "Only AnnData and MerlinData are supported. "
                "If you have passed an instance of MerlinData and still get this error, "
                "make sure merlin is installed as a dependency."
            )
