from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import scvi
import torch
from scipy import sparse
from lightning import LightningDataModule
from scvi import REGISTRY_KEYS
from tqdm import tqdm

from drvi.scvi_tools_based.module._constants import MODULE_KEYS

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from anndata import AnnData

logger = logging.getLogger(__name__)


class GenerativeMixin:
    """Mixin class for generative model interpretation and analysis.

    This mixin provides methods for analyzing the generative part of variational
    autoencoder models. It includes utilities for decoding latent representations,
    iterating over model outputs, and analyzing the effects of different model
    components on the reconstruction.

    Notes
    -----
    This mixin is designed to work with scVI-based models that have a generative
    component. It provides high-level interfaces for:
    - Decoding latent samples to reconstruct data
    - Iterating over model outputs with custom functions
    - Analyzing reconstruction effects of different model splits
    - Computing maximum effects across the data distribution

    All methods operate in inference mode and handle batching automatically
    to manage memory usage with large datasets.
    """

    @torch.inference_mode()
    def iterate_on_decoded_latent_samples(
        self,
        z: np.ndarray,
        lib: np.ndarray | None = None,
        batch_values: np.ndarray | None = None,
        cat_values: np.ndarray | None = None,
        cont_values: np.ndarray | None = None,
        batch_size: int = scvi.settings.batch_size,
        map_cat_values: bool = False,
    ):
        """Iterate over decoder outputs as a generator.

        This method processes latent samples through the generative model in batches
        and yields the generative outputs for each batch.

        Parameters
        ----------
        z
            Latent samples with shape (n_samples, n_latent).
        lib
            Library size array with shape (n_samples,).
            If None, defaults to 1e4 for all samples.
        batch_values
            Batch values with shape (n_samples,).
            If None, defaults to 0 for all samples.
        cat_values
            Categorical covariates with shape (n_samples, n_cat_covs).
            Required if model has categorical covariates.
        cont_values
            Continuous covariates with shape (n_samples, n_cont_covs).
        batch_size
            Minibatch size for data loading into model.
        map_cat_values
            Whether to map categorical covariates to integers based on
            the AnnData manager pipeline.

        Yields
        ------
        dict[str, Any]
            Generative outputs for each batch.

        Notes
        -----
        This method operates in inference mode and processes data in batches
        to manage memory usage. The calling function should handle processing
        and aggregation of the yielded outputs.

        If map_cat_values is True, categorical values are automatically mapped
        to integers using the model's category mappings.

        Examples
        --------
        >>> import numpy as np
        >>> # Process latent samples and aggregate results
        >>> z = np.random.randn(50, 32)  # assuming 32 latent dimensions
        >>> store = []
        >>> for gen_output in model.iterate_on_decoded_latent_samples(z=z):
        ...     store.append(gen_output["px"].mean.detach().cpu())
        >>> result = torch.cat(store, dim=0).numpy()
        >>> print(result.shape)  # (50, n_genes)
        """
        self.module.eval()
        self.module.inspect_mode = True

        if cat_values is not None:
            if cat_values.ndim == 1:  # For a user not noticing cat_values should be 2d!
                cat_values = cat_values.reshape(-1, 1)
            if map_cat_values:
                mapped_values = np.zeros_like(cat_values)
                for i, (_label, map_keys) in enumerate(
                    self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)["mappings"].items()
                ):
                    cat_mapping = dict(zip(map_keys, range(len(map_keys)), strict=False))
                    mapped_values[:, i] = np.vectorize(cat_mapping.get)(cat_values[:, i])
                cat_values = mapped_values.astype(np.int32)

        if batch_values is not None:
            batch_values = batch_values.flatten()
            if map_cat_values:
                map_keys = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)["categorical_mapping"]
                batch_mapping = dict(zip(map_keys, range(len(map_keys)), strict=False))
                batch_values = np.vectorize(batch_mapping.get)(batch_values)
                batch_values = batch_values.astype(np.int32)
            batch_values = batch_values.reshape(-1, 1)

        try:
            for i in np.arange(0, z.shape[0], batch_size):
                slice = np.arange(i, min(i + batch_size, z.shape[0]))
                z_tensor = torch.tensor(z[slice])
                if lib is None:
                    lib_tensor = torch.tensor([1e4] * slice.shape[0])
                else:
                    lib_tensor = torch.tensor(lib[slice])
                cat_tensor = torch.tensor(cat_values[slice]) if cat_values is not None else None
                cont_tensor = torch.tensor(cont_values[slice]) if cont_values is not None else None
                batch_tensor = (
                    torch.tensor(batch_values[slice])
                    if batch_values is not None
                    else torch.zeros((slice.shape[0], 1), dtype=torch.int32)
                )

                if self.module.__class__.__name__ == "DRVIModule":
                    inference_outputs = {
                        MODULE_KEYS.Z_KEY: z_tensor,
                    }
                    library_to_inject = lib_tensor
                else:
                    raise NotImplementedError(f"Module {self.module.__class__.__name__} not supported.")

                gen_input = self.module._get_generative_input(
                    tensors={
                        REGISTRY_KEYS.BATCH_KEY: batch_tensor,
                        REGISTRY_KEYS.LABELS_KEY: None,
                        REGISTRY_KEYS.CONT_COVS_KEY: cont_tensor,
                        REGISTRY_KEYS.CAT_COVS_KEY: cat_tensor,
                    },
                    inference_outputs=inference_outputs,
                    library_to_inject=library_to_inject,
                )
                gen_output = self.module.generative(**gen_input)
                yield gen_output
        finally:
            self.module.inspect_mode = False

    @torch.inference_mode()
    def decode_latent_samples(
        self,
        z: np.ndarray,
        lib: np.ndarray | None = None,
        batch_values: np.ndarray | None = None,
        cat_values: np.ndarray | None = None,
        cont_values: np.ndarray | None = None,
        batch_size: int = scvi.settings.batch_size,
        map_cat_values: bool = False,
        return_in_log_space: bool = True,
    ) -> np.ndarray:
        r"""Return the distribution produced by the decoder for the given latent samples.

        This method computes :math:`p(x \mid z)`, the reconstruction distribution
        for given latent samples. It returns the mean of the reconstruction
        distribution for each sample.

        A user may use `model.get_normalized_expression` to get the normalized expression
        within distribution in count space in a more probabilistic way.

        Parameters
        ----------
        z
            Latent samples with shape (n_samples, n_latent).
        lib
            Library size array with shape (n_samples,).
            If None, defaults to 1e4 for all samples.
        batch_values
            Batch values with shape (n_samples,).
            If None, defaults to 0 for all samples.
        cat_values
            Categorical covariates with shape (n_samples, n_cat_covs).
            Required if model has categorical covariates.
        cont_values
            Continuous covariates with shape (n_samples, n_cont_covs).
        batch_size
            Minibatch size for data loading into model.
        map_cat_values
            Whether to map categorical covariates to integers based on
            the AnnData manager pipeline.
        return_in_log_space
            Whether to return the means in log space.

        Returns
        -------
        np.ndarray
            Reconstructed means with shape (n_samples, n_genes).

        Notes
        -----
        This method is equivalent to computing the expected value of the
        reconstruction distribution :math:`E[p(x \mid z)]`. It's useful for:
        - Generating synthetic data from latent samples
        - Analyzing model reconstructions
        - Visualizing the generative capabilities of the model

        Examples
        --------
        >>> import numpy as np
        >>> # Generate random latent samples
        >>> z = np.random.randn(100, 32)  # assuming 32 latent dimensions
        >>> # Decode to get reconstructed means
        >>> reconstructed = model.decode_latent_samples(z)
        >>> print(reconstructed.shape)  # (100, n_genes)
        >>> # With categorical covariates
        >>> cat_covs = np.array([0, 1, 0, 1] * 25)  # batch labels
        >>> reconstructed = model.decode_latent_samples(z, cat_values=cat_covs)
        """
        store: list[Any] = []
        for gen_output in self.iterate_on_decoded_latent_samples(
            z=z,
            lib=lib,
            batch_values=batch_values,
            cat_values=cat_values,
            cont_values=cont_values,
            batch_size=batch_size,
            map_cat_values=map_cat_values,
        ):
            if return_in_log_space:
                store.append(torch.log(gen_output[MODULE_KEYS.PX_KEY].mean).detach().cpu())
            else:
                store.append(gen_output[MODULE_KEYS.PX_KEY].mean.detach().cpu())
        return torch.cat(store, dim=0).numpy(force=True)

    @torch.inference_mode()
    def iterate_on_ae_output(
        self,
        adata: AnnData,
        datamodule: LightningDataModule | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        deterministic: bool = False,
    ):
        """Iterate over autoencoder outputs as a generator.

        This method processes data through the full autoencoder (encoder + decoder)
        and yields the outputs for each batch.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
            If None, defaults to the AnnData object used to initialize the model.
        datamodule
            LightningDataModule object with equivalent structure to initial AnnData. adata will be ignored if datamodule is provided.
        indices
            Indices of cells in adata to use. If None, all cells are used.
        batch_size
            Minibatch size for data loading into model.
            Defaults to scvi.settings.batch_size.
        deterministic
            Makes model fully deterministic (e.g., no sampling in the bottleneck).

        Yields
        ------
        tuple[dict[str, Any], dict[str, Any], Any]
            Tuple of (inference_outputs, generative_outputs, losses) for each batch.

        Notes
        -----
        This method processes data through both the encoder and decoder components
        of the model. The calling function should handle processing and aggregation
        of the yielded outputs.

        When deterministic=True, the model operates without stochastic sampling,
        which is useful for reproducible analysis.

        Examples
        --------
        >>> import anndata as ad
        >>> # Process data through autoencoder and aggregate results
        >>> store = []
        >>> for inference_outputs, generative_outputs, losses in model.iterate_on_ae_output(
        ...     adata=adata, deterministic=True
        ... ):
        ...     store.append(inference_outputs["qzm"].detach().cpu())
        >>> latents = torch.cat(store, dim=0).numpy()
        >>> print(latents.shape)  # (n_cells, n_latent)
        """
        if datamodule is None:
            adata = self._validate_anndata(adata)
            data_loader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        else:
            datamodule.setup(stage="predict")
            data_loader = datamodule.predict_dataloader()
        self.module.inspect_mode = True

        try:
            if deterministic:
                self.module.fully_deterministic = True
            for tensors in tqdm(data_loader, mininterval=5.0):
                loss_kwargs = {"kl_weight": 1}
                yield self.module(tensors, loss_kwargs=loss_kwargs)
        except Exception as e:
            self.module.fully_deterministic = False
            raise e
        finally:
            self.module.fully_deterministic = False
            self.module.inspect_mode = False


    @torch.inference_mode()
    def iterate_on_encoded_input(
        self,
        adata: AnnData,
        datamodule: LightningDataModule | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        deterministic: bool = False,
    ):
        """Iterate over inference outputs as a generator.

        This method processes data through the encoder (inference) component only
        and yields the inference outputs for each batch.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
            If None, defaults to the AnnData object used to initialize the model.
        datamodule
            LightningDataModule object with equivalent structure to initial AnnData.
            adata will be ignored if datamodule is provided.
        indices
            Indices of cells in adata to use. If None, all cells are used.
        batch_size
            Minibatch size for data loading into model.
            Defaults to scvi.settings.batch_size.
        deterministic
            Makes model fully deterministic (e.g., no sampling in the bottleneck).

        Yields
        ------
        dict[str, Any]
            Inference outputs for each batch, containing latent variables and
            other encoder outputs.

        Notes
        -----
        This method processes data through only the encoder (inference) component
        of the model, without running the decoder. The calling function should handle
        processing and aggregation of the yielded outputs.

        When deterministic=True, the model operates without stochastic sampling,
        which is useful for reproducible analysis.

        Examples
        --------
        >>> import anndata as ad
        >>> # Process data through encoder and aggregate results
        >>> store = []
        >>> for inference_outputs in model.iterate_onencoded_input(
        ...     adata=adata, deterministic=True
        ... ):
        ...     store.append(inference_outputs["qzm"].detach().cpu())
        >>> latents = torch.cat(store, dim=0).numpy()
        >>> print(latents.shape)  # (n_cells, n_latent)
        """
        if datamodule is None:
            adata = self._validate_anndata(adata)
            data_loader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        else:
            datamodule.setup(stage="predict")
            data_loader = datamodule.predict_dataloader()

        try:
            if deterministic:
                self.module.fully_deterministic = True
            for tensors in tqdm(data_loader, mininterval=5.0):
                outputs = self.module.inference(**self.module._get_inference_input(tensors))
                yield outputs
        except Exception as e:
            self.module.fully_deterministic = False
            raise e
        finally:
            self.module.fully_deterministic = False

    @torch.inference_mode()
    def generate_sparse_latent_representation(
        self,
        adata: AnnData | None = None,
        datamodule: LightningDataModule | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        zero_threshold: float = 0.,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Iterate over the data and generate the sparse latent representation for each cell.

        This method computes the sparse latent representation by applying sparsity
        constraints to the latent variables. The sparsity is controlled by the model's sparsity configuration.

        Parameters
        ----------
        adata
            AnnData object with observations. If None, uses the AnnData object
            from model setup.
        datamodule
            LightningDataModule object with equivalent structure to initial AnnData. adata will be ignored if datamodule is provided.
        indices
            Indices of cells to include. If None, all cells are used.
        batch_size
            Minibatch size for data loading. If None, uses the full data.
        zero_threshold
            Threshold for zeroing out the latent variables. Exact zero by default.
        **kwargs
            Additional keyword arguments passed to the iteration method.

        Returns
        -------
        generator[tuple[sparse.csr_matrix, sparse.csr_matrix]]
        returns a generator of (sparse latent means, sparse latent variances) for each cell.
        Both are scipy sparse matrices in CSR format.
        """
        self._check_if_trained(warn=False)

        for inference_outputs in self.iterate_on_encoded_input(
            adata=adata,
            datamodule=datamodule,
            deterministic=True,
            indices=indices,
            batch_size=batch_size,
            **kwargs,
        ):
            qz_m = inference_outputs[MODULE_KEYS.QZM_KEY]
            qz_v = inference_outputs[MODULE_KEYS.QZV_KEY]
            if zero_threshold > 0.:
                qz_m.masked_fill_(qz_m < zero_threshold, 0.)
            qz_v.masked_fill_(qz_m  == 0., 0.)
            qz_m = qz_m.detach().cpu().numpy(force=True)
            qz_v = qz_v.detach().cpu().numpy(force=True)
            qz_m = sparse.csr_matrix(qz_m)
            qz_v = sparse.csr_matrix(qz_v)

            yield qz_m, qz_v

    @torch.inference_mode()
    def get_sparse_latent_representation(
        self,
        adata: AnnData | None = None,
        datamodule: LightningDataModule | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        zero_threshold: float = 0.,
        return_dist: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return the sparse latent representation for each cell.

        This method computes the sparse latent representation by applying sparsity
        constraints to the latent variables. The sparsity is controlled by the model's sparsity configuration.

        Parameters
        ----------
        adata
            AnnData object with observations. If None, uses the AnnData object
            from model setup.
        datamodule
            LightningDataModule object with equivalent structure to initial AnnData. adata will be ignored if datamodule is provided.
        indices
            Indices of cells to include. If None, all cells are used.
        batch_size
            Minibatch size for data loading. If None, uses the full data.
        zero_threshold
            Threshold for zeroing out the latent variables. Exact zero by default.
        return_dist
            Whether to return both mean and variance of the latent distribution.
            If False, only returns the mean.
        **kwargs
            Additional keyword arguments passed to the iteration method.

        Returns
        -------
        sparse.csr_matrix | tuple[sparse.csr_matrix, sparse.csr_matrix]
            If return_dist=False, returns sparse matrix of latent means.
            If return_dist=True, returns tuple of (sparse latent means, sparse latent variances).
        """
        self._check_if_trained(warn=False)

        output = []
        for qz_m, qz_v in self.generate_sparse_latent_representation(
            adata=adata,
            datamodule=datamodule,
            indices=indices,
            batch_size=batch_size,
            zero_threshold=zero_threshold,
            **kwargs,
        ):
            output.append([qz_m, qz_v])

        output = list(zip(*output, strict=False))
        output = [sparse.vstack(output[i]).tocsr() for i in range(len(output))]

        if return_dist:
            return output[0], output[1]
        else:
            return output[0]
