import numpy as np
import pandas as pd
import scvi
from anndata import AnnData
from scipy import sparse

from drvi.model import DRVI


def iterate_dimensions(
    # n_latent: int,
    latent_dims: np.ndarray,
    latent_min: np.ndarray,
    latent_max: np.ndarray,
    n_steps: int = 10 * 2,
    n_samples: int = 100,
):
    assert n_steps % 2 == 0
    # Sometimes it is negligibly not like below
    # assert np.all(latent_min <= 0) & np.all(latent_max >= 0)

    dim_ids = (
        latent_dims.reshape(-1, 1, 1)
        * np.ones(n_steps).astype(int).reshape(1, -1, 1)
        * np.ones(n_samples).astype(int).reshape(1, 1, -1)
    ).reshape(-1)  # n_latent * n_steps * n_samples
    sample_ids = (
        np.ones(len(latent_dims)).astype(int).reshape(-1, 1, 1)
        * np.ones(n_steps).astype(int).reshape(1, -1, 1)
        * np.arange(n_samples).astype(int).reshape(1, 1, -1)
    ).reshape(-1)  # n_latent * n_steps * n_samples
    step_ids = (
        np.ones(len(latent_dims)).astype(int).reshape(-1, 1, 1)
        * np.arange(n_steps).astype(int).reshape(1, -1, 1)
        * np.ones(n_samples).astype(int).reshape(1, 1, -1)
    ).reshape(-1)  # n_latent * n_steps * n_samples
    span_values = (
        np.concatenate(
            [
                np.linspace(latent_min, latent_min * 0.0, num=int(n_steps / 2)),
                np.linspace(latent_max * 0.0, latent_max, num=int(n_steps / 2)),
            ],
            axis=0,
        ).T.reshape(-1, 1)
        * np.ones(n_samples).reshape(1, -1)
    ).reshape(-1)  # n_latent * n_steps * n_samples

    span_vectors = sparse.coo_matrix((span_values, (np.arange(len(dim_ids)), dim_ids)), dtype=np.float32)
    span_vectors = span_vectors.tocsr()  # n_latent * n_steps * n_samples x n_latent

    span_adata = AnnData(
        X=span_vectors,
        obs=pd.DataFrame(
            {
                "original_order": np.arange(span_vectors.shape[0]),
                "dim_id": dim_ids,
                "sample_id": sample_ids,
                "step_id": step_ids,
                "span_value": span_values,
            }
        ),
    )

    return span_adata


def make_traverse_adata(
    model: DRVI,
    embed: AnnData,
    n_steps: int = 10 * 2,
    n_samples: int = 100,
    noise_formula: callable = lambda x: x / 2,
    max_noise_std: float = 0.2,
    copy_adata_var_info: bool = True,
    **kwargs,
):
    # Generate random delta vectors for each dimension
    span_adata = iterate_dimensions(
        latent_dims=embed.var["original_dim_id"].values,
        latent_min=embed.var["min"].values,
        latent_max=embed.var["max"].values,
        n_steps=n_steps,
        n_samples=n_samples,
    )

    # make baseline noise for each sample
    noise_std = noise_formula(embed.var["std"].values).clip(0, max_noise_std).reshape(1, -1)
    sample_wise_noises = np.random.randn(n_samples, embed.n_vars).astype(np.float32) * noise_std
    noise_vector = sample_wise_noises[span_adata.obs["sample_id"]]

    # make categorical covariates for each sample
    if model.adata_manager.get_state_registry(scvi.REGISTRY_KEYS.CAT_COVS_KEY):
        n_cats_per_key = model.adata_manager.get_state_registry(scvi.REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
        sample_wise_cats = np.stack([np.random.randint(0, n_cat, size=n_samples) for n_cat in n_cats_per_key], axis=1)
        cat_vector = sample_wise_cats[span_adata.obs["sample_id"]]
    else:
        cat_vector = None

    if model.adata_manager.get_state_registry(scvi.REGISTRY_KEYS.CONT_COVS_KEY):
        raise NotImplementedError("Interpretability of models with continuous covariates are not implemented yet.")

    # lib size
    lib_vector = np.ones(n_samples) * 1e4
    lib_vector = lib_vector[span_adata.obs["sample_id"]]

    # Control and effect latent data
    control_data = noise_vector
    effect_data = noise_vector + span_adata.X.A

    print("traversing latent ...")

    print(f"Input latent shape: control: {control_data.shape}, effect: {effect_data.shape}")
    # control and effect in mean parameter space
    control_mean_param = model.decode_latent_samples(
        control_data, lib=lib_vector, cat_values=cat_vector, cont_values=None, **kwargs
    )
    effect_mean_param = model.decode_latent_samples(
        effect_data, lib=lib_vector, cat_values=cat_vector, cont_values=None, **kwargs
    )
    print(f"Output mean param shape: control: {control_mean_param.shape}, effect: {effect_mean_param.shape}")

    if copy_adata_var_info:
        traverse_adata_var = model.adata.var.copy()
    else:
        traverse_adata_var = pd.DataFrame(index=model.adata.var_names)
    traverse_adata_var["original_order"] = np.arange(effect_mean_param.shape[1])

    traverse_adata = AnnData(
        X=effect_mean_param - control_mean_param,
        obs=span_adata.obs,
        var=traverse_adata_var,
    )
    traverse_adata.layers["control"] = control_mean_param
    traverse_adata.layers["effect"] = effect_mean_param
    traverse_adata.obsm["control_latent"] = control_data
    traverse_adata.obsm["effect_latent"] = effect_data
    if cat_vector is not None:
        traverse_adata.obsm["cat_covs"] = cat_vector
    traverse_adata.obs["lib_size"] = lib_vector

    return traverse_adata


def get_dimensions_of_traverse_data(traverse_adata: AnnData):
    # Get the number of latent dimensions, steps, samples, and vars
    n_latent = traverse_adata.obs["dim_id"].nunique()
    n_steps = traverse_adata.obs["step_id"].nunique()
    n_samples = traverse_adata.obs["sample_id"].nunique()
    n_vars = traverse_adata.n_vars
    return n_latent, n_steps, n_samples, n_vars


def traverse_latent(
    model: DRVI,
    embed: AnnData,
    n_steps=10 * 2,
    n_samples=100,
    copy_adata_var_info=True,
    **kwargs,
):
    if "original_dim_id" not in embed.var:
        raise ValueError(
            'Column "original_dim_id" not found in `embed.var`. Please run `set_latent_dimension_stats` to set vanished status.'
        )

    traverse_adata = make_traverse_adata(
        model=model,
        embed=embed,
        n_steps=n_steps,
        n_samples=n_samples,
        copy_adata_var_info=copy_adata_var_info,
        **kwargs,
    )

    # enrich traverse_adata with the additional info
    for col in ["title", "vanished", "order"]:
        mapping = dict(zip(embed.var["original_dim_id"].values, embed.var[col].values, strict=False))
        traverse_adata.obs[col] = traverse_adata.obs["dim_id"].map(mapping)

    return traverse_adata
