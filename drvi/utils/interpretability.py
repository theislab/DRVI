from typing import Literal

import numpy as np
import pandas as pd
import scipy
import scipy.special
import scvi
from anndata import AnnData
from scipy import sparse

from drvi import DRVI


def iterate_dimensions(
        # n_latent: int,
        latent_dims: np.ndarray, 
        latent_min: np.ndarray,
        latent_max: np.ndarray,
        n_steps: int = 10 * 2,
        n_samples: int = 100,
    ):
    assert n_steps % 2 == 0
    assert np.all(latent_min <= 0) & np.all(latent_max >= 0)

    dim_ids = (
        latent_dims.reshape(-1, 1, 1) * 
        np.ones(n_steps).astype(int).reshape(1, -1, 1) * 
        np.ones(n_samples).astype(int).reshape(1, 1, -1)
    ).reshape(-1)  # n_latent * n_steps * n_samples
    sample_ids = (
        np.ones(len(latent_dims)).astype(int).reshape(-1, 1, 1) * 
        np.ones(n_steps).astype(int).reshape(1, -1, 1) * 
        np.arange(n_samples).astype(int).reshape(1, 1, -1)
    ).reshape(-1)  # n_latent * n_steps * n_samples
    step_ids = (
        np.ones(len(latent_dims)).astype(int).reshape(-1, 1, 1) * 
        np.arange(n_steps).astype(int).reshape(1, -1, 1) *
        np.ones(n_samples).astype(int).reshape(1, 1, -1)
    ).reshape(-1)  # n_latent * n_steps * n_samples
    span_values = (
        np.concatenate([
            np.linspace(latent_min, latent_min*0., num=int(n_steps / 2)),
            np.linspace(latent_max*0., latent_max, num=int(n_steps / 2)),
        ], axis=0).T.reshape(-1, 1) *
        np.ones(n_samples).reshape(1, -1)
    ).reshape(-1)  # n_latent * n_steps * n_samples

    span_vectors = sparse.coo_matrix((span_values, (np.arange(len(dim_ids)), dim_ids)), dtype=np.float32)
    span_vectors = span_vectors.tocsr()  # n_latent * n_steps * n_samples x n_latent

    span_adata = AnnData(
        X=span_vectors,
        obs=pd.DataFrame({
            'original_order': np.arange(span_vectors.shape[0]),
            'dim_id': dim_ids,
            'sample_id': sample_ids,
            'step_id': step_ids,
            'span_value': span_values,
        }),
    )

    return span_adata


def traverse_the_latent(
        model: DRVI,
        embed: AnnData,
        n_steps: int = 10 * 2,
        n_samples: int = 100,
        noise_formula: callable = lambda x: x / 2,
        max_noise_std: float = 0.2,
        **kwargs,
    ):

    # Generate random delta vectors for each dimension
    span_adata = iterate_dimensions(
        latent_dims=embed.var['original_dim_id'].values,
        latent_min=embed.var['min'].values,
        latent_max=embed.var['max'].values,
        n_steps=n_steps,
        n_samples=n_samples,
    )
    
    # make baseline noise for each sample
    noise_std = noise_formula(embed.var['std'].values).clip(0, max_noise_std).reshape(1, -1)
    sample_wise_noises = np.random.randn(n_samples, embed.n_vars).astype(np.float32) * noise_std
    noise_vector = sample_wise_noises[span_adata.obs['sample_id']]

    # make categorical covariates for each sample
    if model.adata_manager.get_state_registry(scvi.REGISTRY_KEYS.CAT_COVS_KEY):
        n_cats_per_key = model.adata_manager.get_state_registry(scvi.REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
        sample_wise_cats = np.stack([np.random.randint(0, n_cat, size=n_samples) for n_cat in n_cats_per_key], axis=1)
        cat_vector = sample_wise_cats[span_adata.obs['sample_id']]
    else:
        cat_vector = None

    # lib size
    lib_vector = np.ones(n_samples) * 1e4
    lib_vector = lib_vector[span_adata.obs['sample_id']]

    # Control and effect latent data
    control_data = noise_vector
    effect_data = noise_vector + span_adata.X.A
    
    print("traversing latent ...")

    print(f"Input latent shape: control: {control_data.shape}, effect: {effect_data.shape}")
    # control and effect in mean parameter space
    control_mean_param = model.decode_latent_samples(
        control_data, lib=lib_vector, cat_key=cat_vector, cont_key=None, **kwargs)
    effect_mean_param = model.decode_latent_samples(
        effect_data, lib=lib_vector, cat_key=cat_vector, cont_key=None, **kwargs)
    print(f"Output mean param shape: control: {control_mean_param.shape}, effect: {effect_mean_param.shape}")

    traverse_adata = AnnData(
        X=effect_mean_param - control_mean_param,
        obs=span_adata.obs,
        var=pd.DataFrame({
            'original_order': np.arange(effect_mean_param.shape[1]),
        }, index=model.adata.var_names),
    )
    traverse_adata.layers['control'] = control_mean_param
    traverse_adata.layers['effect'] = effect_mean_param
    traverse_adata.obsm['control_latent'] = control_data
    traverse_adata.obsm['effect_latent'] = effect_data
    traverse_adata.obsm['cat_covs'] = cat_vector
    traverse_adata.obs['lib_size'] = lib_vector

    return traverse_adata


def get_dimensions_of_traverse_data(traverse_adata: AnnData):
    # Get the number of latent dimensions, steps, samples, and vars
    n_latent = traverse_adata.obs['dim_id'].nunique()
    n_steps = traverse_adata.obs['step_id'].nunique()
    n_samples = traverse_adata.obs['sample_id'].nunique()
    n_vars = traverse_adata.n_vars
    return n_latent, n_steps, n_samples, n_vars


def find_differential_effects(
        traverse_adata: AnnData, 
        method: Literal['max_possible', 'min_possible'] = 'max_possible', 
        key_added: str = 'effect', 
        add_to_counts: float = .1, 
        relax_max_by: float = 0.,
    ):
    assert method in ['max_possible', 'min_possible']

    # Reorder the traverse_adata to original order
    original_traverse_adata = traverse_adata
    traverse_adata = traverse_adata[traverse_adata.obs.sort_values(['original_order']).index].copy()
    traverse_adata = traverse_adata[:, traverse_adata.var.sort_values(['original_order']).index].copy()

    # Get the number of latent dimensions, steps, samples, and vars
    n_latent, n_steps, n_samples, n_vars = get_dimensions_of_traverse_data(traverse_adata)

    # Get the dim_id and span values
    span_values = traverse_adata.obs['span_value'].values.reshape(n_latent, n_steps, n_samples)
    assert np.allclose(span_values, span_values.max(axis=-1, keepdims=True))
    span_values = span_values[:, :, 0]  # n_latent x n_steps
    dim_ids = traverse_adata.obs['dim_id'].values.reshape(n_latent, n_steps, n_samples)[:, 0, 0]  # n_latent

    # Get the output mean parameters in 4D format
    control_mean_param = traverse_adata.layers['control'].reshape(n_latent, n_steps, n_samples, n_vars)
    effect_mean_param = traverse_adata.layers['effect'].reshape(n_latent, n_steps, n_samples, n_vars)

    # Helper functions
    average_over_samples = lambda x: x.mean(axis=2)
    add_eps_in_count_space = lambda x: scipy.special.logsumexp(np.stack([x, np.log(add_to_counts) * np.ones_like(x)]), axis=0)
    find_relative_effect = lambda x, baseline: (scipy.special.logsumexp(np.stack([x, np.log(add_to_counts) * np.ones_like(x), baseline]), axis=0) - baseline)

    # Find DE for each sample and average over samples
    if method == 'max_possible':
        diff_considering_small_values = average_over_samples(
            add_eps_in_count_space(effect_mean_param) -
            add_eps_in_count_space(control_mean_param)
        )  # n_latent x n_steps x n_vars
    elif method == 'min_possible':
        reduce_dims = (1, 2,)
        max_of_two = np.maximum(
            effect_mean_param.max(axis=reduce_dims, keepdims=True),
            control_mean_param.max(axis=reduce_dims, keepdims=True),
        ) # n_latent x 1 x n_samples, n_vars
        max_cumulative_possible_all = scipy.special.logsumexp(max_of_two, axis=0, keepdims=True) # 1 x 1 x n_samples, n_vars
        max_cumulative_possible_other_dims = np.log(np.exp(max_cumulative_possible_all) - np.exp(max_of_two)) - relax_max_by # n_latent x 1 x n_samples, n_vars
        max_cumulative_possible_other_dims = max_cumulative_possible_other_dims + np.zeros_like(effect_mean_param)  # n_latent x n_steps x n_samples, n_vars
        normalized_effect_mean_param = find_relative_effect(effect_mean_param, max_cumulative_possible_other_dims)  # n_latent x n_steps x n_samples, n_vars
        normalized_control_mean_param = find_relative_effect(control_mean_param, max_cumulative_possible_other_dims)  # n_latent x n_steps x n_samples, n_vars
        diff_considering_small_values = average_over_samples(
            normalized_effect_mean_param -
            normalized_control_mean_param
        ) # n_latent x n_steps x n_vars
    else:
        raise NotImplementedError()

    original_traverse_adata.uns[f"{key_added}_traverse_effect_stepwise"] = diff_considering_small_values

    # Find DE vars in positive and negative directions
    for effect_sign in ['pos', 'neg']:
        mask = np.where(span_values >= 0, 1, 0) if effect_sign == 'pos' else np.where(span_values <= 0, 1, 0)
        max_effect = np.max(np.expand_dims(mask, axis=-1) * diff_considering_small_values, axis=1)  # n_latent x n_vars
        max_effect = pd.DataFrame(max_effect.T, index=traverse_adata.var_names, columns=dim_ids)
        original_traverse_adata.varm[f"{key_added}_traverse_effect_{effect_sign}"] = max_effect.loc[original_traverse_adata.var_names].copy()
        original_traverse_adata.uns[f"{key_added}_traverse_effect_{effect_sign}_dim_ids"] = original_traverse_adata.varm[f"{key_added}_traverse_effect_{effect_sign}"].columns.values


def combine_differential_effects(
        traverse_adata: AnnData,
        keys: list[str],
        key_added: str,
        combine_function: callable,
    ):

    # Reorder the traverse_adata to original order
    original_traverse_adata = traverse_adata
    traverse_adata = traverse_adata[traverse_adata.obs.sort_values(['original_order']).index].copy()
    traverse_adata = traverse_adata[:, traverse_adata.var.sort_values(['original_order']).index].copy()

    # Get the number of latent dimensions, steps, samples, and vars
    n_latent, n_steps, n_samples, n_vars = get_dimensions_of_traverse_data(traverse_adata)

    # Get the dim_id and span values
    span_values = traverse_adata.obs['span_value'].values.reshape(n_latent, n_steps, n_samples)
    assert np.allclose(span_values, span_values.max(axis=-1, keepdims=True))
    span_values = span_values[:, :, 0]  # n_latent x n_steps
    dim_ids = traverse_adata.obs['dim_id'].values.reshape(n_latent, n_steps, n_samples)[:, 0, 0]  # n_latent

    # Combine effects
    combined_traverse_effect_stepwise = combine_function(
        *[traverse_adata.uns[f"{key}_traverse_effect_stepwise"] for key in keys]
    )

    original_traverse_adata.uns[f"{key_added}_traverse_effect_stepwise"] = combined_traverse_effect_stepwise

    # Find DE vars in positive and negative directions
    for effect_sign in ['pos', 'neg']:
        mask = np.where(span_values >= 0, 1, 0) if effect_sign == 'pos' else np.where(span_values <= 0, 1, 0)
        max_effect = np.max(np.expand_dims(mask, axis=-1) * combined_traverse_effect_stepwise, axis=1)  # n_latent x n_vars
        max_effect = pd.DataFrame(max_effect.T, index=traverse_adata.var_names, columns=dim_ids)
        original_traverse_adata.varm[f"{key_added}_traverse_effect_{effect_sign}"] = max_effect.loc[original_traverse_adata.var_names].copy()
        original_traverse_adata.uns[f"{key_added}_traverse_effect_{effect_sign}_dim_ids"] = original_traverse_adata.varm[f"{key_added}_traverse_effect_{effect_sign}"].columns.values

