import numpy as np
from anndata import AnnData

from drvi import DRVI
from drvi.utils import interpretability


def set_latent_dimension_stats(
        model: DRVI,
        embed: AnnData, 
        inplace: bool = True,
        vanished_threshold=0.1,
    ):
    """
    Get the latent dimension statistics of a DRVI model.

    Parameters
    ----------
    model
        DRVI model object.
    embed
        latent representation of the model.
    inplace
        Whether to modify the input AnnData object or return a new one.
    """
    if not inplace:
        embed = embed.copy()

    if 'original_dim_id' not in embed.var:
        embed.var['original_dim_id'] = np.arange(embed.var.shape[0])
    
    embed.var['reconstruction_effect'] = 0
    embed.var['reconstruction_effect'].loc[embed.var.sort_values('original_dim_id').index] = model.get_reconstruction_effect_of_each_split()
    embed.var['order'] = (-embed.var['reconstruction_effect']).argsort().argsort()
    
    embed.var['max_value'] = np.abs(embed.X).max(axis=0)
    embed.var['mean'] = embed.X.mean(axis=0)
    embed.var['min'] = embed.X.min(axis=0)
    embed.var['max'] = embed.X.max(axis=0)
    embed.var['std'] = np.abs(embed.X).std(axis=0)

    embed.var['title'] = 'DR ' + (1 + embed.var['order']).astype(str)
    embed.var['vanished'] = embed.var['max_value'] < vanished_threshold

    if not inplace:
        return embed


def traverse_latent(
        model: DRVI,
        embed: AnnData,
        n_steps = 10 * 2,
        n_samples = 100,
        **kwargs,
    ):
    if 'original_dim_id' not in embed.var:
        raise ValueError('Column "original_dim_id" not found in `embed.var`. Please run `set_latent_dimension_stats` to set vanished status.')
    
    traverse_adata = interpretability.traverse_the_latent(
        model=model,
        embed=embed,
        n_steps=n_steps,
        n_samples=n_samples,
        **kwargs,
    )
    
    # enrich traverse_adata with the additional info
    for col in ['title', 'vanished', 'order']:
        mapping = dict(zip(embed.var['original_dim_id'].values, embed.var[col].values))
        traverse_adata.obs[col] = traverse_adata.obs['dim_id'].map(mapping)

    return traverse_adata


def calculate_differential_vars(
        traverse_adata: AnnData,
        **kwargs
    ):
    print("Finding differential variables per latent dimension ...")
    interpretability.find_differential_effects(traverse_adata, method='max_possible', key_added='max_possible', **kwargs)
    interpretability.find_differential_effects(traverse_adata, method='min_possible', key_added='min_possible', **kwargs)
    
    def combine_function(min_possible, max_possible):
        # min_possible and max_possible dimensions: n_latent x n_steps x n_vars
        keep = (
            (max_possible >= 1.) &
            (
                (max_possible > max_possible.max(axis=(0, 1), keepdims=True) / 2) | 
                (min_possible > min_possible.max(axis=(0, 1), keepdims=True) / 10)
            )
        )
        score = np.where(
            keep,
            max_possible * min_possible,
            0
        )
        return score
    
    # Combine scores with product
    interpretability.combine_differential_effects(
        traverse_adata, 
        keys=['min_possible', 'max_possible'], 
        key_added='combined_score',
        combine_function=combine_function,
    )


def get_split_effects(
        model: DRVI,
        embed: AnnData,
        n_steps = 10 * 2,
        n_samples = 100,
        traverse_kwargs=None,
        de_kwargs=None,
    ):
    """
    Get the split effect of a latent dimension.

    Parameters
    ----------
    model
        DRVI model object.
    embed
        latent representation of the model.
    n_steps
        Number of steps.
    n_samples
        Number of samples.
    traverse_kwargs
        Additional arguments passed to `traverse_the_latent`.
    kwargs
        Additional arguments passed to `find_differential_vars`.
    """
    if traverse_kwargs is None:
        traverse_kwargs = {}
    if de_kwargs is None:
        de_kwargs = {}

    traverse_adata = traverse_latent(model, embed, n_steps=n_steps, n_samples=n_samples, **traverse_kwargs)
    calculate_differential_vars(traverse_adata, **de_kwargs)

    return traverse_adata
