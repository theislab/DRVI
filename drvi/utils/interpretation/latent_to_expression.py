import numpy as np
import math
import torch
import scvi
from drvi import DRVI
import anndata as ad
import pandas as pd



def make_generative_samples_to_inspect(
        model,
        noise_stds = 0.5,
        span_limit = 3,
        n_steps = 10 * 2,
        n_samples = 100,
    ):
    assert n_steps % 2 == 0
    n_latent = model.module.n_latent
    if model.adata_manager.get_state_registry(scvi.REGISTRY_KEYS.CAT_COVS_KEY):
        n_cats_per_key = model.adata_manager.get_state_registry(scvi.REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
    else:
        n_cats_per_key = None
    if isinstance(model, scvi.model.SCVI) and model.summary_stats.n_batch > 0:
        n_cats_per_key = [model.summary_stats.n_batch] + list(n_cats_per_key if n_cats_per_key is not None else [])

    if isinstance(noise_stds, torch.Tensor):
        dim_stds = noise_stds
    elif isinstance(noise_stds, np.ndarray):
        dim_stds = torch.from_numpy(noise_stds)
    else:
        dim_stds = noise_stds * torch.ones(n_latent)

    random_small_samples = torch.randn(1, 1, n_samples, n_latent) * dim_stds  # 1 x 1 x n_samples x n_latent
    if isinstance(span_limit, (int, float)):
        span_limit = [-span_limit * np.ones(n_latent), +span_limit * np.ones(n_latent)]
    assert (span_limit[0] < 0).all() and (span_limit[1] > 0).all()
    latent_min = torch.from_numpy(span_limit[0])
    latent_max = torch.from_numpy(span_limit[1]) 

    per_dim_perturbations_neg = (
        torch.linspace(1, 0, int(n_steps // 2)).unsqueeze(0).unsqueeze(0)  # 1 x 1 x n_steps/2
        * 
        torch.diag(latent_min).unsqueeze(-1)  # n_latent x n_latent x 1
    )  # n_latent x n_latent x n_steps/2
    per_dim_perturbations_pos = (
        torch.linspace(0, 1, int(n_steps // 2)).unsqueeze(0).unsqueeze(0)  # 1 x 1 x n_steps/2
        * 
        torch.diag(latent_max).unsqueeze(-1)  # n_latent x n_latent x 1
    )  # n_latent x n_latent x n_steps/2
    per_dim_perturbations = (
        torch.concat([per_dim_perturbations_neg, per_dim_perturbations_pos], dim=-1)  # n_latent x n_latent x n_steps
        .transpose(-1, -2).unsqueeze(-2)
    )  # n_latent x n_steps x 1 x n_latent

    control_data = random_small_samples + 0 * per_dim_perturbations
    effect_data = random_small_samples + 1 * per_dim_perturbations

    if n_cats_per_key is not None:
        random_cats = torch.from_numpy(np.stack([np.random.randint(0, n_cat, size=n_samples) for n_cat in n_cats_per_key], axis=1))
        random_cats = random_cats.reshape(1, 1, *random_cats.shape).expand(*(per_dim_perturbations.shape[:-2]), -1, -1)
    else:
        random_cats = None
    
    return control_data, effect_data, random_cats


def get_generative_output(model, z, cat_key=None, cont_key=None, batch_size=256):
    z_shape = z.shape
    z = z.reshape(-1, z.shape[-1])
    if cat_key is not None:
        cat_key = cat_key.reshape(-1, cat_key.shape[-1])
    mean_param = []
    model.module.eval()
    with torch.no_grad():
        for slice in torch.split(torch.arange(z.shape[0]), batch_size):
            if isinstance(model, DRVI):
                lib_tensor = torch.tensor([1e4] * slice.shape[0])
                cat_tensor = cat_key[slice] if cat_key is not None else None
                batch_tensor = None
            elif isinstance(model, scvi.model.SCVI):
                lib_tensor = torch.log(torch.tensor([1e4] * slice.shape[0])).reshape(-1, 1)
                cat_tensor = cat_key[slice, 1:] if cat_key is not None else None
                batch_tensor = cat_key[slice, 0:1] if cat_key is not None else None
            
            gen_input = model.module._get_generative_input(
                tensors={
                    scvi.REGISTRY_KEYS.BATCH_KEY: batch_tensor,
                    scvi.REGISTRY_KEYS.LABELS_KEY: None,
                    scvi.REGISTRY_KEYS.CONT_COVS_KEY: torch.log(lib_tensor).unsqueeze(-1) if model.summary_stats.get("n_extra_continuous_covs", 0) == 1 else None,
                    scvi.REGISTRY_KEYS.CAT_COVS_KEY: cat_tensor,
                },
                inference_outputs={
                    'z': z[slice],
                    'library': lib_tensor,
                    'gene_likelihood_additional_info': {},
                },
            )
            gen_output = model.module.generative(**gen_input)
            if isinstance(model, DRVI):
                output = gen_output['params']['mean'].detach().cpu()  # mean in log space
            elif isinstance(model, scvi.model.SCVI):
                output = np.log1p(gen_output['px'].mean.detach().cpu())
            
            mean_param.append(output)
    mean_param = torch.cat(mean_param, dim=0)
    mean_param = mean_param.reshape(*z_shape[:-1], mean_param.shape[-1])
    return mean_param


def make_effect_adata(control_mean_param, effect_mean_param, var_info_df, span_limit, add_to_counts=1.):
    n_latent, n_steps, n_random_samples, n_vars = effect_mean_param.shape

    average_reshape_numpy = lambda x: x.mean(dim=-2).reshape(n_latent * n_steps, n_vars).numpy(force=True)
    add_eps_in_count_space = lambda x: torch.logsumexp(torch.stack([x, math.log(add_to_counts) * torch.ones_like(x)]), dim=0)

    diff_considering_small_values = average_reshape_numpy(
        add_eps_in_count_space(effect_mean_param) -
        add_eps_in_count_space(control_mean_param)
    )

    effect_adata = ad.AnnData(diff_considering_small_values, var=var_info_df)
    effect_adata.layers['diff'] = average_reshape_numpy(effect_mean_param - control_mean_param)
    effect_adata.layers['effect'] = average_reshape_numpy(effect_mean_param)
    effect_adata.layers['control'] = average_reshape_numpy(control_mean_param)
    effect_adata.obs['original_order'] = np.arange(effect_adata.n_obs)
    effect_adata.var['original_order'] = np.arange(effect_adata.n_vars)
    effect_adata.obs['dim_int'] = [1 + i // n_steps for i in range(n_latent * n_steps)]
    effect_adata.obs['dim'] = "Dim " + effect_adata.obs['dim_int'].astype(str)
    if isinstance(span_limit, (int, float)):
        span_limit = [-span_limit * np.ones(n_latent), +span_limit * np.ones(n_latent)]
    effect_adata.obs['value'] = np.concatenate([
        np.linspace(span_limit[0], span_limit[0]*0., num=int(n_steps / 2)),
        np.linspace(span_limit[1]*0., span_limit[1], num=int(n_steps / 2)),
    ], axis=0).T.reshape(-1)

    effect_adata.uns['effect_mean_param'] = effect_mean_param.numpy(force=True)
    effect_adata.uns['control_mean_param'] = control_mean_param.numpy(force=True)
    effect_adata.uns['n_latent'] = n_latent
    effect_adata.uns['n_latent'] = n_latent
    effect_adata.uns['n_steps'] = n_steps
    effect_adata.uns['n_vars'] = n_vars
    effect_adata.uns['span_limit'] = list(span_limit)  # convert tuple to list to be able to save as h5ad
    return effect_adata


def mark_differential_vars(effect_adata, layer=None, key_added='affected_vars', min_lfc=1.):
    original_effect_adata = effect_adata
    original_effect_adata.uns[key_added] = {}
    effect_adata = effect_adata[effect_adata.obs.sort_values(['original_order']).index].copy()
    effect_adata = effect_adata[:, effect_adata.var.sort_values(['original_order']).index].copy()
    change_array = np.expand_dims(effect_adata.obs['value'].values.reshape(effect_adata.uns['n_latent'], effect_adata.uns['n_steps']), 
                                  axis=-1) # n_latent x n_steps x 1
    change_sign_array = np.sign(change_array)  # n_latent x n_steps x 1
    effect_array = effect_adata.X if layer is None else effect_adata.layers[layer]
    effect_array = (
        effect_array.reshape(effect_adata.uns['n_latent'], effect_adata.uns['n_steps'], effect_adata.uns['n_vars'])
    )  # n_latent x n_steps x n_vars
    for change_sign in ['-', '+']:
        sig_genes = {}
        for effect_sign in ['up', 'down']:
            max_effect = np.max(effect_array * (1. if effect_sign == 'up' else -1.) * (change_sign_array == float(f"{change_sign}1")),
                                axis=1) # n_latent x n_vars
            max_indices = np.argsort(-max_effect, axis=1)  # n_latent x n_vars
            num_sig_indices = np.sum(max_effect >= min_lfc, axis=1)  # n_latent
            sig_genes[effect_sign] = [
                list(zip(effect_adata.var.iloc[max_indices[i, :n]].index, max_effect[i, max_indices[i, :n]]))
                for i, n in enumerate(num_sig_indices)
            ]
        for dim in range(effect_adata.uns['n_latent']):
            original_effect_adata.uns[key_added][f'Dim {dim+1}{change_sign}'] = {
                'up': sig_genes['up'][dim],
                'down': sig_genes['down'][dim],
            }


def find_differential_vars(effect_adata, method='log1p', added_layer='effect', add_to_counts=1., relax_max_by=1.):
    assert method in ['log1p', 'relative']
    original_effect_adata = effect_adata
    original_effect_adata.uns[f'affected_vars'] = {}
    effect_adata = effect_adata[effect_adata.obs.sort_values(['original_order']).index].copy()
    effect_adata = effect_adata[:, effect_adata.var.sort_values(['original_order']).index].copy()

    control_mean_param = torch.from_numpy(effect_adata.uns['control_mean_param'])
    effect_mean_param = torch.from_numpy(effect_adata.uns['effect_mean_param'])
    n_latent, n_steps, n_random_samples, n_vars = effect_mean_param.shape

    average_reshape_numpy = lambda x: x.mean(dim=-2).reshape(n_latent * n_steps, n_vars).numpy(force=True)
    add_eps_in_count_space = lambda x: torch.logsumexp(torch.stack([x, math.log(add_to_counts) * torch.ones_like(x)]), dim=0)
    find_relative_effect = lambda x, max_possible: (torch.logsumexp(torch.stack([x, torch.zeros_like(x), max_possible]), dim=0) - max_possible)

    if method == 'log1p':
        diff_considering_small_values = average_reshape_numpy(
            add_eps_in_count_space(effect_mean_param) -
            add_eps_in_count_space(control_mean_param)
        )
    elif method == 'relative':
        max_possible_other_dims = (
            torch.maximum(
                torch.amax(effect_mean_param, dim=(1, 2,), keepdim=True),
                torch.amax(control_mean_param, dim=(1, 2,), keepdim=True),
            ).unsqueeze(0).expand(n_latent, -1, n_steps, n_random_samples, -1)
            .masked_fill(torch.eye(n_latent).bool().reshape(n_latent, n_latent, 1, 1, 1), -float('inf'))
            .amax(dim=1, keepdim=False)
        ) - relax_max_by # n_latent x n_steps x n_samples, n_vars
        normalized_effect_mean_param = find_relative_effect(effect_mean_param, max_possible_other_dims)
        normalized_control_mean_param = find_relative_effect(control_mean_param, max_possible_other_dims)
        diff_considering_small_values = average_reshape_numpy(
            normalized_effect_mean_param -
            normalized_control_mean_param
        )
    else:
        raise NotImplementedError()

    effect_adata.layers[added_layer] = diff_considering_small_values
    original_effect_adata.layers[added_layer] = effect_adata[original_effect_adata.obs.index, original_effect_adata.var.index].layers[added_layer]


def sort_and_filter_effect_adata(effect_adata, optimal_dim_ordering, min_lfc=2.):
    dim_order_mapping = {f"Dim {1 + o}": i for i, o in enumerate(optimal_dim_ordering)}
    effect_adata.obs['optimal_order'] = effect_adata.obs['dim'].apply(lambda x: dim_order_mapping[x]).astype(int).astype('category')
    effect_adata.obs['dim_id'] = effect_adata.obs['optimal_order'].apply(lambda x: f"Dim {1 + optimal_dim_ordering[x]}")
    effect_adata = effect_adata[effect_adata.obs.sort_values(['optimal_order', 'value']).index].copy()
    effect_adata.obs['dim_id_plus'] = np.where(effect_adata.obs['value'] > 0,
                                               effect_adata.obs['dim_id'].astype(str) + ' +',
                                               effect_adata.obs['dim_id'].astype(str) + ' -')
    effect_adata.obs['dim_id_plus'] = pd.Categorical(
        effect_adata.obs['dim_id_plus'], 
        categories=list(effect_adata.obs['dim_id_plus'].drop_duplicates()),
        ordered=True,
    )

    data = effect_adata.X
    n_steps = effect_adata.uns['n_steps']
    
    effect_adata.var['max_effect_change'] = np.array(np.abs(data).argmax(axis=0)).squeeze()
    effect_adata.var['max_effect'] = np.array(np.abs(data).max(axis=0)).squeeze()
    effect_adata.var['max_effect_order'] = np.where(
        effect_adata.var['max_effect'] > min_lfc,
        effect_adata.var['max_effect_change'] // n_steps,
        -1
    )
    effect_adata.var['max_effect_sign'] = np.where(effect_adata.var['max_effect'] == data.max(axis=0), +1, -1)
    order_dim_mapping = {**{v: k for k, v in dim_order_mapping.items()}, -1: 'NONE'}
    effect_adata.var['max_effect_dim'] = effect_adata.var['max_effect_order'].apply(lambda x: order_dim_mapping[x])
    effect_adata.var['max_effect_dim_plus'] = np.where(
        effect_adata.var['max_effect_order'] == -1,
        'NONE',
        effect_adata.obs['dim_id_plus'][effect_adata.var['max_effect_change']]
    )

    effect_adata = effect_adata[:, effect_adata.var.sort_values(["max_effect_order", "max_effect_dim_plus", "max_effect_sign", "max_effect"], ascending=[True, True, True, False]).index].copy()

    return effect_adata


def iterate_and_make_effect_adata(model, adata, n_samples=100, noise_stds=0.5, span_limit=3, n_steps=10 * 2 + 1, min_lfc=1.):
    control_data, effect_data, random_cats = make_generative_samples_to_inspect(
        model, noise_stds=noise_stds, span_limit=span_limit, n_steps=n_steps, n_samples=n_samples,
    )
    print("Input Latent shapes:", control_data.shape, effect_data.shape)
    control_mean_param = get_generative_output(model, control_data, cat_key=random_cats)
    effect_mean_param = get_generative_output(model, effect_data, cat_key=random_cats)
    print("Output shapes:", control_mean_param.shape, effect_mean_param.shape)
    effect_adata = make_effect_adata(control_mean_param, effect_mean_param, adata.var, span_limit)

    # find_differential_vars(effect_adata, method='log1p', added_layer='effect', add_to_counts=1.)
    # mark_differential_vars(effect_adata, layer='effect', min_lfc=min_lfc)
    return effect_adata
