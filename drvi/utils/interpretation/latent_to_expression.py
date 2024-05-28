import numpy as np
import torch
import scvi
from drvi import DRVI
import anndata as ad
import pandas as pd



def make_generative_samples_to_inspect(
        model,
        noise_stds = 0.5,
        span_limit = 3,
        n_steps = 10 * 2 + 1,
        n_samples = 100,
    ):
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
        dim_stds = torch.tensor(torch.from_numpy(noise_stds))
    else:
        dim_stds = noise_stds * torch.ones(n_latent)

    random_small_samples = torch.randn(1, 1, n_samples, n_latent) * dim_stds
    per_dim_perturbations = (
            torch.arange(-span_limit, span_limit + 1e-3, 2 * span_limit / (n_steps - 1)).unsqueeze(0).unsqueeze(0) * 
            torch.eye(n_latent).unsqueeze(-1)
    ).transpose(-1, -2).unsqueeze(-2)

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
                batch_tensor = cat_key[slice, 0:0] if cat_key is not None else None
            
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
            output = model.module.generative(**gen_input)['px'].mean.detach().cpu()
            mean_param.append(output)
    mean_param = torch.cat(mean_param, dim=0)
    mean_param = mean_param.reshape(*z_shape[:-1], mean_param.shape[-1])
    return mean_param


def make_effect_adata(control_mean_param, effect_mean_param, var_info_df, span_limit):
    n_latent, n_steps, n_random_samples, n_vars = effect_mean_param.shape

    average_reshape_numpy = lambda x: x.mean(dim=-2).reshape(n_latent * n_steps, n_vars).numpy(force=True)

    epsilon = 1
    diff_considering_small_values = average_reshape_numpy(
        torch.logsumexp(torch.stack([effect_mean_param, torch.ones_like(effect_mean_param) * epsilon]), dim=0) -
        torch.logsumexp(torch.stack([control_mean_param, torch.ones_like(control_mean_param) * epsilon]), dim=0)
    )

    effect_adata = ad.AnnData(diff_considering_small_values, var=var_info_df)
    effect_adata.layers['diff'] = average_reshape_numpy(effect_mean_param - control_mean_param)
    effect_adata.layers['effect'] = average_reshape_numpy(effect_mean_param)
    effect_adata.layers['control'] = average_reshape_numpy(control_mean_param)
    effect_adata.obs['dummy'] = 'X'
    effect_adata.var['dummy'] = 'X'
    effect_adata.obs['dim'] = [f"Dim {1 + i // n_steps}" for i in range(n_latent * n_steps)]
    effect_adata.obs['value'] = [-span_limit + (i % n_steps) * 2 * span_limit / (n_steps - 1) for i in range(n_latent * n_steps)]
    effect_adata.uns['n_latent'] = n_latent
    effect_adata.uns['n_steps'] = n_steps
    effect_adata.uns['n_vars'] = n_vars
    effect_adata.uns['span_limit'] = span_limit
    return effect_adata


def find_effective_vars(effect_adata, min_lfc=1.):
    original_effect_adata = effect_adata
    effect_adata = effect_adata[effect_adata.obs.sort_values(['dim', 'value']).index].copy()
    original_effect_adata.uns[f'affected_vars'] = [[] for _ in range(effect_adata.uns['n_latent'])]
    for change_sign in ['-', '+']:
        for effect_sign in ['-', '+']:
            change_sign_array = np.expand_dims(np.sign(effect_adata.obs['value']).values.reshape(effect_adata.uns['n_latent'], effect_adata.uns['n_steps']), axis=-1)
            x = (
                effect_adata.X.reshape(effect_adata.uns['n_latent'], effect_adata.uns['n_steps'], effect_adata.uns['n_vars'])
            )
            x = np.max(x * float(f"{effect_sign}1") * (change_sign_array == float(f"{change_sign}1")), axis=1)
            
            max_indices = np.argsort(-x, axis=1)
            num_sig_indices = np.sum(x >= min_lfc, axis=1)
            original_effect_adata.uns[f'affected_vars_{change_sign}_{effect_sign}'] = [effect_adata.var.iloc[max_indices[i, :n]].index for i, n in enumerate(num_sig_indices)]
            original_effect_adata.uns[f'affected_vars'] = [
                original_effect_adata.uns[f'affected_vars'][i] + list(original_effect_adata.uns[f'affected_vars_{change_sign}_{effect_sign}'][i])
                for i in range(original_effect_adata.uns['n_latent'])
            ]


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
    find_effective_vars(effect_adata, min_lfc=min_lfc)
    return effect_adata
