from anndata import AnnData
from typing import Optional


def make_balanced_subsample(adata, col, min_count=10):
    n_sample_per_cond = adata.obs[col].value_counts().min()
    balanced_sample_index = (
        adata.obs
        .groupby(col)
        .sample(n=max(min_count, n_sample_per_cond), random_state=0, replace=n_sample_per_cond < min_count)
        .index
    )
    adata = adata[balanced_sample_index].copy()
    return adata


def iterate_on_top_differential_vars(
        traverse_adata: AnnData,
        key: str,
        title_col: str = 'title',
        order_col: str = 'order',
        gene_symbols: Optional[str] = None,
        score_threshold: float = 0.,
    ):
    """
    Make an iterator of the top differential variables per latent dimension.

    Parameters
    ----------
    traverse_adata
        Anndata object with the split effects.
    key
        Key to the differential variables in traverse_adata.
    title_col
        Title columns defining title of the dimensions.
    order_col
        Order column that defines latent dimension orders.
    gene_symbols
        Column with the gene symbols in traverse_adata.var.
    score_threshold
        A threshold to filter the differential variables.
    """
    df_pos = traverse_adata.varm[f'{key}_traverse_effect_pos'].copy()
    df_neg = traverse_adata.varm[f'{key}_traverse_effect_neg'].copy()

    if gene_symbols is not None:
        gene_name_mapping = dict(zip(traverse_adata.var.index, traverse_adata.var[gene_symbols]))
    else:
        gene_name_mapping = dict(zip(traverse_adata.var.index, traverse_adata.var.index))
    df_pos.index = df_pos.index.map(gene_name_mapping)
    df_neg.index = df_neg.index.map(gene_name_mapping)

    de_info = dict(
        **{(str(k) + "+"): v.sort_values(ascending=False).where(lambda x : x > score_threshold).dropna()
           for k, v in df_pos.to_dict(orient='series').items()
        },
        **{(str(k) + "-"): v.sort_values(ascending=False).where(lambda x : x > score_threshold).dropna()
           for k, v in df_neg.to_dict(orient='series').items()
        },
    )

    return [
        (f"{row[title_col]}{direction}", de_info[f"{row['dim_id']}{direction}"])
        for i, row in traverse_adata.obs[['dim_id', order_col, title_col]].drop_duplicates().sort_values(order_col).iterrows()
        for direction in ['-', '+']
        if len(de_info[f"{row['dim_id']}{direction}"]) > 0
    ]
