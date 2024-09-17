import numpy as np
from anndata import AnnData

from drvi.model import DRVI


def set_latent_dimension_stats(
    model: DRVI,
    embed: AnnData,
    inplace: bool = True,
    vanished_threshold=0.1,
):
    """
    Set the latent dimension statistics of a DRVI model into var of an embedding anndata.

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

    if "original_dim_id" not in embed.var:
        embed.var["original_dim_id"] = np.arange(embed.var.shape[0])

    embed.var["reconstruction_effect"] = 0
    embed.var["reconstruction_effect"].loc[embed.var.sort_values("original_dim_id").index] = (
        model.get_reconstruction_effect_of_each_split()
    )
    embed.var["order"] = (-embed.var["reconstruction_effect"]).argsort().argsort()

    embed.var["max_value"] = np.abs(embed.X).max(axis=0)
    embed.var["mean"] = embed.X.mean(axis=0)
    embed.var["min"] = embed.X.min(axis=0)
    embed.var["max"] = embed.X.max(axis=0)
    embed.var["std"] = np.abs(embed.X).std(axis=0)

    embed.var["title"] = "DR " + (1 + embed.var["order"]).astype(str)
    embed.var["vanished"] = embed.var["max_value"] < vanished_threshold

    if not inplace:
        return embed
