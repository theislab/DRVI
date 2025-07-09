import numpy as np
from anndata import AnnData

from drvi.model import DRVI


def set_latent_dimension_stats(
    model: DRVI,
    embed: AnnData,
    inplace: bool = True,
    vanished_threshold: float = 0.1,
):
    """Set the latent dimension statistics of a DRVI embedding into var of an AnnData.

    This function computes and stores various statistics for each latent dimension
    in the embedding AnnData object. It calculates reconstruction effects, ordering,
    and basic statistical measures (mean, std, min, max) for each dimension.

    Parameters
    ----------
    model
        DRVI model object that has been trained and can compute reconstruction effects.
    embed
        AnnData object containing the latent representation (embedding) of the model.
        The latent dimensions should be in the `.X` attribute.
    inplace
        Whether to modify the input AnnData object in-place or return a new copy.
    vanished_threshold
        Threshold for determining if a latent dimension has "vanished" (become inactive).
        Dimensions with max absolute values below this threshold are marked as vanished.

    Returns
    -------
    AnnData or None
        If `inplace=True` (default), modifies the input AnnData object and returns None.
        If `inplace=False`, returns a new AnnData object with the statistics added.

    Notes
    -----
    The function adds the following columns to `embed.var`:

    - `original_dim_id`: Original dimension indices
    - `reconstruction_effect`: Reconstruction effect scores from the DRVI model
    - `order`: Ranking of dimensions by reconstruction effect (descending)
    - `max_value`: Maximum absolute value across all cells for each dimension
    - `mean`: Mean value across all cells for each dimension
    - `min`: Minimum value across all cells for each dimension
    - `max`: Maximum value across all cells for each dimension
    - `std`: Standard deviation of absolute values across all cells for each dimension
    - `title`: Dimension titles in format "DR {order+1}"
    - `vanished`: Boolean indicating if dimension is considered "vanished" (max_value < threshold)

    Examples
    --------
    >>> # Assuming you have a trained DRVI model and latent representation
    >>> latent_adata = model.get_latent_representation(adata, return_anndata=True)
    >>> set_latent_dimension_stats(model, latent_adata, inplace=True)
    >>> # Now latent_adata.var contains all the statistics
    >>> print(latent_adata.var[["order", "reconstruction_effect", "vanished"]].head())
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
