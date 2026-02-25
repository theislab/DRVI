from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

    from drvi.model import DRVI


def set_latent_dimension_stats(
    model: DRVI,
    embed: AnnData,
    inplace: bool = True,
    vanished_threshold: float = 0.1,
) -> AnnData | None:
    """Set the latent dimension statistics of a DRVI embedding into var of an AnnData.

    This function delegates to the model's :meth:`~drvi.scvi_tools_based.model.base.InterpretabilityMixin.set_latent_dimension_stats`
    method. It is provided for backward compatibility; prefer calling
    ``model.set_latent_dimension_stats(embed, ...)`` directly.

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
    >>> # Backward-compatible usage
    >>> latent_adata = model.get_latent_representation(adata, return_anndata=True)
    >>> set_latent_dimension_stats(model, latent_adata, inplace=True)
    >>> # Preferred: call on the model directly
    >>> model.set_latent_dimension_stats(latent_adata, inplace=True)
    """
    warnings.warn(
        "drvi.utils.tl.set_latent_dimension_stats is deprecated; "
        "use model.set_latent_dimension_stats(embed, ...) instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    if not inplace:
        embed = embed.copy()
    model.set_latent_dimension_stats(embed, vanished_threshold=vanished_threshold)
    if not inplace:
        return embed
