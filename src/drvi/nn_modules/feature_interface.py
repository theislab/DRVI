import re
from collections import namedtuple
from collections.abc import Generator
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

FeatureInfo = namedtuple("FeatureInfo", ["name", "dim", "keywords"])


class FeatureInfoList:
    """A list of feature information for managing feature dimensions and metadata.

    This class parses and manages feature information strings that specify
    feature names, dimensions, and keywords. It's used to organize features
    in multi-modal datasets where different features may have different
    dimensionalities.

    Parameters
    ----------
    feature_info_str_list
        List of feature information strings in the format "name[@dim][!keyword1!keyword2...]".
    axis
        Whether features are stored in AnnData.var or AnnData.obs.
    total_dim
        Total dimensionality to distribute among features with unspecified dimensions.
    default_dim
        Default dimension to assign to features with unspecified dimensions.

    Notes
    -----
    Feature information strings follow the pattern:
    - "name": Feature name only
    - "name@dim": Feature name with specified dimension
    - "name!keyword": Feature name with keyword
    - "name@dim!keyword1!keyword2": Feature name with dimension and keywords

    Examples
    --------
    >>> # Create feature info list with explicit dimensions
    >>> feature_list = FeatureInfoList(["gene@1000", "protein@50", "metadata@10"])
    >>> print(feature_list.names)  # ['gene', 'protein', 'metadata']
    >>> print(feature_list.dims)  # [1000, 50, 10]
    >>> # Create feature info list with total dimension
    >>> feature_list = FeatureInfoList(
    ...     [
    ...         "gene@1000",
    ...         "protein",  # dimension will be inferred
    ...         "metadata",
    ...     ],
    ...     total_dim=1100,
    ... )
    >>> print(feature_list.dims)  # [1000, 50, 50]
    """

    def __init__(
        self,
        feature_info_str_list: list[str],
        axis: str = "var",
        total_dim: int | None = None,
        default_dim: int | None = None,
    ) -> None:
        assert axis in ["var", "obs"]
        assert total_dim is None or default_dim is None
        self.feature_info_list = list(self.parse(feature_info_str_list))
        self.axis = axis
        if any(fi.dim is None for fi in self.feature_info_list):
            if total_dim is None and default_dim is None:
                raise ValueError(f"missing dim in {feature_info_str_list}\nPlease provide `total_dim` or `default_dim`")
            if total_dim is not None:
                self._fill_with_total_dim(total_dim)
            if default_dim is not None:
                self._fill_with_default_dim(default_dim)

    @staticmethod
    def parse(feature_info_list: list[str]) -> Generator[FeatureInfo, int | None, list[tuple[str, ...]] | None]:
        """Parse feature information strings into FeatureInfo objects.

        Parameters
        ----------
        feature_info_list
            List of feature information strings to parse.

        Yields
        ------
        FeatureInfo
            Parsed feature information objects.

        Notes
        -----
        The parsing uses a regex pattern to extract:
        - name: Feature name (required)
        - dim: Dimension (optional, converted to int)
        - keywords: List of keywords (optional, separated by !)
        """
        pattern = re.compile(r"\A(?P<name>\w+)(@(?P<dim>\d*))?(?P<kw>(!\w+)+)?\Z")

        for feature_info in feature_info_list:
            match = pattern.match(feature_info)
            if match is None:
                raise ValueError(f"Invalid feature info format: {feature_info}")
            name, dim, kw = match.group("name"), match.group("dim"), match.group("kw")
            dim = None if dim is None else int(dim)
            kw = () if kw is None else tuple(kw[1:].split("!"))
            yield FeatureInfo(name, dim, kw)

    def _fill_with_default_dim(self, default_dim: int) -> None:
        """Fill unspecified dimensions with a default value.

        Parameters
        ----------
        default_dim
            Default dimension to assign to features with unspecified dimensions.
        """
        for i in range(len(self.feature_info_list)):
            if self.feature_info_list[i].dim is None:
                self.feature_info_list[i] = self.feature_info_list[i]._replace(dim=default_dim)

    def _fill_with_total_dim(self, total_dim: int) -> None:
        """Distribute total dimension among features with unspecified dimensions.

        Parameters
        ----------
        total_dim
            Total dimensionality to distribute.

        Notes
        -----
        This method evenly distributes the remaining dimension among
        features that don't have specified dimensions. The distribution
        must result in integer dimensions.
        """
        n_none_dims = sum([(fi.dim is None) + 0.0 for fi in self.feature_info_list])
        if n_none_dims > 0:
            remaining_dims = total_dim - sum([fi.dim for fi in self.feature_info_list if fi.dim is not None])
            assert remaining_dims >= 0 and remaining_dims % n_none_dims == 0
            fill_value = int(remaining_dims // n_none_dims)
            self._fill_with_default_dim(fill_value)
        assert sum([fi.dim for fi in self.feature_info_list]) == total_dim

    @property
    def names(self) -> list[str]:
        """Get list of feature names.

        Returns
        -------
        list[str]
            List of feature names.
        """
        return [fi.name for fi in self.feature_info_list]

    @property
    def dims(self) -> list[int]:
        """Get list of feature dimensions.

        Returns
        -------
        list[int]
            List of feature dimensions.
        """
        return [fi.dim for fi in self.feature_info_list]

    @property
    def keywords_list(self) -> list[tuple[str, ...]]:
        """Get list of feature keywords.

        Returns
        -------
        list[tuple]
            List of keyword tuples for each feature.
        """
        return [fi.keywords for fi in self.feature_info_list]

    def get_possible_values_array(self, data: dict[str, ad.AnnData] | ad.AnnData) -> np.ndarray:
        """Get unique combinations of feature values across datasets.

        Parameters
        ----------
        data
            Dictionary of AnnData objects or single AnnData object.

        Returns
        -------
        numpy.ndarray
            Array of unique feature value combinations.

        Notes
        -----
        This method extracts unique combinations of feature values from
        the specified axis (obs or var) across all provided datasets.
        It's useful for understanding the range of possible values for
        categorical features.
        """
        keys = self.names
        if keys is None or len(keys) == 0:
            return np.array([])
        possible_vals = []
        if isinstance(data, ad.AnnData):
            data = {"dummy": data}
        for _, adada in data.items():
            if self.axis == "obs":
                df = adada.obs
            elif self.axis == "var":
                df = adada.var
            else:
                raise ValueError("axis should be either `obs` or `var`")
            possible_vals.append(df[list(keys)].drop_duplicates(subset=keys))
        return pd.concat(possible_vals).drop_duplicates(subset=keys).to_numpy()

    def __len__(self) -> int:
        """Get the number of features.

        Returns
        -------
        int
            Number of features in the list.
        """
        return len(self.feature_info_list)

    def __iter__(self) -> Any:
        """Iterate over feature information objects.

        Yields
        ------
        FeatureInfo
            Feature information objects.
        """
        return self.feature_info_list.__iter__()

    def __repr__(self) -> str:
        """String representation of the feature info list.

        Returns
        -------
        str
            String representation showing the feature information.
        """
        return repr(self.feature_info_list)
