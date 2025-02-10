import re
from collections import namedtuple

import anndata as ad
import numpy as np
import pandas as pd

FeatureInfo = namedtuple("FeatureInfo", ["name", "dim", "keywords"])


class FeatureInfoList:
    def __init__(self, feature_info_str_list: list[str], axis="var", total_dim=None, default_dim=None):
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
    def parse(feature_info_list):
        pattern = re.compile(r"\A(?P<name>\w+)(@(?P<dim>\d*))?(?P<kw>(!\w+)+)?\Z")

        for feature_info in feature_info_list:
            match = pattern.match(feature_info)
            name, dim, kw = match.group("name"), match.group("dim"), match.group("kw")
            dim = None if dim is None else int(dim)
            kw = () if kw is None else tuple(kw[1:].split("!"))
            yield FeatureInfo(name, dim, kw)

    def _fill_with_default_dim(self, default_dim):
        for i in range(len(self.feature_info_list)):
            if self.feature_info_list[i].dim is None:
                self.feature_info_list[i] = self.feature_info_list[i]._replace(dim=default_dim)

    def _fill_with_total_dim(self, total_dim):
        n_none_dims = sum([(fi.dim is None) + 0.0 for fi in self.feature_info_list])
        if n_none_dims > 0:
            remaining_dims = total_dim - sum([fi.dim for fi in self.feature_info_list if fi.dim is not None])
            assert remaining_dims >= 0 and remaining_dims % n_none_dims == 0
            fill_value = int(remaining_dims // n_none_dims)
            self._fill_with_default_dim(fill_value)
        assert sum([fi.dim for fi in self.feature_info_list]) == total_dim

    @property
    def names(self):
        return [fi.name for fi in self.feature_info_list]

    @property
    def dims(self):
        return [fi.dim for fi in self.feature_info_list]

    @property
    def keywords_list(self):
        return [fi.keywords for fi in self.feature_info_list]

    def get_possible_values_array(self, data: dict[str, ad.AnnData]):
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

    def __len__(self):
        return len(self.feature_info_list)

    def __iter__(self):
        return self.feature_info_list.__iter__()

    def __repr__(self):
        return repr(self.feature_info_list)
