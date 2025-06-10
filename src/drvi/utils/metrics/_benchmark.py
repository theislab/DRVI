import pickle

import numpy as np
import pandas as pd

from drvi.utils.metrics._aggregation import latent_matching_score, most_similar_averaging_score, most_similar_gap_score
from drvi.utils.metrics._pairwise import (
    discrete_mutual_info_score,
    local_mutual_info_score,
    nn_alignment_score,
    spearman_correlataion_score,
)

AVAILABLE_METRICS = {
    # ASC is generally unsuitable for discrete targets.
    # More info: https://www.biorxiv.org/content/10.1101/2024.11.06.622266v1.full.pdf lines 985 to 989
    "ASC": spearman_correlataion_score,
    "SPN": nn_alignment_score,
    # SMI-cont is not working as expected. More info: https://github.com/scikit-learn/scikit-learn/issues/30772
    "SMI-cont": local_mutual_info_score,
    "SMI-disc": discrete_mutual_info_score,
}


AVAILABLE_AGGREGATION_METHODS = {
    "LMS": latent_matching_score,
    "MSAS": most_similar_averaging_score,
    "MSGS": most_similar_gap_score,
}


class DiscreteDisentanglementBenchmark:
    version = "v2"

    def __init__(
        self,
        embed,
        discrete_target=None,
        one_hot_target=None,
        dim_titles=None,
        metrics=("SMI-disc", "SPN", "ASC"),
        aggregation_methods=("LMS", "MSAS", "MSGS"),
        additional_metric_params=None,
    ):
        if discrete_target is None and one_hot_target is None:
            raise ValueError("Either discrete_target or one_hot_target must be provided.")
        if discrete_target is not None and one_hot_target is not None:
            raise ValueError("Only one of discrete_target or one_hot_target should be provided.")

        if discrete_target is not None:
            if isinstance(discrete_target, pd.Series):
                discrete_target = discrete_target.astype("category")
            elif isinstance(discrete_target, np.ndarray):
                discrete_target = pd.Series(discrete_target, dtype="category")
            else:
                raise ValueError("discrete_target must be a pandas Series or numpy array")
            one_hot_target = pd.DataFrame(
                np.eye(len(discrete_target.cat.categories))[discrete_target.cat.codes],
                columns=discrete_target.cat.categories,
            )

        if isinstance(one_hot_target, pd.DataFrame):
            pass
        elif isinstance(one_hot_target, np.ndarray):
            one_hot_target = pd.DataFrame(
                one_hot_target, columns=[f"process_{i}" for i in range(one_hot_target.shape[1])]
            )
        else:
            raise ValueError("one_hot_target must be a pandas DataFrame or numpy array")

        if dim_titles is None:
            dim_titles = [f"dim_{d}" for d in range(embed.shape[1])]

        self.embed = embed.copy()
        self.one_hot_target = one_hot_target.copy()
        self.dim_titles = dim_titles
        self.metrics = metrics
        self.aggregation_methods = aggregation_methods
        self.additional_metric_params = additional_metric_params if additional_metric_params is not None else {}

        self.results = {}
        self.aggregated_results = {}

    def _compute_metrics(self, embed, one_hot_target, dim_titles=None, metrics=()):
        if dim_titles is None:
            dim_titles = [f"dim_{d}" for d in range(embed.shape[1])]

        results = {}
        for metric_name in metrics:
            metric_params = self.additional_metric_params.get(metric_name, {})
            result_df = pd.DataFrame(
                AVAILABLE_METRICS[metric_name](embed, gt_one_hot=one_hot_target.values, **metric_params),
                index=dim_titles,
                columns=one_hot_target.columns,
            )
            results[metric_name] = result_df

        return results

    @staticmethod
    def _aggregate_metrics(results, aggregation_methods=()):
        aggregated_results = {}
        for aggregation_method in aggregation_methods:
            for metric_name in results:
                aggregated_results[f"{aggregation_method}-{metric_name}"] = AVAILABLE_AGGREGATION_METHODS[
                    aggregation_method
                ](results[metric_name].values)
        return aggregated_results

    def is_complete(self):
        for metric in self.metrics:
            if metric not in self.results:
                return False
        for aggregation_method in self.aggregation_methods:
            for metric in self.metrics:
                if f"{aggregation_method}-{metric}" not in self.aggregated_results:
                    return False
        return True

    def evaluate(self):
        if not self.is_complete():
            remaining_metrics = [metric for metric in self.metrics if metric not in self.results]
            self.results = {
                **self.results,
                **self._compute_metrics(self.embed, self.one_hot_target, self.dim_titles, remaining_metrics),
            }
            # Aggregation is cheap. Do it always.
            self.aggregated_results = {
                **self.aggregated_results,
                **self._aggregate_metrics(self.results, self.aggregation_methods),
            }

    def get_results(self):
        return {
            f"{aggregation_method}-{metric}": self.aggregated_results[f"{aggregation_method}-{metric}"]
            for aggregation_method in self.aggregation_methods
            for metric in self.metrics
        }

    def get_results_details(self):
        return {f"{metric}": self.results[metric] for metric in self.metrics}

    def save(self, path):
        data = {
            "version": self.version,
            "results": self.results,
            "aggregated_results": self.aggregated_results,
            "metrics": self.metrics,
            "aggregation_methods": self.aggregation_methods,
            "dim_titles": self.dim_titles,
            "additional_metric_params": self.additional_metric_params,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path, embed, discrete_target=None, one_hot_target=None, metrics=None, aggregation_methods=None):
        with open(path, "rb") as f:
            data = pickle.load(f)

        assert cls.version == data["version"]
        if metrics is None:
            metrics = data["metrics"]
        if aggregation_methods is None:
            aggregation_methods = data["aggregation_methods"]
        instance = cls(
            embed,
            discrete_target,
            one_hot_target,
            data["dim_titles"],
            metrics,
            aggregation_methods,
            additional_metric_params=data["additional_metric_params"],
        )
        instance.results = data["results"]
        instance.aggregated_results = data["aggregated_results"]
        return instance

    @classmethod
    def load_results(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["aggregated_results"]

    @classmethod
    def load_results_details(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["results"]
