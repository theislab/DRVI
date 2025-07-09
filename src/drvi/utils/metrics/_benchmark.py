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
    """Benchmark for evaluating discrete disentanglement in latent representations.

    This class provides a comprehensive framework for evaluating how well
    latent dimensions capture discrete categorical variables (e.g., cell types,
    experimental conditions, biological processes). It supports multiple
    evaluation metrics and aggregation methods to provide robust assessment
    of disentanglement quality.

    Parameters
    ----------
    embed
        Latent representations to evaluate. Shape should be (n_samples, n_dimensions).
    discrete_target
        Discrete categorical target variable. Should contain categorical labels
        for each sample. Mutually exclusive with `one_hot_target`.
    one_hot_target
        One-hot encoded target variable. Shape should be of shape (n_samples, n_categories).
        Mutually exclusive with `discrete_target`.
    dim_titles
        Titles for each latent dimension. If None, will use "dim_0", "dim_1", etc..
    metrics
        Metrics to compute for evaluation. Available options:
        - "SMI-disc": Discrete mutual information score
        - "SPN": Nearest neighbor alignment score
        - "ASC": Spearman correlation score
        - "SMI-cont": Continuous mutual information score (SMI-cont is not working as expected. More info: https://github.com/scikit-learn/scikit-learn/issues/30772).
    aggregation_methods
        Methods to aggregate metric scores across dimensions. Available options:
        - "LMS": Latent matching score
        - "MSAS": Most similar averaging score
        - "MSGS": Most similar gap score.
    additional_metric_params
        Additional parameters to pass to specific metrics. Keys should be metric
        names, values should be parameter dictionaries.

    Attributes
    ----------
    embed
        Copy of the input latent representations.
    one_hot_target
        One-hot encoded target variable.
    dim_titles
        Titles for each latent dimension.
    metrics
        Metrics used for evaluation.
    aggregation_methods
        Aggregation methods used.
    additional_metric_params
        Additional parameters for metrics.
    results
        Raw metric results for each dimension and category.
    aggregated_results
        Aggregated scores across dimensions.

    Raises
    ------
    ValueError
        If neither `discrete_target` nor `one_hot_target` is provided.
        If both `discrete_target` and `one_hot_target` are provided.
        If `discrete_target` is not a pandas Series or numpy array.
        If `one_hot_target` is not a pandas DataFrame or numpy array.

    Notes
    -----
    The benchmark evaluates disentanglement by measuring how well each latent
    dimension captures information about discrete categorical variables. Higher
    scores indicate better disentanglement.

    **Available Metrics:**

    - **SMI-disc**: Discrete mutual information between latent dimensions and
      categorical targets. Measures how much information each dimension contains
      about the categorical variable.
    - **SPN**: Nearest neighbor alignment score. Measures how well the nearest
      neighbor structure in latent space preserves categorical relationships.
    - **ASC**: Spearman correlation score. Measures linear correlation between
      latent dimensions and categorical targets (less suitable for discrete targets).
    - **SMI-cont**: Continuous mutual information (SMI-cont is not working as expected.
      More info: https://github.com/scikit-learn/scikit-learn/issues/30772).

    **Aggregation Methods:**

    - **LMS**: Latent matching score. Finds the optimal matching between latent
      dimensions and categories. This aggregation discourages presense of multiple
      irrelevant biological processes in a single dimension.
    - **MSAS**: Most similar averaging score. Averages scores for the most
      similar dimension-category pairs.
    - **MSGS**: Most similar gap score. Measures the gap between the best and
      second-best matches. This aggregation discourages redundancy.

    Examples
    --------
    >>> # Basic usage with discrete targets
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Generate sample data
    >>> n_samples, n_dims = 1000, 10
    >>> embed = np.random.randn(n_samples, n_dims)
    >>> cell_types = pd.Series(np.random.choice(["A", "B", "C"], n_samples))
    >>> # Create benchmark
    >>> benchmark = DiscreteDisentanglementBenchmark(
    ...     embed, discrete_target=cell_types, metrics=("SMI-disc", "SPN"), aggregation_methods=("LMS", "MSAS")
    ... )
    >>> # Run evaluation
    >>> benchmark.evaluate()
    >>> results = benchmark.get_results()
    >>> print(f"LMS-SMI-disc: {results['LMS-SMI-disc']:.3f}")
    >>> # With one-hot targets
    >>> one_hot = pd.get_dummies(cell_types)
    >>> benchmark = DiscreteDisentanglementBenchmark(embed, one_hot_target=one_hot)
    """

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
        """Compute evaluation metrics for each dimension and category.

        Parameters
        ----------
        embed
            Latent representations to evaluate.
        one_hot_target
            One-hot encoded target variable.
        dim_titles
            Titles for each latent dimension.
        metrics
            Metrics to compute.

        Returns
        -------
        dict
            Dictionary with metric names as keys and pandas DataFrames as values.
            Each DataFrame has dimensions as rows and categories as columns.
        """
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
        """Aggregate metric scores across dimensions.

        Parameters
        ----------
        results
            Raw metric results from `_compute_metrics`.
        aggregation_methods
            Aggregation methods to apply.

        Returns
        -------
        dict
            Dictionary with "{aggregation_method}-{metric_name}" as keys and
            aggregated scores as values.
        """
        aggregated_results = {}
        for aggregation_method in aggregation_methods:
            for metric_name in results:
                aggregated_results[f"{aggregation_method}-{metric_name}"] = AVAILABLE_AGGREGATION_METHODS[
                    aggregation_method
                ](results[metric_name].values)
        return aggregated_results

    def is_complete(self):
        """Check if all metrics and aggregations have been computed.

        Returns
        -------
        bool
            True if all requested metrics and aggregations are complete.
        """
        for metric in self.metrics:
            if metric not in self.results:
                return False
        for aggregation_method in self.aggregation_methods:
            for metric in self.metrics:
                if f"{aggregation_method}-{metric}" not in self.aggregated_results:
                    return False
        return True

    def evaluate(self):
        """Compute all required metrics and aggregations.

        This method computes any missing metrics and updates the aggregated
        results. It's safe to call multiple times - only missing computations
        will be performed.

        Notes
        -----
        The method performs the following steps:
        1. Identifies any missing metrics that need to be computed
        2. Computes the missing metrics using `_compute_metrics`
        3. Updates the aggregated results using `_aggregate_metrics`

        Aggregation is always performed as it's computationally cheap and
        ensures consistency with any new metric results.

        Examples
        --------
        >>> # Run evaluation
        >>> benchmark.evaluate()
        >>> # Check results
        >>> results = benchmark.get_results()
        >>> print(f"Number of results: {len(results)}")
        """
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
        """Get aggregated benchmark results.

        Returns
        -------
        dict
            Dictionary with "{aggregation_method}-{metric_name}" as keys and
            aggregated scores as values.

        Notes
        -----
        This method returns the final aggregated scores that summarize the
        disentanglement performance across all dimensions. Each key follows
        the pattern "{aggregation_method}-{metric_name}" (e.g., "LMS-SMI-disc").

        Examples
        --------
        >>> benchmark.evaluate()
        >>> results = benchmark.get_results()
        >>> for key, value in results.items():
        ...     print(f"{key}: {value:.3f}")
        """
        return {
            f"{aggregation_method}-{metric}": self.aggregated_results[f"{aggregation_method}-{metric}"]
            for aggregation_method in self.aggregation_methods
            for metric in self.metrics
        }

    def get_results_details(self):
        """Get detailed metric results for each dimension and category.

        Returns
        -------
        dict
            Dictionary with metric names as keys and pandas DataFrames as values.
            Each DataFrame shows scores for each dimension (rows) and category (columns).

        Notes
        -----
        This method returns the raw metric results before aggregation, allowing
        you to examine how each individual dimension performs for each category.
        This is useful for identifying which dimensions capture which categories
        and for debugging or detailed analysis.

        Examples
        --------
        >>> details = benchmark.get_results_details()
        >>> smi_scores = details["SMI-disc"]
        >>> print(f"SMI-disc shape: {smi_scores.shape}")
        >>> print(f"Best dimension for category A: {smi_scores['A'].idxmax()}")
        """
        return {f"{metric}": self.results[metric] for metric in self.metrics}

    def save(self, path):
        """Save benchmark results to a file.

        Parameters
        ----------
        path
            File path where to save the benchmark data.

        Notes
        -----
        The saved data includes:
        - Version information for compatibility
        - Raw metric results
        - Aggregated results
        - Configuration parameters (metrics, aggregation methods, etc.)

        The data is saved using Python's pickle format, which preserves
        all object structures and data types.

        Examples
        --------
        >>> benchmark.evaluate()
        >>> benchmark.save("benchmark_results.pkl")
        >>> # Load later
        >>> loaded_benchmark = DiscreteDisentanglementBenchmark.load(
        ...     "benchmark_results.pkl", embed, discrete_target=cell_types
        ... )
        """
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
        """Load a saved benchmark instance.

        Parameters
        ----------
        path
            File path to the saved benchmark data.
        embed
            Latent representations (must match the original data).
        discrete_target
            Discrete categorical target variable.
        one_hot_target
            One-hot encoded target variable.
        metrics
            Override the metrics from the saved data.
        aggregation_methods
            Override the aggregation methods from the saved data.

        Returns
        -------
        DiscreteDisentanglementBenchmark
            Loaded benchmark instance with all results restored.

        Raises
        ------
        AssertionError
            If the saved version doesn't match the current class version.
        FileNotFoundError
            If the specified file doesn't exist.
        ValueError
            If the target data is invalid (same as constructor).

        Notes
        -----
        This method creates a new benchmark instance with the same configuration
        as the saved one, but allows you to override metrics and aggregation
        methods if needed. The embed and target data must be provided to
        recreate the instance, but the actual results are loaded from the file.

        Examples
        --------
        >>> # Load with same configuration
        >>> benchmark = DiscreteDisentanglementBenchmark.load("results.pkl", embed, discrete_target=cell_types)
        """
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
        """Load only the aggregated results from a saved benchmark.

        Parameters
        ----------
        path
            File path to the saved benchmark data.

        Returns
        -------
        dict
            Dictionary with aggregated results (same format as `get_results()`).

        Notes
        -----
        This is a convenience method for quickly accessing just the final
        aggregated scores without needing to recreate the full benchmark
        instance. Useful when you only need the results for analysis or
        comparison.

        Examples
        --------
        >>> # Quick access to results
        >>> results = DiscreteDisentanglementBenchmark.load_results("results.pkl")
        >>> print(f"LMS-SMI-disc score: {results['LMS-SMI-disc']:.3f}")
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["aggregated_results"]

    @classmethod
    def load_results_details(cls, path):
        """Load only the detailed results from a saved benchmark.

        Parameters
        ----------
        path
            File path to the saved benchmark data.

        Returns
        -------
        dict
            Dictionary with detailed results (same format as `get_results_details()`).

        Notes
        -----
        This is a convenience method for quickly accessing the detailed
        metric results without needing to recreate the full benchmark
        instance. Useful for detailed analysis of dimension-category
        relationships.

        Examples
        --------
        >>> # Quick access to detailed results
        >>> details = DiscreteDisentanglementBenchmark.load_results_details("results.pkl")
        >>> smi_scores = details["SMI-disc"]
        >>> print(f"Best dimension for each category:")
        >>> for col in smi_scores.columns:
        ...     best_dim = smi_scores[col].idxmax()
        ...     print(f"  {col}: {best_dim}")
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["results"]
