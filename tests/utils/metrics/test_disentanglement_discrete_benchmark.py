import numpy as np
import pandas as pd

from drvi.utils.metrics import DiscreteDisentanglementBenchmark


class TestDiscreteDisentanglementBenchmark:
    def make_test_data(self):
        categorical_features = pd.Series(pd.Categorical(np.repeat(["ct1", "ct2", "ct3"], 20)))
        categorical_features_01 = np.eye(len(categorical_features.cat.categories))[categorical_features.cat.codes]

        fit_continuous_latent = np.random.normal(categorical_features_01, 0.1)
        random_continuous_latent = np.random.normal(0, 1, fit_continuous_latent.shape)

        return categorical_features, categorical_features_01, fit_continuous_latent, random_continuous_latent

    def _general_test_pipeline(
        self, continuous_latent, categorical_features=None, categorical_features_01=None, **kwargs
    ):
        METRICS = ["ASC", "SPN", "SMI-cont", "SMI-disc"]
        AGGREGATION_METHODS = ["LMS", "MSAS", "MSGS"]

        benchmark = DiscreteDisentanglementBenchmark(
            continuous_latent,
            discrete_target=categorical_features,
            one_hot_target=categorical_features_01,
            metrics=METRICS,
            aggregation_methods=AGGREGATION_METHODS,
            **kwargs,
        )
        benchmark.evaluate()
        return benchmark

    def test_benchmarker_for_categorical_gt(self):
        categorical_features, categorical_features_01, fit_continuous_latent, random_continuous_latent = (
            self.make_test_data()
        )

        self._general_test_pipeline(
            continuous_latent=fit_continuous_latent,
            categorical_features=categorical_features,
        )

    def test_benchmarker_for_onehot_gt(self):
        categorical_features, categorical_features_01, fit_continuous_latent, random_continuous_latent = (
            self.make_test_data()
        )

        self._general_test_pipeline(
            continuous_latent=fit_continuous_latent, categorical_features_01=categorical_features_01
        )

    def test_benchmarker_make_sure_categorical_and_one_hot_have_same_result(self):
        categorical_features, categorical_features_01, fit_continuous_latent, random_continuous_latent = (
            self.make_test_data()
        )

        benchmark_1 = self._general_test_pipeline(
            continuous_latent=fit_continuous_latent, categorical_features_01=categorical_features_01
        )

        benchmark_2 = self._general_test_pipeline(
            continuous_latent=fit_continuous_latent, categorical_features_01=categorical_features_01
        )

        results_1 = benchmark_1.get_results()
        results_2 = benchmark_2.get_results()

        for metric in results_1:
            assert np.allclose(results_1[metric], results_2[metric]), f"Results for {metric} do not match"

    def test_benchmarker_make_sure_random_latent_have_worse_than_gt_results(self):
        categorical_features, categorical_features_01, fit_continuous_latent, random_continuous_latent = (
            self.make_test_data()
        )

        benchmark_gt = self._general_test_pipeline(
            continuous_latent=fit_continuous_latent,
            categorical_features=categorical_features,
        )

        benchmark_random = self._general_test_pipeline(
            continuous_latent=random_continuous_latent,
            categorical_features=categorical_features,
        )

        results_gt = benchmark_gt.get_results()
        results_random = benchmark_random.get_results()

        for metric in results_gt:
            assert np.all(results_random[metric] < results_gt[metric]), f"Results for {metric} do not match"

    def test_benchmarker_with_additional_metric_params(self):
        categorical_features, categorical_features_01, fit_continuous_latent, random_continuous_latent = (
            self.make_test_data()
        )

        self._general_test_pipeline(
            continuous_latent=fit_continuous_latent,
            categorical_features=categorical_features,
            additional_metric_params={"SMI-disc": {"n_bins": 3}},
        )
