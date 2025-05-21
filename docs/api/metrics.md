# Metrics

We have implemented a user-friendly class for evaluation of disentanglement with respect to known discrete targets.

```{eval-rst}
.. module:: drvi.utils.metrics
.. currentmodule:: drvi.utils

.. autosummary::
    :nosignatures:
    :toctree: generated

    metrics.DiscreteDisentanglementBenchmark
```

The following functions represent the similarity functions used in benchmarking:

```{eval-rst}
.. module:: drvi.utils.metrics
.. currentmodule:: drvi.utils

.. autosummary::
    :nosignatures:
    :toctree: generated

    metrics.nn_alignment_score
    metrics.local_mutual_info_score
    metrics.spearman_correlataion_score
```

The following functions represent the aggregation functions used in benchmarking:

```{eval-rst}
.. module:: drvi.utils.metrics
.. currentmodule:: drvi.utils

.. autosummary::
    :nosignatures:
    :toctree: generated

    metrics.most_similar_averaging_score
    metrics.latent_matching_score
    metrics.most_similar_gap_score
```
