"""Developmental, internal-use-only DRVI extensions.

.. warning::

    Everything in :mod:`drvi.internal` is for **DRVI development only** and may change or be removed
    without notice. For general use, import the maintained model as ``scvi.external.DRVI`` (or its
    alias ``drvi.model.DRVI``).

:class:`drvi.internal.DRVI` subclasses :class:`scvi.external.DRVI` and re-adds a few experimental
features kept from earlier drvi-py: opt-in residual connections between hidden layers, streaming
(online) training metrics, and a sparse latent representation.
"""

from drvi.internal._base_components import DecoderDRVI, enable_residual
from drvi.internal._generative_mixin import SparseLatentMixin
from drvi.internal._metrics import LatentStats, StreamingPairwiseMI
from drvi.internal._model import DRVI
from drvi.internal._module import DRVIModule
from drvi.internal._trainingplan import DRVITrainingPlan

__all__ = [
    "DRVI",
    "DRVIModule",
    "DecoderDRVI",
    "DRVITrainingPlan",
    "SparseLatentMixin",
    "LatentStats",
    "StreamingPairwiseMI",
    "enable_residual",
]
