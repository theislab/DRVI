"""Backward-compatible access to the DRVI model.

The DRVI PyTorch model now lives in scvi-tools and is maintained there. As of
drvi-py ``0.3.0`` this package no longer ships its own model implementation;
``drvi.model.DRVI`` is kept only as an alias for :class:`scvi.external.DRVI`
(requires ``scvi-tools >= 1.5.0``). New code should import the model directly as
``scvi.external.DRVI``. This alias may be deprecated from ``0.4.0``.
"""

from scvi.external import DRVI

__all__ = ["DRVI"]
