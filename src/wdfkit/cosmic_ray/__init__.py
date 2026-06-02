# -*- coding: utf-8 -*-
"""Cosmic-ray removal: :class:`CosmicRayRemover` and helpers."""

from ._1d import remove_cosmic_rays_1d
from ._remover import CosmicRayRemover

__all__ = ["CosmicRayRemover", "remove_cosmic_rays_1d"]
