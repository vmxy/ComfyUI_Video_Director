"""MiniMax H3 model family.

Everything H3 specific lives here: the model contract implementation plus the
latent-level continuity / upscaling / runtime-patch helpers it needs.
"""

from __future__ import annotations

from .model import H3, H3Model

__all__ = ["H3", "H3Model"]
