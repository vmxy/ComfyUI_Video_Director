"""Video models for VideoDirector.

Importing this package registers the bundled models. To add a family (LTX-2,
Wan, ...), write a :class:`~.base.VideoModel` subclass in a new module and call
:func:`register_model` at import time.
"""

from __future__ import annotations

from .base import ConditioningResult, ModelCapabilities, VideoModel
from .registry import (
    active_model,
    capabilities,
    default_model,
    default_model_choice,
    default_model_key,
    get_model,
    model_choices,
    model_keys,
    register_model,
    resolve_model_key,
    use_model,
    validate_plan_features,
)
from .h3 import H3, H3Model

register_model(H3)

__all__ = [
    "ConditioningResult",
    "ModelCapabilities",
    "VideoModel",
    "H3",
    "H3Model",
    "active_model",
    "capabilities",
    "default_model",
    "default_model_choice",
    "default_model_key",
    "get_model",
    "model_choices",
    "model_keys",
    "register_model",
    "resolve_model_key",
    "use_model",
    "validate_plan_features",
]
