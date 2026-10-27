"""Video model registry + "current run" context.

The Director resolves one model per execution from the ``model_family`` widget.
Geometry helpers deep in the pipeline (frame align, canvas snap) read the active
model through :func:`active_model` while the run is in flight.
"""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from typing import Iterator

from .base import ModelCapabilities, VideoModel

log = logging.getLogger("ComfyUI-VideoDirector.models")

_MODELS: dict[str, VideoModel] = {}
_ALIASES: dict[str, str] = {}
_ACTIVE: contextvars.ContextVar[VideoModel | None] = contextvars.ContextVar(
    "videodirector_model", default=None
)


def register_model(model: VideoModel, *, replace: bool = False) -> VideoModel:
    """Add a model to the registry. Re-registering the same key is idempotent."""
    key = str(getattr(model, "key", "") or "").strip().lower()
    if not key:
        raise ValueError("Model key must not be empty.")
    if key in _MODELS and not replace:
        return _MODELS[key]
    model.key = key
    _MODELS[key] = model
    for alias in (model.key, getattr(model, "display_name", ""), *getattr(model, "aliases", ())):
        token = str(alias or "").strip().lower()
        if token:
            _ALIASES[token] = key
    return model


def model_keys() -> list[str]:
    return list(_MODELS)


def default_model_key() -> str:
    for key, model in _MODELS.items():
        if getattr(model, "is_default", False):
            return key
    return next(iter(_MODELS), "")


def default_model() -> VideoModel:
    key = default_model_key()
    if not key:
        raise RuntimeError("No VideoDirector model registered.")
    return _MODELS[key]


def resolve_model_key(value) -> str:
    """Accept a key, display name or alias. Empty → the default model."""
    token = str(value or "").strip().lower()
    if not token:
        return default_model_key()
    return _ALIASES.get(token, "")


def get_model(value=None) -> VideoModel:
    """Resolve a model from a key / display name / alias.

    Unknown values (stale workflow widget slots, old frontends) fall back to the
    default model and log a warning instead of failing the run or a status
    request. Use :func:`resolve_model_key` when the caller needs to know.
    """
    key = resolve_model_key(value)
    if key and key in _MODELS:
        return _MODELS[key]
    fallback = default_model()
    requested = str(value or "").strip()
    if requested:
        log.warning(
            "Unknown model family %r; using %s.",
            requested,
            fallback.display_name,
        )
    return fallback


def model_choices() -> list[str]:
    return [m.display_name for m in _MODELS.values()] or ["MiniMax H3"]


def default_model_choice() -> str:
    return default_model().display_name


@contextmanager
def use_model(model: VideoModel | None) -> Iterator[VideoModel]:
    chosen = model or default_model()
    token = _ACTIVE.set(chosen)
    try:
        yield chosen
    finally:
        _ACTIVE.reset(token)


def active_model() -> VideoModel:
    current = _ACTIVE.get()
    if current is not None:
        return current
    return default_model()


def capabilities() -> ModelCapabilities:
    return active_model().capabilities


def validate_plan_features(plan, model: VideoModel | None = None) -> list[str]:
    """Fail fast when an optional stage is wired for a model that lacks it.

    Returns human-readable warnings for stages that were auto-disabled.
    """
    model = model or active_model()
    caps = model.capabilities
    checks = (
        ("selflift", getattr(plan, "selflift", None), caps.selflift, "SelfLift"),
        ("semantic_bridge", getattr(plan, "semantic_bridge", None), caps.semantic_bridge, "Semantic Bridge"),
        ("refine", getattr(plan, "refine", None), caps.refine, "Refine"),
        ("face_refine", getattr(plan, "face_refine", None), caps.face_refine, "FaceRefine"),
    )
    for _attr, pack, supported, label in checks:
        if isinstance(pack, dict) and pack.get("enabled") and not supported:
            raise ValueError(model.unsupported(label))

    # Continuity is auto-disabled by the executor (and reported in its run log),
    # so it is not duplicated here.
    return []
