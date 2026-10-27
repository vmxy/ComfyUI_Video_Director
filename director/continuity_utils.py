"""Model-agnostic continuity helpers.

Everything here only touches decoded IMAGE frames and the standard ComfyUI
AUDIO dict, so any model family can reuse it. Model-specific latent pinning
lives under ``director/models/<family>/``.
"""

from __future__ import annotations

import math

import torch

DEFAULT_FPS = 24.0

# 「重绘幅度」: higher = more redraw / less copy of the previous tail.
SEAM_REDRAW_DEFAULT = 0.10
SEAM_REDRAW_MIN = 0.0
SEAM_REDRAW_MAX = 0.95


def clamp_seam_min_mask(value) -> float:
    """Clamp the user-facing redraw ratio to a safe range."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        n = SEAM_REDRAW_DEFAULT
    if not math.isfinite(n):
        n = SEAM_REDRAW_DEFAULT
    return max(SEAM_REDRAW_MIN, min(SEAM_REDRAW_MAX, n))


def trim_context_prefix(
    images: torch.Tensor,
    audio: dict | None,
    trim_frames: int,
    *,
    fps: float = DEFAULT_FPS,
    match_tail: bool = True,
) -> tuple[torch.Tensor, dict | None]:
    """Remove pinned head from decoded images/audio; optionally match audio duration."""
    trim = max(0, int(trim_frames))
    if trim > 0:
        if int(images.shape[0]) <= trim:
            raise ValueError(
                f"Director continuity: cannot trim {trim} frames from "
                f"{int(images.shape[0])}-frame decode."
            )
        images = images[trim:]
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return images, audio
    waveform = audio["waveform"]
    sr = int(audio.get("sample_rate") or 32000)
    drop = int(round((trim / float(fps)) * sr)) if trim > 0 else 0
    if drop > 0 and int(waveform.shape[-1]) > drop:
        waveform = waveform[..., drop:]
    if match_tail:
        want = int(round((int(images.shape[0]) / float(fps)) * sr))
        if int(waveform.shape[-1]) > want:
            waveform = waveform[..., :want]
    return images, {"waveform": waveform, "sample_rate": sr}


def trim_export_tail(
    images: torch.Tensor,
    audio: dict | None,
    trim_frames: int,
    *,
    fps: float = DEFAULT_FPS,
) -> tuple[torch.Tensor, dict | None]:
    """Drop trailing frames from a previous export so it ends at the pin window."""
    trim = max(0, int(trim_frames))
    if trim <= 0:
        return images, audio
    keep = int(images.shape[0]) - trim
    if keep < 1:
        raise ValueError(
            f"Director continuity: cannot drop {trim} tail frames from "
            f"{int(images.shape[0])}-frame export."
        )
    images = images[:keep]
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return images, audio
    waveform = audio["waveform"]
    sr = int(audio.get("sample_rate") or 32000)
    want = int(round((keep / float(fps)) * sr))
    if int(waveform.shape[-1]) > want:
        waveform = waveform[..., :want]
    return images, {"waveform": waveform, "sample_rate": sr}


def generation_frame_budget(visible_frames: int, context_frames: int) -> tuple[int, int]:
    """Return ``(sample_length, trim_frames)`` for segment continuity.

    ``context_frames`` must already be on the model's context grid. Contract:
    the UI segment duration equals the exported frame count, so the pinned head
    is trimmed after decode instead of shortening the export.
    """
    from .frame_align import align_frames

    visible = align_frames(max(5, int(visible_frames)))
    ctx = max(0, int(context_frames or 0))
    if ctx <= 0:
        return visible, 0
    sample = align_frames(visible + ctx)
    if ctx >= sample:
        raise ValueError(
            f"Director continuity: context {ctx}f must be smaller than sample "
            f"length {sample}f."
        )
    return sample, ctx


def continuity_export_len(
    *,
    trim_frames: int,
    sample_len: int,
    visible_frames: int,
    target_len: int,
    keep_tail: bool,
) -> int:
    """Frames to keep after dropping the pinned head.

    Default crops the free region back to the UI visible length. ``keep_tail``
    keeps ``sample - trim`` (the align remainder).
    """
    if int(trim_frames) <= 0:
        return int(target_len)
    if keep_tail:
        return max(1, int(sample_len) - int(trim_frames))
    return int(visible_frames)


def handoff_end_frame(*, trim_frames: int, export_frames: int) -> int:
    """Sample-timeline pixel index where the exported segment ends (exclusive)."""
    return max(0, int(trim_frames)) + max(0, int(export_frames))
