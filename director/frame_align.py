"""Exact frame-count alignment for Director merge / cache / preview outputs.

The temporal grid belongs to the active video model: MiniMax H3 uses 17k+5,
most other video models use 4k+1. Shared helpers call :func:`align_frames` so a
new model changes the grid without touching the pipeline.
"""

from __future__ import annotations

import torch


def align_frames(frame_count: int) -> int:
    """Align to the active model's temporal grid."""
    from .models import active_model

    return active_model().align_frames(int(frame_count))


# Legacy names kept for old workflows / third-party imports.
def minimax_align_frame_count(frame_count: int) -> int:
    return align_frames(frame_count)


def wan_align_frame_count(frame_count: int) -> int:
    return align_frames(frame_count)


def pad_or_trim_frames(frames: torch.Tensor, target_len: int) -> torch.Tensor:
    """Trim to at most target_len frames. Does not fabricate last-frame duplicates."""
    target_len = max(0, int(target_len))
    if target_len <= 0:
        return frames[:0]
    if int(frames.shape[0]) > target_len:
        return frames[:target_len]
    return frames
