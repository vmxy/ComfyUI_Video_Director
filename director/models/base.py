"""Video model contract for VideoDirector.

The Director core never talks to a model family directly. Everything model
specific lives here:

* frame grid / canvas stride (geometry),
* text+reference conditioning and the empty AV latent,
* sigma-shift (or whatever the family needs before sampling),
* latent decode (video only, or video + native audio),
* capability flags so unsupported stages are skipped instead of crashing.

Adding a family (LTX-2, Wan, ...) means writing one subclass and registering it
in :mod:`director.models.registry`; the executor, cache and UI stay untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class ModelCapabilities:
    """Feature switches consumed by the Director executor and UI gates."""

    native_audio: bool = False
    keyframes: bool = False
    reference_images: bool = False
    reference_videos: bool = False
    reference_audios: bool = False
    negative_prompt: bool = False
    motion_context: bool = False
    selflift: bool = False
    refine: bool = False
    latent_upscale: bool = False
    face_refine: bool = False
    semantic_bridge: bool = False


@dataclass
class ConditioningResult:
    """Output of one conditioning build for a single segment."""

    positive: Any
    latent: Any
    negative: Any = None
    task_hint: str = ""


class VideoModel:
    """Base class — override the hooks a family needs.

    Only :meth:`build_conditioning` and :meth:`decode` are mandatory; the
    geometry defaults (4k+1 frames, 16-px canvas grid) fit most video models.
    """

    key: ClassVar[str] = ""
    display_name: ClassVar[str] = ""
    aliases: ClassVar[tuple[str, ...]] = ()
    capabilities: ClassVar[ModelCapabilities] = ModelCapabilities()
    canvas_stride: ClassVar[int] = 16
    max_frames: ClassVar[int] = 512
    #: ``(shift_video, shift_audio)`` widget defaults, ``None`` = no shift.
    default_shift: ClassVar[tuple[float, float] | None] = None
    uses_sigma_shift: ClassVar[bool] = False
    #: Fallback model when the widget is empty / unknown.
    is_default: ClassVar[bool] = False
    #: Human-readable sampling chain used in the run report.
    sample_pipeline_label: ClassVar[str] = "model"

    # ── geometry ────────────────────────────────────────────────────────
    def align_frames(self, frame_count: int) -> int:
        """Round up to the family's temporal grid (default 4k+1)."""
        n = max(5, int(frame_count))
        while n % 4 != 1:
            n += 1
        return n

    def snap(self, value: int) -> int:
        stride = max(1, int(self.canvas_stride))
        return max(stride, round(int(value) / stride) * stride)

    def ensure_canvas(self, width: int, height: int) -> tuple[int, int]:
        return self.snap(int(width)), self.snap(int(height))

    def validate_canvas(self, width: int, height: int) -> None:
        w, h = int(width), int(height)
        stride = max(1, int(self.canvas_stride))
        if w > 0 and h > 0 and w % stride == 0 and h % stride == 0:
            return
        nw, nh = self.ensure_canvas(max(w, 1), max(h, 1))
        raise ValueError(
            f"输出分辨率 {w}×{h} 不是 {stride} 的倍数（{self.display_name} 画布要求）。"
            f"请改为 {nw}×{nh}，或在输出面板重新选择最长边/固定尺寸后重跑。"
        )

    # ── sampling ────────────────────────────────────────────────────────
    def shift_model(self, model, *, shift_video: float, shift_audio: float):
        """Model-side preparation before sampling (sigma shift for H3)."""
        return model

    def build_conditioning(
        self,
        *,
        clip,
        vae,
        audio_vae,
        prompt: str,
        width: int,
        height: int,
        length: int,
        task_key: str,
        first_frame=None,
        last_frame=None,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
        ref_image_size: str = "match",
        negative_prompt: str = "",
    ) -> ConditioningResult:
        raise NotImplementedError(
            f"{self.display_name} does not implement build_conditioning()."
        )

    def decode(
        self,
        samples,
        *,
        vae,
        audio_vae=None,
        decode_audio: bool = True,
    ) -> tuple[Any, dict]:
        raise NotImplementedError(
            f"{self.display_name} does not implement decode()."
        )

    # ── capabilities ────────────────────────────────────────────────────
    def supports(self, feature: str) -> bool:
        return bool(getattr(self.capabilities, str(feature), False))

    def unsupported(self, feature: str) -> str:
        return (
            f"{self.display_name} 不支持「{feature}」。"
            "请在导演台把「模型族 (model_family)」切到支持的模型，"
            "或断开对应的可选节点。"
        )

    def describe(self) -> str:
        return f"{self.display_name} ({self.key})"
