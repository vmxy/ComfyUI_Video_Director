"""MiniMax H3 model — the reference implementation of the contract.

All official-H3 specifics live here: the 17k+5 frame grid, the 32-px canvas
grid (VAE ÷16 then 2×2 patchify), ``MiniMaxH3ImageToVideo`` /
``MiniMaxH3ReferenceToVideo`` conditioning, ``MiniMaxH3SigmaShift`` and the
AV-latent decode (``VAEDecode`` + ``VAEDecodeAudio``).
"""

from __future__ import annotations

from typing import Any

from ....lib.task_modes import TASK_DESCRIPTIONS, MiniMaxH3Task, infer_task
from ..base import ConditioningResult, ModelCapabilities, VideoModel


def _load_minimax_nodes():
    try:
        from comfy_extras.nodes_minimax_h3 import (
            MiniMaxH3ImageToVideo,
            MiniMaxH3ReferenceToVideo,
        )
    except ImportError as exc:
        raise RuntimeError(
            "VideoDirector 的 MiniMax H3 模型需要 ComfyUI 官方 MiniMax H3 节点 "
            "(comfy_extras.nodes_minimax_h3)。请升级到含 PR #15224 的 ComfyUI 版本。"
        ) from exc
    return MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo


def _unpack_positive_latent(out):
    args = None
    if hasattr(out, "args"):
        args = out.args
    elif isinstance(out, (tuple, list)):
        args = out
    if args and len(args) >= 2:
        return args[0], args[1]
    raise RuntimeError(f"MiniMax H3 conditioning returned unexpected output: {type(out)!r}")


def _task_hint(task_key: str, ref_images, ref_videos, first_frame=None, last_frame=None) -> str:
    ref_image_count = len(ref_images or {})
    ref_video_count = len(ref_videos or {})
    mode = infer_task(ref_image_count, ref_video_count)
    if mode == MiniMaxH3Task.T2V:
        # Keyframe path: report what was actually locked, not "no keyframes".
        if last_frame is not None:
            mode = MiniMaxH3Task.FL2V
        elif first_frame is not None:
            mode = MiniMaxH3Task.I2V
    desc = TASK_DESCRIPTIONS[mode]
    if mode == MiniMaxH3Task.FL2V and first_frame is None:
        desc = "Last keyframe AV"
    hint = f"{task_key or mode.value} — {desc} (MiniMax H3)"
    if ref_image_count or ref_video_count:
        hint += f" (~{ref_image_count} ref image(s), {ref_video_count} ref video(s))"
    return hint


class H3Model(VideoModel):
    key = "minimax_h3"
    display_name = "MiniMax H3"
    aliases = ("minimax", "h3", "minimax-h3", "minimax h3")
    capabilities = ModelCapabilities(
        native_audio=True,
        keyframes=True,
        reference_images=True,
        reference_videos=True,
        reference_audios=True,
        negative_prompt=False,
        motion_context=True,
        selflift=True,
        refine=True,
        latent_upscale=True,
        face_refine=True,
        semantic_bridge=True,
    )
    canvas_stride = 32
    max_frames = 512
    default_shift = (12.0, 3.0)
    uses_sigma_shift = True
    is_default = True
    sample_pipeline_label = "MiniMaxH3SigmaShift(model)"

    # ── geometry ────────────────────────────────────────────────────────
    def align_frames(self, frame_count: int) -> int:
        """Round up to the H3 17k+5 grid (5, 22, 39, …)."""
        n = max(5, int(frame_count))
        while n % 17 != 5:
            n += 1
        return n

    def validate_canvas(self, width: int, height: int) -> None:
        w, h = int(width), int(height)
        if w > 0 and h > 0 and w % self.canvas_stride == 0 and h % self.canvas_stride == 0:
            return
        nw, nh = self.ensure_canvas(max(w, 1), max(h, 1))
        raise ValueError(
            f"输出分辨率 {w}×{h} 不是 32 的倍数（MiniMax H3 要求：VAE÷16 后再 2×2 patch）。"
            f"请改为 {nw}×{nh}，或在输出面板重新选择最长边/固定尺寸后重跑。"
            "若同时安装了独立的 ComfyUI-H3-Motion-Context，请卸载或禁用后重启 ComfyUI"
            "（与 Director 内置连贯冲突）。"
        )

    # ── sampling ────────────────────────────────────────────────────────
    def shift_model(self, model, *, shift_video: float, shift_audio: float):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

        shifted = MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio))
        if hasattr(shifted, "args") and shifted.args:
            return shifted.args[0]
        if isinstance(shifted, (tuple, list)):
            return shifted[0]
        raise RuntimeError(f"MiniMaxH3SigmaShift returned unexpected output: {type(shifted)!r}")

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
        MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo = _load_minimax_nodes()

        ref_images = {k: v for k, v in (ref_images or {}).items() if v is not None} or None
        ref_videos = {k: v for k, v in (ref_videos or {}).items() if v is not None} or None

        use_reference = bool(
            task_key in {"r2v", "v2v", "rv2v"}
            or ref_images
            or ref_videos
            or ref_audios
            or ref_video_audios
        )

        if use_reference:
            if audio_vae is None:
                raise ValueError(
                    "MiniMax H3 r2v/v2v/rv2v / reference conditioning requires audio_vae."
                )
            # Keyword args: official execute() moved vae/audio_vae after
            # ref_image_size. Positional calls send prompt into height
            # (`TypeError: ... for //: 'str' and 'int'` in _empty_av_latent).
            out = MiniMaxH3ReferenceToVideo.execute(
                clip=clip,
                prompt=prompt,
                width=width,
                height=height,
                length=length,
                ref_image_size=ref_image_size,
                vae=vae,
                audio_vae=audio_vae,
                ref_images=ref_images,
                ref_videos=ref_videos,
                ref_video_audios=ref_video_audios,
                ref_audios=ref_audios,
            )
        else:
            out = MiniMaxH3ImageToVideo.execute(
                clip=clip,
                vae=vae,
                prompt=prompt,
                width=width,
                height=height,
                length=length,
                first_frame=first_frame,
                last_frame=last_frame,
            )

        positive, latent = _unpack_positive_latent(out)
        return ConditioningResult(
            positive=positive,
            latent=latent,
            negative=None,
            task_hint=_task_hint(task_key, ref_images, ref_videos, first_frame, last_frame),
        )

    def decode(
        self,
        samples,
        *,
        vae,
        audio_vae=None,
        decode_audio: bool = True,
    ) -> tuple[Any, dict]:
        """Same as official r2v: both decoders take the AV latent.

        ``VAEDecode`` unbinds the video stream; ``VAEDecodeAudio`` unbinds the
        audio stream. Official VAE already writes pixels to
        ``intermediate_device()`` (CPU by default).
        """
        from nodes import VAEDecode

        from ...audio_export import empty_audio_dict

        images, = VAEDecode().decode(vae, samples)
        if not decode_audio or audio_vae is None:
            return images, empty_audio_dict()
        try:
            from comfy_extras.nodes_audio import VAEDecodeAudio
        except ImportError:
            from comfy_extras.nodes_lt import VAEDecodeAudio  # type: ignore

        audio_out = VAEDecodeAudio.execute(audio_vae, samples)
        if hasattr(audio_out, "args"):
            audio = audio_out.args[0]
        else:
            audio = audio_out[0]
        return images, audio


H3 = H3Model()
