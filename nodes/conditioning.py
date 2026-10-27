"""VideoDirector conditioning nodes — delegate to the selected video model.

The model family is chosen with the ``model_family`` widget. The H3 model
wraps ComfyUI's official MiniMax H3 nodes; new families register themselves in
:mod:`director.models` and reuse the same nodes.
"""

from __future__ import annotations

from ..director.models import (
    active_model,
    model_choices,
    default_model_choice,
    get_model,
)
from ..lib.ref_images import MAX_REFERENCE_IMAGES, REF_IMAGE_KEY_PREFIX, flatten_reference_kwargs


def _shared_optional_inputs() -> dict:
    return {
        "first_frame": (
            "IMAGE",
            {"tooltip": "Optional first keyframe (i2v / fl2v)."},
        ),
        "last_frame": (
            "IMAGE",
            {"tooltip": "Optional last keyframe (fl2v)."},
        ),
        **{
            f"{REF_IMAGE_KEY_PREFIX}{index}": (
                "IMAGE",
                {
                    "tooltip": (
                        f"Reference image for <Picture {index + 1}> in prompt (r2v). "
                        "Native aspect; the model's ref_image_size applies at encode time."
                    ),
                },
            )
            for index in range(MAX_REFERENCE_IMAGES)
        },
        "ref_image_size": (
            ["match", "max"],
            {
                "default": "match",
                "tooltip": "Reference image sizing for reference-to-video models.",
            },
        ),
        "model_family": (
            model_choices(),
            {
                "default": default_model_choice(),
                "tooltip": (
                    "模型族：决定条件编码 / 帧对齐 / 采样前置 / latent 解码走哪个模型。"
                    "MiniMax H3 为默认模型；其它模型通过 director/models 注册。"
                ),
            },
        ),
    }


def _reference_images_dict_from_kwargs(kwargs: dict) -> dict | None:
    nested = kwargs.get("ref_images")
    if isinstance(nested, dict) and nested:
        out = {k: v for k, v in nested.items() if v is not None}
        return out or None

    refs = flatten_reference_kwargs(kwargs)
    out: dict[str, object] = {}
    for key, value in refs.items():
        if value is None:
            continue
        idx = key.removeprefix(REF_IMAGE_KEY_PREFIX)
        out[f"ref_image_{idx}"] = value
    return out or None


def run_minimax_conditioning(
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
    vmodel=None,
    **kwargs,
):
    """Build positive conditioning + latent through the active video model.

    The function name is kept for compatibility with the H3-only releases; the
    implementation is now model-driven.
    """
    vmodel = vmodel or active_model()
    ref_images = ref_images or _reference_images_dict_from_kwargs(kwargs)
    ref_videos = {k: v for k, v in (ref_videos or {}).items() if v is not None} or None

    result = vmodel.build_conditioning(
        clip=clip,
        vae=vae,
        audio_vae=audio_vae,
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        task_key=task_key,
        first_frame=first_frame,
        last_frame=last_frame,
        ref_images=ref_images,
        ref_videos=ref_videos,
        ref_video_audios=ref_video_audios,
        ref_audios=ref_audios,
        ref_image_size=ref_image_size,
        negative_prompt=negative_prompt,
    )
    return result.positive, list(result.negative or []), result.latent, result.task_hint


def _task_key_for(kwargs: dict) -> str:
    return "r2v" if kwargs.get("ref_images") or any(
        k.startswith(REF_IMAGE_KEY_PREFIX) for k in kwargs
    ) else "t2v"


class MiniMaxH3DirectorConditioning:
    """Thin wrapper around the model conditioning path (positive + latent)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 864, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 480, "min": 32, "max": 8192, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "Required for r2v / v2v / rv2v / reference video+audio."}),
                **_shared_optional_inputs(),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "apply"
    CATEGORY = "MiniMaxH3"

    def apply(self, clip, vae, prompt, width, height, length, audio_vae=None, model_family=None, **kwargs):
        positive, _, latent, _ = run_minimax_conditioning(
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            task_key=_task_key_for(kwargs),
            vmodel=get_model(model_family),
            **kwargs,
        )
        return positive, latent


class MiniMaxH3DirectorPlannerConditioning:
    """Model conditioning plus a task_mode string for planning UIs."""

    @classmethod
    def INPUT_TYPES(cls):
        return MiniMaxH3DirectorConditioning.INPUT_TYPES()

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "latent", "task_mode")
    FUNCTION = "apply"
    CATEGORY = "MiniMaxH3"

    def apply(self, clip, vae, prompt, width, height, length, audio_vae=None, model_family=None, **kwargs):
        positive, _, latent, hint = run_minimax_conditioning(
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            task_key=_task_key_for(kwargs),
            vmodel=get_model(model_family),
            **kwargs,
        )
        return positive, latent, hint
