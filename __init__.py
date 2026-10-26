"""ComfyUI VideoDirector — timeline plugin for MiniMax-H3 AV generation.

Based on ComfyUI official MiniMax H3 support (PR #15224 / #15228).
Licensed under the Apache License, Version 2.0. See LICENSE.
"""

from .nodes.conditioning import (
    MiniMaxH3DirectorConditioning,
    MiniMaxH3DirectorPlannerConditioning,
)
from .nodes.director import MiniMaxH3Director
from .nodes.director_refine import MiniMaxH3DirectorRefine
from .nodes.director_selflift import MiniMaxH3DirectorSelfLift
from .nodes.director_semantic_bridge import MiniMaxH3DirectorSemanticBridge
from .nodes.director_face_refine import MiniMaxH3DirectorFaceRefine
from .nodes.director_groups import (
    MiniMaxH3DirectorGroupImageToVideo,
    MiniMaxH3DirectorGroupReferenceToVideo,
    MiniMaxH3DirectorGroupsCombine,
)

NODE_CLASS_MAPPINGS = {
    "VD": MiniMaxH3Director,
    "VDRefine": MiniMaxH3DirectorRefine,
    "VDSelfLift": MiniMaxH3DirectorSelfLift,
    "VDSemanticBridge": MiniMaxH3DirectorSemanticBridge,
    "VDFaceRefine": MiniMaxH3DirectorFaceRefine,
    "VDConditioning": MiniMaxH3DirectorConditioning,
    "VDPlannerConditioning": MiniMaxH3DirectorPlannerConditioning,
    "VDGroupImageToVideo": MiniMaxH3DirectorGroupImageToVideo,
    "VDGroupReferenceToVideo": MiniMaxH3DirectorGroupReferenceToVideo,
    # Must stay in NODE_CLASS_MAPPINGS: ComfyUI skips comfy_entrypoint when
    # NODE_CLASS_MAPPINGS is present (if/elif in load_custom_node).
    "VDGroupsCombine": MiniMaxH3DirectorGroupsCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VD": "VideoDirector",
    "VDRefine": "VideoDirector Refine",
    "VDSelfLift": "VideoDirector SelfLift",
    "VDSemanticBridge": "VideoDirector Semantic Bridge",
    "VDFaceRefine": "VideoDirector FaceRefine",
    "VDConditioning": "VideoDirector Conditioning",
    "VDPlannerConditioning": "VideoDirector Planner Conditioning",
    "VDGroupImageToVideo": "VideoDirector Group (Image to Video)",
    "VDGroupReferenceToVideo": "VideoDirector Group (Reference to Video)",
    "VDGroupsCombine": "VideoDirector Groups Combine",
}

WEB_DIRECTORY = "./web/js"

import logging

_log = logging.getLogger("ComfyUI-VideoDirector")

try:
    from .director.http_routes import register_routes as _register_director_routes

    if not _register_director_routes():
        _log.warning(
            "VideoDirector HTTP routes deferred (PromptServer not ready). "
            "Restart ComfyUI if /videodirector/* returns 404."
        )
except Exception as _director_routes_exc:
    _log.warning("VideoDirector HTTP routes failed to load: %s", _director_routes_exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
