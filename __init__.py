from .nodes import (
    MiniMaxH3Easy,
    MiniMaxH3EasyAreaSwitch,
    MiniMaxH3EasyAudioLock,
    MiniMaxH3EasyChromaContext,
    MiniMaxH3EasyFaceRefine,
    MiniMaxH3EasyFrameInterpolation,
    MiniMaxH3EasyLoader,
    MiniMaxH3EasyMediaLoader,
    MiniMaxH3EasyMultiSet,
    MiniMaxH3EasyOutput,
    MiniMaxH3EasyReplaceVideoFrames,
    MiniMaxH3EasySaveVideo,
    MiniMaxH3EasySeamStabilizer,
)
from .face_refine_nodes import (
    NODE_CLASS_MAPPINGS as FACE_REFINE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FACE_REFINE_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {
    **FACE_REFINE_NODE_CLASS_MAPPINGS,
    "MiniMaxH3EasyLoader": MiniMaxH3EasyLoader,
    "MiniMaxH3EasyMediaLoader": MiniMaxH3EasyMediaLoader,
    "MiniMaxH3EasyMultiSet": MiniMaxH3EasyMultiSet,
    "MiniMaxH3Easy": MiniMaxH3Easy,
    "MiniMaxH3EasyAreaSwitch": MiniMaxH3EasyAreaSwitch,
    "MiniMaxH3EasyOutput": MiniMaxH3EasyOutput,
    "MiniMaxH3EasyFaceRefine": MiniMaxH3EasyFaceRefine,
    "MiniMaxH3EasyAudioLock": MiniMaxH3EasyAudioLock,
    "MiniMaxH3EasyReplaceVideoFrames": MiniMaxH3EasyReplaceVideoFrames,
    "MiniMaxH3EasySaveVideo": MiniMaxH3EasySaveVideo,
    "MiniMaxH3EasyFrameInterpolation": MiniMaxH3EasyFrameInterpolation,
    "MiniMaxH3EasyChromaContext": MiniMaxH3EasyChromaContext,
    "MiniMaxH3EasySeamStabilizer": MiniMaxH3EasySeamStabilizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **FACE_REFINE_NODE_DISPLAY_NAME_MAPPINGS,
    "MiniMaxH3EasyLoader": "MiniMax H3 Easy Loader",
    "MiniMaxH3EasyMediaLoader": "MiniMax H3 Easy 多媒体加载",
    "MiniMaxH3EasyMultiSet": "Multi Set",
    "MiniMaxH3Easy": "MiniMax H3 Easy",
    "MiniMaxH3EasyAreaSwitch": "MiniMax H3 Easy 功能区路由",
    "MiniMaxH3EasyOutput": "MiniMax H3 Easy Output",
    "MiniMaxH3EasyFaceRefine": "MiniMax H3 Easy Distant Face Refine",
    "MiniMaxH3EasyAudioLock": "MiniMax H3 Easy Audio Lock",
    "MiniMaxH3EasyReplaceVideoFrames": "MiniMax H3 Easy Replace Video Frames",
    "MiniMaxH3EasySaveVideo": "MiniMax H3 Easy Save Video",
    "MiniMaxH3EasyFrameInterpolation": "MiniMax H3 Easy Frame Interpolation",
    "MiniMaxH3EasyChromaContext": "MiniMax H3 Easy Chroma Context",
    "MiniMaxH3EasySeamStabilizer": "MiniMax H3 Easy Seam Stabilizer",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
