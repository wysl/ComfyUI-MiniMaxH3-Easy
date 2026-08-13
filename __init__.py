from .nodes import (
    MiniMaxH3Easy,
    MiniMaxH3EasyFrameInterpolation,
    MiniMaxH3EasyLoader,
    MiniMaxH3EasyOutput,
    MiniMaxH3EasySaveVideo,
)

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3EasyLoader": MiniMaxH3EasyLoader,
    "MiniMaxH3Easy": MiniMaxH3Easy,
    "MiniMaxH3EasyOutput": MiniMaxH3EasyOutput,
    "MiniMaxH3EasySaveVideo": MiniMaxH3EasySaveVideo,
    "MiniMaxH3EasyFrameInterpolation": MiniMaxH3EasyFrameInterpolation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3EasyLoader": "MiniMax H3 Easy Loader",
    "MiniMaxH3Easy": "MiniMax H3 Easy",
    "MiniMaxH3EasyOutput": "MiniMax H3 Easy Output",
    "MiniMaxH3EasySaveVideo": "MiniMax H3 Easy Save Video",
    "MiniMaxH3EasyFrameInterpolation": "MiniMax H3 Easy Frame Interpolation",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
