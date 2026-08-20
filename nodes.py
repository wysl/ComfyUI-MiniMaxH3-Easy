"""A compact MiniMax H3 entry point for ComfyUI.

The node intentionally keeps the graph contract small: one loader bundle, one
mode-aware conditioning node, and standard ComfyUI outputs for the sampler
chain. The browser extension supplies the ordered virtual media inputs.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache, partial
from typing import Any

import torch
import torch.nn.functional as functional
import torchaudio

import comfy.nested_tensor
import comfy.model_management
import comfy.samplers
import comfy.utils
import folder_paths
import node_helpers
import nodes
from comfy.cli_args import args
from comfy_api.latest import InputImpl, Types
from comfy_execution.graph_utils import GraphBuilder
from comfy_extras import nodes_minimax_h3 as h3


MODE_IMAGE = "image"
MODE_REFERENCE = "reference"
KEYFRAME_FIRST = "first"
KEYFRAME_LAST = "last"
REF_IMAGE_1K = "1k"
REF_IMAGE_2K = "2k"
REFERENCE_MENTION_FILENAME = "filename"
REFERENCE_MENTION_INDEX = "index"
NONE_MODEL = "none"
NONE_MODEL_DISPLAY_VALUES = (NONE_MODEL, "None", "无")
NONE_MODEL_ALIASES = {value.lower() for value in NONE_MODEL_DISPLAY_VALUES}
FACE_REFINE_SINGLE = "single person"
FACE_REFINE_TWO = "two people"
FACE_REFINE_PROMPT_IDENTITY_ONLY = "identity only (recommended)"
FACE_REFINE_PROMPT_FULL_SCENE = "full scene prompt"
FACE_REFINE_SEED_IDENTITY_LOCKED = "identity locked (recommended)"
FACE_REFINE_SEED_INPUT = "input seed"
RESOLUTION_360 = "360P"
RESOLUTION_416 = "416P"
RESOLUTION_480 = "480P"
RESOLUTION_540 = "540P"
RESOLUTION_640 = "640P"
RESOLUTION_720 = "720P"
RESOLUTION_768 = "768P"
RESOLUTION_832 = "832P"
RESOLUTION_928 = "928P"
RESOLUTION_1024 = "1024P"
RESOLUTION_1080 = "1080P"
RESOLUTION_CUSTOM = "custom"
ASPECT_SQUARE = "1:1"
ASPECT_PHOTO_PORTRAIT = "2:3"
ASPECT_PHOTO = "3:2"
ASPECT_STANDARD_PORTRAIT = "3:4"
ASPECT_STANDARD = "4:3"
ASPECT_WIDESCREEN_PORTRAIT = "9:16"
ASPECT_WIDESCREEN = "16:9"
ASPECT_ULTRAWIDE = "21:9"
RESOLUTION_MEGAPIXELS = {
    RESOLUTION_360: 0.2,
    RESOLUTION_416: 0.3,
    RESOLUTION_480: 0.4,
    RESOLUTION_540: 0.5,
    RESOLUTION_640: 0.7,
    RESOLUTION_720: 0.9,
    RESOLUTION_768: 1.0,
    RESOLUTION_832: 1.2,
    RESOLUTION_928: 1.5,
    RESOLUTION_1024: 1.8,
    RESOLUTION_1080: 2.0,
}
RESOLUTIONS = (*RESOLUTION_MEGAPIXELS, RESOLUTION_CUSTOM)
REFERENCE_IMAGE_SHORT_EDGES = {
    REF_IMAGE_1K: 1024,
    REF_IMAGE_2K: h3.REF_IMAGE_SHORT_EDGE,
}
ASPECT_RATIOS = {
    ASPECT_SQUARE: (1, 1),
    ASPECT_PHOTO_PORTRAIT: (2, 3),
    ASPECT_PHOTO: (3, 2),
    ASPECT_STANDARD_PORTRAIT: (3, 4),
    ASPECT_STANDARD: (4, 3),
    ASPECT_WIDESCREEN_PORTRAIT: (9, 16),
    ASPECT_WIDESCREEN: (16, 9),
    ASPECT_ULTRAWIDE: (21, 9),
}
MAX_MEDIA = 15
MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_AUDIOS = 3
MIN_SECONDS = 4.0
MAX_SECONDS = 20.0
REFERENCE_PLACEHOLDER_RE = re.compile(r"__MINIMAX_H3_REF_(\d+)__")
UNRESOLVED_REFERENCE_RE = re.compile(r"__MINIMAX_H3_UNRESOLVED_REF_[^_]+__")
MODEL_FILE_EXTENSIONS = {".safetensors", ".gguf"}
H3_MULTI_SET_MAX_PAIRS = 64


class _H3AnyType(str):
    def __ne__(self, _value: object) -> bool:
        return False


H3_ANY_TYPE = _H3AnyType("*")


def _normalise_model_name(name: str) -> str:
    """Turn community naming variants into comparable tokens.

    MiniMax H3 files appear with underscores, dashes, camel case and sometimes
    only a role folder (for example ``FL2VA/model.safetensors``). Matching the
    normalised path rather than one exact filename keeps the loader useful for
    community quantisations without admitting every unrelated model.
    """
    value = str(name or "").replace("\\", "/").lower()
    value = re.sub(r"([a-z])([0-9])", r"\1 \2", value)
    value = re.sub(r"([0-9])([a-z])", r"\1 \2", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _model_tokens(name: str) -> set[str]:
    return set(_normalise_model_name(name).split())


def _is_minimax_h3_name(normalised: str, compact: str, tokens: set[str]) -> bool:
    """Require an explicit MiniMax H3 identity before matching shared roles."""
    return "minimaxh3" in compact or ("minimax" in tokens and "h3" in compact)


def _is_weight_file(name: str) -> bool:
    return os.path.splitext(str(name or ""))[1].lower() in MODEL_FILE_EXTENSIONS


def _is_gguf_file(name: str) -> bool:
    return str(name or "").lower().endswith(".gguf")


def _category_names(category: str) -> list[str]:
    """Read a ComfyUI filename category without assuming it exists."""
    try:
        return [str(name) for name in folder_paths.get_filename_list(category)]
    except Exception:
        return []


def _category_paths(category: str) -> list[str]:
    try:
        entry = folder_paths.folder_names_and_paths.get(category)
        if not entry:
            return []
        paths = entry[0]
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        return [os.fspath(path) for path in paths]
    except Exception:
        return []


def _filesystem_weight_names(categories: tuple[str, ...]) -> list[str]:
    """Find GGUF files even when ComfyUI has no GGUF extension category yet."""
    names: list[str] = []
    for category in categories:
        for base in _category_paths(category):
            if not os.path.isdir(base):
                continue
            try:
                for root, _dirs, files in os.walk(base):
                    for filename in files:
                        if os.path.splitext(filename)[1].lower() not in MODEL_FILE_EXTENSIONS:
                            continue
                        full_path = os.path.join(root, filename)
                        relative = os.path.relpath(full_path, base).replace(os.sep, "/")
                        names.append(relative)
            except OSError:
                continue
    return names


@lru_cache(maxsize=16)
def _collect_weight_names(categories: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for category in categories:
        for name in _category_names(category):
            if not _is_weight_file(name):
                continue
            key = name.replace("\\", "/")
            if key not in seen:
                seen.add(key)
                names.append(key)
    # The normal ComfyUI categories may not advertise .gguf until the optional
    # GGUF node is loaded, so supplement them from the actual model folders.
    for name in _filesystem_weight_names(categories):
        key = name.replace("\\", "/")
        if key not in seen:
            seen.add(key)
            names.append(key)
    return names


def _has_role(name: str, role: str) -> bool:
    normalised = _normalise_model_name(name)
    compact = normalised.replace(" ", "")
    tokens = set(normalised.split())
    if role == "fl2va":
        if "minimax" not in tokens and "h3" not in compact:
            return False
        if "ref2va" in compact or "ref2v" in compact:
            return False
        return "fl2va" in compact or "fl2v" in compact
    if role == "ref2va":
        if "minimax" not in tokens and "h3" not in compact:
            return False
        return "ref2va" in compact or "ref2v" in compact
    if role == "text_encoder":
        if ("qwen3vl" in compact or ("qwen3" in tokens and "vl" in tokens)) and (
            "32b" in tokens or "32" in tokens
        ):
            return True
        # Some community H3 exports omit "minimax_h3" from the encoder
        # filename but retain the characteristic INT8/ConvRot or NVFP4/AWQ
        # variant naming.
        if (
            "qwen3" in tokens
            and "vl" in tokens
            and ("32b" in tokens or "32" in tokens)
            and (("int8" in tokens and "convrot" in tokens) or ("nvfp4" in tokens and "awq" in tokens))
        ):
            return True
        # A few community exports use only text_encoder.safetensors, but keep
        # the match scoped to an H3-named path to avoid generic CLIP files.
        return "text encoder" in normalised and ("minimax" in tokens or "h3" in compact)
    if role == "video_vae":
        is_minimax_h3 = _is_minimax_h3_name(normalised, compact, tokens)
        is_video_vae = (
            ("video" in tokens and "vae" in tokens)
            or "videovae" in compact
            # Diffusers-style exports may use MiniMax-H3/vae/... without the
            # word "video". In H3, an unqualified VAE is the visual VAE.
            or ("vae" in tokens and "audio" not in tokens and "audiovae" not in compact)
        )
        return is_minimax_h3 and is_video_vae and "tae" not in tokens and "approx" not in tokens
    if role == "audio_vae":
        is_minimax_h3 = _is_minimax_h3_name(normalised, compact, tokens)
        is_audio_vae = (
            ("audio" in tokens and "vae" in tokens)
            or "audiovae" in compact
        )
        return is_minimax_h3 and is_audio_vae and "tae" not in tokens and "approx" not in tokens
    return False


def _is_h3_transformer_name(name: str) -> bool:
    """Return whether a diffusion model name explicitly identifies H3."""
    compact = _normalise_model_name(name).replace(" ", "")
    return "h3" in compact


def _sort_model_names(names: list[str]) -> list[str]:
    def sort_key(name: str) -> tuple[int, int, str]:
        normalised = _normalise_model_name(name)
        # Keep safetensors first for the native path, followed by GGUF. Within
        # each group use a deterministic name order for stable workflows.
        extension_rank = 1 if _is_gguf_file(name) else 0
        official_rank = 0 if "minimax" in normalised and "h3" in normalised else 1
        return extension_rank, official_rank, normalised

    return sorted(names, key=sort_key)


def _is_none_model(value: Any) -> bool:
    return str(value or "").strip().lower() in NONE_MODEL_ALIASES


def _role_choices(role: str, categories: tuple[str, ...], fallback: str) -> list[str]:
    names = _collect_weight_names(categories)
    selected = [name for name in names if _has_role(name, role)]
    return _sort_model_names(selected) or [fallback]


def _optional_role_choices(role: str, categories: tuple[str, ...]) -> list[str]:
    names = _collect_weight_names(categories)
    selected = _sort_model_names([name for name in names if _has_role(name, role)])
    # ComfyUI validates combo values before invoking the node. The frontend
    # localizes the sentinel to either "None" or "无", so all display values
    # must also be accepted by the server-side combo definition.
    return [*selected, *NONE_MODEL_DISPLAY_VALUES]


def _filtered_choices(category: str, needles: tuple[str, ...], fallback: str) -> list[str]:
    names = _collect_weight_names((category,))
    selected = [name for name in names if any(needle.lower() in _normalise_model_name(name).replace(" ", "") for needle in needles)]
    return _sort_model_names(selected) or [fallback]


def _model_choices() -> list[str]:
    return _h3_transformer_choices()


def _ref_model_choices() -> list[str]:
    return _h3_transformer_choices()


def _h3_transformer_choices() -> list[str]:
    names = _collect_weight_names(("diffusion_models", "unet", "unet_gguf"))
    selected = _sort_model_names([name for name in names if _is_h3_transformer_name(name)])
    # Both transformer slots intentionally expose every H3-named weight. Some
    # community exports omit the FL2VA/REF2VA role in the filename.
    return [*selected, *NONE_MODEL_DISPLAY_VALUES]


def _face_refine_detector_choices() -> list[str]:
    from .face_refine_nodes import _detector_list

    return _detector_list()


def _face_refine_lora_choices() -> list[str]:
    names = _collect_weight_names(("loras",))
    selected = [
        name for name in names
        if "h3" in name.lower() and "turbo" in name.lower()
    ]
    return [*_sort_model_names(selected), NONE_MODEL]


def _face_refine_sampler_choices() -> list[str]:
    choices = list(getattr(comfy.samplers.KSampler, "SAMPLERS", ()))
    return choices or ["res_multistep", "euler"]


def _face_refine_scheduler_choices() -> list[str]:
    choices = list(getattr(comfy.samplers.KSampler, "SCHEDULERS", ()))
    return choices or ["simple"]


def _face_refine_step_count(lora_name: str, requested_steps: int) -> int:
    requested_steps = int(requested_steps)
    if requested_steps > 0:
        return requested_steps
    if str(lora_name or "").strip().lower() in NONE_MODEL_ALIASES:
        return 20
    match = re.search(r"(\d+)\s*[_-]?\s*steps?", str(lora_name or ""), re.IGNORECASE)
    return max(1, int(match.group(1))) if match else 8


def _clip_choices() -> list[str]:
    return _role_choices("text_encoder", ("text_encoders", "clip", "clip_gguf"), "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")


def _vae_choices(needles: tuple[str, ...], fallback: str) -> list[str]:
    role = "video_vae" if any("video" in needle.lower() for needle in needles) else "audio_vae"
    return _role_choices(role, ("vae",), fallback)


@lru_cache(maxsize=16)
def _registered_node_class(*names: str):
    """Find an optional custom-node class without importing it unconditionally."""
    mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
    for name in names:
        node_class = mappings.get(name) if hasattr(mappings, "get") else None
        if node_class is not None:
            return node_class
        node_class = getattr(nodes, name, None)
        if node_class is not None:
            return node_class
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        for name in names:
            node_class = getattr(module, name, None)
            if node_class is not None:
                return node_class
    return None


def _load_gguf_unet(model_name: str):
    loader_class = _registered_node_class("UnetLoaderGGUF", "UNETLoaderGGUF", "UnetLoaderGGUFAdvanced")
    if loader_class is None:
        raise RuntimeError(
            "检测到 GGUF MiniMax H3 主模型，但当前 ComfyUI 未安装 GGUF 加载节点。"
            "请安装 ComfyUI-GGUF 后重启 ComfyUI。"
        )
    loader = loader_class()
    return loader.load_unet(model_name)[0]


def _load_text_encoder(text_encoder: str):
    if not _is_gguf_file(text_encoder):
        return nodes.CLIPLoader().load_clip(text_encoder, "minimax", "default")[0]

    loader_class = _registered_node_class("CLIPLoaderGGUF", "CLIPLoaderGGUFAdvanced")
    if loader_class is None:
        raise RuntimeError(
            "检测到 GGUF MiniMax H3 文本编码器，但当前 ComfyUI 未安装 GGUF 加载节点。"
            "请安装 ComfyUI-GGUF 后重启 ComfyUI。"
        )
    loader = loader_class()
    try:
        return loader.load_clip(text_encoder, "minimax")[0]
    except TypeError:
        return loader.load_clip(text_encoder, type="minimax")[0]


@dataclass
class MiniMaxH3Bundle:
    fl2va_model_name: str
    ref2va_model_name: str
    clip_name: str
    video_vae_name: str
    audio_vae_name: str
    clip: Any
    video_vae: Any
    audio_vae: Any

    def __post_init__(self) -> None:
        self._model = None
        self._model_kind = ""
        self._model_name = ""
        self._lock = threading.RLock()

    def _model_name_for(self, kind: str) -> str:
        """Return the preferred model, falling back to the other H3 model.

        FL2VA and REF2VA are exposed as separate choices when both are
        installed, but a user may intentionally install only one of them for
        testing. In that case, let the remaining transformer serve either
        generation path instead of rejecting the mode before execution.
        """
        requested_kind = "ref2va" if kind == "ref2va" else "fl2va"
        preferred = self.ref2va_model_name if requested_kind == "ref2va" else self.fl2va_model_name
        if not _is_none_model(preferred):
            return preferred

        fallback = self.fl2va_model_name if requested_kind == "ref2va" else self.ref2va_model_name
        if not _is_none_model(fallback):
            return fallback

        if requested_kind == "ref2va":
            raise ValueError("Reference Video mode requires at least one MiniMax H3 transformer model.")
        raise ValueError("Text-to-video and I2V or First/Last Frame mode require at least one MiniMax H3 transformer model.")

    def model_for(self, kind: str):
        kind = "ref2va" if kind == "ref2va" else "fl2va"
        with self._lock:
            model_name = self._model_name_for(kind)
            if self._model is not None and self._model_name == model_name:
                return self._model

            if self._model is not None:
                self._model = None
                self._model_kind = ""
                self._model_name = ""
                comfy.model_management.soft_empty_cache()

            if _is_gguf_file(model_name):
                self._model = _load_gguf_unet(model_name)
            else:
                self._model, = nodes.UNETLoader().load_unet(model_name, "default")
            self._model_kind = kind
            self._model_name = model_name
            return self._model


@dataclass(frozen=True)
class MiniMaxH3Context:
    conditioning: Any
    latent: Any
    video_vae: Any
    audio_vae: Any
    fps: float
    prompt: str = ""
    mode: str = MODE_IMAGE
    media: tuple[tuple[str, Any], ...] = ()
    keyframe_roles: tuple[str, ...] = ()
    _prompt_encoder: Any = None
    seconds: float = 5.0
    width: int = 0
    height: int = 0

    def prompt_assistant_payload(self) -> dict[str, Any]:
        """Expose the original generation inputs without importing either plugin."""
        synchronized_audios = []
        synchronized_audio_video_indices = []
        video_index = 0
        for media_type, value in self.media:
            if media_type != "video":
                continue
            video_index += 1
            try:
                _frames, soundtrack, _fps = _video_parts(value)
            except (TypeError, ValueError, AttributeError):
                soundtrack = None
            if soundtrack is not None:
                synchronized_audios.append(soundtrack)
                synchronized_audio_video_indices.append(video_index)
        standalone_audios = [
            value for media_type, value in self.media if media_type == "audio"
        ]
        return {
            "prompt": self.prompt,
            "mode": self.mode,
            "duration_seconds": float(self.seconds),
            "images": [value for media_type, value in self.media if media_type == "image"],
            "videos": [value for media_type, value in self.media if media_type == "video"],
            "audios": [*synchronized_audios, *standalone_audios],
            "synchronized_audio_count": len(synchronized_audios),
            "synchronized_audio_video_indices": synchronized_audio_video_indices,
            "keyframe_roles": list(self.keyframe_roles),
        }

    def encode_prompt(self, prompt: str):
        if not callable(self._prompt_encoder):
            raise ValueError("This H3 Context cannot encode a replacement prompt")
        return self._prompt_encoder(str(prompt))


@dataclass(frozen=True)
class _MediaInput:
    input_index: int
    media_type: str
    value: Any


class MiniMaxH3EasyLoader:
    CATEGORY = "MiniMax H3 Easy"
    FUNCTION = "load"
    RETURN_TYPES = ("MINIMAX_H3_BUNDLE",)
    RETURN_NAMES = ("h3_bundle",)
    DESCRIPTION = "Load either or both MiniMax H3 transformers, plus the text encoder and both AV VAEs."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fl2va_model": (_model_choices(),),
                "ref2va_model": (_ref_model_choices(),),
                "text_encoder": (_clip_choices(),),
                "video_vae": (_vae_choices(("minimax_h3_video_vae",), "minimax_h3_video_vae_fp16.safetensors"),),
                "audio_vae": (_vae_choices(("minimax_h3_audio_vae",), "minimax_h3_audio_vae_fp32.safetensors"),),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return "|".join(str(kwargs.get(key, "")) for key in ("fl2va_model", "ref2va_model", "text_encoder", "video_vae", "audio_vae"))

    def load(self, fl2va_model, ref2va_model, text_encoder, video_vae, audio_vae):
        if _is_none_model(fl2va_model) and _is_none_model(ref2va_model):
            raise ValueError("Select at least one MiniMax H3 transformer: FL2VA or REF2VA.")
        clip = _load_text_encoder(text_encoder)
        video_vae_obj, = nodes.VAELoader().load_vae(video_vae)
        audio_vae_obj, = nodes.VAELoader().load_vae(audio_vae)
        return (MiniMaxH3Bundle(
            fl2va_model_name=fl2va_model,
            ref2va_model_name=ref2va_model,
            clip_name=text_encoder,
            video_vae_name=video_vae,
            audio_vae_name=audio_vae,
            clip=clip,
            video_vae=video_vae_obj,
            audio_vae=audio_vae_obj,
        ),)


def _infer_media_type(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        return "image"
    if isinstance(value, Mapping) and "waveform" in value:
        return "audio"
    if hasattr(value, "get_components"):
        return "video"
    return "video"


def _audio_sample_rate(audio: Mapping) -> int:
    return int(audio.get("sample_rate") or audio.get("samplerate") or audio.get("sampler_rate") or 32000)


def _video_parts(value: Any) -> tuple[torch.Tensor, dict | None, float]:
    if hasattr(value, "get_components"):
        components = value.get_components()
        return components.images, components.audio, float(components.frame_rate or 24.0)
    if isinstance(value, Mapping):
        frames = value.get("images")
        if frames is None:
            frames = value.get("frames")
        if isinstance(frames, torch.Tensor):
            return frames, value.get("audio"), float(value.get("fps") or value.get("frame_rate") or 24.0)
    if isinstance(value, torch.Tensor) and value.ndim == 4:
        return value, None, 24.0
    raise ValueError("Unsupported reference video payload")


def _resample_video_frames(frames: torch.Tensor, source_fps: float) -> torch.Tensor:
    if not source_fps or abs(source_fps - h3.FPS) < 0.01:
        return frames
    count = max(1, round(frames.shape[0] * h3.FPS / source_fps))
    indexes = torch.linspace(0, frames.shape[0] - 1, count, device=frames.device).round().long()
    return frames[indexes]


def _encode_reference_audio(audio_vae, audio: Mapping):
    waveform = audio["waveform"]
    sample_rate = _audio_sample_rate(audio)
    vae_sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sample_rate != vae_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)
    latent = audio_vae.encode(waveform[:1].movedim(1, -1))
    return latent, latent.shape[-1]


def _resolve_reference_prompt(
    prompt: str,
    tag_by_input: dict[int, str],
    soundtrack_pairs: list[tuple[int, int]],
    video_count: int,
    standalone_audio_count: int,
) -> str:
    if UNRESOLVED_REFERENCE_RE.search(str(prompt or "")):
        raise ValueError("Prompt contains a disconnected media reference. Reconnect the media or remove the @ reference.")
    resolved = REFERENCE_PLACEHOLDER_RE.sub(
        lambda match: tag_by_input.get(int(match.group(1)), ""),
        str(prompt or ""),
    )
    if soundtrack_pairs and (video_count > 1 or standalone_audio_count > 0):
        provenance = [
            f"<Audio {audio_index}> is the synchronized audio track of <Video {video_index}>."
            for audio_index, video_index in soundtrack_pairs
        ]
        return "\n".join((*provenance, resolved))
    return resolved


def _align_canvas_dimension(value: float) -> int:
    return max(h3.CANVAS_MULTIPLE, round(float(value) / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)


def _canvas_dimensions(resolution: str, aspect_ratio: str, custom_width: int, custom_height: int) -> tuple[int, int]:
    if str(resolution) == RESOLUTION_CUSTOM:
        return _align_canvas_dimension(custom_width), _align_canvas_dimension(custom_height)

    megapixels = RESOLUTION_MEGAPIXELS.get(str(resolution), RESOLUTION_MEGAPIXELS[RESOLUTION_480])
    ratio_w, ratio_h = ASPECT_RATIOS.get(str(aspect_ratio), ASPECT_RATIOS[ASPECT_WIDESCREEN])
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (ratio_w * ratio_h))
    return _align_canvas_dimension(ratio_w * scale), _align_canvas_dimension(ratio_h * scale)


def _frame_length(seconds: float, fps: float) -> int:
    target_frames = max(5.0, float(seconds) * float(fps))
    block_count = max(0, math.ceil((target_frames - 5) / 17))
    return block_count * 17 + 5


def _empty_image_conditioning(bundle, prompt, width, height, length, first_frame=None, last_frame=None):
    latent, frame_count = h3._empty_av_latent(width, height, length)
    images = []
    keyframes = []
    if first_frame is not None:
        image = h3._resize(first_frame[:1], width, height, "disabled")
        images.append(image)
        keyframes.append({"resolved_frame_index": 0, "image": image})
    if last_frame is not None:
        image = h3._resize(last_frame[:1], width, height, "center")
        images.append(image)
        keyframes.append({"resolved_frame_index": frame_count - 1, "image": image})

    tokens = bundle.clip.tokenize(prompt, images=images)
    conditioning = bundle.clip.encode_from_tokens_scheduled(tokens)
    if keyframes:
        for keyframe in keyframes:
            keyframe["latent"] = bundle.video_vae.encode(keyframe.pop("image"))
        conditioning = node_helpers.conditioning_set_values(conditioning, {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
        })
    return conditioning, latent


def _reference_conditioning(
    bundle,
    prompt,
    width,
    height,
    length,
    ref_image_size,
    items: list[_MediaInput],
    return_prompt: bool = False,
):
    latent, frame_count = h3._empty_av_latent(width, height, length)
    ref_items = []
    ref_blocks = []
    tag_by_input: dict[int, str] = {}
    soundtrack_pairs: list[tuple[int, int]] = []
    images = [item for item in items if item.media_type == "image"]
    videos = [item for item in items if item.media_type == "video"]
    audios = [item for item in items if item.media_type == "audio"]
    audio_ordinal = 0

    # Match the official H3 presentation order: images, videos (with each
    # synchronized soundtrack immediately before its video), standalone audio.
    for picture_ordinal, item in enumerate(images, start=1):
        image = item.value
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("Image references must be IMAGE tensors")
        image_h, image_w = image.shape[1], image.shape[2]
        short_edge_limit = REFERENCE_IMAGE_SHORT_EDGES.get(str(ref_image_size), REFERENCE_IMAGE_SHORT_EDGES[REF_IMAGE_1K])
        scale = min(1.0, short_edge_limit / max(1, min(image_w, image_h)))
        target_w = max(h3.CANVAS_MULTIPLE, round(image_w * scale / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
        target_h = max(h3.CANVAS_MULTIPLE, round(image_h * scale / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
        resized = h3._resize(image[:1], target_w, target_h, "disabled")
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({"kind": "image", "latent_h": target_h // 16, "latent_w": target_w // 16, "latent": bundle.video_vae.encode(resized)})
        tag_by_input[item.input_index] = f"<Picture {picture_ordinal}>"

    for video_ordinal, item in enumerate(videos, start=1):
        frames, soundtrack, source_fps = _video_parts(item.value)
        frames = _resample_video_frames(frames, source_fps)
        video_h, video_w = frames.shape[1], frames.shape[2]
        canvas_w, canvas_h = h3.adapt_canvas(video_w, video_h)
        if video_w * video_h < canvas_w * canvas_h:
            canvas_w = max(h3.CANVAS_MULTIPLE, round(video_w / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
            canvas_h = max(h3.CANVAS_MULTIPLE, round(video_h / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
        frames = h3._resize(frames, canvas_w, canvas_h, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        count = frames.shape[0]
        if count < 5:
            raise ValueError("Reference videos need at least 5 frames")
        while count % 17 != 5:
            count -= 1
        frames = frames[:count]
        video_latent = bundle.video_vae.encode(frames)
        audio_latent = None
        audio_t = 0
        if soundtrack is not None:
            audio_latent, audio_t = _encode_reference_audio(bundle.audio_vae, soundtrack)
            audio_ordinal += 1
            soundtrack_pairs.append((audio_ordinal, video_ordinal))
            ref_items.append({"type": "audio"})
        sample_indexes = list(range(0, frames.shape[0], h3.FPS // 2))
        ref_items.append({
            "type": "video",
            "data": frames[sample_indexes],
            "timestamps": [i / 2.0 for i in range(len(sample_indexes))],
        })
        ref_blocks.append({
            "kind": "video_audio" if audio_t else "video",
            "latent_t": video_latent.shape[2],
            "latent_h": canvas_h // 16,
            "latent_w": canvas_w // 16,
            "ref_audio_t": audio_t,
            "latent": video_latent,
            "audio_latent": audio_latent,
        })
        tag_by_input[item.input_index] = f"<Video {video_ordinal}>"

    for item in audios:
        if not isinstance(item.value, Mapping) or "waveform" not in item.value:
            raise ValueError("Audio references must be AUDIO payloads")
        audio_latent, audio_t = _encode_reference_audio(bundle.audio_vae, item.value)
        audio_ordinal += 1
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": audio_t, "audio_latent": audio_latent})
        tag_by_input[item.input_index] = f"<Audio {audio_ordinal}>"

    if not ref_items or all(item.get("type") == "audio" for item in ref_items):
        raise ValueError("Reference mode needs at least one image or video")

    resolved_prompt = _resolve_reference_prompt(
        prompt,
        tag_by_input,
        soundtrack_pairs,
        len(videos),
        len(audios),
    )

    tokens = bundle.clip.tokenize(resolved_prompt, minimax_ref_items=ref_items)
    conditioning = bundle.clip.encode_from_tokens_scheduled(tokens)
    conditioning = node_helpers.conditioning_set_values(conditioning, {"minimax_refs": ref_blocks})
    if return_prompt:
        return conditioning, latent, resolved_prompt
    return conditioning, latent


class MiniMaxH3Easy:
    CATEGORY = "MiniMax H3 Easy"
    FUNCTION = "generate"
    RETURN_TYPES = ("MODEL", "MINIMAX_H3_CONTEXT")
    RETURN_NAMES = ("model", "h3_context")
    DESCRIPTION = "One MiniMax H3 node for text, image and reference video workflows."

    @classmethod
    def INPUT_TYPES(cls):
        optional = {"media": ("*",)}
        for index in range(1, MAX_MEDIA + 1):
            optional[f"media_{index}"] = ("*",)
            optional[f"media_type_{index}"] = ("STRING", {"default": ""})
        return {
            "required": {
                "h3_bundle": ("MINIMAX_H3_BUNDLE",),
                "mode": ([MODE_IMAGE, MODE_REFERENCE], {"default": MODE_IMAGE}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "resolution": (list(RESOLUTIONS), {"default": RESOLUTION_480}),
                "aspect_ratio": (list(ASPECT_RATIOS), {"default": ASPECT_WIDESCREEN}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                "seconds": ("FLOAT", {"default": 5.0, "min": MIN_SECONDS, "max": MAX_SECONDS, "step": 1.0}),
                "advanced": ("BOOLEAN", {"default": False}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "keyframe_role": ([KEYFRAME_FIRST, KEYFRAME_LAST], {"default": KEYFRAME_FIRST}),
                "ref_image_size": ([REF_IMAGE_1K, REF_IMAGE_2K], {"default": REF_IMAGE_1K}),
                "reference_mention_mode": ([REFERENCE_MENTION_FILENAME, REFERENCE_MENTION_INDEX], {"default": REFERENCE_MENTION_INDEX}),
            },
            "optional": optional,
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @staticmethod
    def _collect_media(kwargs: dict) -> list[_MediaInput]:
        items = []
        direct = kwargs.get("media")
        if direct is not None:
            items.append(_MediaInput(0, _infer_media_type(direct), direct))
        for index in range(1, MAX_MEDIA + 1):
            value = kwargs.get(f"media_{index}")
            if value is None:
                continue
            media_type = str(kwargs.get(f"media_type_{index}") or "").strip().lower()
            resolved_type = media_type if media_type in {"image", "video", "audio"} else _infer_media_type(value)
            items.append(_MediaInput(index, resolved_type, value))
        return items

    @staticmethod
    def _keyframes(items, role):
        images = [item.value for item in items if item.media_type == "image"]
        if any(item.media_type != "image" for item in items):
            raise ValueError("Image mode accepts image resources only")
        if len(images) > 2:
            raise ValueError("Image mode accepts at most two images")
        if not images:
            return None, None
        if len(images) == 1:
            if role == KEYFRAME_LAST:
                return None, images[0]
            return images[0], None
        if role == KEYFRAME_LAST:
            return images[1], images[0]
        return images[0], images[1]

    @classmethod
    def generate(cls, h3_bundle, mode, prompt, resolution, aspect_ratio, width, height, seconds, advanced, fps, keyframe_role, ref_image_size, reference_mention_mode, **kwargs):
        if not isinstance(h3_bundle, MiniMaxH3Bundle):
            raise ValueError("Connect a MiniMax H3 Easy Loader bundle")
        mode = str(mode)
        keyframe_role = KEYFRAME_LAST if str(keyframe_role) == KEYFRAME_LAST else KEYFRAME_FIRST
        width, height = _canvas_dimensions(resolution, aspect_ratio, width, height)
        seconds = min(MAX_SECONDS, max(MIN_SECONDS, float(seconds)))
        length = _frame_length(seconds, fps)
        items = cls._collect_media(kwargs)
        if mode == MODE_REFERENCE and items:
            if len(items) > MAX_MEDIA:
                raise ValueError("Reference mode accepts at most fifteen media resources")
            counts = {"image": 0, "video": 0, "audio": 0}
            for item in items:
                if item.media_type not in counts:
                    raise ValueError("Unsupported media resource")
                counts[item.media_type] += 1
            if counts["image"] > MAX_IMAGES or counts["video"] > MAX_VIDEOS or counts["audio"] > MAX_AUDIOS:
                raise ValueError("Reference mode media limits are 9 images, 3 videos and 3 audio clips")
            if counts["image"] == 0 and counts["video"] == 0:
                raise ValueError("Reference mode needs an image or video in addition to audio")
            model = h3_bundle.model_for("ref2va")
            conditioning, latent, context_prompt = _reference_conditioning(
                h3_bundle,
                prompt,
                width,
                height,
                length,
                ref_image_size,
                items,
                return_prompt=True,
            )
            context_mode = MODE_REFERENCE
            context_media = tuple((item.media_type, item.value) for item in items)
            keyframe_roles = ()
            prompt_encoder = partial(
                _reference_conditioning,
                h3_bundle,
                width=width,
                height=height,
                length=length,
                ref_image_size=ref_image_size,
                items=items,
            )
        else:
            first_frame, last_frame = cls._keyframes(items, keyframe_role)
            model = h3_bundle.model_for("fl2va")
            conditioning, latent = _empty_image_conditioning(h3_bundle, prompt, width, height, length, first_frame, last_frame)
            context_mode = MODE_IMAGE
            context_media_items = []
            keyframe_role_items = []
            if first_frame is not None:
                context_media_items.append(("image", first_frame))
                keyframe_role_items.append(KEYFRAME_FIRST)
            if last_frame is not None:
                context_media_items.append(("image", last_frame))
                keyframe_role_items.append(KEYFRAME_LAST)
            context_media = tuple(context_media_items)
            keyframe_roles = tuple(keyframe_role_items)
            prompt_encoder = partial(
                _empty_image_conditioning,
                h3_bundle,
                width=width,
                height=height,
                length=length,
                first_frame=first_frame,
                last_frame=last_frame,
            )
            context_prompt = str(prompt or "")
        context = MiniMaxH3Context(
            conditioning=conditioning,
            latent=latent,
            video_vae=h3_bundle.video_vae,
            audio_vae=h3_bundle.audio_vae,
            fps=float(fps),
            seconds=float(seconds),
            width=int(width),
            height=int(height),
            prompt=context_prompt,
            mode=context_mode,
            media=context_media,
            keyframe_roles=keyframe_roles,
            _prompt_encoder=prompt_encoder,
        )
        return model, context


class MiniMaxH3EasyPrompt:
    """Standalone prompt editor source for a MiniMax H3 Easy node.

    The browser extension turns its ``@`` mentions into H3 runtime reference
    placeholders when this output is connected to an H3 prompt input. Keeping
    the backend contract as a regular STRING lets it compose with native
    ComfyUI text nodes and avoids duplicating media inputs.
    """

    CATEGORY = "MiniMax H3 Easy"
    FUNCTION = "get_prompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    DESCRIPTION = "Edit a MiniMax H3 prompt in a separate node. Type @ to reference media connected to the target H3 node."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "",
                    },
                ),
            }
        }

    @staticmethod
    def get_prompt(prompt):
        return (str(prompt or ""),)


class MiniMaxH3EasyOutput:
    CATEGORY = "MiniMax H3 Easy"
    FUNCTION = "unpack"
    RETURN_TYPES = ("CONDITIONING", "LATENT", "VAE", "VAE", "FLOAT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "positive",
        "latent",
        "video_vae",
        "audio_vae",
        "fps",
        "original_width",
        "original_height",
        "scaled_width",
        "scaled_height",
    )
    DESCRIPTION = "Unpack the non-model outputs from a MiniMax H3 Easy context."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_context": ("MINIMAX_H3_CONTEXT",),
                "size_multiplier": (["1.2", "1.4", "1.5", "1.6", "2.0"], {"default": "1.5"}),
            },
            "optional": {
                "optimized_prompt": ("STRING", {"forceInput": True}),
            },
        }

    @staticmethod
    def unpack(h3_context, optimized_prompt=None, size_multiplier="1.5"):
        if not isinstance(h3_context, MiniMaxH3Context):
            raise ValueError("Connect the H3 Context output from a MiniMax H3 Easy node")
        while isinstance(optimized_prompt, (list, tuple)):
            optimized_prompt = optimized_prompt[0] if optimized_prompt else None
        conditioning = h3_context.conditioning
        latent = h3_context.latent
        if optimized_prompt is not None:
            optimized_prompt = str(optimized_prompt).strip()
            if not optimized_prompt:
                raise ValueError("The connected optimized prompt is empty")
            conditioning, latent = h3_context.encode_prompt(optimized_prompt)
        original_width = int(h3_context.width)
        original_height = int(h3_context.height)
        multiplier = float(size_multiplier)
        scaled_width = _align_canvas_dimension(original_width * multiplier)
        scaled_height = _align_canvas_dimension(original_height * multiplier)
        return (
            conditioning,
            latent,
            h3_context.video_vae,
            h3_context.audio_vae,
            h3_context.fps,
            original_width,
            original_height,
            scaled_width,
            scaled_height,
        )


def _face_refine_prompt(
    prompt: str,
    identity_reference: bool,
    prompt_mode: str = FACE_REFINE_PROMPT_IDENTITY_ONLY,
) -> str:
    prompt = str(prompt or "").strip()
    if not identity_reference:
        return prompt

    instruction = (
        "Refine facial details only. Preserve the exact facial identity from <Picture 1> "
        "and the source video, including facial geometry, proportions, skin tone, exposure, "
        "head pose, expression, gaze, lighting, motion and temporal continuity. Do not alter "
        "the hairstyle, clothing, background, camera, composition or scene lighting."
    )
    if prompt_mode == FACE_REFINE_PROMPT_FULL_SCENE:
        prompt = re.sub(
            r"<(?:Picture|Video|Audio)\s+\d+>",
            "",
            prompt,
            flags=re.IGNORECASE,
        )
        prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()
        if prompt:
            return f"<Picture 1>\n{prompt}\n{instruction}"
    return f"<Picture 1>\n{instruction}"


def _face_refine_seed(
    seed: int,
    seed_mode: str,
    identity_reference,
    subject_index: int = 0,
) -> int:
    input_seed = (int(seed) + int(subject_index)) & 0xFFFFFFFFFFFFFFFF
    if (
        seed_mode != FACE_REFINE_SEED_IDENTITY_LOCKED
        or not isinstance(identity_reference, torch.Tensor)
        or identity_reference.ndim != 4
        or identity_reference.shape[0] < 1
        or identity_reference.shape[1] < 1
        or identity_reference.shape[2] < 1
        or identity_reference.shape[3] < 1
    ):
        return input_seed

    height = int(identity_reference.shape[1])
    width = int(identity_reference.shape[2])
    channels = min(3, int(identity_reference.shape[3]))
    sample_height = min(8, height)
    sample_width = min(8, width)
    y_indices = torch.linspace(
        0,
        height - 1,
        sample_height,
        device=identity_reference.device,
    ).round().to(dtype=torch.long)
    x_indices = torch.linspace(
        0,
        width - 1,
        sample_width,
        device=identity_reference.device,
    ).round().to(dtype=torch.long)
    sampled = (
        identity_reference[0]
        .index_select(0, y_indices)
        .index_select(1, x_indices)
    )
    quantized = (
        sampled[..., :channels]
        .detach()
        .to(dtype=torch.float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(device="cpu", dtype=torch.int64)
        .reshape(-1)
    )
    weights = torch.arange(1, quantized.numel() + 1, dtype=torch.int64)
    checksum = int((quantized * weights).sum().item())
    mixed = (
        checksum
        ^ (height << 32)
        ^ width
        ^ (channels << 48)
        ^ ((int(subject_index) + 1) * 0x9E3779B97F4A7C15)
    ) & 0xFFFFFFFFFFFFFFFF
    mixed ^= mixed >> 30
    mixed = (mixed * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    mixed ^= mixed >> 27
    mixed = (mixed * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    mixed ^= mixed >> 31
    return mixed & 0xFFFFFFFFFFFFFFFF


def _face_refine_condition_inputs(
    h3_bundle,
    h3_context,
    prompt,
    width,
    height,
    length,
    identity_reference=None,
    prompt_mode=FACE_REFINE_PROMPT_IDENTITY_ONLY,
):
    inputs = {
        "clip": h3_bundle.clip,
        "vae": h3_bundle.video_vae,
        "audio_vae": h3_bundle.audio_vae,
        "prompt": _face_refine_prompt(
            prompt,
            identity_reference is not None,
            prompt_mode,
        ),
        "width": width,
        "height": height,
        "length": int(length),
        "ref_image_size": "max",
    }
    if identity_reference is not None:
        inputs["ref_images.ref_image_0"] = identity_reference
        return inputs

    image_index = video_index = audio_index = 0
    image_tags = []
    for media_type, value in h3_context.media:
        if media_type == "image":
            inputs[f"ref_images.ref_image_{image_index}"] = value
            image_index += 1
            image_tags.append(f"<Picture {image_index}>")
        elif media_type == "video":
            frames, soundtrack, source_fps = _video_parts(value)
            inputs[f"ref_videos.ref_video_{video_index}"] = _resample_video_frames(frames, source_fps)
            if soundtrack is not None:
                inputs[f"ref_video_audios.ref_video_audio_{video_index}"] = soundtrack
            video_index += 1
        elif media_type == "audio":
            inputs[f"ref_audios.ref_audio_{audio_index}"] = value
            audio_index += 1

    if h3_context.mode == MODE_IMAGE and image_tags:
        existing = str(inputs["prompt"])
        missing_tags = [tag for tag in image_tags if tag not in existing]
        if missing_tags:
            inputs["prompt"] = "\n".join([*missing_tags, existing]).strip()
    return inputs


class MiniMaxH3EasyAudioLock:
    CATEGORY = "MiniMax H3 Easy/Face Refine"
    FUNCTION = "lock"
    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("model", "av_latent")
    DESCRIPTION = "Lock the source audio latent while the face-refine video stream is denoised."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "av_latent": ("LATENT",),
                "audio_vae": ("VAE",),
                "audio": ("AUDIO",),
            }
        }

    @staticmethod
    def lock(model, av_latent, audio_vae, audio):
        samples = av_latent.get("samples")
        if samples is None or not (
            isinstance(samples, comfy.nested_tensor.NestedTensor)
            or getattr(samples, "is_nested", False)
        ):
            raise ValueError("Face refinement requires a MiniMax H3 joint AV latent")
        members = list(samples.unbind())
        if len(members) < 2:
            raise ValueError("MiniMax H3 AV latent is missing its audio stream")

        encoded, _ = _encode_reference_audio(audio_vae, audio)
        target = members[1]
        if tuple(encoded.shape[:-1]) != tuple(target.shape[:-1]):
            raise ValueError(
                f"Encoded audio shape {tuple(encoded.shape)} does not match H3 audio latent "
                f"shape {tuple(target.shape)}"
            )
        encoded = encoded.to(device=target.device, dtype=target.dtype)
        target_t = int(target.shape[-1])
        if encoded.shape[-1] > target_t:
            encoded = encoded[..., :target_t]
        elif encoded.shape[-1] < target_t:
            encoded = torch.cat(
                (
                    encoded,
                    torch.zeros(
                        (*encoded.shape[:-1], target_t - encoded.shape[-1]),
                        device=encoded.device,
                        dtype=encoded.dtype,
                    ),
                ),
                dim=-1,
            )
        members[1] = encoded

        out = dict(av_latent)
        out["samples"] = comfy.nested_tensor.NestedTensor(tuple(members))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (torch.ones_like(members[0]), torch.zeros_like(members[1]))
        )
        return model, out


class MiniMaxH3EasyReplaceVideoFrames:
    CATEGORY = "MiniMax H3 Easy/Face Refine"
    FUNCTION = "replace"
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",), "images": ("IMAGE",)}}

    @staticmethod
    def replace(video, images):
        components = video.get_components()
        if int(images.shape[0]) != int(components.images.shape[0]):
            raise ValueError(
                f"Face-refined frame count {images.shape[0]} does not match source video "
                f"frame count {components.images.shape[0]}"
            )
        output = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=images,
                audio=components.audio,
                frame_rate=components.frame_rate,
                metadata=getattr(components, "metadata", None),
                alpha=getattr(components, "alpha", None),
            ),
            bit_depth=video.get_bit_depth(),
        )
        return (output,)


class MiniMaxH3EasyFaceRefine:
    CATEGORY = "MiniMax H3 Easy"
    FUNCTION = "refine"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    DESCRIPTION = (
        "Track distant faces, regenerate stabilised face crops with MiniMax H3, and stitch "
        "them back while preserving the source video's audio, FPS and metadata."
    )

    @classmethod
    def INPUT_TYPES(cls):
        loras = _face_refine_lora_choices()
        samplers = _face_refine_sampler_choices()
        schedulers = _face_refine_scheduler_choices()
        positions = ["自动（最大脸）", "最左人物", "左起第2人", "左起第3人", "最右人物"]
        return {
            "required": {
                "video": ("VIDEO",),
                "h3_bundle": ("MINIMAX_H3_BUNDLE",),
                "h3_context": ("MINIMAX_H3_CONTEXT",),
                "subject_mode": ([FACE_REFINE_SINGLE, FACE_REFINE_TWO], {"default": FACE_REFINE_SINGLE}),
                "detector": (_face_refine_detector_choices(),),
                "turbo_lora": (loras, {"default": loras[0]}),
                "lora_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 30, "step": 1}),
                "denoise": ("FLOAT", {"default": 0.32, "min": 0.01, "max": 1.0, "step": 0.01}),
                "sampler": (samplers, {"default": "res_multistep" if "res_multistep" in samplers else samplers[0]}),
                "scheduler": (schedulers, {"default": "simple" if "simple" in schedulers else schedulers[0]}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "identity_position_1": (positions, {"default": positions[0]}),
                "identity_position_2": (positions, {"default": positions[0]}),
                "prompt_mode": (
                    [FACE_REFINE_PROMPT_IDENTITY_ONLY, FACE_REFINE_PROMPT_FULL_SCENE],
                    {"default": FACE_REFINE_PROMPT_IDENTITY_ONLY},
                ),
                "seed_mode": (
                    [FACE_REFINE_SEED_IDENTITY_LOCKED, FACE_REFINE_SEED_INPUT],
                    {"default": FACE_REFINE_SEED_IDENTITY_LOCKED},
                ),
            },
            "optional": {
                "identity_reference_1": ("IMAGE",),
                "identity_reference_2": ("IMAGE",),
                "optimized_prompt": ("STRING", {"forceInput": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @staticmethod
    def refine(
        video,
        h3_bundle,
        h3_context,
        subject_mode,
        detector,
        turbo_lora,
        lora_strength,
        steps,
        denoise,
        sampler,
        scheduler,
        seed,
        identity_position_1,
        identity_position_2,
        prompt_mode=FACE_REFINE_PROMPT_IDENTITY_ONLY,
        seed_mode=FACE_REFINE_SEED_IDENTITY_LOCKED,
        identity_reference_1=None,
        identity_reference_2=None,
        optimized_prompt=None,
    ):
        if not isinstance(h3_bundle, MiniMaxH3Bundle):
            raise ValueError("Connect the H3 Bundle output from MiniMax H3 Easy Loader")
        if not isinstance(h3_context, MiniMaxH3Context):
            raise ValueError("Connect the H3 Context output from MiniMax H3 Easy")
        if _is_none_model(h3_bundle.ref2va_model_name):
            raise ValueError("Distant-face refinement requires a model selected in the REF2VA slot")

        components = video.get_components()
        frames = components.images
        if int(frames.shape[0]) < 5:
            raise ValueError("Distant-face refinement requires at least five video frames")
        fps = float(components.frame_rate or h3_context.fps or h3.FPS)
        if abs(fps - h3.FPS) > 0.01:
            raise ValueError(
                f"Distant-face refinement requires a 24 FPS source, received {fps:g} FPS. "
                "Place this node before frame interpolation."
            )
        if subject_mode == FACE_REFINE_TWO and (
            identity_reference_1 is None or identity_reference_2 is None
        ):
            raise ValueError("Two-person refinement requires both identity reference inputs")

        while isinstance(optimized_prompt, (list, tuple)):
            optimized_prompt = optimized_prompt[0] if optimized_prompt else None
        prompt = str(optimized_prompt or h3_context.prompt or "").strip()
        effective_steps = _face_refine_step_count(turbo_lora, steps)

        g = GraphBuilder()
        model = h3_bundle.model_for("ref2va")
        if not _is_none_model(turbo_lora) and float(lora_strength) > 0:
            model = g.node(
                "LoraLoaderModelOnly",
                model=model,
                lora_name=turbo_lora,
                strength_model=float(lora_strength),
            ).out(0)

        sampler_node = g.node("KSamplerSelect", sampler_name=str(sampler))
        identities = [(identity_reference_1, identity_position_1)]
        if subject_mode == FACE_REFINE_TWO:
            identities.append((identity_reference_2, identity_position_2))

        final_images = frames
        last_report = None
        for subject_index, (identity_image, identity_position) in enumerate(identities):
            identity_face = None
            if identity_image is not None:
                identity_face = g.node(
                    "MiniMaxH3EasySelectIdentityFace",
                    image=identity_image,
                    detector=detector,
                    selection=identity_position,
                    confidence=0.35,
                    padding=0.55,
                ).out(0)

            tracker_inputs = {
                "images": final_images,
                "detector": detector,
                "confidence": 0.35,
                "crop_factor": 2.5,
                "canvas_width": 512,
                "canvas_height": 512,
                "canvas_mode": "auto_capped_768",
                "smooth_window": 11,
                "size_smooth_window": 81,
                "smooth_method": "gaussian",
                "size_mode": "per_frame",
                "identity_track": True,
                "identity_threshold": 0.28,
                "select": "largest",
                "fallback_detector": "none",
                "fallback_head_frac": 0.5,
            }
            if identity_face is not None:
                tracker_inputs["identity_reference"] = identity_face
            tracker = g.node("MiniMaxH3EasyFaceTrackCrop", **tracker_inputs)
            last_report = tracker.out(3)

            condition_inputs = _face_refine_condition_inputs(
                h3_bundle,
                h3_context,
                prompt,
                tracker.out(4),
                tracker.out(5),
                int(frames.shape[0]),
                identity_face,
                prompt_mode,
            )
            prepared = g.node("MiniMaxH3ReferenceToVideo", **condition_inputs)
            injected = g.node(
                "MiniMaxH3EasyInjectVideoLatent",
                av_latent=prepared.out(1),
                images=tracker.out(0),
                vae=h3_bundle.video_vae,
            )

            active_model = model
            active_latent = injected.out(0)
            if components.audio is not None:
                audio_locked = g.node(
                    "MiniMaxH3EasyAudioLock",
                    model=active_model,
                    av_latent=active_latent,
                    audio_vae=h3_bundle.audio_vae,
                    audio=components.audio,
                )
                active_model = audio_locked.out(0)
                active_latent = audio_locked.out(1)

            per_frame = g.node(
                "MiniMaxH3EasyPerFrameDenoise",
                av_latent=active_latent,
                transform=tracker.out(1),
                strength_small_face=0.80,
                strength_large_face=0.30,
                scale_mode="absolute_px",
                face_px_small=30.0,
                face_px_large=120.0,
                gamma=1.0,
                smooth_frames=25,
            )
            guider = g.node("BasicGuider", model=active_model, conditioning=prepared.out(0))
            sigmas = g.node(
                "BasicScheduler",
                model=active_model,
                scheduler=str(scheduler),
                steps=effective_steps,
                denoise=float(denoise),
            )
            noise = g.node(
                "RandomNoise",
                noise_seed=_face_refine_seed(
                    seed,
                    seed_mode,
                    identity_image,
                    subject_index,
                ),
            )
            sampled = g.node(
                "SamplerCustomAdvanced",
                noise=noise.out(0),
                guider=guider.out(0),
                sampler=sampler_node.out(0),
                sigmas=sigmas.out(0),
                latent_image=per_frame.out(0),
            )
            decoded = g.node("VAEDecode", samples=sampled.out(0), vae=h3_bundle.video_vae)
            stitched = g.node(
                "MiniMaxH3EasyFaceStitch",
                base_images=final_images,
                refined_crops=decoded.out(0),
                transform=tracker.out(1),
                paste_region="face_only",
                mask_dilation=24,
                feather=28,
                colour_match=1.0,
                blend=0.80,
                undetected_frames="fade_out",
                feather_scales_with_crop=False,
            )
            final_images = stitched.out(0)

        output_video = g.node("MiniMaxH3EasyReplaceVideoFrames", video=video, images=final_images)
        return {
            "result": (output_video.out(0), last_report),
            "expand": g.finalize(),
        }


def _per_second_frame_indices(seconds: float, fps: float, frame_count: int) -> list[int]:
    seconds = float(seconds)
    fps = float(fps)
    frame_count = int(frame_count)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Video seconds must be greater than zero")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video FPS must be greater than zero")
    if frame_count <= 0:
        raise ValueError("Video frame count must be greater than zero")

    indexes = []
    for second in range(math.ceil(seconds)):
        index = math.floor(second * fps)
        if index >= frame_count:
            break
        indexes.append(index)
    return indexes


def _extract_video_output_frames(frames, seconds: float, fps: float):
    actual_frame_count = int(frames.shape[0])
    if actual_frame_count <= 0:
        raise ValueError("The connected video contains no frames")

    indexes = _per_second_frame_indices(seconds, fps, actual_frame_count)
    return frames[indexes], frames[-1:]


H3_CHROMA_CONTEXT_FRAME_OPTIONS = ("1", "5", "22", "39")
H3_CHROMA_PALETTE = (
    (185, 115, 215),
    (115, 195, 140),
    (150, 148, 162),
    (205, 150, 192),
    (138, 182, 148),
    (160, 120, 175),
)
H3_CHROMA_GRID = (36, 64)
H3_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
H3_SEAM_ANALYSIS_SIZE = 384
H3_SEAM_MOTION_HISTORY = 4
H3_SEAM_COLOR_GRID = 8


def _h3_chroma_taper_alphas(
    context_frames: int,
    start_alpha: float,
    end_alpha: float,
    taper_frames: int,
) -> tuple[float, ...]:
    if context_frames < 1:
        raise ValueError("context_frames must be positive")
    if taper_frames < 1 or taper_frames > context_frames:
        raise ValueError("taper_frames must be between 1 and context_frames")
    if not 0.0 <= start_alpha <= 1.0 or not 0.0 <= end_alpha <= 1.0:
        raise ValueError("start_alpha and end_alpha must be between 0 and 1")
    if end_alpha > start_alpha:
        raise ValueError("end_alpha must not exceed start_alpha for a taper")

    alphas = []
    for position in range(context_frames):
        from_end = context_frames - 1 - position
        if from_end >= taper_frames:
            alphas.append(start_alpha)
            continue
        amount = start_alpha + (end_alpha - start_alpha) * (
            taper_frames - from_end
        ) / taper_frames
        alphas.append(amount)
    return tuple(alphas)


def _h3_luminance(images: torch.Tensor) -> torch.Tensor:
    weights = images.new_tensor(H3_LUMA_WEIGHTS)
    return (images * weights).sum(dim=-1, keepdim=True)


def _preserve_h3_chroma_luminance(
    clean_images: torch.Tensor,
    noisy_images: torch.Tensor,
) -> torch.Tensor:
    target_luma = _h3_luminance(clean_images)
    noisy_luma = _h3_luminance(noisy_images)
    chroma = noisy_images.sub_(noisy_luma)

    # Keep the noisy chroma direction, but shrink its strength where necessary
    # so rebuilding it around the clean luma remains inside RGB gamut.
    epsilon = 1e-6
    chroma_scale = torch.ones_like(target_luma)
    for channel_index in range(3):
        channel = chroma[..., channel_index:channel_index + 1]
        channel_bound = torch.full_like(target_luma, float("inf"))
        channel_bound = torch.where(
            channel > epsilon,
            (1.0 - target_luma) / channel.clamp_min(epsilon),
            channel_bound,
        )
        channel_bound = torch.where(
            channel < -epsilon,
            target_luma / (-channel).clamp_min(epsilon),
            channel_bound,
        )
        torch.minimum(chroma_scale, channel_bound, out=chroma_scale)
    chroma.mul_(chroma_scale).add_(target_luma).clamp_(0.0, 1.0)
    return chroma


def _prepare_h3_chroma_context(
    images: torch.Tensor,
    context_frames: int = 22,
    start_alpha: float = 0.45,
    end_alpha: float = 0.10,
    taper_frames: int = 3,
    seed: int = 0,
    preserve_luminance: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    alphas = _h3_chroma_taper_alphas(
        context_frames,
        start_alpha,
        end_alpha,
        taper_frames,
    )
    if not isinstance(images, torch.Tensor):
        raise TypeError("VIDEO images must be a torch.Tensor")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("VIDEO images must use [frames, height, width, 3] layout")
    if not images.is_floating_point():
        raise TypeError("VIDEO images must use a floating-point dtype")
    if images.shape[0] < context_frames:
        raise ValueError(
            f"The source VIDEO has {images.shape[0]} frames, but "
            f"context_frames={context_frames} was requested"
        )
    if images.shape[1] < 1 or images.shape[2] < 1:
        raise ValueError("VIDEO images must have a positive height and width")

    clean_context = images[-context_frames:].contiguous()
    noisy_context = clean_context.clone()
    rng = random.Random(int(seed))
    grid_width, grid_height = H3_CHROMA_GRID
    palette = torch.tensor(H3_CHROMA_PALETTE, dtype=torch.float32).div_(255.0)

    for frame_index, alpha in enumerate(alphas):
        palette_indexes = torch.tensor(
            [
                rng.randrange(len(H3_CHROMA_PALETTE))
                for _ in range(grid_width * grid_height)
            ],
            dtype=torch.long,
        ).reshape(grid_height, grid_width)
        grid = palette[palette_indexes].permute(2, 0, 1).unsqueeze(0)
        noise = functional.interpolate(
            grid,
            size=(images.shape[1], images.shape[2]),
            mode="nearest",
        ).squeeze(0).permute(1, 2, 0)
        noise = noise.to(device=images.device, dtype=images.dtype)
        noisy_context[frame_index].mul_(1.0 - alpha).add_(noise, alpha=alpha)

    if preserve_luminance:
        noisy_context = _preserve_h3_chroma_luminance(
            clean_context,
            noisy_context,
        )
    return clean_context, noisy_context.clamp_(0.0, 1.0)


class MiniMaxH3EasyChromaContext:
    CATEGORY = "MiniMax H3 Easy"
    FUNCTION = "create_context"
    RETURN_TYPES = ("VIDEO", "VIDEO")
    RETURN_NAMES = ("clean_context", "noisy_context")
    DESCRIPTION = (
        "Extract the clean tail of a video and create a tapered chroma-noise "
        "copy for the next MiniMax H3 Motion Context segment."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "context_frames": (
                    list(H3_CHROMA_CONTEXT_FRAME_OPTIONS),
                    {"default": "22"},
                ),
                "start_alpha": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_alpha": (
                    "FLOAT",
                    {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "taper_frames": (
                    "INT",
                    {"default": 3, "min": 1, "max": 39, "step": 1},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "preserve_luminance": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Preserve each clean context pixel's Rec.709 luminance while retaining chroma noise.",
                    },
                ),
            }
        }

    @staticmethod
    def create_context(
        video,
        context_frames="22",
        start_alpha=0.45,
        end_alpha=0.10,
        taper_frames=3,
        seed=0,
        preserve_luminance=True,
    ):
        try:
            frame_count = int(context_frames)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unsupported H3 context frame count: {context_frames!r}"
            ) from exc
        if str(frame_count) not in H3_CHROMA_CONTEXT_FRAME_OPTIONS:
            raise ValueError("context_frames must be one of 1, 5, 22 or 39")

        try:
            components = video.get_components()
            frame_rate = Fraction(components.frame_rate)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError("Failed to read the input VIDEO") from exc

        clean_images, noisy_images = _prepare_h3_chroma_context(
            images=components.images,
            context_frames=frame_count,
            start_alpha=float(start_alpha),
            end_alpha=float(end_alpha),
            taper_frames=int(taper_frames),
            seed=int(seed),
            preserve_luminance=bool(preserve_luminance),
        )
        bit_depth = video.get_bit_depth()

        def make_video(images):
            return InputImpl.VideoFromComponents(
                Types.VideoComponents(
                    images=images,
                    audio=None,
                    frame_rate=frame_rate,
                    metadata=getattr(components, "metadata", None),
                ),
                bit_depth=bit_depth,
            )

        return make_video(clean_images), make_video(noisy_images)


def _h3_phase_peak_offset(values: torch.Tensor, index: int) -> float:
    size = int(values.shape[0])
    center = values[index]
    before = values[(index - 1) % size]
    after = values[(index + 1) % size]
    denominator = before - 2.0 * center + after
    if abs(float(denominator)) < 1e-8:
        return 0.0
    offset = 0.5 * (before - after) / denominator
    return float(offset.clamp(-0.5, 0.5))


def _estimate_h3_frame_translation(
    source: torch.Tensor,
    target: torch.Tensor,
    max_analysis_size: int = H3_SEAM_ANALYSIS_SIZE,
) -> tuple[float, float]:
    if source.shape != target.shape or source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("H3 seam translation requires two matching RGB images")

    height, width = int(source.shape[0]), int(source.shape[1])
    analysis_height, analysis_width = height, width
    longest_side = max(height, width)
    if longest_side > max_analysis_size:
        scale = max_analysis_size / longest_side
        analysis_height = max(16, round(height * scale))
        analysis_width = max(16, round(width * scale))

    pair = torch.stack((source, target)).to(dtype=torch.float32)
    gray = _h3_luminance(pair).permute(0, 3, 1, 2)
    if (analysis_height, analysis_width) != (height, width):
        gray = functional.interpolate(
            gray,
            size=(analysis_height, analysis_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    gray = gray[:, 0]
    gray = gray - gray.mean(dim=(-2, -1), keepdim=True)
    window_y = torch.hann_window(
        analysis_height,
        periodic=False,
        device=gray.device,
        dtype=gray.dtype,
    )
    window_x = torch.hann_window(
        analysis_width,
        periodic=False,
        device=gray.device,
        dtype=gray.dtype,
    )
    gray = gray * window_y[:, None] * window_x[None, :]

    source_spectrum = torch.fft.rfft2(gray[0])
    target_spectrum = torch.fft.rfft2(gray[1])
    cross_power = target_spectrum * source_spectrum.conj()
    cross_power = cross_power / cross_power.abs().clamp_min(1e-8)
    correlation = torch.fft.irfft2(
        cross_power,
        s=(analysis_height, analysis_width),
    ).real
    flat_index = int(correlation.argmax())
    peak_y, peak_x = divmod(flat_index, analysis_width)
    offset_x = _h3_phase_peak_offset(correlation[peak_y], peak_x)
    offset_y = _h3_phase_peak_offset(correlation[:, peak_x], peak_y)
    if peak_x > analysis_width // 2:
        peak_x -= analysis_width
    if peak_y > analysis_height // 2:
        peak_y -= analysis_height

    shift_x = (peak_x + offset_x) * width / analysis_width
    shift_y = (peak_y + offset_y) * height / analysis_height
    return float(shift_x), float(shift_y)


def _translate_h3_frames(
    images: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("H3 seam frames must use [frames, height, width, 3] layout")
    if shifts.shape != (images.shape[0], 2):
        raise ValueError("H3 seam shifts must use [frames, 2] layout")

    source_dtype = images.dtype
    working = images.permute(0, 3, 1, 2).to(dtype=torch.float32)
    shifts = shifts.to(device=working.device, dtype=working.dtype)
    height, width = int(images.shape[1]), int(images.shape[2])
    theta = torch.zeros(
        (images.shape[0], 2, 3),
        device=working.device,
        dtype=working.dtype,
    )
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    if width > 1:
        theta[:, 0, 2] = -2.0 * shifts[:, 0] / (width - 1)
    if height > 1:
        theta[:, 1, 2] = -2.0 * shifts[:, 1] / (height - 1)
    grid = functional.affine_grid(theta, working.shape, align_corners=True)
    translated = functional.grid_sample(
        working,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return translated.permute(0, 2, 3, 1).to(dtype=source_dtype)


def _h3_seam_decay_weights(
    frame_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if frame_count < 1:
        raise ValueError("H3 seam correction requires at least one frame")
    if frame_count == 1:
        return torch.ones(1, device=device, dtype=dtype)
    positions = torch.linspace(0.0, 1.0, frame_count, device=device, dtype=dtype)
    return 0.5 * (1.0 + torch.cos(math.pi * positions))


def _stabilize_h3_seam(
    images: torch.Tensor,
    context_frames: int = 22,
    correction_frames: int = 12,
    position_strength: float = 0.5,
    color_strength: float = 0.75,
    max_position_shift: float = 4.0,
    max_color_shift: float = 0.08,
) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        raise TypeError("H3 seam input must be a torch.Tensor")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("H3 seam images must use [frames, height, width, 3] layout")
    if not images.is_floating_point():
        raise TypeError("H3 seam images must use a floating-point dtype")
    if context_frames == 0:
        return images
    if context_frames < 0 or context_frames >= int(images.shape[0]):
        raise ValueError("context_frames must leave at least one generated frame")
    if correction_frames < 1:
        raise ValueError("correction_frames must be positive")
    if not 0.0 <= position_strength <= 1.0:
        raise ValueError("position_strength must be between 0 and 1")
    if not 0.0 <= color_strength <= 1.0:
        raise ValueError("color_strength must be between 0 and 1")
    if max_position_shift < 0.0 or max_color_shift < 0.0:
        raise ValueError("H3 seam correction limits must not be negative")

    boundary = int(context_frames)
    corrected_count = min(int(correction_frames), int(images.shape[0]) - boundary)
    output = images.clone()
    reference = images[boundary - 1]

    history_count = min(H3_SEAM_MOTION_HISTORY, boundary - 1)
    history_shifts = []
    for frame_index in range(boundary - history_count, boundary):
        history_shifts.append(
            _estimate_h3_frame_translation(
                images[frame_index - 1],
                images[frame_index],
            )
        )
    if history_shifts:
        history_tensor = images.new_tensor(history_shifts, dtype=torch.float32)
        expected_shift = history_tensor.median(dim=0).values
    else:
        expected_shift = images.new_zeros(2, dtype=torch.float32)

    actual_shift = images.new_tensor(
        _estimate_h3_frame_translation(reference, images[boundary]),
        dtype=torch.float32,
    )
    position_correction = (expected_shift - actual_shift) * float(position_strength)
    position_correction.clamp_(
        min=-float(max_position_shift),
        max=float(max_position_shift),
    )
    weights = _h3_seam_decay_weights(
        corrected_count,
        device=images.device,
        dtype=torch.float32,
    )
    shifts = weights[:, None] * position_correction[None, :]
    source_frames = images[boundary:boundary + corrected_count]
    if float(position_correction.abs().max()) > 1e-6:
        corrected = _translate_h3_frames(source_frames, shifts)
    else:
        corrected = source_frames.clone()

    if color_strength > 0.0 and max_color_shift > 0.0:
        predicted_reference = _translate_h3_frames(
            reference.unsqueeze(0),
            expected_shift.reshape(1, 2),
        )[0].to(dtype=torch.float32)
        first_corrected = corrected[0].to(dtype=torch.float32)
        color_delta = (predicted_reference - first_corrected).permute(2, 0, 1).unsqueeze(0)
        grid_height = min(H3_SEAM_COLOR_GRID, int(images.shape[1]))
        grid_width = min(H3_SEAM_COLOR_GRID, int(images.shape[2]))
        color_delta = functional.adaptive_avg_pool2d(
            color_delta,
            output_size=(grid_height, grid_width),
        )
        color_delta = functional.interpolate(
            color_delta,
            size=(int(images.shape[1]), int(images.shape[2])),
            mode="bilinear",
            align_corners=False,
        ).clamp_(-float(max_color_shift), float(max_color_shift))
        color_delta = color_delta[0].permute(1, 2, 0)
        color_weights = weights * float(color_strength)
        corrected = corrected.to(dtype=torch.float32)
        corrected.add_(color_delta[None, ...] * color_weights[:, None, None, None])
        corrected.clamp_(0.0, 1.0)
        corrected = corrected.to(dtype=images.dtype)

    if corrected_count > 1:
        corrected[-1] = source_frames[-1]
    output[boundary:boundary + corrected_count] = corrected
    return output


class MiniMaxH3EasySeamStabilizer:
    CATEGORY = "MiniMax H3 Easy"
    FUNCTION = "stabilize"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = (
        "Stabilize the transition after H3 Motion Context overlap frames, "
        "then pass the unchanged frame count to the existing trim node."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "context_frames": (
                    "INT",
                    {"forceInput": True, "min": 0, "max": 39},
                ),
                "correction_frames": (
                    "INT",
                    {"default": 12, "min": 1, "max": 48, "step": 1},
                ),
                "position_strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "color_strength": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "max_position_shift": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 32.0, "step": 0.25},
                ),
                "max_color_shift": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.0, "max": 0.25, "step": 0.01},
                ),
            }
        }

    @staticmethod
    def stabilize(
        images,
        context_frames=0,
        correction_frames=12,
        position_strength=0.5,
        color_strength=0.75,
        max_position_shift=4.0,
        max_color_shift=0.08,
    ):
        frame_count = int(context_frames)
        return (
            _stabilize_h3_seam(
                images=images,
                context_frames=frame_count,
                correction_frames=int(correction_frames),
                position_strength=float(position_strength),
                color_strength=float(color_strength),
                max_position_shift=float(max_position_shift),
                max_color_shift=float(max_color_shift),
            ),
        )


def _parse_h3_media_manifest(value: Any) -> dict[str, list[str]]:
    empty = {"images": [], "audios": [], "videos": []}
    if isinstance(value, list):
        value = value[0] if value else ""
    try:
        payload = value if isinstance(value, Mapping) else json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return empty
    if not isinstance(payload, Mapping):
        return empty

    result = {}
    for key in empty:
        items = payload.get(key, [])
        if not isinstance(items, list):
            items = []
        result[key] = [str(item).strip() for item in items if str(item).strip()]
    return result


H3_MEDIA_EXTENSIONS = {
    "images": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"},
    "audios": {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"},
    "videos": {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"},
}


def _validate_h3_media_manifest(manifest: Mapping[str, list[str]]) -> str | None:
    for media_type, extensions in H3_MEDIA_EXTENSIONS.items():
        for name in manifest.get(media_type, []):
            if os.path.splitext(str(name))[1].lower() not in extensions:
                return f"Invalid {media_type[:-1]} file type: {name}"
            if not folder_paths.exists_annotated_filepath(name):
                return f"Missing {media_type[:-1]} file: {name}"
    return None


def _h3_media_target_size(
    width: int,
    height: int,
    scale: float,
    resize_mode: str = "倍率",
    edge_length: int = 1024,
    divisible_by: int = 1,
) -> tuple[int, int]:
    source_width = max(1, int(width))
    source_height = max(1, int(height))
    target_edge = max(1, int(edge_length))
    mode = str(resize_mode).strip().lower()

    if mode in {"不缩放", "原图", "none", "original", "disabled"}:
        return source_width, source_height

    if mode in {"长边", "long edge", "long_edge"}:
        if source_width >= source_height:
            target_width, target_height = target_edge, max(1, round(source_height * target_edge / source_width))
        else:
            target_width, target_height = max(1, round(source_width * target_edge / source_height)), target_edge
    elif mode in {"短边", "short edge", "short_edge"}:
        if source_width <= source_height:
            target_width, target_height = target_edge, max(1, round(source_height * target_edge / source_width))
        else:
            target_width, target_height = max(1, round(source_width * target_edge / source_height)), target_edge
    else:
        factor = max(0.01, float(scale))
        target_width = max(1, round(source_width * factor))
        target_height = max(1, round(source_height * factor))

    size_factor = max(1, int(divisible_by))
    if size_factor > 1:
        target_width = max(size_factor, target_width - (target_width % size_factor))
        target_height = max(size_factor, target_height - (target_height % size_factor))
    return target_width, target_height


def _resize_h3_media_image(
    image: torch.Tensor,
    scale: float,
    method: str,
    resize_mode: str = "倍率",
    edge_length: int = 1024,
    divisible_by: int = 1,
) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise TypeError("Loaded images must use ComfyUI's [batch, height, width, channels] layout")
    source_height, source_width = int(image.shape[1]), int(image.shape[2])
    target_width, target_height = _h3_media_target_size(
        source_width,
        source_height,
        scale,
        resize_mode,
        edge_length,
        divisible_by,
    )
    if target_width == source_width and target_height == source_height:
        return image
    return comfy.utils.common_upscale(
        image.movedim(-1, 1),
        target_width,
        target_height,
        str(method),
        "disabled",
    ).movedim(1, -1)


def _select_h3_multi_set_outputs(
    input_values: Mapping[str, Any],
    pair_count: int,
) -> tuple[Any, ...]:
    """Route repeated Comfy lists to one item per connected Multi Set input."""
    source_positions: dict[int, int] = {}
    outputs = []
    for slot in range(pair_count):
        values = input_values.get(f"value_{slot + 1}", [])
        if not isinstance(values, (list, tuple)):
            values = [values]
        if not values:
            outputs.append(None)
            continue
        if len(values) == 1:
            outputs.append(values[0])
            continue

        source_id = id(values)
        source_index = source_positions.get(source_id, 0)
        if source_index >= len(values):
            raise ValueError(
                "Multi Set has more connections to one list output than that list has items"
            )
        outputs.append(values[source_index])
        source_positions[source_id] = source_index + 1
    return tuple(outputs)


class MiniMaxH3EasyMultiSet:
    CATEGORY = "MiniMax H3 Easy/Utilities"
    FUNCTION = "route_values"
    RETURN_TYPES = (H3_ANY_TYPE,) * H3_MULTI_SET_MAX_PAIRS
    RETURN_NAMES = tuple(f"value_{index + 1}" for index in range(H3_MULTI_SET_MAX_PAIRS))
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False,) * H3_MULTI_SET_MAX_PAIRS
    DESCRIPTION = (
        "Collect dynamically typed values for KJ Get nodes. Repeated connections from "
        "the same Comfy list output are distributed in order, one item per port."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                f"value_{index + 1}": (H3_ANY_TYPE, {"forceInput": True})
                for index in range(H3_MULTI_SET_MAX_PAIRS)
            }
        }

    @staticmethod
    def route_values(**kwargs):
        return _select_h3_multi_set_outputs(kwargs, H3_MULTI_SET_MAX_PAIRS)


class MiniMaxH3EasyAreaSwitch:
    """Select one function-area output without evaluating the other branch."""

    CATEGORY = "MiniMax H3 Easy/Utilities"
    FUNCTION = "route"
    RETURN_TYPES = (H3_ANY_TYPE,)
    RETURN_NAMES = ("selected",)
    DESCRIPTION = (
        "Route one of two function-area outputs. Inputs are lazy, so the bypassed "
        "area is never evaluated. Auto sync follows the Ignore Groups GuHai node."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_first": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "反推",
                        "label_off": "不反推",
                    },
                ),
                "first_area": (
                    "STRING",
                    {"default": "反推", "multiline": False},
                ),
                "second_area": (
                    "STRING",
                    {"default": "不反推", "multiline": False},
                ),
                "auto_sync": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "自动跟随忽略多组",
                        "label_off": "手动控制",
                    },
                ),
            },
            "optional": {
                "first": (H3_ANY_TYPE, {"lazy": True}),
                "second": (H3_ANY_TYPE, {"lazy": True}),
            },
        }

    @staticmethod
    def check_lazy_status(use_first, first=None, second=None, **_kwargs):
        selected_name = "first" if bool(use_first) else "second"
        selected_value = first if selected_name == "first" else second
        return [selected_name] if selected_value is None else []

    @staticmethod
    def route(use_first, first=None, second=None, **_kwargs):
        return (first if bool(use_first) else second,)


class MiniMaxH3EasyMediaLoader:
    CATEGORY = "MiniMax H3 Easy"
    FUNCTION = "load_media"
    RETURN_TYPES = ("IMAGE", "AUDIO", "VIDEO")
    RETURN_NAMES = ("multi output", "audio output", "video output")
    OUTPUT_IS_LIST = (True, True, True)
    DESCRIPTION = (
        "Load ordered image, audio, and video lists without mixing media types. "
        "Image resize mode is the only active resize rule: scale mode uses the custom "
        "factor, while long-edge and short-edge modes use only the target edge length."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media_manifest": (
                    "STRING",
                    {"default": '{"version":1,"images":[],"audios":[],"videos":[]}'},
                ),
                "image_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01},
                ),
                "scale_method": (
                    ["lanczos", "bicubic", "bilinear", "area", "nearest-exact"],
                    {"default": "lanczos"},
                ),
            },
            "optional": {
                "image_resize_mode": (
                    ["不缩放", "倍率", "长边", "短边"],
                    {"default": "倍率"},
                ),
                "image_edge_length": (
                    "INT",
                    {"default": 1024, "min": 1, "max": 16384, "step": 8},
                ),
                "image_divisible_by": (
                    "INT",
                    {"default": 1, "min": 1, "max": 512, "step": 1},
                ),
            },
        }

    @staticmethod
    def load_media(
        media_manifest,
        image_scale=1.0,
        scale_method="lanczos",
        image_resize_mode="倍率",
        image_edge_length=1024,
        image_divisible_by=1,
    ):
        manifest = _parse_h3_media_manifest(media_manifest)
        validation_error = _validate_h3_media_manifest(manifest)
        if validation_error:
            raise ValueError(validation_error)

        image_outputs = []
        image_loader = nodes.LoadImage()
        for name in manifest["images"]:
            image, _mask = image_loader.load_image(name)
            image_outputs.append(
                _resize_h3_media_image(
                    image,
                    float(image_scale),
                    str(scale_method),
                    str(image_resize_mode),
                    int(image_edge_length),
                    int(image_divisible_by),
                )
            )

        from comfy_extras.nodes_audio import load as load_audio_file

        audio_outputs = []
        for name in manifest["audios"]:
            path = folder_paths.get_annotated_filepath(name)
            waveform, sample_rate = load_audio_file(path)
            audio_outputs.append(
                {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            )

        video_outputs = [
            InputImpl.VideoFromFile(folder_paths.get_annotated_filepath(name))
            for name in manifest["videos"]
        ]
        return image_outputs, audio_outputs, video_outputs

    @classmethod
    def IS_CHANGED(
        cls,
        media_manifest,
        image_scale=1.0,
        scale_method="lanczos",
        image_resize_mode="倍率",
        image_edge_length=1024,
        image_divisible_by=1,
    ):
        manifest = _parse_h3_media_manifest(media_manifest)
        signature = [
            str(float(image_scale)),
            str(scale_method),
            str(image_resize_mode),
            str(int(image_edge_length)),
            str(int(image_divisible_by)),
        ]
        for media_type in ("images", "audios", "videos"):
            for name in manifest[media_type]:
                try:
                    path = folder_paths.get_annotated_filepath(name)
                    stat = os.stat(path)
                    signature.extend((media_type, name, str(stat.st_mtime_ns), str(stat.st_size)))
                except OSError:
                    signature.extend((media_type, name, "missing"))
        return "|".join(signature)

    @classmethod
    def VALIDATE_INPUTS(cls, media_manifest, **_kwargs):
        manifest = _parse_h3_media_manifest(media_manifest)
        return _validate_h3_media_manifest(manifest) or True


class MiniMaxH3EasySaveVideo:
    CATEGORY = "MiniMax H3 Easy"
    FUNCTION = "save"
    RETURN_TYPES = ("VIDEO", "IMAGE", "IMAGE")
    RETURN_NAMES = ("video", "frames_per_second", "last_frame")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Save a ComfyUI VIDEO with a manually resizable preview, and output the first "
        "frame of each second plus the video's actual last frame."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "seconds": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 86400.0, "step": 0.01}),
                "fps": ("FLOAT", {"forceInput": True}),
                "filename_prefix": (
                    "STRING",
                    {"default": "video/MiniMaxH3", "multiline": False},
                ),
                "format": (["auto", "mp4"], {"default": "auto"}),
                "codec": (["auto", "h264"], {"default": "auto"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    @staticmethod
    def save(
        video,
        seconds,
        fps,
        filename_prefix,
        format,
        codec,
        prompt=None,
        extra_pnginfo=None,
    ):
        components = video.get_components()
        sampled_frames, last_frame = _extract_video_output_frames(
            components.images,
            seconds,
            fps,
        )

        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height,
        )
        metadata = None
        if not args.disable_metadata:
            metadata_items = {}
            if extra_pnginfo is not None:
                metadata_items.update(extra_pnginfo)
            if prompt is not None:
                metadata_items["prompt"] = prompt
            if metadata_items:
                metadata = metadata_items

        extension = Types.VideoContainer.get_extension(format)
        output_file = f"{filename}_{counter:05}_.{extension}"
        video.save_to(
            os.path.join(full_output_folder, output_file),
            format=Types.VideoContainer(format),
            codec=str(codec),
            metadata=metadata,
        )

        return {
            "ui": {
                "h3_saved_video": [
                    {
                        "filename": output_file,
                        "subfolder": subfolder,
                        "type": "output",
                    }
                ]
            },
            "result": (video, sampled_frames, last_frame),
        }


def _rife_vfi_node_class():
    node_class = nodes.NODE_CLASS_MAPPINGS.get("RIFE_VFI_Opt")
    if node_class is None:
        raise RuntimeError(
            "MiniMax H3 Easy Frame Interpolation requires the WhiteRabbit custom node "
            "with its RIFE 4.7 model. Install or enable comfyui-WhiteRabbit, then restart ComfyUI."
        )
    return node_class


class MiniMaxH3EasyFrameInterpolation:
    CATEGORY = "MiniMax H3 Easy"
    FUNCTION = "interpolate"
    RETURN_TYPES = ("VIDEO", "FLOAT")
    RETURN_NAMES = ("video", "fps")
    DESCRIPTION = (
        "Double a video's frame rate with WhiteRabbit RIFE 4.7 while preserving "
        "the original resolution and audio."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",)}}

    @staticmethod
    def interpolate(video):
        components = video.get_components()
        frames = components.images
        if int(frames.shape[0]) < 2:
            raise ValueError("Frame interpolation requires a video with at least two frames")

        source_fps = float(components.frame_rate)
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Video FPS must be greater than zero")

        interpolated_frames, = _rife_vfi_node_class()().vfi(
            ckpt_name="rife47.pth",
            frames=frames,
            multiplier=2,
            scale_factor=1.0,
            ensemble=False,
            clear_cache_after_n_frames=10,
        )
        # RIFE returns 2N-1 frames. Holding the original final frame once keeps
        # the encoded 2x-FPS video duration (and therefore its audio) unchanged.
        interpolated_frames = torch.cat(
            (
                interpolated_frames,
                frames[-1:].to(
                    device=interpolated_frames.device,
                    dtype=interpolated_frames.dtype,
                ),
            ),
            dim=0,
        )
        output_fps = source_fps * 2.0
        output_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=interpolated_frames,
                audio=components.audio,
                frame_rate=Fraction(round(output_fps * 1000), 1000),
                metadata=getattr(components, "metadata", None),
            ),
            bit_depth=video.get_bit_depth(),
        )
        return output_video, output_fps


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3EasyLoader": MiniMaxH3EasyLoader,
    "MiniMaxH3EasyPrompt": MiniMaxH3EasyPrompt,
    "MiniMaxH3Easy": MiniMaxH3Easy,
    "MiniMaxH3EasyOutput": MiniMaxH3EasyOutput,
    "MiniMaxH3EasyFaceRefine": MiniMaxH3EasyFaceRefine,
    "MiniMaxH3EasyAudioLock": MiniMaxH3EasyAudioLock,
    "MiniMaxH3EasyReplaceVideoFrames": MiniMaxH3EasyReplaceVideoFrames,
    "MiniMaxH3EasySaveVideo": MiniMaxH3EasySaveVideo,
    "MiniMaxH3EasyFrameInterpolation": MiniMaxH3EasyFrameInterpolation,
    "MiniMaxH3EasyChromaContext": MiniMaxH3EasyChromaContext,
    "MiniMaxH3EasySeamStabilizer": MiniMaxH3EasySeamStabilizer,
    "MiniMaxH3EasyMediaLoader": MiniMaxH3EasyMediaLoader,
    "MiniMaxH3EasyMultiSet": MiniMaxH3EasyMultiSet,
}
