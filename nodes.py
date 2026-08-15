"""A compact MiniMax H3 entry point for ComfyUI.

The node intentionally keeps the graph contract small: one loader bundle, one
mode-aware conditioning node, and standard ComfyUI outputs for the sampler
chain. The browser extension supplies the ordered virtual media inputs.
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache, partial
from typing import Any

import torch
import torchaudio

import comfy.nested_tensor
import comfy.model_management
import comfy.samplers
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


def _face_refine_prompt(prompt: str, identity_reference: bool) -> str:
    prompt = str(prompt or "").strip()
    if not identity_reference:
        return prompt
    prompt = re.sub(r"<(?:Picture|Video|Audio)\s+\d+>", "", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()
    instruction = (
        "Preserve the facial identity from <Picture 1>, the original head pose, expression, "
        "lighting, motion and temporal continuity. Refine facial details only."
    )
    return f"<Picture 1>\n{prompt}\n{instruction}" if prompt else f"<Picture 1>\n{instruction}"


def _face_refine_condition_inputs(
    h3_bundle,
    h3_context,
    prompt,
    width,
    height,
    length,
    identity_reference=None,
):
    inputs = {
        "clip": h3_bundle.clip,
        "vae": h3_bundle.video_vae,
        "audio_vae": h3_bundle.audio_vae,
        "prompt": _face_refine_prompt(prompt, identity_reference is not None),
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
            noise = g.node("RandomNoise", noise_seed=(int(seed) + subject_index) & 0xFFFFFFFFFFFFFFFF)
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
    "MiniMaxH3Easy": MiniMaxH3Easy,
    "MiniMaxH3EasyOutput": MiniMaxH3EasyOutput,
    "MiniMaxH3EasyFaceRefine": MiniMaxH3EasyFaceRefine,
    "MiniMaxH3EasyAudioLock": MiniMaxH3EasyAudioLock,
    "MiniMaxH3EasyReplaceVideoFrames": MiniMaxH3EasyReplaceVideoFrames,
    "MiniMaxH3EasySaveVideo": MiniMaxH3EasySaveVideo,
    "MiniMaxH3EasyFrameInterpolation": MiniMaxH3EasyFrameInterpolation,
}
