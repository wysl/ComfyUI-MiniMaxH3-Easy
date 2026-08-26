"""Small HTTP helpers used by the media loader browser widget."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from aiohttp import web

import folder_paths
from server import PromptServer

from .nodes import H3_MEDIA_EXTENSIONS


INPUT_MEDIA_ROUTE = "/minimax_h3_easy/input-media"
INPUT_PREVIEW_ROUTE = "/minimax_h3_easy/input-preview"


def _list_input_media(kind: str) -> list[str]:
    """Return safe, relative media paths below ComfyUI's configured input folder."""
    extensions = H3_MEDIA_EXTENSIONS.get(kind)
    if extensions is None:
        return []

    root = Path(folder_paths.get_input_directory()).resolve()
    files: list[str] = []
    for candidate in root.rglob("*"):
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if not resolved.is_file() or resolved.suffix.lower() not in extensions:
            continue
        files.append(resolved.relative_to(root).as_posix())
    return sorted(files, key=str.casefold)


def _resolve_input_image(filename: str) -> Path | None:
    """Resolve an image below the configured input directory without path traversal."""
    normalized = str(filename or "").replace("\\", "/").lstrip("/")
    if not normalized or Path(normalized).suffix.lower() not in H3_MEDIA_EXTENSIONS["images"]:
        return None

    root = Path(folder_paths.get_input_directory()).resolve()
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def register_media_routes() -> None:
    """Register the input browser route once when the custom node is loaded."""
    server = PromptServer.instance
    if server is None:
        return

    @server.routes.get(INPUT_MEDIA_ROUTE)
    async def list_input_media(request):
        kind = str(request.query.get("kind", "images")).strip().lower()
        if kind not in H3_MEDIA_EXTENSIONS:
            return web.json_response({"error": "Unsupported media category"}, status=400)
        return web.json_response({"kind": kind, "files": _list_input_media(kind)})

    @server.routes.get(INPUT_PREVIEW_ROUTE)
    async def preview_input_image(request):
        image_path = _resolve_input_image(request.query.get("filename", ""))
        if image_path is None:
            return web.Response(status=404)

        try:
            from PIL import Image

            with Image.open(image_path) as source:
                preview = source.convert("RGB")
                resampling = getattr(Image, "Resampling", Image).BILINEAR
                preview.thumbnail((256, 256), resampling)
                buffer = BytesIO()
                preview.save(buffer, format="JPEG", quality=82, optimize=True)
        except (OSError, ValueError):
            return web.Response(status=404)

        return web.Response(
            body=buffer.getvalue(),
            content_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=60"},
        )
