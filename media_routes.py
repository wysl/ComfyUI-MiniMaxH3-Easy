"""Small HTTP helpers used by the media loader browser widget."""

from __future__ import annotations

from pathlib import Path

from aiohttp import web

import folder_paths
from server import PromptServer

from .nodes import H3_MEDIA_EXTENSIONS


INPUT_MEDIA_ROUTE = "/minimax_h3_easy/input-media"


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

