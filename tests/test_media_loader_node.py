from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest


PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_PATH = PROJECT_ROOT / "nodes.py"
WEB_PATH = PROJECT_ROOT / "web" / "media_loader.js"


def load_media_loader_symbols():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    names = {
        "_parse_h3_media_manifest",
        "_validate_h3_media_manifest",
        "_h3_media_target_size",
        "MiniMaxH3EasyMediaLoader",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    namespace = {
        "Any": Any,
        "Mapping": Mapping,
        "json": json,
        "os": __import__("os"),
        "folder_paths": None,
        "H3_MEDIA_EXTENSIONS": {
            "images": {".png", ".jpg"},
            "audios": {".wav", ".mp3"},
            "videos": {".mp4", ".webm"},
        },
    }
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace


class MediaLoaderNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.symbols = load_media_loader_symbols()
        cls.web_source = WEB_PATH.read_text(encoding="utf-8")

    def test_manifest_keeps_media_categories_and_selection_order_separate(self):
        parse = self.symbols["_parse_h3_media_manifest"]
        manifest = parse(
            json.dumps(
                {
                    "version": 1,
                    "images": ["images/02.png", "images/01.png"],
                    "audios": ["audio/theme.wav"],
                    "videos": ["video/a.mp4", "video/b.mp4"],
                }
            )
        )

        self.assertEqual(manifest["images"], ["images/02.png", "images/01.png"])
        self.assertEqual(manifest["audios"], ["audio/theme.wav"])
        self.assertEqual(manifest["videos"], ["video/a.mp4", "video/b.mp4"])

    def test_invalid_manifest_returns_three_empty_categories(self):
        parse = self.symbols["_parse_h3_media_manifest"]

        self.assertEqual(
            parse("not json"),
            {"images": [], "audios": [], "videos": []},
        )

    def test_backend_rejects_media_in_the_wrong_category(self):
        validate = self.symbols["_validate_h3_media_manifest"]
        validate.__globals__["folder_paths"] = SimpleNamespace(
            exists_annotated_filepath=lambda _name: True
        )

        error = validate(
            {"images": ["clip.mp4"], "audios": [], "videos": []}
        )

        self.assertEqual(error, "Invalid image file type: clip.mp4")

    def test_scale_preserves_aspect_ratio(self):
        target_size = self.symbols["_h3_media_target_size"]

        self.assertEqual(target_size(640, 360, 1.5), (960, 540))
        self.assertEqual(target_size(101, 51, 0.5), (50, 26))

    def test_outputs_are_three_independent_comfy_lists(self):
        node = self.symbols["MiniMaxH3EasyMediaLoader"]

        self.assertEqual(node.RETURN_TYPES, ("IMAGE", "AUDIO", "VIDEO"))
        self.assertEqual(
            node.RETURN_NAMES,
            ("multi output", "audio output", "video output"),
        )
        self.assertEqual(node.OUTPUT_IS_LIST, (True, True, True))

    def test_image_scaling_controls_are_built_in(self):
        required = self.symbols["MiniMaxH3EasyMediaLoader"].INPUT_TYPES()["required"]

        self.assertIn("image_scale", required)
        self.assertIn("scale_method", required)
        self.assertEqual(required["image_scale"][1]["default"], 1.0)
        self.assertIn("lanczos", required["scale_method"][0])

    def test_frontend_supports_multi_select_preview_order_and_resize(self):
        self.assertIn("input.multiple = true;", self.web_source)
        self.assertIn("h3-media-image-grid", self.web_source)
        self.assertIn("number.textContent = String(index + 1);", self.web_source)
        self.assertIn('element.addEventListener("dragstart"', self.web_source)
        self.assertIn("node.resizable = true;", self.web_source)
        self.assertIn("getMinHeight: () => 200", self.web_source)

    def test_frontend_uses_separate_upload_tabs_and_server_paths(self):
        self.assertIn('activeKind: "images"', self.web_source)
        self.assertIn('form.append("subfolder", `minimax_h3_easy/media_loader/${kind}`);', self.web_source)
        self.assertIn('api.fetchApi("/upload/image"', self.web_source)
        self.assertNotIn("URL.createObjectURL", self.web_source)

    def test_configure_does_not_resize_the_node(self):
        start = self.web_source.index(
            "nodeType.prototype.onConfigure = function onConfigureH3MediaLoader(info)"
        )
        configure_source = self.web_source[start:]

        self.assertNotIn("setSize", configure_source)

    def test_python_and_frontend_registrations_exist(self):
        init_source = (PROJECT_ROOT / "__init__.py").read_text(encoding="utf-8")

        self.assertIn('"MiniMaxH3EasyMediaLoader": MiniMaxH3EasyMediaLoader', init_source)
        self.assertIn('name: "MiniMaxH3Easy.MediaLoader"', self.web_source)


if __name__ == "__main__":
    unittest.main()
