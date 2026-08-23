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
        "_concatenate_h3_media_images",
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
        "torch": __import__("torch"),
        "folder_paths": None,
        "H3_MEDIA_RESIZE_METHOD": "lanczos",
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

    def test_long_edge_resize_preserves_aspect_ratio(self):
        target_size = self.symbols["_h3_media_target_size"]

        self.assertEqual(target_size(1920, 1080, 1.0, "长边", 1024), (1024, 576))
        self.assertEqual(target_size(800, 1200, 1.0, "long_edge", 600), (400, 600))

    def test_short_edge_resize_preserves_aspect_ratio(self):
        target_size = self.symbols["_h3_media_target_size"]

        self.assertEqual(target_size(1920, 1080, 1.0, "短边", 720), (1280, 720))
        self.assertEqual(target_size(800, 1200, 1.0, "short edge", 400), (400, 600))

    def test_no_resize_mode_keeps_original_size(self):
        target_size = self.symbols["_h3_media_target_size"]

        self.assertEqual(target_size(1920, 1080, 4.0, "不缩放", 256), (1920, 1080))

    def test_custom_divisible_factor_aligns_both_dimensions(self):
        target_size = self.symbols["_h3_media_target_size"]

        self.assertEqual(target_size(1920, 1080, 1.0, "倍率", 1024, 32), (1920, 1056))
        self.assertEqual(target_size(800, 1200, 1.0, "长边", 1000, 64), (640, 960))

    def test_no_resize_mode_ignores_divisible_factor(self):
        target_size = self.symbols["_h3_media_target_size"]

        self.assertEqual(target_size(101, 51, 2.0, "不缩放", 1024, 32), (101, 51))

    def test_outputs_keep_media_lists_and_add_concat_image(self):
        node = self.symbols["MiniMaxH3EasyMediaLoader"]

        self.assertEqual(node.RETURN_TYPES, ("IMAGE", "AUDIO", "VIDEO", "IMAGE"))
        self.assertEqual(
            node.RETURN_NAMES,
            ("multi output", "audio output", "video output", "拼接图片"),
        )
        self.assertEqual(node.OUTPUT_IS_LIST, (True, True, True, False))

    def test_image_scaling_controls_are_built_in(self):
        input_types = self.symbols["MiniMaxH3EasyMediaLoader"].INPUT_TYPES()
        required = input_types["required"]
        optional = input_types["optional"]

        self.assertIn("image_resize_mode", required)
        self.assertIn("image_scale", required)
        self.assertIn("scale_method", required)
        self.assertIn("image_edge_length", optional)
        self.assertIn("image_divisible_by", optional)
        self.assertEqual(required["image_scale"][1]["default"], 0.5)
        self.assertEqual(required["image_scale"][1]["step"], 0.01)
        self.assertEqual(required["scale_method"][0], ["lanczos"])
        self.assertEqual(required["scale_method"][1]["default"], "lanczos")
        self.assertEqual(required["image_resize_mode"][0], ["不缩放", "倍率", "长边", "短边"])
        self.assertEqual(optional["image_edge_length"][1]["default"], 1024)
        self.assertEqual(optional["image_divisible_by"][1]["default"], 1)
        self.assertEqual(
            list(required)[:2],
            ["media_manifest", "image_resize_mode"],
        )

    def test_concatenated_image_output_preserves_order_and_pads_width(self):
        concatenate = self.symbols["_concatenate_h3_media_images"]
        torch = self.symbols["torch"]
        first = torch.full((1, 2, 3, 3), 0.25)
        second = torch.full((1, 1, 2, 3), 0.75)

        result = concatenate([first, second])

        self.assertEqual(tuple(result.shape), (1, 3, 3, 3))
        self.assertTrue(torch.allclose(result[:, :2, :, :], first))
        self.assertTrue(torch.allclose(result[:, 2:, :2, :], second))
        self.assertTrue(torch.all(result[:, 2:, 2:, :] == 0))

    def test_frontend_labels_edge_resize_controls(self):
        self.assertIn('widgetByName(node, "image_resize_mode")', self.web_source)
        self.assertIn('widgetByName(node, "image_edge_length")', self.web_source)
        self.assertIn('resizeModeWidget.label = "图片缩放模式（唯一生效规则）"', self.web_source)
        self.assertIn('"自定义缩放倍率（当前生效）"', self.web_source)
        self.assertIn('"目标边长/像素（当前生效）"', self.web_source)
        self.assertIn('"当前：保持原图尺寸（其他缩放参数均不生效）"', self.web_source)
        self.assertIn('widgetByName(node, "image_divisible_by")', self.web_source)
        self.assertIn('"尺寸因数（宽高可整除，1=关闭）"', self.web_source)
        self.assertIn('watchResizeControls(node);', self.web_source)

    def test_frontend_applies_resize_defaults_and_repairs_legacy_order(self):
        self.assertIn("const DEFAULT_SCALE = 0.5;", self.web_source)
        self.assertIn("const DEFAULT_EDGE_LENGTH = 1024;", self.web_source)
        self.assertIn('const FIXED_SCALE_METHOD = "lanczos";', self.web_source)
        self.assertIn("repairLegacyResizeWidgetValues(node);", self.web_source)
        self.assertIn("if (mode === \"scale\") changed = setWidgetValue(scaleWidget, DEFAULT_SCALE)", self.web_source)
        self.assertIn("setWidgetValue(edgeLengthWidget, DEFAULT_EDGE_LENGTH)", self.web_source)
        self.assertIn("setWidgetValue(methodWidget, FIXED_SCALE_METHOD)", self.web_source)

    def test_frontend_supports_multi_select_preview_order_and_resize(self):
        self.assertIn("input.multiple = true;", self.web_source)
        self.assertIn("h3-media-image-grid", self.web_source)
        self.assertIn("number.textContent = String(index + 1);", self.web_source)
        self.assertIn('element.addEventListener("dragstart"', self.web_source)
        self.assertIn("node.resizable = true;", self.web_source)
        self.assertIn("getMinHeight: () => 200", self.web_source)
        self.assertIn("setResizeWidgetVisibility", self.web_source)
        self.assertIn('widget.type = visible ? widget.__h3MediaLoaderOriginalType : "hidden";', self.web_source)

    def test_frontend_uses_separate_upload_tabs_and_server_paths(self):
        self.assertIn('activeKind: "images"', self.web_source)
        self.assertIn('form.append("subfolder", `minimax_h3_easy/media_loader/${kind}`);', self.web_source)
        self.assertIn('api.fetchApi("/upload/image"', self.web_source)
        self.assertNotIn("URL.createObjectURL", self.web_source)

    def test_frontend_supports_input_browser_and_file_drop(self):
        self.assertIn("/minimax_h3_easy/input-media?kind=", self.web_source)
        self.assertIn("h3-media-input-picker", self.web_source)
        self.assertIn("event.dataTransfer.files", self.web_source)
        self.assertIn("addLocalFiles(event.dataTransfer.files", self.web_source)

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
