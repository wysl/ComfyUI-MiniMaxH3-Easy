from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


PROJECT_ROOT = Path(__file__).parents[1]


def load_lightroom_symbols(legacy_video_api=False):
    source_path = PROJECT_ROOT / "nodes.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "_lightroom_parameter_values",
        "_lightroom_has_adjustments",
        "_lightroom_rgb_to_hsl",
        "_lightroom_hsl_to_rgb",
        "_lightroom_adjust_rgb",
        "_apply_lightroom_to_image",
        "_lightroom_preview_data",
        "_lightroom_video_from_components",
        "_lightroom_input_types",
        "_lightroom_ui_result",
        "MiniMaxH3EasyLightroomImage",
        "MiniMaxH3EasyLightroomVideo",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))

    @dataclass
    class VideoComponents:
        images: torch.Tensor
        audio: object = None
        frame_rate: object = 24.0
        metadata: object = None
        alpha: object = None

    if legacy_video_api:
        class VideoFromComponents:
            def __init__(self, components, bit_depth=8):
                self.components = components
                self.bit_depth = bit_depth
                self.color_space = None
    else:
        class VideoFromComponents:
            def __init__(self, components, bit_depth=8, color_space="sRGB"):
                self.components = components
                self.bit_depth = bit_depth
                self.color_space = color_space

        def get_components(self):
            return self.components

    class FakeVideo:
        def __init__(self, components, bit_depth=10, color_space="Rec.709"):
            self.components = components
            self.bit_depth = bit_depth
            self.color_space = color_space

        def get_components(self):
            return self.components

        def get_bit_depth(self):
            return self.bit_depth

        def get_color_space(self):
            return self.color_space

    namespace = {
        "Any": object,
        "Mapping": dict,
        "math": math,
        "torch": torch,
        "functional": torch.nn.functional,
        "base64": __import__("base64"),
        "io": __import__("io"),
        "LIGHTROOM_HSL_ZONES": (
            ("red", 0.0), ("orange", 30.0), ("yellow", 60.0), ("green", 120.0),
            ("aqua", 180.0), ("blue", 210.0), ("purple", 270.0), ("magenta", 330.0),
        ),
        "LIGHTROOM_PARAMETER_NAMES": (
            "temperature", "tint", "exposure", "contrast", "highlights", "shadows",
            "whites", "blacks", "texture", "clarity", "dehaze", "vibrance", "saturation",
            *(f"{zone}_{control}" for zone, _ in (
                ("red", 0.0), ("orange", 30.0), ("yellow", 60.0), ("green", 120.0),
                ("aqua", 180.0), ("blue", 210.0), ("purple", 270.0), ("magenta", 330.0),
            ) for control in ("hue", "saturation", "lightness")),
        ),
        "InputImpl": SimpleNamespace(VideoFromComponents=VideoFromComponents),
        "Types": SimpleNamespace(VideoComponents=VideoComponents),
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    namespace["FakeVideo"] = FakeVideo
    namespace["VideoComponents"] = VideoComponents
    return namespace


class LightroomNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.symbols = load_lightroom_symbols()

    def test_all_controls_default_to_zero(self):
        for node_name in ("MiniMaxH3EasyLightroomImage", "MiniMaxH3EasyLightroomVideo"):
            controls = self.symbols[node_name].INPUT_TYPES()["required"]
            self.assertGreaterEqual(len(controls), 37)
            for name, spec in controls.items():
                if name in {"image", "video"}:
                    continue
                self.assertEqual(spec[1]["default"], 0.0, name)

    def test_zero_controls_return_the_original_image_object(self):
        image = torch.rand(2, 8, 10, 4)
        result = self.symbols["_apply_lightroom_to_image"](image, {})
        self.assertIs(result, image)

    def test_exposure_changes_rgb_and_preserves_alpha(self):
        image = torch.full((1, 4, 4, 4), 0.25)
        image[..., 3] = 0.37
        result = self.symbols["_apply_lightroom_to_image"](image, {"exposure": 1.0})
        self.assertTrue(torch.all(result[..., :3] > image[..., :3]))
        self.assertTrue(torch.equal(result[..., 3], image[..., 3]))

    def test_hsl_red_adjustment_does_not_change_green_pixel(self):
        image = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])
        result = self.symbols["_apply_lightroom_to_image"](
            image, {"red_hue": 50.0, "red_saturation": -100.0}
        )
        self.assertFalse(torch.allclose(result[..., 0, :], image[..., 0, :]))
        self.assertTrue(torch.allclose(result[..., 1, :], image[..., 1, :], atol=1e-5))

    def test_video_adjustment_preserves_video_components(self):
        audio = object()
        metadata = {"source": "test"}
        alpha = torch.ones(2, 4, 4, 1)
        source = torch.full((2, 4, 4, 3), 0.25)
        video = self.symbols["FakeVideo"](
            self.symbols["VideoComponents"](source, audio, 23.976, metadata, alpha),
            bit_depth=10,
        )
        result = self.symbols["MiniMaxH3EasyLightroomVideo"].adjust(video, exposure=1.0)
        output = result["result"][0]
        self.assertEqual(output.components.images.shape, source.shape)
        self.assertIs(output.components.audio, audio)
        self.assertEqual(output.components.frame_rate, 23.976)
        self.assertIs(output.components.metadata, metadata)
        self.assertIs(output.components.alpha, alpha)
        self.assertEqual(output.bit_depth, 10)
        self.assertEqual(output.color_space, "Rec.709")

    def test_video_adjustment_falls_back_for_legacy_video_api(self):
        symbols = load_lightroom_symbols(legacy_video_api=True)
        source = torch.full((2, 4, 4, 3), 0.25)
        video = symbols["FakeVideo"](symbols["VideoComponents"](source), bit_depth=10)
        result = symbols["MiniMaxH3EasyLightroomVideo"].adjust(video, exposure=1.0)
        self.assertEqual(result["result"][0].components.images.shape, source.shape)
        self.assertEqual(result["result"][0].bit_depth, 10)

    def test_frontend_preview_and_registrations_exist(self):
        init_source = (PROJECT_ROOT / "__init__.py").read_text(encoding="utf-8")
        web_source = (PROJECT_ROOT / "web" / "lightroom_adjustment.js").read_text(encoding="utf-8")
        self.assertIn('"MiniMaxH3EasyLightroomImage": MiniMaxH3EasyLightroomImage', init_source)
        self.assertIn('"MiniMaxH3EasyLightroomVideo": MiniMaxH3EasyLightroomVideo', init_source)
        self.assertIn("h3_lightroom_preview", web_source)
        self.assertIn("renderPreview", web_source)


if __name__ == "__main__":
    unittest.main()
