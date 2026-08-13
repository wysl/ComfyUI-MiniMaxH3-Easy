from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest


PROJECT_ROOT = Path(__file__).parents[1]


def load_frame_interpolation_symbols():
    source_path = PROJECT_ROOT / "nodes.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {"_rife_vfi_node_class", "MiniMaxH3EasyFrameInterpolation"}
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))

    @dataclass
    class VideoComponents:
        images: object
        frame_rate: Fraction
        audio: object = None
        metadata: object = None
        alpha: object = None

    class VideoFromComponents:
        def __init__(self, components, bit_depth=8):
            self.components = components
            self.bit_depth = bit_depth

    namespace = {
        "Fraction": Fraction,
        "InputImpl": SimpleNamespace(VideoFromComponents=VideoFromComponents),
        "Types": SimpleNamespace(VideoComponents=VideoComponents),
        "math": __import__("math"),
        "nodes": SimpleNamespace(NODE_CLASS_MAPPINGS={}),
        "torch": SimpleNamespace(cat=lambda values, dim: FakeFrames(
            sum(value.shape[0] for value in values), marker="interpolated"
        )),
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


class FakeFrames:
    def __init__(self, count, marker="input"):
        self.shape = (count, 4, 6, 3)
        self.marker = marker

    @property
    def device(self):
        return "cpu"

    @property
    def dtype(self):
        return "float32"

    def __getitem__(self, _item):
        return FakeFrames(1, marker=self.marker)

    def to(self, **_kwargs):
        return self


class FakeVideo:
    def __init__(self, components, bit_depth=10):
        self.components = components
        self.bit_depth = bit_depth

    def get_components(self):
        return self.components

    def get_bit_depth(self):
        return self.bit_depth


class FakeRifeNode:
    calls = []

    def vfi(self, **kwargs):
        self.calls.append(kwargs)
        return (FakeFrames(9, marker="interpolated"),)


class FrameInterpolationNodeTests(unittest.TestCase):
    def setUp(self):
        self.symbols = load_frame_interpolation_symbols()
        self.symbols["nodes"].NODE_CLASS_MAPPINGS["RIFE_VFI_Opt"] = FakeRifeNode
        FakeRifeNode.calls.clear()

    def test_node_is_video_to_video_with_fps_output(self):
        node = self.symbols["MiniMaxH3EasyFrameInterpolation"]

        self.assertEqual(node.INPUT_TYPES()["required"], {"video": ("VIDEO",)})
        self.assertEqual(node.RETURN_TYPES, ("VIDEO", "FLOAT"))
        self.assertEqual(node.RETURN_NAMES, ("video", "fps"))

    def test_doubles_fps_and_preserves_audio_metadata_and_bit_depth(self):
        components_type = self.symbols["Types"].VideoComponents
        audio = object()
        metadata = {"source": "test"}
        input_frames = FakeFrames(5)
        video = FakeVideo(
            components_type(input_frames, Fraction(24), audio, metadata),
            bit_depth=10,
        )

        output_video, output_fps = self.symbols[
            "MiniMaxH3EasyFrameInterpolation"
        ].interpolate(video)

        self.assertEqual(output_fps, 48.0)
        self.assertEqual(output_video.components.images.shape[0], 10)
        self.assertIs(output_video.components.audio, audio)
        self.assertIs(output_video.components.metadata, metadata)
        self.assertEqual(output_video.components.frame_rate, Fraction(48))
        self.assertEqual(output_video.bit_depth, 10)

    def test_uses_fixed_fast_quality_balanced_rife_settings(self):
        components_type = self.symbols["Types"].VideoComponents
        video = FakeVideo(components_type(FakeFrames(5), Fraction(24)))

        self.symbols["MiniMaxH3EasyFrameInterpolation"].interpolate(video)

        self.assertEqual(
            FakeRifeNode.calls,
            [{
                "ckpt_name": "rife47.pth",
                "frames": video.components.images,
                "multiplier": 2,
                "scale_factor": 1.0,
                "ensemble": False,
                "clear_cache_after_n_frames": 10,
            }],
        )

    def test_missing_whiterabbit_has_an_actionable_error(self):
        self.symbols["nodes"].NODE_CLASS_MAPPINGS.clear()

        with self.assertRaisesRegex(RuntimeError, "WhiteRabbit"):
            self.symbols["_rife_vfi_node_class"]()


if __name__ == "__main__":
    unittest.main()
