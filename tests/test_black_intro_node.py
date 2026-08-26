from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest


PROJECT_ROOT = Path(__file__).parents[1]


class FakeFrames:
    def __init__(self, values, marker="source"):
        self.values = list(values)
        self.marker = marker
        self.shape = (len(self.values), 4, 6, 3)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeFrames(self.values[item], self.marker)
        return self.values[item]


class FakeTorch:
    @staticmethod
    def zeros_like(frames):
        return FakeFrames([0] * len(frames.values), marker="black")

    @staticmethod
    def cat(values, dim=0):
        assert dim == 0
        merged = []
        for value in values:
            merged.extend(value.values)
        return FakeFrames(merged, marker="composite")


@dataclass
class VideoComponents:
    images: FakeFrames
    frame_rate: Fraction
    audio: object = None
    metadata: object = None
    alpha: object = None


class FakeVideo:
    def __init__(self, components, bit_depth=10, color_space="sRGB"):
        self.components = components
        self.bit_depth = bit_depth
        self.color_space = color_space

    def get_components(self):
        return self.components

    def get_bit_depth(self):
        return self.bit_depth

    def get_color_space(self):
        return self.color_space


def load_black_intro_node(legacy_video_api=False):
    source_path = PROJECT_ROOT / "nodes.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3EasyBlackIntro"
    ]
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))

    if legacy_video_api:
        class VideoFromComponents:
            def __init__(self, components, bit_depth=8):
                self.components = components
                self.bit_depth = bit_depth
    else:
        class VideoFromComponents:
            def __init__(self, components, bit_depth=8, color_space="sRGB"):
                self.components = components
                self.bit_depth = bit_depth
                self.color_space = color_space

    namespace = {
        "math": __import__("math"),
        "torch": FakeTorch,
        "InputImpl": SimpleNamespace(VideoFromComponents=VideoFromComponents),
        "Types": SimpleNamespace(VideoComponents=VideoComponents),
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["MiniMaxH3EasyBlackIntro"]


class BlackIntroNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = load_black_intro_node()

    def test_node_exposes_video_and_seconds_input(self):
        required = self.node.INPUT_TYPES()["required"]
        self.assertEqual(required["video"], ("VIDEO",))
        self.assertEqual(required["seconds"][1]["default"], 0.05)

    def test_replaces_first_seconds_with_black_and_preserves_video_properties(self):
        audio = object()
        metadata = {"source": "test"}
        video = FakeVideo(
            VideoComponents(FakeFrames(range(10)), Fraction(24), audio, metadata),
            bit_depth=10,
        )

        output, = self.node.blacken(video, seconds=0.05)

        self.assertEqual(output.components.images.values[:2], [0, 0])
        self.assertEqual(output.components.images.values[2:], list(range(2, 10)))
        self.assertEqual(output.components.frame_rate, Fraction(24))
        self.assertIs(output.components.audio, audio)
        self.assertIs(output.components.metadata, metadata)
        self.assertEqual(output.bit_depth, 10)
        self.assertEqual(output.color_space, "sRGB")

    def test_zero_duration_returns_original_video(self):
        video = FakeVideo(VideoComponents(FakeFrames(range(3)), Fraction(24)))
        output, = self.node.blacken(video, seconds=0)
        self.assertIs(output, video)

    def test_falls_back_for_older_video_api_without_color_space(self):
        node = load_black_intro_node(legacy_video_api=True)
        video = FakeVideo(VideoComponents(FakeFrames(range(4)), Fraction(24)))
        output, = node.blacken(video, seconds=0.01)
        self.assertEqual(output.components.images.values[0], 0)
        self.assertEqual(output.bit_depth, 10)


if __name__ == "__main__":
    unittest.main()
