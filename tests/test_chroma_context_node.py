from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import random
import unittest

import torch
import torch.nn.functional as functional


PROJECT_ROOT = Path(__file__).parents[1]


def load_chroma_context_symbols():
    source_path = PROJECT_ROOT / "nodes.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "H3_CHROMA_CONTEXT_FRAME_OPTIONS",
        "H3_CHROMA_PALETTE",
        "H3_CHROMA_GRID",
        "H3_LUMA_WEIGHTS",
        "_h3_chroma_taper_alphas",
        "_h3_luminance",
        "_preserve_h3_chroma_luminance",
        "_prepare_h3_chroma_context",
        "MiniMaxH3EasyChromaContext",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.ClassDef))
        and any(name in names for name in _defined_names(node))
    ]
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))

    @dataclass
    class VideoComponents:
        images: torch.Tensor
        frame_rate: Fraction
        audio: object = None
        metadata: object = None

    class VideoFromComponents:
        def __init__(self, components, bit_depth=8):
            self.components = components
            self.bit_depth = bit_depth

    namespace = {
        "Fraction": Fraction,
        "InputImpl": SimpleNamespace(VideoFromComponents=VideoFromComponents),
        "Types": SimpleNamespace(VideoComponents=VideoComponents),
        "functional": functional,
        "random": random,
        "torch": torch,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


def _defined_names(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return (node.name,)
    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
    return tuple(target.id for target in targets if isinstance(target, ast.Name))


class FakeVideo:
    def __init__(self, components, bit_depth=10):
        self.components = components
        self.bit_depth = bit_depth

    def get_components(self):
        return self.components

    def get_bit_depth(self):
        return self.bit_depth


class ChromaContextNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.symbols = load_chroma_context_symbols()

    def test_taper_schedule_matches_expected_tail(self):
        alphas = self.symbols["_h3_chroma_taper_alphas"](22, 0.45, 0.10, 3)

        self.assertEqual(len(alphas), 22)
        self.assertTrue(all(alpha == 0.45 for alpha in alphas[:19]))
        for actual, expected in zip(alphas[19:], (1 / 3, 13 / 60, 0.10)):
            self.assertAlmostEqual(actual, expected)

    def test_node_outputs_deterministic_visual_only_contexts(self):
        components_type = self.symbols["Types"].VideoComponents
        images = torch.linspace(0.0, 1.0, 30 * 8 * 6 * 3).reshape(30, 8, 6, 3)
        source = FakeVideo(
            components_type(
                images=images,
                frame_rate=Fraction(24000, 1001),
                audio=object(),
                metadata={"source": "test"},
            )
        )

        node = self.symbols["MiniMaxH3EasyChromaContext"]
        clean, noisy = node.create_context(source, seed=730002)
        clean_repeat, noisy_repeat = node.create_context(source, seed=730002)

        self.assertTrue(torch.equal(clean.components.images, images[-22:]))
        self.assertTrue(torch.equal(clean.components.images, clean_repeat.components.images))
        self.assertTrue(torch.equal(noisy.components.images, noisy_repeat.components.images))
        self.assertFalse(torch.equal(noisy.components.images, clean.components.images))
        self.assertEqual(clean.components.frame_rate, Fraction(24000, 1001))
        self.assertIsNone(clean.components.audio)
        self.assertIsNone(noisy.components.audio)
        self.assertEqual(noisy.bit_depth, 10)

    def test_luminance_lock_preserves_brightness_and_chroma_noise(self):
        prepare = self.symbols["_prepare_h3_chroma_context"]
        luminance = self.symbols["_h3_luminance"]
        images = torch.linspace(0.05, 0.95, 30 * 8 * 6 * 3).reshape(30, 8, 6, 3)

        clean, unlocked = prepare(
            images,
            context_frames=22,
            seed=730002,
            preserve_luminance=False,
        )
        _, locked = prepare(
            images,
            context_frames=22,
            seed=730002,
            preserve_luminance=True,
        )

        unlocked_error = (luminance(unlocked) - luminance(clean)).abs().mean()
        locked_error = (luminance(locked) - luminance(clean)).abs().mean()
        self.assertGreater(unlocked_error.item(), 0.01)
        self.assertLess(locked_error.item(), 1e-6)
        self.assertFalse(torch.equal(locked, clean))
        self.assertGreater((locked - clean).abs().mean().item(), 0.001)

    def test_luminance_lock_is_enabled_by_default(self):
        config = self.symbols["MiniMaxH3EasyChromaContext"].INPUT_TYPES()

        self.assertTrue(config["required"]["preserve_luminance"][1]["default"])

    def test_rejects_more_context_frames_than_video_contains(self):
        prepare = self.symbols["_prepare_h3_chroma_context"]

        with self.assertRaisesRegex(ValueError, "has 5 frames"):
            prepare(torch.zeros(5, 8, 6, 3), context_frames=22)


if __name__ == "__main__":
    unittest.main()
