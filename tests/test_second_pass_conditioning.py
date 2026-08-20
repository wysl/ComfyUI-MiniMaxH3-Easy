from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "nodes.py"


class _NodeHelpers:
    @staticmethod
    def conditioning_set_values(conditioning, values):
        return {"conditioning": conditioning, "values": values}


class _VideoVae:
    def encode(self, image):
        return torch.zeros(
            (1, 16, 1, image.shape[1] // 16, image.shape[2] // 16),
            dtype=image.dtype,
        )


def load_classes():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    names = {
        "_MiniMaxH3KeyframeSource",
        "MiniMaxH3Context",
        "MiniMaxH3EasyAspectRatio",
        "MiniMaxH3EasySecondPassConditioning",
    }
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    namespace = {
        "Any": Any,
        "Mapping": Mapping,
        "MODE_IMAGE": "image",
        "ASPECT_WIDESCREEN": "16:9",
        "ASPECT_SELECTOR_LABELS": {"16:9": "16:9 (Widescreen)"},
        "H3_ANY_TYPE": "*",
        "dataclass": dataclass,
        "torch": torch,
        "node_helpers": _NodeHelpers,
        "h3": SimpleNamespace(
            _resize=lambda image, width, height, _crop: torch.zeros(
                (1, height, width, 3), dtype=image.dtype
            )
        ),
    }
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return (
        namespace["_MiniMaxH3KeyframeSource"],
        namespace["MiniMaxH3Context"],
        namespace["MiniMaxH3EasyAspectRatio"],
        namespace["MiniMaxH3EasySecondPassConditioning"],
    )


class SecondPassConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.keyframe_source,
            cls.context_class,
            cls.aspect_ratio_class,
            cls.second_pass_class,
        ) = load_classes()

    def make_context(self, keyframe_sources=()):
        return self.context_class(
            conditioning="original conditioning",
            latent="original latent",
            video_vae=_VideoVae(),
            audio_vae="audio vae",
            fps=24.0,
            keyframe_sources=keyframe_sources,
        )

    def test_aspect_ratio_outputs_resolution_selector_label(self):
        context = self.make_context()
        self.assertEqual(self.aspect_ratio_class.extract(context), ("16:9 (Widescreen)",))

    def test_second_pass_target_requires_24_channel_video_latent(self):
        latent = {"samples": torch.zeros((1, 24, 5, 60, 80))}
        width, height, shape = self.second_pass_class._target_dimensions(latent)
        self.assertEqual((width, height), (1280, 960))
        self.assertEqual(tuple(shape), (60, 80))

        with self.assertRaisesRegex(ValueError, "24-channel"):
            self.second_pass_class._target_dimensions(
                {"samples": torch.zeros((1, 56, 5, 60, 80))}
            )

    def test_second_pass_preserves_conditioning_without_keyframes(self):
        context = self.make_context()
        result = self.second_pass_class.rebuild(
            context,
            {"samples": torch.zeros((1, 24, 5, 60, 80))},
        )
        self.assertEqual(result[0]["conditioning"], "original conditioning")
        self.assertEqual(result[0]["values"], {})

    def test_second_pass_reencodes_keyframes_at_target_canvas(self):
        source = self.keyframe_source(
            resolved_frame_index=0,
            image=torch.zeros((1, 320, 544, 3)),
        )
        context = self.make_context((source,))
        result = self.second_pass_class.rebuild(
            context,
            {"samples": torch.zeros((1, 24, 5, 60, 80))},
        )
        keyframes = result[0]["values"]["minimax_keyframes"]
        self.assertEqual(keyframes[0]["resolved_frame_index"], 0)
        self.assertEqual(tuple(keyframes[0]["latent"].shape[-2:]), (60, 80))


if __name__ == "__main__":
    unittest.main()
