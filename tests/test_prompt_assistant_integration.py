from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest
from typing import Any


PROJECT_ROOT = Path(__file__).parents[1]


def load_integration_classes():
    source_path = PROJECT_ROOT / "nodes.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {"MiniMaxH3Context", "MiniMaxH3EasyOutput"}
    definitions = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ClassDef)
            and node.name in names
        ) or (
            isinstance(node, ast.FunctionDef)
            and node.name == "_align_canvas_dimension"
        )
    ]
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    namespace = {
        "Any": Any,
        "MODE_IMAGE": "image",
        "dataclass": dataclass,
        "_video_parts": lambda _value: (None, None, 24.0),
        "h3": SimpleNamespace(CANVAS_MULTIPLE=32),
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["MiniMaxH3Context"], namespace["MiniMaxH3EasyOutput"]


class PromptAssistantIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context_class, cls.output_class = load_integration_classes()

    def make_context(self, encoder=None):
        image = object()
        video = object()
        audio = object()
        context = self.context_class(
            conditioning="original conditioning",
            latent="original latent",
            video_vae="video vae",
            audio_vae="audio vae",
            fps=24.0,
            seconds=8.0,
            width=1344,
            height=768,
            prompt="original prompt",
            mode="reference",
            media=(("image", image), ("video", video), ("audio", audio)),
            _prompt_encoder=encoder,
        )
        return context, image, video, audio

    def test_context_exposes_prompt_assistant_payload(self):
        context, image, video, audio = self.make_context()

        payload = context.prompt_assistant_payload()

        self.assertEqual(payload["prompt"], "original prompt")
        self.assertEqual(payload["mode"], "reference")
        self.assertEqual(payload["duration_seconds"], 8.0)
        self.assertEqual(payload["images"], [image])
        self.assertEqual(payload["videos"], [video])
        self.assertEqual(payload["audios"], [audio])
        self.assertEqual(payload["synchronized_audio_count"], 0)
        self.assertEqual(payload["synchronized_audio_video_indices"], [])

    def test_context_places_video_soundtracks_before_standalone_audio(self):
        soundtrack = object()
        context, _image, _video, standalone_audio = self.make_context()
        method_globals = self.context_class.prompt_assistant_payload.__globals__
        original_video_parts = method_globals["_video_parts"]
        method_globals["_video_parts"] = lambda _value: (None, soundtrack, 24.0)
        try:
            payload = context.prompt_assistant_payload()
        finally:
            method_globals["_video_parts"] = original_video_parts

        self.assertEqual(payload["audios"], [soundtrack, standalone_audio])
        self.assertEqual(payload["synchronized_audio_count"], 1)
        self.assertEqual(payload["synchronized_audio_video_indices"], [1])

    def test_output_reencodes_connected_optimized_prompt(self):
        received = []

        def encoder(prompt):
            received.append(prompt)
            return "optimized conditioning", "optimized latent"

        context, _image, _video, _audio = self.make_context(encoder)

        result = self.output_class.unpack(context, "  optimized prompt  ", "1.5")

        self.assertEqual(received, ["optimized prompt"])
        self.assertEqual(result[0], "optimized conditioning")
        self.assertEqual(result[1], "optimized latent")
        self.assertEqual(
            result[2:],
            ("video vae", "audio vae", 24.0, 1344, 768, 2016, 1152),
        )

    def test_output_keeps_original_conditioning_without_override(self):
        context, _image, _video, _audio = self.make_context()

        result = self.output_class.unpack(context)

        self.assertEqual(result[0], "original conditioning")
        self.assertEqual(result[1], "original latent")

    def test_output_exposes_aligned_dimensions_for_each_multiplier(self):
        context, _image, _video, _audio = self.make_context()
        expected = {
            "1.2": (1600, 928),
            "1.4": (1888, 1088),
            "1.5": (2016, 1152),
            "1.6": (2144, 1216),
            "2.0": (2688, 1536),
        }

        for multiplier, scaled_size in expected.items():
            with self.subTest(multiplier=multiplier):
                result = self.output_class.unpack(context, size_multiplier=multiplier)
                self.assertEqual(result[5:7], (1344, 768))
                self.assertEqual(result[7:9], scaled_size)

    def test_output_keeps_existing_output_order_and_multiplier_choices(self):
        self.assertEqual(
            self.output_class.RETURN_NAMES[:5],
            ("positive", "latent", "video_vae", "audio_vae", "fps"),
        )
        multiplier_options = self.output_class.INPUT_TYPES()["required"]["size_multiplier"][0]
        self.assertEqual(multiplier_options, ["1.2", "1.4", "1.5", "1.6", "2.0"])


if __name__ == "__main__":
    unittest.main()
