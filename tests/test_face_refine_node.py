from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as functional


PROJECT_ROOT = Path(__file__).parents[1]


class FakeNode:
    def __init__(self, node_id, class_type, inputs):
        self.node_id = node_id
        self.class_type = class_type
        self.inputs = inputs

    def out(self, index):
        return [self.node_id, index]


class FakeGraphBuilder:
    latest = None

    def __init__(self):
        self.nodes = []
        FakeGraphBuilder.latest = self

    def node(self, class_type, **inputs):
        node = FakeNode(str(len(self.nodes) + 1), class_type, inputs)
        self.nodes.append(node)
        return node

    def finalize(self):
        return {
            node.node_id: {"class_type": node.class_type, "inputs": node.inputs}
            for node in self.nodes
        }


class MiniMaxH3Bundle:
    def __init__(self):
        self.ref2va_model_name = "minimax_h3_ref2va.safetensors"
        self.video_vae = object()
        self.audio_vae = object()
        self.clip = object()
        self.requested_kinds = []

    def model_for(self, kind):
        self.requested_kinds.append(kind)
        return object()


class MiniMaxH3Context:
    def __init__(self):
        self.prompt = "original prompt"
        self.fps = 24.0
        self.media = ()
        self.mode = "image"


class FakeFrames:
    shape = (22, 64, 96, 3)


class FakeVideo:
    def __init__(self, fps=24.0, audio=None):
        self.components = SimpleNamespace(
            images=FakeFrames(),
            audio=object() if audio is None else audio,
            frame_rate=fps,
        )

    def get_components(self):
        return self.components


def load_face_refine_symbols():
    source_path = PROJECT_ROOT / "nodes.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "_face_refine_prompt",
        "_face_refine_seed",
        "_face_refine_step_count",
        "MiniMaxH3EasyFaceRefine",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    def fake_condition_inputs(
        bundle,
        context,
        prompt,
        width,
        height,
        length,
        identity,
        prompt_mode=None,
    ):
        return {
            "clip": bundle.clip,
            "vae": bundle.video_vae,
            "audio_vae": bundle.audio_vae,
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": length,
        }

    namespace = {
        "FACE_REFINE_SINGLE": "single person",
        "FACE_REFINE_TWO": "two people",
        "FACE_REFINE_PROMPT_IDENTITY_ONLY": "identity only (recommended)",
        "FACE_REFINE_PROMPT_FULL_SCENE": "full scene prompt",
        "FACE_REFINE_SEED_IDENTITY_LOCKED": "identity locked (recommended)",
        "FACE_REFINE_SEED_INPUT": "input seed",
        "NONE_MODEL_ALIASES": {"none", "无"},
        "GraphBuilder": FakeGraphBuilder,
        "MiniMaxH3Bundle": MiniMaxH3Bundle,
        "MiniMaxH3Context": MiniMaxH3Context,
        "_face_refine_condition_inputs": fake_condition_inputs,
        "_face_refine_detector_choices": lambda: ["face_yolov8m.pt"],
        "_face_refine_lora_choices": lambda: ["minimax_h3_turbo_8step.safetensors", "none"],
        "_face_refine_sampler_choices": lambda: ["res_multistep", "euler"],
        "_face_refine_scheduler_choices": lambda: ["simple"],
        "_is_none_model": lambda value: str(value).lower() in {"none", "无"},
        "h3": SimpleNamespace(FPS=24),
        "torch": torch,
        "functional": functional,
        "re": re,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


class FaceRefineNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.symbols = load_face_refine_symbols()
        cls.node = cls.symbols["MiniMaxH3EasyFaceRefine"]

    def refine(self, **overrides):
        values = {
            "video": FakeVideo(),
            "h3_bundle": MiniMaxH3Bundle(),
            "h3_context": MiniMaxH3Context(),
            "subject_mode": "single person",
            "detector": "face_yolov8m.pt",
            "turbo_lora": "minimax_h3_turbo_8step.safetensors",
            "lora_strength": 0.75,
            "steps": 0,
            "denoise": 0.32,
            "sampler": "res_multistep",
            "scheduler": "simple",
            "seed": 7,
            "identity_position_1": "自动（最大脸）",
            "identity_position_2": "自动（最大脸）",
        }
        values.update(overrides)
        return self.node.refine(**values), values

    def test_single_person_expands_the_complete_refine_chain(self):
        result, values = self.refine()
        graph = list(result["expand"].values())
        types = [node["class_type"] for node in graph]

        for expected in (
            "MiniMaxH3EasyFaceTrackCrop",
            "MiniMaxH3ReferenceToVideo",
            "MiniMaxH3EasyInjectVideoLatent",
            "MiniMaxH3EasyAudioLock",
            "MiniMaxH3EasyPerFrameDenoise",
            "SamplerCustomAdvanced",
            "VAEDecode",
            "MiniMaxH3EasyFaceStitch",
            "MiniMaxH3EasyReplaceVideoFrames",
        ):
            self.assertIn(expected, types)

        scheduler = next(node for node in graph if node["class_type"] == "BasicScheduler")
        self.assertEqual(scheduler["inputs"]["steps"], 8)
        self.assertEqual(values["h3_bundle"].requested_kinds, ["ref2va"])

    def test_two_people_builds_two_sequential_refine_passes(self):
        image_1, image_2 = object(), object()
        result, _ = self.refine(
            subject_mode="two people",
            identity_reference_1=image_1,
            identity_reference_2=image_2,
        )
        types = [node["class_type"] for node in result["expand"].values()]

        self.assertEqual(types.count("MiniMaxH3EasySelectIdentityFace"), 2)
        self.assertEqual(types.count("MiniMaxH3EasyFaceTrackCrop"), 2)
        self.assertEqual(types.count("MiniMaxH3EasyFaceStitch"), 2)

    def test_manual_steps_override_lora_filename(self):
        result, _ = self.refine(steps=6)
        scheduler = next(
            node for node in result["expand"].values()
            if node["class_type"] == "BasicScheduler"
        )
        self.assertEqual(scheduler["inputs"]["steps"], 6)

    def test_full_model_uses_twenty_steps_in_auto_mode(self):
        result, _ = self.refine(turbo_lora="none")
        scheduler = next(
            node for node in result["expand"].values()
            if node["class_type"] == "BasicScheduler"
        )
        self.assertEqual(scheduler["inputs"]["steps"], 20)

    def test_rejects_post_interpolation_video(self):
        with self.assertRaisesRegex(ValueError, "before frame interpolation"):
            self.refine(video=FakeVideo(fps=48.0))

    def test_identity_prompt_ignores_scene_changes_by_default(self):
        prompt = self.symbols["_face_refine_prompt"](
            "High angle, bright golden scene with a new camera move.",
            True,
        )

        self.assertNotIn("High angle", prompt)
        self.assertIn("Preserve the exact facial identity", prompt)
        self.assertIn("Do not alter", prompt)

    def test_full_scene_prompt_mode_remains_available(self):
        prompt = self.symbols["_face_refine_prompt"](
            "<Picture 8> High angle continuation.",
            True,
            "full scene prompt",
        )

        self.assertIn("High angle continuation.", prompt)
        self.assertNotIn("<Picture 8>", prompt)
        self.assertTrue(prompt.startswith("<Picture 1>"))

    def test_identity_locked_seed_is_stable_across_segment_seeds(self):
        identity = torch.linspace(0.0, 1.0, 32 * 24 * 3).reshape(1, 32, 24, 3)
        seed_for_segment_1 = self.symbols["_face_refine_seed"](
            31001,
            "identity locked (recommended)",
            identity,
        )
        seed_for_segment_3 = self.symbols["_face_refine_seed"](
            31003,
            "identity locked (recommended)",
            identity,
        )

        self.assertEqual(seed_for_segment_1, seed_for_segment_3)

    def test_input_seed_mode_preserves_existing_seed_control(self):
        identity = torch.zeros((1, 16, 16, 3))
        resolved = self.symbols["_face_refine_seed"](
            31003,
            "input seed",
            identity,
            subject_index=1,
        )

        self.assertEqual(resolved, 31004)

    def test_identity_lock_defaults_are_exposed(self):
        required = self.node.INPUT_TYPES()["required"]

        self.assertEqual(
            required["prompt_mode"][1]["default"],
            "identity only (recommended)",
        )
        self.assertEqual(
            required["seed_mode"][1]["default"],
            "identity locked (recommended)",
        )


if __name__ == "__main__":
    unittest.main()
