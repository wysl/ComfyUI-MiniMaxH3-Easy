from __future__ import annotations

import ast
import math
from pathlib import Path
import unittest

import torch
import torch.nn.functional as functional


PROJECT_ROOT = Path(__file__).parents[1]


def _defined_names(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return (node.name,)
    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
    return tuple(target.id for target in targets if isinstance(target, ast.Name))


def load_seam_stabilizer_symbols():
    source_path = PROJECT_ROOT / "nodes.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "H3_CHROMA_CONTEXT_FRAME_OPTIONS",
        "H3_LUMA_WEIGHTS",
        "H3_SEAM_ANALYSIS_SIZE",
        "H3_SEAM_MOTION_HISTORY",
        "H3_SEAM_COLOR_GRID",
        "_h3_luminance",
        "_h3_phase_peak_offset",
        "_estimate_h3_frame_translation",
        "_translate_h3_frames",
        "_h3_seam_decay_weights",
        "_stabilize_h3_seam",
        "MiniMaxH3EasySeamStabilizer",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.ClassDef))
        and any(name in names for name in _defined_names(node))
    ]
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    namespace = {
        "functional": functional,
        "math": math,
        "torch": torch,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


def translated(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    return torch.roll(image, shifts=(dy, dx), dims=(0, 1))


class SeamStabilizerNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.symbols = load_seam_stabilizer_symbols()

    def test_translation_estimator_detects_known_shift(self):
        estimate = self.symbols["_estimate_h3_frame_translation"]
        generator = torch.Generator().manual_seed(730002)
        source = torch.rand((72, 96, 3), generator=generator)
        target = translated(source, dx=3, dy=-2)

        shift_x, shift_y = estimate(source, target)

        self.assertAlmostEqual(shift_x, 3.0, delta=0.2)
        self.assertAlmostEqual(shift_y, -2.0, delta=0.2)

    def test_stabilizer_reduces_boundary_motion_jump(self):
        stabilize = self.symbols["_stabilize_h3_seam"]
        estimate = self.symbols["_estimate_h3_frame_translation"]
        generator = torch.Generator().manual_seed(730003)
        source = torch.rand((72, 96, 3), generator=generator)
        context = [translated(source, dx=index, dy=0) for index in range(22)]
        delivered = [translated(source, dx=26 + index, dy=0) for index in range(8)]
        images = torch.stack(context + delivered)

        stabilized = stabilize(
            images,
            context_frames=22,
            correction_frames=6,
            color_strength=0.0,
            max_position_shift=8.0,
        )
        before_shift, _ = estimate(images[21], images[22])
        after_shift, _ = estimate(stabilized[21], stabilized[22])

        self.assertTrue(torch.equal(stabilized[:22], images[:22]))
        self.assertGreater(before_shift, 4.5)
        self.assertLess(abs(after_shift - 1.0), abs(before_shift - 1.0))
        self.assertAlmostEqual(after_shift, 3.0, delta=0.3)
        self.assertTrue(torch.equal(stabilized[27], images[27]))

    def test_color_correction_reduces_drift_and_fades_to_zero(self):
        stabilize = self.symbols["_stabilize_h3_seam"]
        generator = torch.Generator().manual_seed(730004)
        source = torch.rand((64, 80, 3), generator=generator) * 0.65 + 0.15
        context = [source.clone() for _ in range(22)]
        drift = source.new_tensor((0.08, 0.04, -0.03))
        delivered = [(source + drift).clamp(0.0, 1.0) for _ in range(6)]
        images = torch.stack(context + delivered)

        stabilized = stabilize(
            images,
            context_frames=22,
            correction_frames=4,
            position_strength=0.0,
            color_strength=1.0,
            max_color_shift=0.10,
        )
        original_error = (images[22] - images[21]).abs().mean()
        corrected_error = (stabilized[22] - stabilized[21]).abs().mean()

        self.assertLess(corrected_error.item(), original_error.item() * 0.2)
        self.assertTrue(torch.equal(stabilized[:22], images[:22]))
        self.assertTrue(torch.equal(stabilized[25:], images[25:]))

    def test_node_defaults_match_long_video_workflow(self):
        config = self.symbols["MiniMaxH3EasySeamStabilizer"].INPUT_TYPES()
        required = config["required"]

        self.assertTrue(required["context_frames"][1]["forceInput"])
        self.assertEqual(required["correction_frames"][1]["default"], 12)
        self.assertEqual(required["position_strength"][1]["default"], 0.5)
        self.assertEqual(required["color_strength"][1]["default"], 0.75)

    def test_rejects_input_without_generated_frames(self):
        stabilize = self.symbols["_stabilize_h3_seam"]

        with self.assertRaisesRegex(ValueError, "leave at least one generated frame"):
            stabilize(torch.zeros((22, 8, 8, 3)), context_frames=22)

    def test_zero_context_frames_passes_first_segment_through(self):
        stabilize = self.symbols["_stabilize_h3_seam"]
        images = torch.rand((24, 8, 8, 3))

        stabilized = stabilize(images, context_frames=0)

        self.assertIs(stabilized, images)


if __name__ == "__main__":
    unittest.main()
