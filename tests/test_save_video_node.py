from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


PROJECT_ROOT = Path(__file__).parents[1]


def load_save_video_symbols():
    source_path = PROJECT_ROOT / "nodes.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "_per_second_frame_indices",
        "_extract_video_output_frames",
        "MiniMaxH3EasySaveVideo",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    class FakeVideoContainer:
        def __new__(cls, value):
            return str(value)

        @classmethod
        def get_extension(cls, _value):
            return "mp4"

    namespace = {
        "args": SimpleNamespace(disable_metadata=False),
        "folder_paths": SimpleNamespace(),
        "math": __import__("math"),
        "os": os,
        "Types": SimpleNamespace(VideoContainer=FakeVideoContainer),
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


class FakeFrames:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values), 1, 1, 3)

    def __getitem__(self, item):
        if isinstance(item, list):
            return [self.values[index] for index in item]
        if isinstance(item, slice):
            return self.values[item]
        return self.values[item]


class FakeVideo:
    def __init__(self, frames):
        self.frames = frames
        self.saved = None

    def get_components(self):
        return SimpleNamespace(images=self.frames)

    def get_dimensions(self):
        return 1920, 1080

    def save_to(self, path, **kwargs):
        self.saved = (path, kwargs)


class SaveVideoNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.symbols = load_save_video_symbols()

    def test_five_seconds_samples_the_first_frame_of_each_second(self):
        indexes = self.symbols["_per_second_frame_indices"](5.0, 24.0, 121)

        self.assertEqual(indexes, [0, 24, 48, 72, 96])

    def test_partial_final_second_is_included(self):
        indexes = self.symbols["_per_second_frame_indices"](5.5, 24.0, 121)

        self.assertEqual(indexes, [0, 24, 48, 72, 96, 120])

    def test_extracts_sample_batch_and_actual_last_frame(self):
        frames = FakeFrames(range(121))

        sampled, last = self.symbols["_extract_video_output_frames"](
            frames, seconds=5.0, fps=24.0
        )

        self.assertEqual(sampled, [0, 24, 48, 72, 96])
        self.assertEqual(last, [120])

    def test_uses_the_actual_video_frame_count_as_an_automatic_bound(self):
        frames = FakeFrames(range(121))

        sampled, last = self.symbols["_extract_video_output_frames"](
            frames, seconds=10.0, fps=24.0
        )

        self.assertEqual(sampled, [0, 24, 48, 72, 96, 120])
        self.assertEqual(last, [120])

    def test_node_exposes_video_sample_batch_and_last_frame_outputs(self):
        node = self.symbols["MiniMaxH3EasySaveVideo"]

        self.assertEqual(node.RETURN_TYPES, ("VIDEO", "IMAGE", "IMAGE"))
        self.assertEqual(
            node.RETURN_NAMES,
            ("video", "frames_per_second", "last_frame"),
        )
        self.assertTrue(node.OUTPUT_NODE)
        required = node.INPUT_TYPES()["required"]
        self.assertIn("video", required)
        self.assertIn("seconds", required)
        self.assertIn("fps", required)
        self.assertNotIn("frame_count", required)
        self.assertTrue(required["fps"][1]["forceInput"])

    def test_save_uses_comfy_video_api_and_returns_all_three_outputs(self):
        frames = FakeFrames(range(121))
        video = FakeVideo(frames)
        node = self.symbols["MiniMaxH3EasySaveVideo"]
        node.save.__globals__["folder_paths"] = SimpleNamespace(
            get_output_directory=lambda: "output-root",
            get_save_image_path=lambda *_args: ("output-root/video", "clip", 7, "video", "ignored"),
        )

        output = node.save(
            video,
            seconds=5.0,
            fps=24.0,
            filename_prefix="video/clip",
            format="mp4",
            codec="h264",
            prompt={"1": {"class_type": "Example"}},
            extra_pnginfo={"workflow": {"nodes": []}},
        )

        self.assertEqual(video.saved[0], os.path.join("output-root/video", "clip_00007_.mp4"))
        self.assertEqual(video.saved[1]["format"], "mp4")
        self.assertEqual(video.saved[1]["codec"], "h264")
        self.assertIn("prompt", video.saved[1]["metadata"])
        self.assertEqual(output["result"], (video, [0, 24, 48, 72, 96], [120]))
        self.assertEqual(output["ui"]["h3_saved_video"][0]["filename"], "clip_00007_.mp4")

    def test_python_and_frontend_registrations_exist(self):
        init_source = (PROJECT_ROOT / "__init__.py").read_text(encoding="utf-8")
        web_source = (PROJECT_ROOT / "web" / "minimax_h3_easy_ui.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('"MiniMaxH3EasySaveVideo": MiniMaxH3EasySaveVideo', init_source)
        self.assertIn('const SAVE_CLASS = "MiniMaxH3EasySaveVideo";', web_source)
        self.assertIn("installSaveVideoNode(nodeType, nodeData);", web_source)


if __name__ == "__main__":
    unittest.main()
