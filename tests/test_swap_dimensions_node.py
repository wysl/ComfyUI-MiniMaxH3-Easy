from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


PROJECT_ROOT = Path(__file__).parents[1]


def load_swap_node():
    source_path = PROJECT_ROOT / "nodes.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3EasySwapDimensions"
    ]
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    namespace = {
        "nodes": SimpleNamespace(MAX_RESOLUTION=8192),
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["MiniMaxH3EasySwapDimensions"]


class SwapDimensionsNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = load_swap_node()

    def test_node_exposes_width_height_and_named_switch(self):
        required = self.node.INPUT_TYPES()["required"]
        self.assertEqual(required["width"][0], "INT")
        self.assertEqual(required["height"][0], "INT")
        self.assertFalse(required["swap_dimensions"][1]["default"])
        self.assertEqual(required["swap_dimensions"][1]["label_on"], "交换长宽")

    def test_passes_dimensions_without_swap(self):
        self.assertEqual(self.node.swap(1920, 1080, False), (1920, 1080))

    def test_swaps_dimensions_when_enabled(self):
        self.assertEqual(self.node.swap(1920, 1080, True), (1080, 1920))


if __name__ == "__main__":
    unittest.main()
