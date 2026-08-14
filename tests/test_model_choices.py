from __future__ import annotations

import ast
from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).parents[1]


def load_h3_name_matcher():
    source_path = PROJECT_ROOT / "nodes.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {"_normalise_model_name", "_is_h3_transformer_name"}
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    namespace = {"re": re}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_is_h3_transformer_name"]


class H3ModelChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.is_h3_transformer_name = staticmethod(load_h3_name_matcher())

    def test_any_h3_named_transformer_is_eligible(self):
        matcher = self.is_h3_transformer_name

        self.assertTrue(matcher("custom/H3_model.safetensors"))
        self.assertTrue(matcher("MiniMax-H3-quantized.gguf"))

    def test_non_h3_transformer_is_not_eligible(self):
        self.assertFalse(self.is_h3_transformer_name("other_video_model.safetensors"))


if __name__ == "__main__":
    unittest.main()
