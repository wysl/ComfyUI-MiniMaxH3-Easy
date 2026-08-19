from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).parents[1]
PYTHON_SOURCE_PATH = PROJECT_ROOT / "nodes.py"
WEB_SOURCE_PATH = PROJECT_ROOT / "web" / "area_switch.js"


def load_area_switch_class():
    tree = ast.parse(PYTHON_SOURCE_PATH.read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MiniMaxH3EasyAreaSwitch"
    )
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    namespace = {"H3_ANY_TYPE": "*"}
    exec(compile(module, str(PYTHON_SOURCE_PATH), "exec"), namespace)
    return namespace["MiniMaxH3EasyAreaSwitch"]


class AreaSwitchNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = load_area_switch_class()
        cls.source = WEB_SOURCE_PATH.read_text(encoding="utf-8")

    def test_exposes_lazy_function_area_inputs(self):
        inputs = self.node.INPUT_TYPES()
        self.assertEqual(inputs["required"]["use_first"][0], "BOOLEAN")
        self.assertTrue(inputs["optional"]["first"][1]["lazy"])
        self.assertTrue(inputs["optional"]["second"][1]["lazy"])

    def test_requests_only_selected_branch(self):
        self.assertEqual(self.node.check_lazy_status(True, first=None, second="unused"), ["first"])
        self.assertEqual(self.node.check_lazy_status(False, first="unused", second=None), ["second"])
        self.assertEqual(self.node.route(True, first="reverse", second="normal"), ("reverse",))
        self.assertEqual(self.node.route(False, first="reverse", second="normal"), ("normal",))

    def test_frontend_syncs_from_ignore_groups_state(self):
        self.assertIn('const NODE_TYPE = "MiniMaxH3EasyAreaSwitch";', self.source)
        self.assertIn("guhai_ig_active_set", self.source)
        self.assertIn("guhai_ig_active", self.source)
        self.assertIn("auto_sync", self.source)
        self.assertIn("syncRouteNode(this)", self.source)


if __name__ == "__main__":
    unittest.main()
