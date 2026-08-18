import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import unittest


PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_PATH = PROJECT_ROOT / "web" / "multi_set.js"
PYTHON_SOURCE_PATH = PROJECT_ROOT / "nodes.py"


def load_multi_set_selector():
    tree = ast.parse(PYTHON_SOURCE_PATH.read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_select_h3_multi_set_outputs"
    )
    namespace = {"Any": Any, "Mapping": Mapping}
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(module, str(PYTHON_SOURCE_PATH), "exec"), namespace)
    return namespace["_select_h3_multi_set_outputs"]


class MultiSetNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")
        cls.python_source = PYTHON_SOURCE_PATH.read_text(encoding="utf-8")
        cls.select_outputs = staticmethod(load_multi_set_selector())

    def test_registers_an_executable_multi_set_node(self):
        self.assertIn('const NODE_TYPE = "MiniMaxH3EasyMultiSet";', self.source)
        self.assertIn('name: "MiniMaxH3Easy.MultiSet"', self.source)
        self.assertIn("beforeRegisterNodeDef(nodeType, nodeData)", self.source)
        self.assertNotIn("isVirtualNode = true", self.source)
        self.assertIn("class MiniMaxH3EasyMultiSet:", self.python_source)
        self.assertIn("INPUT_IS_LIST = True", self.python_source)
        self.assertIn("OUTPUT_IS_LIST = (False,)", self.python_source)

    def test_starts_with_two_input_output_pairs(self):
        self.assertIn("const MIN_PAIRS = 2;", self.source)
        self.assertIn("ensurePairs(node, MIN_PAIRS);", self.source)
        self.assertIn("node.addInput(inputName(slot), \"*\");", self.source)
        self.assertIn("node.addOutput(", self.source)

    def test_adds_an_empty_pair_after_all_inputs_are_connected(self):
        self.assertIn(
            "&& this.inputs.every((input) => input.link != null)",
            self.source,
        )

    def test_automatically_copies_source_name_and_type(self):
        self.assertIn("const source = sourceInfo(this, slot);", self.source)
        self.assertIn('const type = source?.type || targetType(this, slot) || "*";', self.source)
        self.assertIn("this.commitName(slot, source?.name", self.source)

    def test_integrates_with_existing_get_nodes(self):
        self.assertIn('const GET_NODE_TYPE = "GetNode";', self.source)
        self.assertIn("findMultiSetEntry(graph", self.source)
        self.assertIn("prototype.getInputLink = function getMultiSetInputLink", self.source)
        self.assertIn("prototype.resolveVirtualOutput = function resolveMultiSetOutput", self.source)
        self.assertIn("{ node: entry.node, slot: entry.slot }", self.source)
        self.assertIn("...multiSetNames(node.graph)", self.source)

    def test_new_get_nodes_refresh_after_they_are_added_to_a_graph(self):
        self.assertIn("function scheduleCreatedGetRefresh(node)", self.source)
        self.assertIn("if (!node?.graph) return;", self.source)
        self.assertIn("queueMicrotask(refresh);", self.source)
        self.assertIn("scheduleCreatedGetRefresh(node);", self.source)

    def test_repeated_list_connection_is_distributed_in_order(self):
        shared = ["image 1", "image 2"]

        result = self.select_outputs(
            {"value_1": shared, "value_2": shared},
            2,
        )

        self.assertEqual(result, ("image 1", "image 2"))

    def test_separate_inputs_and_scalar_values_are_not_accidentally_split(self):
        result = self.select_outputs(
            {"value_1": ["first source"], "value_2": ["second source"]},
            3,
        )

        self.assertEqual(result, ("first source", "second source", None))

    def test_repeated_list_connections_cannot_silently_duplicate_last_item(self):
        shared = ["image 1", "image 2"]

        with self.assertRaisesRegex(ValueError, "more connections"):
            self.select_outputs(
                {"value_1": shared, "value_2": shared, "value_3": shared},
                3,
            )

    def test_backend_keeps_internal_input_names_stable(self):
        self.assertIn('return `value_${slot + 1}`;', self.source)
        self.assertIn("input.name = inputName(slot);", self.source)
        self.assertIn("input.label = visibleName;", self.source)

    def test_get_combo_is_remounted_when_multi_set_values_change(self):
        self.assertNotIn("options.__h3MultiSetValues) return", self.source)
        self.assertIn("widget.options = wrapped;", self.source)
        self.assertIn("node.widgets.splice(index, 1);", self.source)
        self.assertIn("node.widgets.splice(index, 0, widget);", self.source)

    def test_kjnodes_instance_refresh_keeps_multi_set_values(self):
        self.assertIn("function installGetNodeInstanceCompatibility(node)", self.source)
        self.assertIn('const refreshCombo = node._refreshComboOptions;', self.source)
        self.assertIn("wrapGetCombo(this);", self.source)
        self.assertIn("wrappedRefresh.__h3MultiSetWrapped = true;", self.source)
        self.assertIn("refreshGetNode(node);", self.source)

    def test_kjnodes_legacy_menu_uses_merged_widget_values(self):
        self.assertIn("function installGetWidgetClickCompatibility(node, widget)", self.source)
        self.assertIn("const values = getComboValues(widget);", self.source)
        self.assertIn("new LiteGraph.ContextMenu(labels", self.source)
        self.assertIn("installGetWidgetClickCompatibility(node, widget);", self.source)

    def test_has_no_control_after_generate_option(self):
        self.assertNotIn("addValueControlWidgets", self.source)
        self.assertNotIn("control_after_generate", self.source)


if __name__ == "__main__":
    unittest.main()
