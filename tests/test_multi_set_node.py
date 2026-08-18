from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_PATH = PROJECT_ROOT / "web" / "multi_set.js"


class MultiSetNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")

    def test_registers_a_virtual_multi_set_node(self):
        self.assertIn('const NODE_TYPE = "MiniMaxH3EasyMultiSet";', self.source)
        self.assertIn('name: "MiniMaxH3Easy.MultiSet"', self.source)
        self.assertIn("this.isVirtualNode = true;", self.source)

    def test_starts_with_two_input_output_pairs(self):
        self.assertIn("const MIN_PAIRS = 2;", self.source)
        self.assertIn("this.ensurePairs(MIN_PAIRS);", self.source)
        self.assertIn("this.addInput(", self.source)
        self.assertIn("this.addOutput(", self.source)

    def test_adds_an_empty_pair_after_all_inputs_are_connected(self):
        self.assertIn(
            "if (this.inputs.every((input) => input.link != null)) this.addPair();",
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
        self.assertIn("...multiSetNames(node.graph)", self.source)

    def test_get_combo_is_remounted_when_multi_set_values_change(self):
        self.assertNotIn("options.__h3MultiSetValues) return", self.source)
        self.assertIn("widget.options = wrapped;", self.source)
        self.assertIn("node.widgets.splice(index, 1);", self.source)
        self.assertIn("node.widgets.splice(index, 0, widget);", self.source)

    def test_has_no_control_after_generate_option(self):
        self.assertNotIn("addValueControlWidgets", self.source)
        self.assertNotIn("control_after_generate", self.source)


if __name__ == "__main__":
    unittest.main()
