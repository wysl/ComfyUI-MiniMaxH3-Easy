from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_PATH = PROJECT_ROOT / "web" / "multi_primitive.js"


class MultiPrimitiveNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")

    def test_node_registers_as_a_custom_frontend_extension(self):
        self.assertIn('name: "MiniMaxH3Easy.MultiPrimitive"', self.source)
        self.assertIn("registerCustomNodes()", self.source)
        self.assertIn("LiteGraph.registerNodeType(", self.source)

    def test_node_is_virtual_and_starts_with_two_outputs(self):
        self.assertIn('const MIN_OUTPUTS = 2;', self.source)
        self.assertIn('this.isVirtualNode = true;', self.source)
        self.assertIn('this.ensureMinimumOutputs();', self.source)

    def test_output_slots_expand_when_every_slot_is_connected(self):
        self.assertIn('if (this.outputs.every(outputHasLink)) this.addEmptyOutput();', self.source)
        self.assertIn('this.outputs.length > MIN_OUTPUTS', self.source)

    def test_each_output_has_an_independent_widget_value(self):
        self.assertIn('const name = `value_${slot + 1}`;', self.source)
        self.assertIn('widget.__h3MultiPrimitiveSlot = slot;', self.source)
        self.assertIn('this.applySlotToGraph(slot);', self.source)

    def test_does_not_add_control_after_generate_widgets(self):
        self.assertNotIn('addValueControlWidgets', self.source)
        self.assertNotIn('control_after_generate', self.source)

    def test_saved_widget_values_are_restored_after_graph_load(self):
        self.assertIn('Array.isArray(this.widgets_values)', self.source)
        self.assertIn('this.rebuildWidgets(savedValues);', self.source)


if __name__ == "__main__":
    unittest.main()
