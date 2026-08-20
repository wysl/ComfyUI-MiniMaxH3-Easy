import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class H3WebUILayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (PROJECT_ROOT / "web" / "minimax_h3_easy_ui.js").read_text(
            encoding="utf-8"
        )

    def test_conditional_height_compensation_is_idempotent(self):
        self.assertIn(
            "function syncModeWidgets(node, preserveNodeSize = false)",
            self.source,
        )
        self.assertIn(
            "desiredHeightAdjustment - previousHeightAdjustment",
            self.source,
        )
        self.assertNotIn(
            "adjustNodeHeight(node, visible ? rowHeight : -rowHeight);",
            self.source,
        )

    def test_configure_preserves_serialized_node_size(self):
        configure_start = self.source.index(
            "nodeType.prototype.onConfigure = function onConfigureH3Easy(info)"
        )
        configure_end = self.source.index(
            "const originalConnectionsChange", configure_start
        )
        configure_source = self.source[configure_start:configure_end]
        self.assertIn("syncModeWidgets(this, true);", configure_source)

    def test_external_prompt_traces_selected_area_switch_branch(self):
        self.assertIn('const AREA_SWITCH_CLASS = "MiniMaxH3EasyAreaSwitch";', self.source)
        self.assertIn("function resolvePromptSourceNode(node, visited = new Set())", self.source)
        self.assertIn('const selectedInput = useFirst ? "first" : "second";', self.source)
        self.assertIn("return Number(promptSourceNode(candidate)?.id) === sourceId;", self.source)
        self.assertIn("function installAreaSwitchReferenceWatcher(nodeType, nodeData)", self.source)
        self.assertIn("refreshIfRouteChanged(this);", self.source)


if __name__ == "__main__":
    unittest.main()
