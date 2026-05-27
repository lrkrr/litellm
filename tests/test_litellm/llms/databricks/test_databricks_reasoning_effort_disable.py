"""
Regression test for ``reasoning_effort="disable"`` on Claude via Databricks.

Pins the explicit ``thinking: {"type": "disabled"}`` wire format (rather than
omitting the param) so disable works correctly on models where
omit-defaults-to-off is not documented.
"""

from litellm.llms.databricks.chat.transformation import DatabricksConfig


class TestDatabricksReasoningEffortDisable:
    def test_disable_sends_explicit_disabled_thinking(self):
        config = DatabricksConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "disable"},
            optional_params={},
            model="databricks-claude-sonnet-4",
            drop_params=False,
        )
        assert optional_params["thinking"] == {"type": "disabled"}

    def test_none_sends_explicit_disabled_thinking(self):
        config = DatabricksConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="databricks-claude-sonnet-4",
            drop_params=False,
        )
        assert optional_params["thinking"] == {"type": "disabled"}

    def test_low_still_sets_thinking(self):
        config = DatabricksConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "low"},
            optional_params={},
            model="databricks-claude-3-7-sonnet",
            drop_params=False,
        )
        thinking = optional_params.get("thinking")
        assert isinstance(thinking, dict)
        assert thinking.get("type") == "enabled"
