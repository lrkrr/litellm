"""
Regression test for ``reasoning_effort="disable"`` on Claude via Databricks.

Same root cause as the Bedrock regression: our fork's
``AnthropicConfig._map_reasoning_effort`` returns ``None`` for ``"disable"``,
and the Databricks helper used to write that ``None`` straight into
``optional_params["thinking"]``.
"""

from litellm.llms.databricks.chat.transformation import DatabricksConfig


class TestDatabricksReasoningEffortDisable:
    def test_disable_does_not_set_thinking_key(self):
        config = DatabricksConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "disable"},
            optional_params={},
            model="databricks-claude-sonnet-4",
            drop_params=False,
        )
        assert "thinking" not in optional_params

    def test_none_does_not_set_thinking_key(self):
        config = DatabricksConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="databricks-claude-sonnet-4",
            drop_params=False,
        )
        assert "thinking" not in optional_params

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
