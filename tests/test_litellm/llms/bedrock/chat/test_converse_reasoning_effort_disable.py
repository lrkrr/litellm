"""
Regression tests for ``reasoning_effort="disable"`` on Anthropic-via-Bedrock.

Our fork's ``AnthropicConfig._map_reasoning_effort`` now returns
``{"type": "disabled"}`` for ``"none"`` and ``"disable"`` (was previously
``None``, which both (a) crashed ``is_thinking_enabled`` downstream and
(b) relied on omit-defaults-to-off semantics that are not documented for
every Claude model). These tests pin the explicit-disabled wire format.
"""

from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig


class TestBedrockReasoningEffortDisable:
    def test_disable_sends_explicit_disabled_thinking(self):
        config = AmazonConverseConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "disable"},
            optional_params={},
            model="anthropic.claude-sonnet-4-6-v1",
            drop_params=False,
        )
        assert optional_params["thinking"] == {"type": "disabled"}

    def test_disable_does_not_crash_with_max_tokens_resolution(self):
        """Repro of the original 400: update_optional_params_with_thinking_tokens
        used to AttributeError when thinking was None."""
        config = AmazonConverseConfig()
        config.map_openai_params(
            non_default_params={"reasoning_effort": "disable"},
            optional_params={},
            model="anthropic.claude-sonnet-4-6-v1",
            drop_params=False,
        )

    def test_none_sends_explicit_disabled_thinking(self):
        config = AmazonConverseConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="anthropic.claude-sonnet-4-6-v1",
            drop_params=False,
        )
        assert optional_params["thinking"] == {"type": "disabled"}

    def test_low_still_sets_thinking(self):
        config = AmazonConverseConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "low"},
            optional_params={},
            model="anthropic.claude-3-7-sonnet-20250219",
            drop_params=False,
        )
        thinking = optional_params.get("thinking")
        assert isinstance(thinking, dict)
        assert thinking.get("type") == "enabled"
        assert thinking.get("budget_tokens", 0) > 0


class TestIsThinkingEnabledDefensive:
    """``is_thinking_enabled`` must treat ``disabled`` and missing/None as off."""

    def test_none_thinking_value_is_not_enabled(self):
        config = AmazonConverseConfig()
        assert config.is_thinking_enabled({"thinking": None}) is False

    def test_disabled_dict_is_not_enabled(self):
        config = AmazonConverseConfig()
        assert (
            config.is_thinking_enabled({"thinking": {"type": "disabled"}}) is False
        )

    def test_enabled_dict_is_enabled(self):
        config = AmazonConverseConfig()
        assert (
            config.is_thinking_enabled(
                {"thinking": {"type": "enabled", "budget_tokens": 1024}}
            )
            is True
        )
