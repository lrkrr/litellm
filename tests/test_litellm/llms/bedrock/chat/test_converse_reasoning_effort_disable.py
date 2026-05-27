"""
Regression tests for ``reasoning_effort="disable"`` on Anthropic-via-Bedrock.

Background: our fork's ``AnthropicConfig._map_reasoning_effort`` returns
``None`` for ``"none"`` and ``"disable"``. The Bedrock Converse helper used
to write that ``None`` straight into ``optional_params["thinking"]``, which
caused ``is_thinking_enabled`` to crash with::

    'NoneType' object has no attribute 'get'

The fix is twofold:

1. The Bedrock helper now only sets ``thinking`` when the mapping is non-None.
2. The base ``is_thinking_enabled`` is defensive against ``thinking=None``.
"""

from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig


class TestBedrockReasoningEffortDisable:
    def test_disable_does_not_set_thinking_key(self):
        config = AmazonConverseConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "disable"},
            optional_params={},
            model="anthropic.claude-sonnet-4-6-v1",
            drop_params=False,
        )
        # No thinking key, no crash. Either absent or non-None.
        assert optional_params.get("thinking") is None
        assert "thinking" not in optional_params

    def test_disable_does_not_crash_with_max_tokens_resolution(self):
        """Repro of the original 400: update_optional_params_with_thinking_tokens
        used to AttributeError on ``thinking=None``."""
        config = AmazonConverseConfig()
        # Should not raise.
        config.map_openai_params(
            non_default_params={"reasoning_effort": "disable"},
            optional_params={},
            model="anthropic.claude-sonnet-4-6-v1",
            drop_params=False,
        )

    def test_none_does_not_set_thinking_key(self):
        config = AmazonConverseConfig()
        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "none"},
            optional_params={},
            model="anthropic.claude-sonnet-4-6-v1",
            drop_params=False,
        )
        assert "thinking" not in optional_params

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
    """``thinking`` may be present with value ``None`` after upstream mapping."""

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
