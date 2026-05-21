"""
Tests for reasoning_effort mapping in AnthropicConfig.

Covers _map_reasoning_effort, map_openai_params dict normalization (#24599),
and reasoning_effort='disable'.
"""

from litellm.constants import DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams


class TestMapReasoningEffort:
    def test_none_returns_none_for_opus_4_6(self):
        """reasoning_effort=None should return None for Opus 4.6, not adaptive."""
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort=None, model="claude-opus-4-6"
        )
        assert result is None

    def test_none_returns_none_for_other_models(self):
        """reasoning_effort=None should return None for non-Opus models."""
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort=None, model="claude-4-sonnet-20250514"
        )
        assert result is None

    def test_opus_4_6_returns_adaptive_for_low(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="low", model="claude-opus-4-6"
        )
        assert result["type"] == "adaptive"

    def test_opus_4_6_returns_adaptive_for_high(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="high", model="claude-opus-4-6"
        )
        assert result["type"] == "adaptive"

    def test_other_model_low_returns_enabled_with_budget(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="low", model="claude-4-sonnet-20250514"
        )
        assert result["type"] == "enabled"
        assert "budget_tokens" in result

    def test_other_model_high_returns_enabled_with_budget(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="high", model="claude-4-sonnet-20250514"
        )
        assert result["type"] == "enabled"
        assert "budget_tokens" in result

    def test_none_string_returns_none_for_opus_4_6(self):
        """reasoning_effort='none' should return None for Opus 4.6."""
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="none", model="claude-opus-4-6"
        )
        assert result is None

    def test_none_string_returns_none_for_other_models(self):
        """reasoning_effort='none' should return None for non-Opus models."""
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="none", model="claude-4-sonnet-20250514"
        )
        assert result is None

    def test_disable_returns_none(self):
        result = AnthropicConfig._map_reasoning_effort(
            reasoning_effort="disable", model="claude-sonnet-4-6"
        )
        assert result is None


class TestMapOpenAIParamsReasoningEffort:
    """Regression tests for BerriAI/litellm#24599 (dict-valued reasoning_effort)."""

    def test_dict_with_summary_enables_thinking_for_sonnet_4_6(self):
        result = AnthropicConfig().map_openai_params(
            non_default_params={
                "reasoning_effort": {"effort": "medium", "summary": "auto"}
            },
            optional_params={},
            model="claude-sonnet-4-6",
            drop_params=False,
        )
        assert result["thinking"] == {"type": "adaptive"}
        assert result["output_config"] == {"effort": "medium"}

    def test_dict_with_summary_enables_thinking_for_non_4_6(self):
        result = AnthropicConfig().map_openai_params(
            non_default_params={
                "reasoning_effort": {"effort": "low", "summary": "auto"}
            },
            optional_params={},
            model="claude-4-sonnet-20250514",
            drop_params=False,
        )
        assert result["thinking"]["type"] == "enabled"
        assert "output_config" not in result

    def test_disable_string_disables_thinking(self):
        result = AnthropicConfig().map_openai_params(
            non_default_params={"reasoning_effort": "disable"},
            optional_params={},
            model="claude-sonnet-4-6",
            drop_params=False,
        )
        assert "thinking" not in result
        assert "output_config" not in result

    def test_disable_dict_disables_thinking(self):
        result = AnthropicConfig().map_openai_params(
            non_default_params={
                "reasoning_effort": {"effort": "disable", "summary": "auto"}
            },
            optional_params={},
            model="claude-sonnet-4-6",
            drop_params=False,
        )
        assert "thinking" not in result
        assert "output_config" not in result

    def test_medium_string_enables_budget_for_non_4_6(self):
        result = AnthropicConfig().map_openai_params(
            non_default_params={"reasoning_effort": "medium"},
            optional_params={},
            model="claude-4-sonnet-20250514",
            drop_params=False,
        )
        assert result["thinking"] == {
            "type": "enabled",
            "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
        }


class TestOpus47AdaptiveThinking:
    def test_opus_4_7_medium_uses_adaptive_not_budget_tokens(self):
        """Opus 4.7 rejects manual budget_tokens; must use adaptive + output_config."""
        result = AnthropicConfig().map_openai_params(
            non_default_params={"reasoning_effort": "medium"},
            optional_params={},
            model="claude-opus-4-7",
            drop_params=False,
        )
        assert result["thinking"] == {"type": "adaptive"}
        assert result["output_config"] == {"effort": "medium"}
        assert "budget_tokens" not in result.get("thinking", {})

    def test_opus_4_7_disable_has_no_thinking(self):
        result = AnthropicConfig().map_openai_params(
            non_default_params={"reasoning_effort": "disable"},
            optional_params={},
            model="claude-opus-4-7",
            drop_params=False,
        )
        assert "thinking" not in result
        assert "output_config" not in result


class TestResponsesApiReasoningWithSummary:
    def test_reasoning_with_summary_extracts_string_effort(self):
        """show_reasoning must not pass a dict reasoning_effort to chat completion."""
        responses_api_request: ResponsesAPIOptionalRequestParams = {
            "reasoning": {"effort": "medium", "summary": "auto"},
        }
        result = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
                model="claude-sonnet-4-6",
                input="hi",
                responses_api_request=responses_api_request,
            )
        )
        assert result["reasoning_effort"] == "medium"
        assert isinstance(result["reasoning_effort"], str)
