"""
Regression tests for Responses API reasoning / reasoning_effort bridging.

Locks behavior for #24599 (dict dropped on Anthropic), disable, and non-Anthropic
providers so bridge changes cannot regress silently.
"""

import copy

import pytest

from litellm.constants import DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams


def _bridge(model: str, reasoning: dict | str) -> dict:
    responses_api_request: ResponsesAPIOptionalRequestParams = {
        "reasoning": reasoning,
    }
    return LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model=model,
        input="hi",
        responses_api_request=responses_api_request,
    )


class TestResponsesBridgeNeverPassesDict:
    """Completion request must never receive dict reasoning_effort (#24599)."""

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-6",
            "gemini/gemini-2.5-pro",
        ],
    )
    def test_summary_plus_effort_is_string_only(self, model: str):
        result = _bridge(model, {"effort": "medium", "summary": "auto"})
        assert result.get("reasoning_effort") == "medium"
        assert isinstance(result.get("reasoning_effort"), str)

    def test_responses_request_reasoning_unchanged_after_bridge(self):
        """summary stays on responses_api_request for response transformation."""
        responses_api_request: ResponsesAPIOptionalRequestParams = {
            "reasoning": {"effort": "high", "summary": "detailed"},
        }
        original = copy.deepcopy(responses_api_request)
        LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="claude-sonnet-4-6",
            input="hi",
            responses_api_request=responses_api_request,
        )
        assert responses_api_request["reasoning"] == original["reasoning"]

    def test_summary_without_effort_omits_reasoning_effort(self):
        result = _bridge("claude-sonnet-4-6", {"summary": "auto"})
        assert "reasoning_effort" not in result


class TestAnthropicEndToEndFromBridge:
    """Bridge string + Anthropic mapper must enable thinking (show_reasoning fix)."""

    def test_medium_with_summary_enables_adaptive_thinking(self):
        completion = _bridge("claude-sonnet-4-6", {"effort": "medium", "summary": "auto"})
        anthropic = AnthropicConfig().map_openai_params(
            non_default_params={"reasoning_effort": completion["reasoning_effort"]},
            optional_params={},
            model="claude-sonnet-4-6",
            drop_params=False,
        )
        assert anthropic["thinking"] == {"type": "adaptive"}
        assert anthropic["output_config"] == {"effort": "medium"}

    def test_string_medium_without_summary_unchanged(self):
        completion = _bridge("claude-sonnet-4-6", {"effort": "medium"})
        anthropic = AnthropicConfig().map_openai_params(
            non_default_params={"reasoning_effort": completion["reasoning_effort"]},
            optional_params={},
            model="claude-sonnet-4-6",
            drop_params=False,
        )
        assert anthropic["thinking"] == {"type": "adaptive"}
        assert anthropic["output_config"] == {"effort": "medium"}

    def test_disable_from_bridge_disables_thinking(self):
        completion = _bridge("claude-sonnet-4-6", {"effort": "disable", "summary": "auto"})
        anthropic = AnthropicConfig().map_openai_params(
            non_default_params={"reasoning_effort": completion["reasoning_effort"]},
            optional_params={},
            model="claude-sonnet-4-6",
            drop_params=False,
        )
        assert "thinking" not in anthropic
        assert "output_config" not in anthropic


class TestGeminiEndToEndFromBridge:
    """Gemini must still map string effort from bridge (not dict)."""

    def test_high_with_summary_maps_thinking_config(self):
        completion = _bridge("gemini/gemini-2.5-pro", {"effort": "high", "summary": "auto"})
        gemini = VertexGeminiConfig()
        anthropic_params = gemini.map_openai_params(
            non_default_params={"reasoning_effort": completion["reasoning_effort"]},
            optional_params={},
            model="gemini-2.5-pro",
            drop_params=False,
        )
        assert "thinkingConfig" in anthropic_params


class TestLegacyAnthropicBudgetModels:
    """Pre-4.6 adaptive models keep fixed budget_tokens mapping."""

    def test_sonnet_45_medium_budget_via_bridge(self):
        completion = _bridge(
            "claude-4-sonnet-20250514", {"effort": "medium", "summary": "auto"}
        )
        anthropic = AnthropicConfig().map_openai_params(
            non_default_params={"reasoning_effort": completion["reasoning_effort"]},
            optional_params={},
            model="claude-4-sonnet-20250514",
            drop_params=False,
        )
        assert anthropic["thinking"] == {
            "type": "enabled",
            "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
        }
        assert "output_config" not in anthropic


class TestIsThinkingEnabledRegression:
    def test_disable_dict_not_thinking_enabled(self):
        config = AnthropicConfig()
        assert (
            config.is_thinking_enabled(
                {"reasoning_effort": {"effort": "disable", "summary": "auto"}}
            )
            is False
        )

    def test_medium_dict_is_thinking_enabled(self):
        config = AnthropicConfig()
        assert (
            config.is_thinking_enabled(
                {"reasoning_effort": {"effort": "medium", "summary": "auto"}}
            )
            is True
        )
