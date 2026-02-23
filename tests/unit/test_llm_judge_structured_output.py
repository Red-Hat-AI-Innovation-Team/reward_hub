#!/usr/bin/env python3
"""Tests for structured output mode in LLM judges."""

import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from reward_hub.llm_judge import (
    create_groupwise_judge,
    create_pointwise_judge,
    get_structured_output_fallback_stats,
    log_structured_output_fallback_stats,
)
from reward_hub.llm_judge.utils import (
    get_response_format_fallback_counter,
    get_response_format_fallback_counter_display,
    increment_response_format_fallback_counter,
    is_response_format_cached_as_unsupported,
    mark_response_format_as_unsupported,
    reset_response_format_fallback_state,
)


def _mock_completion_response(content: str) -> Mock:
    return Mock(choices=[Mock(message=Mock(content=content))])


@pytest.fixture(autouse=True)
def _reset_structured_output_state():
    reset_response_format_fallback_state()
    yield
    reset_response_format_fallback_state()


class TestPointwiseStructuredOutput:
    def test_auto_mode_sends_response_format_schema(self):
        with patch("reward_hub.llm_judge.pointwise.validate_api_configuration"):
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = _mock_completion_response(
                    '{"score": 8.5, "reasoning": "clear and correct"}'
                )

                judge = create_pointwise_judge(
                    model="gpt-4o-mini",
                    criterion="overall_quality",
                    api_key="test-key",
                    structured_output_mode="auto",
                )

                conversation = [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test answer"},
                ]

                score = judge.score(conversation)
                assert score == 8.5

                kwargs = mock_completion.call_args.kwargs
                assert "response_format" in kwargs
                assert kwargs["response_format"]["type"] == "json_schema"

    def test_auto_mode_falls_back_when_response_format_is_unsupported(self):
        with patch("reward_hub.llm_judge.pointwise.validate_api_configuration"):
            with patch("litellm.completion") as mock_completion:

                def side_effect(*args, **kwargs):
                    if kwargs.get("response_format") is not None:
                        raise ValueError("response_format not supported")
                    return _mock_completion_response(
                        '{"score": 7.0, "reasoning": "ok"}'
                    )

                mock_completion.side_effect = side_effect

                judge = create_pointwise_judge(
                    model="gpt-4o-mini",
                    criterion="overall_quality",
                    api_key="test-key",
                    structured_output_mode="auto",
                )

                conversation = [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test answer"},
                ]

                score = judge.score(conversation)
                assert score == 7.0
                assert mock_completion.call_count == 2
                assert (
                    mock_completion.call_args_list[0].kwargs.get("response_format")
                    is not None
                )
                assert (
                    mock_completion.call_args_list[1].kwargs.get("response_format")
                    is None
                )

    def test_strict_mode_fails_when_response_format_is_unsupported(self):
        with patch("reward_hub.llm_judge.pointwise.validate_api_configuration"):
            with patch("litellm.completion") as mock_completion:

                def side_effect(*args, **kwargs):
                    if kwargs.get("response_format") is not None:
                        raise ValueError("response_format not supported")
                    return _mock_completion_response(
                        '{"score": 7.0, "reasoning": "ok"}'
                    )

                mock_completion.side_effect = side_effect

                judge = create_pointwise_judge(
                    model="gpt-4o-mini",
                    criterion="overall_quality",
                    api_key="test-key",
                    structured_output_mode="strict",
                )

                conversation = [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test answer"},
                ]

                with pytest.raises(ValueError, match="response_format not supported"):
                    judge.score(conversation)

    def test_auto_mode_does_not_fallback_on_unrelated_error(self):
        with patch("reward_hub.llm_judge.pointwise.validate_api_configuration"):
            with patch("litellm.completion") as mock_completion:
                mock_completion.side_effect = RuntimeError("rate limit exceeded")

                judge = create_pointwise_judge(
                    model="gpt-4o-mini",
                    criterion="overall_quality",
                    api_key="test-key",
                    structured_output_mode="auto",
                )

                conversation = [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test answer"},
                ]

                with pytest.raises(RuntimeError, match="rate limit exceeded"):
                    judge.score(conversation)

                assert mock_completion.call_count == 3
                assert all(
                    call.kwargs.get("response_format") is not None
                    for call in mock_completion.call_args_list
                )
                assert get_response_format_fallback_counter() == 0
                assert not is_response_format_cached_as_unsupported(
                    model="gpt-4o-mini",
                    base_url=None,
                )

    def test_auto_mode_uses_cached_unsupported_response_format_on_subsequent_calls(
        self,
    ):
        with patch("reward_hub.llm_judge.pointwise.validate_api_configuration"):
            with patch("litellm.completion") as mock_completion:

                def side_effect(*args, **kwargs):
                    if kwargs.get("response_format") is not None:
                        raise ValueError("response_format not supported")
                    return _mock_completion_response(
                        '{"score": 7.0, "reasoning": "ok"}'
                    )

                mock_completion.side_effect = side_effect

                judge = create_pointwise_judge(
                    model="gpt-4o-mini",
                    criterion="overall_quality",
                    api_key="test-key",
                    structured_output_mode="auto",
                )

                conversation = [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test answer"},
                ]

                assert judge.score(conversation) == 7.0
                assert judge.score(conversation) == 7.0

                assert mock_completion.call_count == 3
                assert (
                    mock_completion.call_args_list[0].kwargs.get("response_format")
                    is not None
                )
                assert (
                    mock_completion.call_args_list[1].kwargs.get("response_format")
                    is None
                )
                assert (
                    mock_completion.call_args_list[2].kwargs.get("response_format")
                    is None
                )
                assert get_response_format_fallback_counter() == 2

    @pytest.mark.asyncio
    async def test_async_auto_mode_uses_cached_unsupported_response_format(self):
        with patch("reward_hub.llm_judge.pointwise.validate_api_configuration"):
            with patch(
                "litellm.acompletion", new_callable=AsyncMock
            ) as mock_acompletion:

                async def side_effect(*args, **kwargs):
                    if kwargs.get("response_format") is not None:
                        raise ValueError("response_format not supported")
                    return _mock_completion_response(
                        '{"score": 6.0, "reasoning": "ok"}'
                    )

                mock_acompletion.side_effect = side_effect

                judge = create_pointwise_judge(
                    model="gpt-4o-mini",
                    criterion="overall_quality",
                    api_key="test-key",
                    structured_output_mode="auto",
                )

                conversation = [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test answer"},
                ]

                assert await judge.ascore(conversation) == 6.0
                assert await judge.ascore(conversation) == 6.0

                assert mock_acompletion.call_count == 3
                assert (
                    mock_acompletion.call_args_list[0].kwargs.get("response_format")
                    is not None
                )
                assert (
                    mock_acompletion.call_args_list[1].kwargs.get("response_format")
                    is None
                )
                assert (
                    mock_acompletion.call_args_list[2].kwargs.get("response_format")
                    is None
                )
                assert get_response_format_fallback_counter() == 2

    def test_off_mode_never_sends_response_format(self):
        with patch("reward_hub.llm_judge.pointwise.validate_api_configuration"):
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = _mock_completion_response(
                    '{"score": 8.0, "reasoning": "ok"}'
                )

                judge = create_pointwise_judge(
                    model="gpt-4o-mini",
                    criterion="overall_quality",
                    api_key="test-key",
                    structured_output_mode="off",
                )

                conversation = [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test answer"},
                ]

                assert judge.score(conversation) == 8.0
                assert mock_completion.call_count == 1
                assert mock_completion.call_args.kwargs.get("response_format") is None


class TestGroupwiseStructuredOutput:
    def test_auto_mode_sends_response_format_schema(self):
        with patch("reward_hub.llm_judge.groupwise.validate_api_configuration"):
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = _mock_completion_response(
                    '{"selected_indices": [1], "reasoning": "Response 1 is better"}'
                )

                judge = create_groupwise_judge(
                    model="gpt-4o-mini",
                    criterion="multi_step_tool_judge",
                    api_key="test-key",
                    structured_output_mode="auto",
                )

                conversations = [
                    [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "Response A"},
                    ],
                    [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "Response B"},
                    ],
                ]

                scores = judge.score(conversations, top_n=1)
                assert scores == [0.0, 1.0]

                kwargs = mock_completion.call_args.kwargs
                assert "response_format" in kwargs
                assert kwargs["response_format"]["type"] == "json_schema"

    def test_strict_mode_rejects_invalid_groupwise_indices(self):
        with patch("reward_hub.llm_judge.groupwise.validate_api_configuration"):
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value = _mock_completion_response(
                    '{"selected_indices": [0, 0], "reasoning": "duplicate indices"}'
                )

                judge = create_groupwise_judge(
                    model="gpt-4o-mini",
                    criterion="multi_step_tool_judge",
                    api_key="test-key",
                    structured_output_mode="strict",
                )

                conversations = [
                    [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "Response A"},
                    ],
                    [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "Response B"},
                    ],
                ]

                with pytest.raises(ValueError, match="selected_indices"):
                    judge.score(conversations, top_n=2)

    def test_groupwise_auto_mode_falls_back_when_response_format_is_unsupported(self):
        with patch("reward_hub.llm_judge.groupwise.validate_api_configuration"):
            with patch("litellm.completion") as mock_completion:

                def side_effect(*args, **kwargs):
                    if kwargs.get("response_format") is not None:
                        raise ValueError("response_format not supported")
                    return _mock_completion_response(
                        '{"selected_indices": [1], "reasoning": "ok"}'
                    )

                mock_completion.side_effect = side_effect

                judge = create_groupwise_judge(
                    model="gpt-4o-mini",
                    criterion="multi_step_tool_judge",
                    api_key="test-key",
                    structured_output_mode="auto",
                )

                conversations = [
                    [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "Response A"},
                    ],
                    [
                        {"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "Response B"},
                    ],
                ]

                assert judge.score(conversations, top_n=1) == [0.0, 1.0]
                assert mock_completion.call_count == 2
                assert (
                    mock_completion.call_args_list[0].kwargs.get("response_format")
                    is not None
                )
                assert (
                    mock_completion.call_args_list[1].kwargs.get("response_format")
                    is None
                )
                assert get_response_format_fallback_counter() == 1


class TestStructuredOutputFallbackState:
    def test_fallback_counter_saturates_at_99999(self):
        for _ in range(100500):
            increment_response_format_fallback_counter()

        assert get_response_format_fallback_counter() == 99999
        assert get_response_format_fallback_counter_display() == "99999+"

    def test_response_format_unsupported_cache_is_bounded(self):
        for index in range(300):
            mark_response_format_as_unsupported(model=f"model-{index}", base_url=None)

        assert not is_response_format_cached_as_unsupported(
            model="model-0", base_url=None
        )
        assert is_response_format_cached_as_unsupported(
            model="model-299", base_url=None
        )

    def test_structured_output_fallback_stats_reports_counter_and_cache(self):
        mark_response_format_as_unsupported(model="gpt-4o-mini", base_url=None)
        increment_response_format_fallback_counter()

        stats = get_structured_output_fallback_stats()
        assert stats["fallback_count"] == 1
        assert stats["fallback_count_display"] == "1"
        assert stats["unsupported_cache_size"] == 1
        assert stats["unsupported_cache_maxsize"] == 256

    def test_structured_output_fallback_stats_can_be_logged(self, caplog):
        mark_response_format_as_unsupported(model="gpt-4o-mini", base_url=None)
        increment_response_format_fallback_counter()

        with caplog.at_level(logging.INFO, logger="reward_hub.llm_judge.utils"):
            log_structured_output_fallback_stats()

        assert any(
            "fallback_count=1" in record.message
            and "unsupported_cache_size=1" in record.message
            for record in caplog.records
        )
