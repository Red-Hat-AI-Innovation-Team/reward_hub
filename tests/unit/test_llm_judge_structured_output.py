#!/usr/bin/env python3
"""Tests for structured output mode in LLM judges."""

from unittest.mock import Mock, patch

import pytest

from reward_hub.llm_judge import create_groupwise_judge, create_pointwise_judge


def _mock_completion_response(content: str) -> Mock:
    return Mock(choices=[Mock(message=Mock(content=content))])


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
                    return _mock_completion_response('{"score": 7.0, "reasoning": "ok"}')

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
                assert mock_completion.call_args_list[0].kwargs.get("response_format") is not None
                assert mock_completion.call_args_list[1].kwargs.get("response_format") is None

    def test_strict_mode_fails_when_response_format_is_unsupported(self):
        with patch("reward_hub.llm_judge.pointwise.validate_api_configuration"):
            with patch("litellm.completion") as mock_completion:

                def side_effect(*args, **kwargs):
                    if kwargs.get("response_format") is not None:
                        raise ValueError("response_format not supported")
                    return _mock_completion_response('{"score": 7.0, "reasoning": "ok"}')

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
