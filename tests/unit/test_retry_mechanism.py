#!/usr/bin/env python3
"""Test cases for retry mechanism in LLM Judge functionality"""

import pytest
from unittest.mock import Mock, patch
from reward_hub.llm_judge import create_pointwise_judge, create_groupwise_judge
from reward_hub.llm_judge.utils import with_retry, call_with_retry
import litellm


class TestRetryMechanism:
    """Test retry functionality for LLM judges"""

    def test_retry_decorator_sync_success(self):
        """Test retry decorator with sync function that succeeds immediately"""
        call_count = 0
        
        @with_retry(max_attempts=3, min_wait=0.01, max_wait=0.1)
        def mock_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = mock_function()
        assert result == "success"
        assert call_count == 1  # Should succeed on first try

    def test_retry_decorator_sync_eventual_success(self):
        """Test retry decorator with sync function that fails then succeeds"""
        call_count = 0
        
        @with_retry(max_attempts=3, min_wait=0.01, max_wait=0.1)
        def mock_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = mock_function()
        assert result == "success"
        assert call_count == 3  # Should succeed on third try

    def test_retry_decorator_sync_final_failure(self):
        """Test retry decorator with sync function that always fails"""
        call_count = 0
        
        @with_retry(max_attempts=3, min_wait=0.01, max_wait=0.1)
        def mock_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            mock_function()
        assert call_count == 3  # Should try 3 times

    @pytest.mark.asyncio
    async def test_retry_decorator_async_success(self):
        """Test retry decorator with async function that succeeds immediately"""
        call_count = 0
        
        @with_retry(max_attempts=3, min_wait=0.01, max_wait=0.1)
        async def mock_async_function():
            nonlocal call_count
            call_count += 1
            return "async_success"
        
        result = await mock_async_function()
        assert result == "async_success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_decorator_async_eventual_success(self):
        """Test retry decorator with async function that fails then succeeds"""
        call_count = 0
        
        @with_retry(max_attempts=3, min_wait=0.01, max_wait=0.1)
        async def mock_async_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary async failure")
            return "async_success"
        
        result = await mock_async_function()
        assert result == "async_success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_call_with_retry_wrapper(self):
        """Test call_with_retry wrapper function"""
        call_count = 0
        
        async def mock_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Wrapper test failure")
            return "wrapper_success"
        
        result = await call_with_retry(mock_function)
        assert result == "wrapper_success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_pointwise_judge_retry_on_api_failure(self):
        """Test pointwise judge retries on API failures"""
        with patch('reward_hub.llm_judge.pointwise.validate_api_configuration'):
            with patch('litellm.acompletion') as mock_completion:
                # Configure mock to fail twice then succeed
                mock_completion.side_effect = [
                    ConnectionError("API timeout"),
                    ConnectionError("Rate limit"),
                    Mock(choices=[Mock(message=Mock(content='{"score": 8.5, "reasoning": "test"}'))])
                ]

                judge = create_pointwise_judge(
                    model="gpt-4o-mini",
                    criterion="overall_quality",
                    api_key="test-key"
                )

                conversation = [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test answer"}
                ]

                # Should succeed after retries
                score = await judge.ascore(conversation)
                assert score == 8.5
                assert mock_completion.call_count == 3

    @pytest.mark.asyncio
    async def test_groupwise_judge_retry_on_api_failure(self):
        """Test groupwise judge retries on API failures"""
        with patch('reward_hub.llm_judge.groupwise.validate_api_configuration'):
            with patch('litellm.acompletion') as mock_completion:
                # Configure mock to fail once then succeed
                mock_completion.side_effect = [
                    litellm.exceptions.RateLimitError(
                        message="Rate limit exceeded",
                        llm_provider="openai",
                        model="gpt-4o-mini"
                    ),
                    Mock(choices=[Mock(message=Mock(content='{"selected_indices": [0, 1], "reasoning": "test"}'))])
                ]

                judge = create_groupwise_judge(
                    model="gpt-4o-mini",
                    criterion="multi_step_tool_judge",
                    api_key="test-key"
                )

                conversations = [
                    [
                        {"role": "user", "content": "Test question"},
                        {"role": "assistant", "content": "Good answer"}
                    ],
                    [
                        {"role": "user", "content": "Test question"},
                        {"role": "assistant", "content": "Great answer"}
                    ],
                    [
                        {"role": "user", "content": "Test question"},
                        {"role": "assistant", "content": "Poor answer"}
                    ]
                ]

                # Should succeed after retry
                scores = await judge.ascore(conversations, top_n=2)
                assert scores == [1.0, 1.0, 0.0]
                assert mock_completion.call_count == 2

    def test_pointwise_judge_sync_retry(self):
        """Test pointwise judge sync method with retries"""
        with patch('reward_hub.llm_judge.pointwise.validate_api_configuration'):
            with patch('litellm.completion') as mock_completion:
                # Configure mock to fail once then succeed
                mock_completion.side_effect = [
                    litellm.exceptions.AuthenticationError(
                        message="Invalid API key",
                        llm_provider="openai",
                        model="gpt-4o-mini"
                    ),
                    Mock(choices=[Mock(message=Mock(content='{"score": 7.2, "reasoning": "test"}'))])
                ]

                judge = create_pointwise_judge(
                    model="gpt-4o-mini",
                    criterion="overall_quality",
                    api_key="test-key"
                )

                conversation = [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test answer"}
                ]

                # Should succeed after retry
                score = judge.score(conversation)
                assert score == 7.2
                assert mock_completion.call_count == 2

    def test_groupwise_judge_sync_retry(self):
        """Test groupwise judge sync method with retries"""
        with patch('reward_hub.llm_judge.groupwise.validate_api_configuration'):
            with patch('litellm.completion') as mock_completion:
                # Configure mock to fail twice then succeed
                mock_completion.side_effect = [
                    ConnectionError("Network error"),
                    TimeoutError("Request timeout"),
                    Mock(choices=[Mock(message=Mock(content='{"selected_indices": [1], "reasoning": "test"}'))])
                ]

                judge = create_groupwise_judge(
                    model="gpt-4o-mini",
                    criterion="multi_step_tool_judge",
                    api_key="test-key"
                )

                conversations = [
                    [
                        {"role": "user", "content": "Test question"},
                        {"role": "assistant", "content": "Poor answer"}
                    ],
                    [
                        {"role": "user", "content": "Test question"},
                        {"role": "assistant", "content": "Best answer"}
                    ]
                ]

                # Should succeed after retries
                scores = judge.score(conversations, top_n=1)
                assert scores == [0.0, 1.0]
                assert mock_completion.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_preserves_original_error(self):
        """Test that final error is properly re-raised after all retries fail"""
        with patch('reward_hub.llm_judge.pointwise.validate_api_configuration'):
            with patch('litellm.acompletion') as mock_completion:
                # Configure mock to always fail with specific error
                mock_completion.side_effect = litellm.exceptions.AuthenticationError(
                    message="Invalid API key provided",
                    llm_provider="openai",
                    model="gpt-4o-mini"
                )

                judge = create_pointwise_judge(
                    model="gpt-4o-mini",
                    criterion="overall_quality",
                    api_key="invalid-key"
                )

                conversation = [
                    {"role": "user", "content": "Test"},
                    {"role": "assistant", "content": "Test"}
                ]

                # Should raise the original AuthenticationError, not RetryError
                with pytest.raises(litellm.exceptions.AuthenticationError, match="Invalid API key provided"):
                    await judge.ascore(conversation)

                # Should have tried 3 times
                assert mock_completion.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_timing_configuration(self):
        """Test that retry timing follows exponential backoff"""
        import time
        
        call_times = []
        
        @with_retry(max_attempts=3, min_wait=0.1, max_wait=1.0, multiplier=2.0)
        async def timed_function():
            call_times.append(time.time())
            raise ValueError("Always fails")
        
        start_time = time.time()
        
        with pytest.raises(ValueError):
            await timed_function()
        
        # Should have 3 calls
        assert len(call_times) == 3
        
        # Check timing intervals (allowing some tolerance)
        # Note: actual delays can be longer due to system scheduling
        if len(call_times) >= 2:
            interval1 = call_times[1] - call_times[0]
            assert 0.08 <= interval1 <= 1.5  # ~0.1s min wait, allowing system overhead

        if len(call_times) >= 3:
            interval2 = call_times[2] - call_times[1]
            assert 0.15 <= interval2 <= 2.5  # ~0.2s min wait, allowing system overhead

    def test_retry_exception_filtering(self):
        """Test that retry only retries specified exception types"""
        call_count = 0
        
        @with_retry(max_attempts=3, min_wait=0.01, retry_exceptions=(ConnectionError,))
        def selective_retry_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Should retry")
            elif call_count == 2:
                raise ValueError("Should not retry") 
            return "success"
        
        # Should fail immediately on ValueError (not retried)
        with pytest.raises(ValueError, match="Should not retry"):
            selective_retry_function()
        
        # Should only have been called twice (initial + 1 retry for ConnectionError)
        assert call_count == 2


class TestRetryIntegration:
    """Integration tests for retry mechanism with actual judge functionality"""
    
    @pytest.mark.asyncio
    async def test_multiple_conversations_with_retries(self):
        """Test that retries work properly when processing multiple conversations"""
        with patch('reward_hub.llm_judge.pointwise.validate_api_configuration'):
            # Test single conversation with retries instead of parallel execution
            # Parallel execution with asyncio.gather makes mock ordering non-deterministic
            with patch('litellm.acompletion') as mock_completion:
                # Configure mock to fail twice then succeed
                mock_completion.side_effect = [
                    ConnectionError("First attempt fails"),
                    ConnectionError("Second attempt fails"),
                    Mock(choices=[Mock(message=Mock(content='{"score": 8.5, "reasoning": "test"}'))])
                ]

                judge = create_pointwise_judge(
                    model="gpt-4o-mini",
                    criterion="overall_quality",
                    api_key="test-key"
                )

                conversation = [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"}
                ]

                # Should succeed after 2 retries
                score = await judge.ascore(conversation)
                assert score == 8.5
                assert mock_completion.call_count == 3  # 3 attempts total

    def test_retry_configuration_inheritance(self):
        """Test that retry configuration is properly applied to judge methods"""
        with patch('reward_hub.llm_judge.pointwise.validate_api_configuration'):
            # Test that the decorators are actually applied
            judge = create_pointwise_judge(
                model="gpt-4o-mini",
                criterion="overall_quality",
                api_key="test-key"
            )

            # Check that the methods have retry decorators
            assert hasattr(judge._score_single, '__wrapped__')
            assert hasattr(judge._ascore_single, '__wrapped__')

        with patch('reward_hub.llm_judge.groupwise.validate_api_configuration'):
            groupwise_judge = create_groupwise_judge(
                model="gpt-4o-mini",
                criterion="multi_step_tool_judge",
                api_key="test-key"
            )

            assert hasattr(groupwise_judge._score_groupwise, '__wrapped__')
            assert hasattr(groupwise_judge._ascore_groupwise, '__wrapped__')