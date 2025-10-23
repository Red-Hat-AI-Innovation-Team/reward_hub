"""Unit tests for LLM judge deduplication logic"""

import pytest
from reward_hub.llm_judge.utils import deduplicate_conversations, extract_message_content


class TestExtractMessageContent:
    """Test extract_message_content function"""

    def test_extract_content_only(self):
        """Test extracting message with only content field"""
        message = {"role": "assistant", "content": "Hello world"}
        result = extract_message_content(message)
        assert result == "Hello world"

    def test_extract_tool_calls_only(self):
        """Test extracting message with only tool_calls"""
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}'
                    }
                }
            ]
        }
        result = extract_message_content(message)
        assert "Tool call: get_weather" in result
        assert '{"location": "NYC"}' in result

    def test_extract_content_and_tool_calls(self):
        """Test extracting message with both content and tool_calls"""
        message = {
            "role": "assistant",
            "content": "Let me check the weather",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "NYC"}'
                    }
                }
            ]
        }
        result = extract_message_content(message)
        assert "Let me check the weather" in result
        assert "Tool call: get_weather" in result

    def test_extract_empty_message(self):
        """Test extracting empty message"""
        message = {"role": "assistant"}
        result = extract_message_content(message)
        assert result == ""


class TestDeduplicateConversations:
    """Test deduplicate_conversations function"""

    def test_no_duplicates(self):
        """Test deduplication with no duplicates"""
        conversations = [
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hey"}],
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Greetings"}],
        ]
        unique, mapping = deduplicate_conversations(conversations, last_message_only=True)

        assert len(unique) == 3
        assert mapping == [0, 1, 2]

    def test_all_duplicates(self):
        """Test deduplication with all duplicates"""
        conversations = [
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
        ]
        unique, mapping = deduplicate_conversations(conversations, last_message_only=True)

        assert len(unique) == 1
        assert mapping == [0, 0, 0]

    def test_partial_duplicates(self):
        """Test deduplication with partial duplicates"""
        conversations = [
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hey"}],
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],  # Duplicate of first
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hey"}],  # Duplicate of second
        ]
        unique, mapping = deduplicate_conversations(conversations, last_message_only=True)

        assert len(unique) == 2
        assert mapping == [0, 1, 0, 1]

    def test_last_message_only_mode(self):
        """Test deduplication based on last message only (groupwise mode)"""
        conversations = [
            [{"role": "user", "content": "What's the weather?"}, {"role": "assistant", "content": "It's sunny"}],
            [{"role": "user", "content": "Tell me the weather"}, {"role": "assistant", "content": "It's sunny"}],  # Same response
        ]
        unique, mapping = deduplicate_conversations(conversations, last_message_only=True)

        # Should deduplicate because last messages are identical
        assert len(unique) == 1
        assert mapping == [0, 0]

    def test_full_conversation_mode(self):
        """Test deduplication based on full conversation (pointwise mode)"""
        conversations = [
            [{"role": "user", "content": "What's the weather?"}, {"role": "assistant", "content": "It's sunny"}],
            [{"role": "user", "content": "Tell me the weather"}, {"role": "assistant", "content": "It's sunny"}],  # Same response, different question
        ]
        unique, mapping = deduplicate_conversations(conversations, last_message_only=False)

        # Should NOT deduplicate because full conversations are different
        assert len(unique) == 2
        assert mapping == [0, 1]

    def test_tool_calls_deduplication(self):
        """Test deduplication with tool calls"""
        conversations = [
            [
                {"role": "user", "content": "Get weather"},
                {"role": "assistant", "content": "Checking", "tool_calls": [{"function": {"name": "weather", "arguments": '{"loc":"NYC"}'}}]}
            ],
            [
                {"role": "user", "content": "Get weather"},
                {"role": "assistant", "content": "Checking", "tool_calls": [{"function": {"name": "weather", "arguments": '{"loc":"NYC"}'}}]}
            ],
            [
                {"role": "user", "content": "Get weather"},
                {"role": "assistant", "content": "Checking", "tool_calls": [{"function": {"name": "weather", "arguments": '{"loc":"LA"}'}}]}
            ],
        ]
        unique, mapping = deduplicate_conversations(conversations, last_message_only=True)

        # First two should be duplicates, third is different
        assert len(unique) == 2
        assert mapping == [0, 0, 1]

    def test_preserves_original_order(self):
        """Test that deduplication preserves the first occurrence"""
        conversations = [
            [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A"}],
            [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "B"}],
            [{"role": "user", "content": "Q3"}, {"role": "assistant", "content": "A"}],  # Dup of first
            [{"role": "user", "content": "Q4"}, {"role": "assistant", "content": "C"}],
            [{"role": "user", "content": "Q5"}, {"role": "assistant", "content": "B"}],  # Dup of second
        ]
        unique, mapping = deduplicate_conversations(conversations, last_message_only=True)

        assert len(unique) == 3
        assert mapping == [0, 1, 0, 2, 1]
        # Check that unique preserves first occurrence
        assert extract_message_content(unique[0][-1]) == "A"
        assert extract_message_content(unique[1][-1]) == "B"
        assert extract_message_content(unique[2][-1]) == "C"

    def test_empty_content_handling(self):
        """Test handling of messages with empty content"""
        conversations = [
            [{"role": "user", "content": "Hello"}, {"role": "assistant"}],
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": ""}],
            [{"role": "user", "content": "Hello"}, {"role": "assistant"}],
        ]
        unique, mapping = deduplicate_conversations(conversations, last_message_only=True)

        # All should be considered duplicates (empty content)
        assert len(unique) == 1
        assert mapping == [0, 0, 0]
