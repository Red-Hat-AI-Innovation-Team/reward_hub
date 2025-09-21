import asyncio
import logging

import aiohttp
import backoff
import requests

from .base import AbstractLanguageModel
from .error_handling import (
    RETRYABLE_ERRORS,
    APIError,
    enhanced_on_backoff,
    format_non_retryable_error,
    parse_api_error,
    should_retry,
)
from .types import ChatMessage



class OpenAICompatibleLanguageModel(AbstractLanguageModel):
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model_name: str,
        system_prompt: str | None = None,
        is_async: bool = False,
        # default runtime parameters
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_tries: int = 8,
        max_concurrency: int = -1,
        replace_error_with_message: str | None = None,
    ):
        assert max_concurrency == -1 or max_concurrency > 0, (
            "max_concurrency must be -1 (unlimited concurrency) or a positive integer"
        )

        self.endpoint = endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.is_async = is_async
        self.max_tries = max_tries
        self.max_concurrency = max_concurrency
        self.replace_error_with_message = replace_error_with_message

        # runtime parameters
        self.stop = stop
        self.max_tokens = max_tokens
        self.temperature = temperature

        # set up headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # endpoint type
        self.endpoint_type = "openai" if "openai" in self.endpoint else "vllm"

    @property
    def _chat_completion_endpoint(self) -> str:
        return self.endpoint.rstrip("/") + "/chat/completions"

    def _prepare_request_data(
        self,
        messages: list[ChatMessage],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict:
        # helper method to prepare request data for both sync and async methods
        # Convert dict messages to Message objects if needed
        messages = [
            msg if isinstance(msg, ChatMessage) else ChatMessage(**msg)
            for msg in messages
        ]

        if self.system_prompt:
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                *messages,
            ]

        request_data = {
            "model": self.model_name,
            "messages": [msg.to_dict() for msg in messages],
        }

        if self.endpoint_type == "vllm":
            request_data["extra_body"] = {}
            if messages[-1].role == "assistant":
                request_data["extra_body"]["add_generation_prompt"] = False
                request_data["extra_body"]["continue_final_message"] = True
                request_data["add_generation_prompt"] = False
                request_data["continue_final_message"] = True
            if include_stop_str_in_output is not None:
                request_data["extra_body"]["include_stop_str_in_output"] = (
                    include_stop_str_in_output
                )
                request_data["include_stop_str_in_output"] = include_stop_str_in_output
        else:
            logging.info(
                "openai endpoint does not support add_generation_prompt, continue_final_message, or include_stop_str_in_output"
            )
            if include_stop_str_in_output is not None:
                logging.warning(
                    "include_stop_str_in_output parameter is not supported with OpenAI endpoints and will be ignored"
                )

        # set default runtime parameters
        if self.stop is not None:
            request_data["stop"] = self.stop
        if self.max_tokens is not None:
            request_data["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            request_data["temperature"] = self.temperature

        # override runtime parameters
        if stop is not None:
            request_data["stop"] = stop
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        if temperature is not None:
            request_data["temperature"] = temperature

        # add tools and tool_choice if provided
        if tools is not None:
            request_data["tools"] = tools
        if tool_choice is not None:
            request_data["tool_choice"] = tool_choice

        return request_data

    async def _generate(
        self,
        messages_lst: list[list[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> list[dict]:
        # limit concurrency to max_concurrency using a semaphore
        semaphore = asyncio.Semaphore(
            len(messages_lst) if self.max_concurrency == -1 else self.max_concurrency
        )

        # create a single session for all requests in this call
        async with aiohttp.ClientSession() as session:

            @backoff.on_exception(
                backoff.expo,
                RETRYABLE_ERRORS,
                max_tries=self.max_tries,
                on_backoff=enhanced_on_backoff,
                giveup=lambda e: not should_retry(e),
            )
            async def fetch_response(
                messages: list[ChatMessage], _temperature: float | None
            ) -> dict:
                async with semaphore:
                    request_data = self._prepare_request_data(
                        messages,
                        stop,
                        max_tokens,
                        _temperature,
                        include_stop_str_in_output,
                        tools,
                        tool_choice,
                    )

                    async with session.post(
                        self._chat_completion_endpoint,
                        headers=self.headers,
                        json=request_data,
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            api_error = parse_api_error(response.status, error_text)
                            if not should_retry(api_error):
                                logging.error(format_non_retryable_error(api_error))
                            raise api_error
                        response_json = await response.json()
                        # Return the full message object to preserve tool calls
                        return response_json["choices"][0]["message"]

            async def safe_fetch_response(
                messages: list[ChatMessage], _temperature: float | None
            ) -> dict:
                if self.replace_error_with_message is not None:
                    try:
                        return await fetch_response(messages, _temperature)
                    except (aiohttp.ClientError, TimeoutError) as e:
                        logging.error(f"Network error during async generation: {e}")
                        return {
                            "role": "assistant",
                            "content": self.replace_error_with_message,
                        }
                    except APIError as e:
                        logging.error(f"API error during async generation: {e}")
                        return {
                            "role": "assistant",
                            "content": self.replace_error_with_message,
                        }
                else:
                    return await fetch_response(messages, _temperature)

            # gather all responses asynchronously, with concurrency limited to max_concurrency
            temperature_lst = (
                temperature
                if isinstance(temperature, list)
                else [temperature] * len(messages_lst)
            )
            return await asyncio.gather(
                *(
                    safe_fetch_response(messages, _temperature)
                    for messages, _temperature in zip(messages_lst, temperature_lst)
                )
            )

    def generate(
        self,
        messages_or_messages_lst: list[ChatMessage] | list[list[ChatMessage]],
        stop: str | None = None,
        max_tokens: int | None = None,
        temperature: float | list[float] | None = None,
        include_stop_str_in_output: bool | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | list[dict]:
        # Check if we have a single list of messages or a list of message lists
        # Single list: [{"role": "user", "content": "..."}] or [Message(...)]
        # Multiple lists: [[{"role": "user", "content": "..."}], [{"role": "user", "content": "..."}]]
        is_single = not isinstance(messages_or_messages_lst[0], list)
        messages_lst = (
            [messages_or_messages_lst] if is_single else messages_or_messages_lst
        )
        if self.is_async:
            loop = asyncio.get_event_loop()
            response_or_responses = loop.run_until_complete(
                self._generate(
                    messages_lst,
                    stop,
                    max_tokens,
                    temperature,
                    include_stop_str_in_output,
                    tools,
                    tool_choice,
                )
            )
        else:

            @backoff.on_exception(
                backoff.expo,
                RETRYABLE_ERRORS,
                max_tries=self.max_tries,
                on_backoff=enhanced_on_backoff,
                giveup=lambda e: not should_retry(e),
            )
            def fetch_single_response(
                messages: list[ChatMessage], _temperature: float | None
            ) -> dict:
                request_data = self._prepare_request_data(
                    messages,
                    stop,
                    max_tokens,
                    _temperature,
                    include_stop_str_in_output,
                    tools,
                    tool_choice,
                )

                response = requests.post(
                    self._chat_completion_endpoint,
                    headers=self.headers,
                    json=request_data,
                )

                if response.status_code != 200:
                    api_error = parse_api_error(response.status_code, response.text)
                    if not should_retry(api_error):
                        logging.error(format_non_retryable_error(api_error))
                    raise api_error

                response_json = response.json()
                # Return the full message object to preserve tool calls
                return response_json["choices"][0]["message"]

            def safe_fetch_single_response(
                messages: list[ChatMessage], _temperature: float | None
            ) -> dict:
                if self.replace_error_with_message is not None:
                    try:
                        return fetch_single_response(messages, _temperature)
                    except requests.RequestException as e:
                        logging.error(f"Network error during sync generation: {e}")
                        return {
                            "role": "assistant",
                            "content": self.replace_error_with_message,
                        }
                    except APIError as e:
                        logging.error(f"API error during sync generation: {e}")
                        return {
                            "role": "assistant",
                            "content": self.replace_error_with_message,
                        }
                else:
                    return fetch_single_response(messages, _temperature)

            temperature_lst = (
                temperature
                if isinstance(temperature, list)
                else [temperature] * len(messages_lst)
            )
            response_or_responses = [
                safe_fetch_single_response(messages, _temperature)
                for messages, _temperature in zip(messages_lst, temperature_lst)
            ]
        return response_or_responses[0] if is_single else response_or_responses

    # TODO implement evaluation
    def evaluate(self, prompt: str, generation: str) -> list[float]:
        raise NotImplementedError("evaluate method not implemented")


# TODO(GX) implement local VLLM-based language model
class LocalVLLMLanguageModel(AbstractLanguageModel):
    pass


# TODO implement transformers-based language model
class TransformersLanguageModel(AbstractLanguageModel):
    pass
