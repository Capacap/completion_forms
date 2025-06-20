"""A resilient client for making requests to an OpenAI-compatible API."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import httpx

from .exceptions import (
    ClientConfigurationError,
    MaxRetriesExceededError,
    ResponseParsingError,
)
from .form import CompletionForm


@dataclass(frozen=True)
class CompletionClientSettings:
    """Settings for configuring the CompletionClient.

    Attributes:
        model: The model name to use for the completion.
        api_key: The API key for authentication. Can be None for local servers.
        base_url: The base URL of the API. Defaults to OpenAI's API.
        endpoint: The endpoint for chat completions.
        max_retries: The maximum number of times to retry a failed request.
        temperature: The sampling temperature for the completion.
        max_tokens: The maximum number of tokens to generate.
        top_p: The nucleus sampling probability.
        frequency_penalty: The penalty for repeating tokens.
        presence_penalty: The penalty for introducing new tokens.
        timeout: The request timeout in seconds.
        backoff_base: The base for the exponential backoff delay.
        backoff_jitter: Whether to add a random jitter to the backoff delay.
    """
    model: str
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    endpoint: str = "/chat/completions"
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 60
    backoff_base: float = 2
    backoff_jitter: bool = True

    def __post_init__(self):
        """Performs validation on settings after initialization."""
        if not isinstance(self.model, str) or not self.model:
            raise ClientConfigurationError("model must be a non-empty string.")
        if not isinstance(self.base_url, str) or not self.base_url:
            raise ClientConfigurationError("base_url must be a non-empty string.")
        if not isinstance(self.endpoint, str) or not self.endpoint:
            raise ClientConfigurationError("endpoint must be a non-empty string.")
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ClientConfigurationError(
                "api_key must be a string if provided."
            )
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ClientConfigurationError(
                "max_retries must be a non-negative integer."
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise ClientConfigurationError(
                "temperature must be a float between 0.0 and 2.0."
            )
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ClientConfigurationError(
                "max_tokens must be a positive integer."
            )
        if not (0.0 <= self.top_p <= 1.0):
            raise ClientConfigurationError(
                "top_p must be a float between 0.0 and 1.0."
            )
        if not (-2.0 <= self.frequency_penalty <= 2.0):
            raise ClientConfigurationError(
                "frequency_penalty must be a float between -2.0 and 2.0."
            )
        if not (-2.0 <= self.presence_penalty <= 2.0):
            raise ClientConfigurationError(
                "presence_penalty must be a float between -2.0 and 2.0."
            )
        if not isinstance(self.timeout, (int, float)) or self.timeout < 0:
            raise ClientConfigurationError("timeout must be a non-negative float.")
        if not isinstance(self.backoff_base, (int, float)) or self.backoff_base < 1:
            raise ClientConfigurationError(
                "backoff_base must be a float greater than or equal to 1."
            )


class CompletionClient:
    """A client for making requests to a completion API, with retry logic."""

    def __init__(self, settings: CompletionClientSettings):
        """Initializes the CompletionClient.

        Args:
            settings: A CompletionClientSettings object containing the
                configuration for the client.

        Raises:
            ClientConfigurationError: If the settings object is not a valid
                instance of CompletionClientSettings.
        """
        if not isinstance(settings, CompletionClientSettings):
            raise ClientConfigurationError(
                "settings must be an instance of CompletionClientSettings."
            )
        self.settings = settings

        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        self._client = httpx.Client(
            base_url=self.settings.base_url,
            headers=headers,
            timeout=self.settings.timeout,
        )

    def complete(
        self,
        form: CompletionForm,
        stream_handler: Callable[[str], None] | None = None
    ) -> Dict[str, Any]:
        """Executes a completion request using a formatted CompletionForm.

        This method sends the formatted request to the completion API, handles
        retries with exponential backoff, and processes the response. It
        supports both regular and streaming responses.

        Args:
            form: An instance of CompletionForm containing the payload.
            stream_handler: An optional callable that receives content chunks as
                they arrive when streaming.

        Returns:
            A dictionary containing the parsed response from the API.

        Raises:
            MaxRetriesExceededError: If the request fails after all configured
                retries have been attempted.
            TypeError: If form is not a CompletionForm instance or if
                stream_handler is not a callable.
        """
        if not isinstance(form, CompletionForm):
            raise TypeError("form must be an instance of CompletionForm.")
        if stream_handler is not None and not callable(stream_handler):
            raise TypeError("stream_handler must be a callable or None.")

        payload = self._build_request_payload(form, stream=bool(stream_handler))

        last_exception = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                if stream_handler:
                    return self._stream_completion(payload, stream_handler, form)
                else:
                    return self._standard_completion(payload, form)
            except httpx.HTTPStatusError as e:
                last_exception = e
                error_text = e.response.text if e.response.is_closed else ""
                print(
                    f"API request failed with status {e.response.status_code}:"
                    f" {error_text}"
                )
            except Exception as e:
                last_exception = e
                print(f"An unexpected error occurred: {e}")

            if attempt < self.settings.max_retries:
                backoff_time = self.settings.backoff_base**attempt
                if self.settings.backoff_jitter:
                    backoff_time += random.uniform(0, 1)
                print(
                    f"Attempt {attempt + 1}/{self.settings.max_retries + 1} failed. "
                    f"Retrying in {backoff_time:.2f} seconds..."
                )
                time.sleep(backoff_time)

        raise MaxRetriesExceededError(
            "Completion failed after all retries.", last_exception
        )

    def _stream_completion(
        self,
        payload: Dict[str, Any],
        stream_handler: Callable[[str], None],
        form: CompletionForm
    ) -> Dict[str, Any]:
        """Handles the logic for a streaming completion."""
        full_content = []
        with self._client.stream(
            "POST", self.settings.endpoint, json=payload
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data:"):
                    content = line[len("data: ") :].strip()
                    if content == "[DONE]":
                        break
                    chunk = json.loads(content)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        stream_handler(delta)
                        full_content.append(delta)
        raw_content = "".join(full_content)
        return form.parse_response(raw_content)

    def _standard_completion(
        self, payload: Dict[str, Any], form: CompletionForm
    ) -> Dict[str, Any]:
        """Handles the logic for a standard (non-streaming) completion."""
        response = self._client.post(self.settings.endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        raw_content = data["choices"][0]["message"]["content"]
        return form.parse_response(raw_content)

    def _build_request_payload(
        self, form: CompletionForm, stream: bool
    ) -> Dict[str, Any]:
        """Builds the request payload dictionary."""
        payload = {
            "model": self.settings.model,
            "messages": form.get_messages(),
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "top_p": self.settings.top_p,
            "frequency_penalty": self.settings.frequency_penalty,
            "presence_penalty": self.settings.presence_penalty,
            "stream": stream,
        }

        response_format = form.get_response_format()
        if response_format:
            payload["response_format"] = {"type": "json_object"}

        return payload

    def close(self) -> None:
        """Closes the underlying httpx client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 