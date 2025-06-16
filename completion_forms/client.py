from __future__ import annotations

import json
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional
import httpx

from .form import CompletionForm
from .exceptions import (
    ClientConfigurationError,
    MaxRetriesExceededError,
    ResponseParsingError,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_COMPLETION_ENDPOINT = "/chat/completions"


@dataclass(frozen=True)
class CompletionClientSettings:
    """Settings for configuring the CompletionClient, with built-in validation."""
    model: str
    base_url: str = DEFAULT_BASE_URL
    api_key: Optional[str] = None
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
        if self.base_url is not None and (not isinstance(self.base_url, str) or not self.base_url):
            raise ClientConfigurationError("base_url must be a non-empty string if provided.")
        if self.api_key is not None and (not isinstance(self.api_key, str) or not self.api_key):
            raise ClientConfigurationError("api_key must be a non-empty string if provided.")
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ClientConfigurationError("max_retries must be a non-negative integer.")
        if not isinstance(self.temperature, (int, float)) or not (0.0 <= self.temperature <= 2.0):
            raise ClientConfigurationError("temperature must be a float between 0.0 and 2.0.")
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ClientConfigurationError("max_tokens must be a positive integer.")
        if not isinstance(self.top_p, (int, float)) or not (0.0 <= self.top_p <= 1.0):
            raise ClientConfigurationError("top_p must be a float between 0.0 and 1.0.")
        if not isinstance(self.frequency_penalty, (int, float)) or not (-2.0 <= self.frequency_penalty <= 2.0):
            raise ClientConfigurationError("frequency_penalty must be a float between -2.0 and 2.0.")
        if not isinstance(self.presence_penalty, (int, float)) or not (-2.0 <= self.presence_penalty <= 2.0):
            raise ClientConfigurationError("presence_penalty must be a float between -2.0 and 2.0.")
        if not isinstance(self.timeout, (int, float)) or self.timeout < 0:
            raise ClientConfigurationError("timeout must be a non-negative float.")
        if not isinstance(self.backoff_base, (int, float)) or self.backoff_base < 1:
            raise ClientConfigurationError("backoff_base must be a float greater than or equal to 1.")


class CompletionClient:
    """A client for making requests to a completion API, with retry logic."""
    def __init__(self, settings: CompletionClientSettings):
        """Initializes the CompletionClient."""
        if not isinstance(settings, CompletionClientSettings):
            raise ClientConfigurationError("settings must be an instance of CompletionClientSettings.")
        self.settings = settings
        self._client = httpx.Client(
            base_url=self.settings.base_url,
            headers={"Authorization": f"Bearer {self.settings.api_key}"},
            timeout=self.settings.timeout,
        )

    def complete(self, form: CompletionForm, stream_handler: Callable[[str], None] | None = None) -> Dict[str, Any]:
        """Executes a completion request using a formatted CompletionForm."""
        if not isinstance(form, CompletionForm):
            raise TypeError("form must be an instance of CompletionForm.")
        if stream_handler is not None and not callable(stream_handler):
            raise TypeError("stream_handler must be a callable or None.")

        payload = self._build_request_payload(form, stream=bool(stream_handler))
        
        last_exception = None
        for attempt in range(self.settings.max_retries):
            try:
                if stream_handler:
                    full_content = []
                    with self._client.stream("POST", DEFAULT_COMPLETION_ENDPOINT, json=payload) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if line.startswith("data:"):
                                content = line[len("data: "):].strip()
                                if content == "[DONE]":
                                    break
                                chunk = json.loads(content)
                                delta = chunk["choices"][0]["delta"].get("content", "")
                                stream_handler(delta)
                                full_content.append(delta)
                    raw_content = "".join(full_content)
                else:
                    response = self._client.post(DEFAULT_COMPLETION_ENDPOINT, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    raw_content = data["choices"][0]["message"]["content"]
                
                return form.parse_response(raw_content)

            except httpx.HTTPStatusError as e:
                last_exception = e
                # For streaming responses, content may not have been read.
                error_text = e.response.text if not e.response.is_stream_consumed else ""
                print(f"API request failed with status {e.response.status_code}: {error_text}")
            except Exception as e:
                last_exception = e
                print(f"An unexpected error occurred: {e}")

            if attempt < self.settings.max_retries - 1:
                backoff_time = self.settings.backoff_base ** attempt
                if self.settings.backoff_jitter:
                    backoff_time += random.uniform(0, 1)
                print(f"Attempt {attempt + 1}/{self.settings.max_retries} failed. Retrying in {backoff_time:.2f} seconds...")
                time.sleep(backoff_time)

        raise MaxRetriesExceededError("Completion failed after all retries.", last_exception)

    def _build_request_payload(self, form: CompletionForm, stream: bool) -> Dict[str, Any]:
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
            # OpenAI expects `response_format` to be {"type": "json_object"}
            # for schema-enforced JSON. The schema itself goes in the system prompt.
            # `CompletionForm` already handles injecting the schema.
            payload["response_format"] = {"type": "json_object"}

        return payload

    def close(self) -> None:
        """Closes the underlying httpx client."""
        self._client.close()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 