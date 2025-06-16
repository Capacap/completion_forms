from __future__ import annotations

import json
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

from litellm import completion
from .form import CompletionForm
from .exceptions import (
    ClientConfigurationError,
    MaxRetriesExceededError,
    ResponseParsingError,
)


@dataclass(frozen=True)
class CompletionClientSettings:
    """Settings for configuring the CompletionClient, with built-in validation."""
    model: str
    base_url: Optional[str] = None
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
        """
        Initializes the CompletionClient.

        Args:
            settings: An instance of CompletionClientSettings containing the
                      configuration for the client (e.g., model, API key, retry
                      settings).

        Raises:
            ClientConfigurationError: If the settings object is not a valid
                                      instance of CompletionClientSettings.
        """
        if not isinstance(settings, CompletionClientSettings):
            raise ClientConfigurationError("settings must be an instance of CompletionClientSettings.")
        self.settings = settings

    def complete(self, form: CompletionForm, stream_handler: Callable[[str], None] | None = None) -> Dict[str, Any]:
        """
        Executes a completion request using a formatted CompletionForm.

        This method sends the formatted request to the completion API, handles
        retries with exponential backoff, and processes the response. It supports
        both regular and streaming responses.

        Args:
            form: An instance of CompletionForm containing the payload
                     and response parser.
            stream_handler: An optional callable that receives content chunks as
                            they arrive when streaming. If provided, the response
                            will be streamed.

        Returns:
            A dictionary containing the parsed response from the API. If the
            response was a JSON object, it's parsed. If it was plain text, it's
            returned under the appropriate response key.

        Raises:
            TypeError: If request is not a CompletionRequest instance or if
                       stream_handler is not a callable.
            MaxRetriesExceededError: If the request fails after all configured
                                     retries have been attempted.
            ResponseParsingError: If the API response cannot be parsed (e.g.,
                                  invalid JSON).
        """
        if not isinstance(form, CompletionForm):
            raise TypeError("form must be an instance of CompletionForm.")
        if stream_handler is not None and not callable(stream_handler):
            raise TypeError("stream_handler must be a callable or None.")
        
        settings_dict = asdict(self.settings)
        # Pop keys not used by litellm.completion
        for key in ["max_retries", "backoff_base", "backoff_jitter"]:
            settings_dict.pop(key, None)

        completion_kwargs = {
            **settings_dict,
            "messages": form.get_messages(),
            "response_format": form.get_response_format(),
        }

        last_exception = None
        for attempt in range(self.settings.max_retries):
            try:
                if stream_handler:
                    completion_kwargs["stream"] = True
                    response_litellm = completion(**completion_kwargs)
                    
                    full_content = []
                    for chunk in response_litellm:
                        delta = chunk.choices[0].delta.content if chunk.choices[0].delta and chunk.choices[0].delta.content else ""
                        stream_handler(delta)
                        full_content.append(delta)
                    raw_content = "".join(full_content).strip()
                else:
                    response_litellm = completion(**completion_kwargs)
                    raw_content = response_litellm.choices[0].message.content.strip()

                return form.parse_response(raw_content)

            except Exception as e:
                last_exception = e
                if attempt < self.settings.max_retries - 1:
                    backoff_time = self.settings.backoff_base ** attempt
                    if self.settings.backoff_jitter:
                        backoff_time += random.uniform(0, 1)
                    print(f"Attempt {attempt + 1} failed. Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                else:
                    raise MaxRetriesExceededError("Completion failed after all retries.", last_exception)

        raise MaxRetriesExceededError("Completion failed unexpectedly after all retries.", last_exception) 