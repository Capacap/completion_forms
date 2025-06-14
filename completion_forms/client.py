from __future__ import annotations

import json
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

from litellm import completion
from .form import CompletionForm


@dataclass(frozen=True)
class CompletionClientSettings:
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


class CompletionClient:
    def __init__(self, settings: CompletionClientSettings):
        self.settings = settings

    def complete(self, completion_form: CompletionForm, stream_handler: Callable[[str], None] | None = None) -> Dict[str, Any]:
        messages, response_format, raw_response_template = completion_form.format()
        
        settings_dict = asdict(self.settings)
        # Pop keys not used by litellm.completion
        for key in ["max_retries", "backoff_base", "backoff_jitter"]:
            settings_dict.pop(key, None)

        completion_kwargs = {
            **settings_dict,
            "messages": messages,
            "response_format": response_format,
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

                if response_format is None:
                    parsed_response = self._parse_text_response(raw_content, raw_response_template)
                else:
                    try:
                        parsed_response = json.loads(raw_content)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Failed to decode JSON response: {e}. Raw content: {raw_content}")
                
                return parsed_response

            except Exception as e:
                last_exception = e
                if attempt < self.settings.max_retries - 1:
                    backoff_time = self.settings.backoff_base ** attempt
                    if self.settings.backoff_jitter:
                        backoff_time += random.uniform(0, 1)
                    print(f"Attempt {attempt + 1} failed. Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                else:
                    raise

        raise last_exception if last_exception else RuntimeError("Completion failed after all retries.")

    # Internal/Helper Methods
    def _parse_text_response(self, raw_content: str, raw_response_template: Dict) -> Dict[str, Any]:
        response_key = next(iter(raw_response_template.keys()), None)
        if not response_key:
            raise ValueError("Text response template missing a key.")

        # Regex to find content between <think> and </think>
        thinking_match = re.search(r"<think>(.*?)</think>(.*)", raw_content, re.DOTALL)

        parsed_response = {}
        if thinking_match:
            parsed_response["thinking"] = thinking_match.group(1).strip()
            parsed_response[response_key] = thinking_match.group(2).strip()
        else:
            parsed_response[response_key] = raw_content.strip()
        return parsed_response 