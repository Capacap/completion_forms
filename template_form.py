from __future__ import annotations

import json
import random
import re
import string
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from litellm import completion

@dataclass(frozen=True)
class CompletionClientSettings:
    model: str
    base_url: Optional[str] = None
    api_key: str = "empty"
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 60
    backoff_base: float = 2
    backoff_jitter: bool = True

class CompletionForm:
    __slots__ = ('_template', '_key_map', '_keys', '_data')

    # Public Methods
    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, Dict]]) -> 'CompletionForm':
        return cls(data)

    @classmethod
    def from_json_file(cls, file_path: str) -> 'CompletionForm':
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return cls(data)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at '{file_path}' was not found.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in '{file_path}': {e}")

    def __init__(self, template: Dict[str, Union[str, Dict]]):
        self._template: Dict[str, Union[str, Dict]]
        self._key_map: Dict[str, int]
        self._keys: List[str]
        self._data: Dict[str, str] = {}
        self._template, self._key_map, self._keys = self._parse_template(template)

    def put(self, key: str, value: str) -> None:
        if key not in self._keys:
            raise KeyError(f"Invalid key '{key}'. Valid keys are: {self._keys}")
        self._data[key] = value

    def format(self) -> tuple[List[Dict[str, str]], Dict | None, Dict | None]:
        self._validate_data()
        messages = self._build_messages()
        raw_response_template = self._template.get("response", {})
        response_format = self._build_response_format()
        return messages, response_format, raw_response_template

    @property
    def keys(self) -> list[str]:
        return self._keys

    # Internal/Helper Methods
    def _parse_template(self, template: Dict[str, Union[str, Dict]]) -> tuple[dict[str, Union[str, Dict]], dict[str, int], list[str]]:
        parts: dict[str, Union[str, Dict]] = {}
        key_map: dict[str, int] = {}
        keys: list[str] = []
        
        for role, content in template.items():
            if role == "thinking":
                raise ValueError("The key 'thinking' is reserved for internal use and cannot be used as a top-level role.")

            parts[role] = content
            if role == "response":
                if isinstance(content, dict):
                    for name, details in content.items():
                        if name == "thinking":
                            raise ValueError("The key 'thinking' is reserved for internal use and cannot be used in the response schema.")
                        if isinstance(details, dict) and details.get("type") == "text":
                            if len(content) > 1:
                                raise ValueError("Text response type must be the only key in the response template.")
                continue

            if isinstance(content, str):
                formatter = string.Formatter()
                for _, field_name, _, _ in formatter.parse(content):
                    if field_name is not None and field_name not in key_map:
                        key_map[field_name] = len(keys)
                        keys.append(field_name)
        
        return parts, key_map, keys

    def _validate_data(self) -> None:
        if len(self._data) != len(self._keys):
            provided = set(self._data)
            required = set(self._keys)
            
            missing = sorted(list(required - provided))
            extra = sorted(list(provided - required))
            
            errors = [
                f"missing keys: {missing}" if missing else "",
                f"unexpected keys: {extra}" if extra else "",
            ]
            
            error_message = "; ".join(filter(None, errors))
            raise TypeError(f"format() got invalid arguments; {error_message}")

    def _build_messages(self) -> List[Dict[str, str]]:
        messages = []
        for role, template in self._template.items():
            if role == "response":
                continue
            if not isinstance(template, str):
                continue
                
            content = template.format(**self._data)
            
            messages.append({
                "role": role,
                "content": content.strip()
            })
        return messages

    def _build_response_format(self) -> Dict | None:
        response_data = self._template.get("response", {})
        if not response_data:
            return None

        properties = {
            name: {"type": details["type"], **({"description": details["description"]} if "description" in details else {})}
            for name, details in response_data.items()
            if name != "thinking" and isinstance(details, dict) and "type" in details and details["type"] != "text"
        }
        required = sorted(list(set(
            name
            for name, details in response_data.items()
            if name != "thinking" and isinstance(details, dict) and "type" in details and details["type"] != "text"
        )))

        if not properties:
            return None
       
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
            "strict": True
        }

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

def _test_json_completion(client: CompletionClient):
    json_template_data = {
        "system": (
            "You are a helpful assistant.\n"
            "\n"
            "Instructions:\n"
            "{instructions}"
            "\n"
            "Requirements:\n"
            "{requirements}"
        ),
        "user": ("I have a problem: {problem}"),
        "response": {
            "solution": {"type": "string", "description": "Your solution."}
        }
    }
    json_form = CompletionForm(json_template_data)
    json_form.put("instructions", "1. Analyze the problem.\n2. Solve the problem.\n")
    json_form.put("requirements", "- Response MUST be below 1024 tokens.\n")
    json_form.put("problem", "There is a wild animal in my livingroom.")
    messages, response_format, _ = json_form.format()
    response = client.complete(json_form)
    
    print("--- JSON ---")
    print("\n--- SYSTEM ---")
    print(messages[0]["content"])
    print("\n--- USER ---")
    print(messages[1]["content"])
    print("\n---FORMAT---")
    print(response_format)
    print("\n---RESPONSE---")
    print(response["solution"].strip())

def _test_text_completion(client: CompletionClient):
    text_template_data = {
        "system": (
            "You are a helpful assistant.\n"
            "\n"
            "Instructions:\n"
            "{instructions}"
            "\n"
            "Requirements:\n"
            "{requirements}"
        ),
        "user": ("I have a problem: {problem}"),
        "response": {"summary": {"type": "text"}}
    }
    text_form = CompletionForm(text_template_data)
    text_form.put("instructions", "1. Analyze the problem.\n2. Solve the problem.\n")
    text_form.put("requirements", "- Response MUST be below 1024 tokens.\n")
    text_form.put("problem", "There is a wild animal in my livingroom.")
    messages, response_format, _ = text_form.format()
    response = client.complete(text_form)
 
    print("\n--- TEXT ---")
    print("\n--- SYSTEM ---")
    print(messages[0]["content"])
    print("\n--- USER ---")
    print(messages[1]["content"])
    print("\n---FORMAT---")
    print(response_format)
    print("\n---RESPONSE---")
    print(response["summary"].strip())

def _test_form_validation():
    json_template_data = {
        "system": "You are a helpful assistant. Instructions: {instructions}",
        "user": "I have a problem: {problem}",
        "response": {"solution": {"type": "string", "description": "Your solution."}}
    }
    # Test key validation
    try:
        form = CompletionForm(json_template_data)
        form.put("instructions", "test")
        form.format()
    except Exception as e:
        print(f"\nCaught expected error for missing keys: {e}")

    try:
        form = CompletionForm(json_template_data)
        form.put("instructions", "test")
        form.put("requirements", "test")
        form.put("problem", "test")
        form.put("extra", "bad")
        form.format()
    except Exception as e:
        print(f"Caught expected error for extra keys: {e}")

    # Test duplicate key in template
    try:
        CompletionForm({"system": "Hello {name}, and again, {name}"})
    except Exception as e:
        print(f"Caught expected error for duplicate keys: {e}")

    # Test text response with multiple keys
    try:
        CompletionForm({
            "system": "Hello",
            "response": {
                "summary": {"type": "text"},
                "extra": {"type": "string"}
            }
        })
    except ValueError as e:
        print(f"\nCaught expected error for text response with multiple keys: {e}")

    # Test reserved 'thinking' key in response schema
    try:
        CompletionForm({
            "system": "Test",
            "response": {"thinking": {"type": "string", "description": "Reserved thinking key."}}
        })
    except ValueError as e:
        print(f"\nCaught expected error for reserved 'thinking' key in response schema: {e}")

    # Test reserved 'thinking' key as top-level role
    try:
        CompletionForm({"thinking": "This is a reserved role."})
    except ValueError as e:
        print(f"\nCaught expected error for reserved 'thinking' key as top-level role: {e}")

def _test_streaming_completion(client: CompletionClient):
    text_template_data = {
        "system": (
            "You are a helpful assistant.\n"
            "\n"
            "Instructions:\n"
            "{instructions}"
            "\n"
            "Requirements:\n"
            "{requirements}"
        ),
        "user": ("I have a problem: {problem}"),
        "response": {"summary": {"type": "text"}}
    }
    print("\n--- STREAMING TEXT RESPONSE ---")
    streaming_text_form = CompletionForm(text_template_data)
    streaming_text_form.put("instructions", "1. Analyze the problem.\n2. Solve the problem.\n")
    streaming_text_form.put("requirements", "- Response MUST be below 1024 tokens.\n")
    streaming_text_form.put("problem", "There is a wild animal in my livingroom.")

    print("Streaming content: ")
    streamed_response = client.complete(streaming_text_form, lambda chunk: print(chunk, end="", flush=True))
    print("\nFull streamed response:", streamed_response.get("summary", "").strip())
    if "thinking" in streamed_response:
        print("Full streamed response (thinking):", streamed_response["thinking"].strip())

    assert "summary" in streamed_response and len(streamed_response["summary"]) > 0
    assert "thinking" in streamed_response and len(streamed_response["thinking"]) > 0
    print("Streaming text response test passed.")

def _test_factory_methods():
    dict_data = {"test_role": "This is a test: {test_field}"}
    dict_form = CompletionForm.from_dict(dict_data)
    dict_form.put("test_field", "Hello from dict!")
    messages, _, _ = dict_form.format()
    assert len(messages) == 1 and messages[0]["content"] == "This is a test: Hello from dict!"
    print("\nfrom_dict class method test passed.")

    test_json_file = "test_template.json"
    try:
        with open(test_json_file, 'w') as f:
            json.dump(dict_data, f)
        json_file_form = CompletionForm.from_json_file(test_json_file)
        json_file_form.put("test_field", "Hello from json file!")
        messages, _, _ = json_file_form.format()
        assert len(messages) == 1 and messages[0]["content"] == "This is a test: Hello from json file!"
        print("from_json_file class method test passed.")
    finally:
        import os
        if os.path.exists(test_json_file):
            os.remove(test_json_file)

def main():
    settings = CompletionClientSettings(
        model="openai/grayline-qwen3-8b",
        base_url="http://0.0.0.0:1234/v1"
    )
    client = CompletionClient(settings)
    
    _test_json_completion(client)
    _test_text_completion(client)
    _test_form_validation()
    _test_streaming_completion(client)
    _test_factory_methods()

if __name__ == "__main__":
    main()