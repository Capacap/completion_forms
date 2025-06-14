from __future__ import annotations

import json
import string
from typing import Dict, List, Union


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
            # "strict": True
        } 