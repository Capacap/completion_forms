from __future__ import annotations

import json
import pprint
import re
import string
from typing import Any, Dict, List, Union

from .exceptions import (
    FormFileNotFoundError,
    FormValidationError,
    InvalidJSONError,
    InvalidKeyError,
    InvalidTemplateError,
    InvalidValueError,
    ReservedKeyError,
    ResponseParsingError,
)


class CompletionForm:
    """A class to define, validate, and format structured completion requests."""

    __slots__ = ("_template", "_key_map", "_keys", "_data")
    VALID_RESPONSE_TYPES = {"string", "number", "integer", "object", "array", "boolean", "text"}

    def __init__(self, template: Dict[str, Union[str, Dict]]):
        """
        Initializes a new CompletionForm instance.

        Args:
            template: A dictionary representing the completion template. It must
                      contain 'user' and 'response' keys. Placeholders for data
                      should be specified using standard Python string formatting
                      syntax (e.g., '{key}').

        Raises:
            InvalidTemplateError: If the template is invalid (e.g., not a
                                  non-empty dictionary, missing required keys,
                                  or structurally incorrect).
            ReservedKeyError: If a reserved key like 'thinking' is used improperly.
        """
        if not isinstance(template, dict) or not template:
            raise InvalidTemplateError("CompletionForm template must be a non-empty dictionary.")
        self._template: Dict[str, Union[str, Dict]]
        self._key_map: Dict[str, int]
        self._keys: List[str]
        self._data: Dict[str, str] = {}
        self._template, self._key_map, self._keys = self._parse_template(template)

    def __repr__(self) -> str:
        """Provides an unambiguous string representation of the CompletionForm."""
        return f"CompletionForm(template={pprint.pformat(self._template)})"

    # ------------------
    # Properties
    # ------------------
    @property
    def keys(self) -> list[str]:
        """Returns a list of all placeholder keys defined in the template."""
        return self._keys

    # ------------------
    # Alternative Constructors
    # ------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, Dict]]) -> "CompletionForm":
        """
        Creates a CompletionForm instance from a dictionary.

        This class method provides a convenient way to instantiate a CompletionForm
        directly from a Python dictionary that represents the form template.

        Args:
            data: A dictionary containing the form template. The dictionary must
                  not be empty and should conform to the expected template structure.

        Returns:
            A new instance of CompletionForm.

        Raises:
            InvalidTemplateError: If the provided data is not a non-empty dictionary.
        """
        if not isinstance(data, dict) or not data:
            raise InvalidTemplateError("from_dict expects a non-empty dictionary.")
        return cls(data)

    @classmethod
    def from_json_file(cls, file_path: str) -> "CompletionForm":
        """
        Creates a CompletionForm instance from a JSON file.

        This class method reads a JSON file from the specified path, parses it,
        and uses the resulting dictionary to create a CompletionForm instance.

        Args:
            file_path: The path to the JSON file. Must be a non-empty string.

        Returns:
            A new instance of CompletionForm.

        Raises:
            InvalidValueError: If the file_path is not a non-empty string.
            FormFileNotFoundError: If the file at the specified path does not exist.
            InvalidJSONError: If the file contains malformed JSON.
        """
        if not isinstance(file_path, str) or not file_path:
            raise InvalidValueError("from_json_file expects a non-empty file path string.")
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return cls(data)
        except FileNotFoundError:
            raise FormFileNotFoundError(f"The file at '{file_path}' was not found.")
        except json.JSONDecodeError as e:
            raise InvalidJSONError(f"Invalid JSON in '{file_path}': {e}")

    # ------------------
    # Core Public API
    # ------------------
    def put(self, key: str, value: str) -> None:
        """
        Populates the form with a key-value pair.

        This method is used to provide the data that will be used to fill in
        the placeholders in the template.

        Args:
            key: The name of the placeholder in the template. Must be a non-empty string.
            value: The string value to substitute for the placeholder.

        Raises:
            InvalidKeyError: If the key is not a non-empty string or is not a
                             valid key defined in the template.
            InvalidValueError: If the value is not a string.
        """
        if not isinstance(key, str) or not key:
            raise InvalidKeyError("Key must be a non-empty string.")
        if not isinstance(value, str):
            raise InvalidValueError("Value must be a string.")
        if key not in self._keys:
            raise InvalidKeyError(f"Invalid key '{key}'. Valid keys are: {self._keys}")
        self._data[key] = value

    def get_messages(self) -> List[Dict[str, str]]:
        """Validates the form data and returns the list of messages for the API."""
        self._validate_data()
        return self._build_messages()

    def get_response_format(self) -> Dict | None:
        """Returns the response_format dictionary for the API."""
        return self._build_response_format()

    def get_response_schema(self) -> Dict:
        """Returns the response schema part of the original template."""
        return self._template.get("response", {})

    def get_messages_schema(self) -> Dict[str, str]:
        """Returns the messages schema part of the original template."""
        return {
            role: content
            for role, content in self._template.items()
            if role != "response" and isinstance(content, str)
        }

    def parse_response(self, raw_content: str) -> Dict[str, Any]:
        """
        Parses the raw string response from the API.

        This method intelligently handles both JSON and plain text responses
        based on the form's configuration.

        Args:
            raw_content: The raw string content returned by the API.

        Returns:
            A dictionary containing the parsed data.

        Raises:
            ResponseParsingError: If parsing fails for any reason.
        """
        is_text_response = self.get_response_format() is None
        if is_text_response:
            return self._parse_text_response(raw_content)

        try:
            return json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise ResponseParsingError(f"Failed to decode JSON response: {e}", raw_content)

    # ------------------
    # Public Utility Methods
    # ------------------
    def pprint_messages(self, raw: bool = False) -> None:
        """
        Displays the raw or formatted messages.

        Args:
            raw: If True, displays the raw, unformatted message templates.
                 If False (default), displays the messages after populating
                 the form with data.
        """
        if raw:
            print("--- Raw Messages ---")
            for role, content in self.get_messages_schema().items():
                print(f"[{role.capitalize()}]")
                print(content)
                print()
        else:
            self._validate_data()
            print("--- Formatted Messages ---")
            for message in self._build_messages():
                print(f"[{message['role'].capitalize()}]")
                print(message["content"])
                print()

    def pprint_response_format(self) -> None:
        """
        Displays the generated response format for the API.
        """
        print("--- Response Format ---")
        response_format = self._build_response_format()
        if response_format:
            pprint.pprint(response_format)
        else:
            # For text responses, show the schema which just contains the response key
            pprint.pprint(self.get_response_schema())
        print()

    # ------------------
    # Internal Methods
    # ------------------
    def _parse_template(self, template: Dict[str, Union[str, Dict]]) -> tuple[dict, dict, list]:
        """Parses the template to extract roles, content, and placeholder keys."""
        if "user" not in template:
            raise InvalidTemplateError("Template must include a 'user' key.")
        if "response" not in template:
            raise InvalidTemplateError("Template must include a 'response' key.")

        parts: dict[str, Union[str, Dict]] = {}
        key_map: dict[str, int] = {}
        keys: list[str] = []

        for role, content in template.items():
            if not isinstance(role, str) or not role:
                raise InvalidTemplateError("Template keys (roles) must be non-empty strings.")
            if not isinstance(content, (str, dict)):
                raise InvalidTemplateError(f"Content for role '{role}' must be a string or dict.")

            if role == "thinking":
                raise ReservedKeyError("The 'thinking' key is reserved and cannot be a role.")

            parts[role] = content
            if role == "response":
                if not isinstance(content, dict) or not content:
                    raise InvalidTemplateError("The 'response' must be a non-empty dictionary.")

                is_text_response = any(
                    isinstance(v, dict) and v.get("type") == "text" for v in content.values()
                )
                if is_text_response and len(content) > 1:
                    raise InvalidTemplateError("A 'text' response cannot be mixed with others.")

                for name, details in content.items():
                    if name == "thinking":
                        raise ReservedKeyError("'thinking' is reserved in the response schema.")
                    self._validate_response_schema(details, f"response.{name}")
                continue

            if isinstance(content, str):
                formatter = string.Formatter()
                for _, field, _, _ in formatter.parse(content):
                    if field is not None and field not in key_map:
                        key_map[field] = len(keys)
                        keys.append(field)

        return parts, key_map, keys

    def _validate_response_schema(self, schema_node: dict, path: str):
        """Recursively validates the response schema structure and types."""
        if not isinstance(schema_node, dict):
            raise InvalidTemplateError(f"The schema at '{path}' must be a dictionary.")
        if "type" not in schema_node:
            raise InvalidTemplateError(f"The schema at '{path}' must include a 'type' key.")

        response_type = schema_node["type"]
        if not isinstance(response_type, str) or not response_type:
            raise InvalidTemplateError(f"The 'type' at '{path}' must be a non-empty string.")
        if response_type not in self.VALID_RESPONSE_TYPES:
            raise InvalidTemplateError(
                f"Invalid response type '{response_type}' at '{path}'. Valid types are: {self.VALID_RESPONSE_TYPES}"
            )

        if response_type == "object":
            if "properties" not in schema_node or not isinstance(schema_node["properties"], dict):
                raise InvalidTemplateError(f"Object schema '{path}' needs 'properties' dict.")
            for name, prop in schema_node["properties"].items():
                self._validate_response_schema(prop, f"{path}.properties.{name}")
            if "required" in schema_node and not isinstance(schema_node["required"], list):
                raise InvalidTemplateError(f"'required' at '{path}' must be a list of strings.")

        if response_type == "array":
            if "items" not in schema_node or not isinstance(schema_node["items"], dict):
                raise InvalidTemplateError(f"Array schema '{path}' needs an 'items' dict.")
            self._validate_response_schema(schema_node["items"], f"{path}.items")

    def _validate_data(self) -> None:
        """Ensures all required data keys are provided and no extra keys exist."""
        provided, required = set(self._data), set(self._keys)
        if provided == required:
            return

        missing = sorted(list(required - provided))
        extra = sorted(list(provided - required))
        errors = [
            f"missing keys: {missing}" if missing else "",
            f"unexpected keys: {extra}" if extra else "",
        ]
        error_message = "; ".join(filter(None, errors))
        raise FormValidationError(f"format() got invalid arguments; {error_message}")

    def _build_messages(self) -> List[Dict[str, str]]:
        """Builds the list of message dictionaries by formatting template strings."""
        messages = []
        for role, template in self.get_messages_schema().items():
            content = template.format(**self._data)
            messages.append({"role": role, "content": content.strip()})
        return messages

    def _build_response_format(self) -> Dict | None:
        """Constructs the JSON schema for the response format if not a text response."""
        response_data = self.get_response_schema()
        if not response_data or any(
            isinstance(d, dict) and d.get("type") == "text" for d in response_data.values()
        ):
            return None

        properties = {
            name: self._build_properties_recursively(details)
            for name, details in response_data.items()
            if name != "thinking" and isinstance(details, dict)
        }

        if not properties:
            return None

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": sorted(properties.keys()),
                },
            },
        }

    def _build_properties_recursively(self, schema_part: dict) -> dict:
        """Recursively builds the JSON schema for nested properties."""
        processed_schema = {
            k: v for k, v in schema_part.items() if k in ("type", "description")
        }

        if schema_part.get("type") == "object" and "properties" in schema_part:
            processed_schema["properties"] = {
                name: self._build_properties_recursively(prop)
                for name, prop in schema_part["properties"].items()
            }
            if "required" in schema_part:
                processed_schema["required"] = sorted(list(set(schema_part["required"])))

        if schema_part.get("type") == "array" and "items" in schema_part:
            processed_schema["items"] = self._build_properties_recursively(schema_part["items"])

        return processed_schema

    def _parse_text_response(self, raw_content: str) -> Dict[str, Any]:
        """Parses a plain text response, extracting thinking and content parts."""
        response_schema = self.get_response_schema()
        if not response_schema:
            raise ResponseParsingError("Cannot parse text response without a template.", raw_content)

        response_key = next(iter(response_schema.keys()), None)
        if not response_key:
            raise ResponseParsingError("Text response template has no key.", raw_content)

        match = re.search(r"<think>(.*?)</think>(.*)", raw_content, re.DOTALL)
        if match:
            return {"thinking": match.group(1).strip(), response_key: match.group(2).strip()}
        
        return {response_key: raw_content.strip()} 