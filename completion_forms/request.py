from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .exceptions import ResponseParsingError


class CompletionRequest:
    """
    A self-contained, ready-to-send request object for the completion API.

    This class encapsulates both the data to be sent to the API (messages and
    response_format) and the logic required to parse the corresponding raw
    response from the API.
    """
    __slots__ = ('messages', 'response_format', '_is_text_response', '_response_template')

    def __init__(
        self,
        messages: List[Dict[str, str]],
        response_format: Dict | None,
        is_text_response: bool,
        response_template: Dict | None
    ):
        """
        Initializes the CompletionRequest.

        Args:
            messages: The list of message dictionaries for the API.
            response_format: The JSON schema for the response, or None for text.
            is_text_response: A flag indicating if the response is plain text.
            response_template: The raw response template, used for text parsing.
        """
        self.messages = messages
        self.response_format = response_format
        self._is_text_response = is_text_response
        self._response_template = response_template

    def parse_response(self, raw_content: str) -> Dict[str, Any]:
        """

        Parses the raw string response from the API.

        This method intelligently handles both JSON and plain text responses
        based on how the request was initially configured.

        Args:
            raw_content: The raw string content returned by the API.

        Returns:
            A dictionary containing the parsed data.

        Raises:
            ResponseParsingError: If parsing fails for any reason (e.g.,
                                  invalid JSON, missing text response key).
        """
        if self._is_text_response:
            return self._parse_text_response(raw_content)
        
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise ResponseParsingError(f"Failed to decode JSON response: {e}", raw_content)

    def _parse_text_response(self, raw_content: str) -> Dict[str, Any]:
        """Parses a plain text response, extracting thinking and content parts."""
        if not self._response_template:
            raise ResponseParsingError("Cannot parse text response without a response template.", raw_content)
            
        response_key = next(iter(self._response_template.keys()), None)
        if not response_key:
            raise ResponseParsingError("Text response template missing a key.", raw_content)

        thinking_match = re.search(r"<think>(.*?)</think>(.*)", raw_content, re.DOTALL)

        parsed_response = {}
        if thinking_match:
            parsed_response["thinking"] = thinking_match.group(1).strip()
            parsed_response[response_key] = thinking_match.group(2).strip()
        else:
            parsed_response[response_key] = raw_content.strip()
        return parsed_response 