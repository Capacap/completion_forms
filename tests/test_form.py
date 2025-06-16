"""Tests for the CompletionForm class."""

import os
import pytest
import json
from completion_forms import CompletionForm
from completion_forms.exceptions import (
    InvalidTemplateError,
    InvalidKeyError,
    InvalidValueError,
    FormValidationError,
    ReservedKeyError,
    ResponseParsingError
)

# Setup paths for test data
TEST_FORMS_DIR = os.path.join(os.path.dirname(__file__), "test_forms")
BASIC_FORM_PATH = os.path.join(TEST_FORMS_DIR, "basic_form.json")
TEXT_FORM_PATH = os.path.join(TEST_FORMS_DIR, "text_form.json")
NESTED_FORM_PATH = os.path.join(TEST_FORMS_DIR, "nested_form.json")


@pytest.fixture
def basic_template():
    """Provides a basic template for testing."""
    return {
        "system": "You are a helpful assistant.",
        "user": "Hello, my name is {name} and I live in {city}.",
        "response": {
            "greeting": {"type": "string", "description": "A welcome message."},
            "location_echo": {"type": "string", "description": "Echo back the city."}
        }
    }


@pytest.fixture
def text_template():
    """Provides a template for testing text-only responses."""
    return {
        "system": "Summarize the following text.",
        "user": "{text_to_summarize}",
        "response": {
            "summary": {"type": "text"}
        }
    }


def test_form_instantiation_from_dict(basic_template):
    """Tests that a form can be created from a dictionary."""
    form = CompletionForm.from_dict(basic_template)
    assert isinstance(form, CompletionForm)
    assert sorted(form.keys) == ["city", "name"]


def test_form_instantiation_from_json():
    """Tests that a form can be created from a JSON file."""
    form = CompletionForm.from_json_file(BASIC_FORM_PATH)
    assert isinstance(form, CompletionForm)
    assert sorted(form.keys) == ["city", "name"]


def test_put_and_get_messages(basic_template):
    """Tests that data can be put into the form and formatted correctly."""
    form = CompletionForm(basic_template)
    form.put("name", "Alice").put("city", "Wonderland")
    messages = form.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello, my name is Alice and I live in Wonderland."


def test_get_response_format_json(basic_template):
    """Tests that the correct response format is generated for JSON schemas."""
    form = CompletionForm(basic_template)
    response_format = form.get_response_format()
    assert response_format is not None
    assert response_format["type"] == "json_schema"
    assert "json_schema" in response_format


def test_get_response_format_text(text_template):
    """Tests that no response format is generated for text-only responses."""
    form = CompletionForm(text_template)
    assert form.get_response_format() is None


def test_get_schemas(basic_template):
    """Tests that the message and response schemas can be retrieved."""
    form = CompletionForm(basic_template)
    response_schema = form.get_response_schema()
    messages_schema = form.get_messages_schema()

    assert response_schema == basic_template["response"]
    assert "user" in messages_schema
    assert "system" in messages_schema
    assert "response" not in messages_schema
    assert messages_schema["user"] == "Hello, my name is {name} and I live in {city}."


def test_parse_json_response(basic_template):
    """Tests that a valid JSON string is parsed correctly."""
    form = CompletionForm(basic_template)
    raw_response = '{"greeting": "Hello Alice!", "location_echo": "Wonderland"}'
    parsed = form.parse_response(raw_response)
    assert parsed["greeting"] == "Hello Alice!"
    assert parsed["location_echo"] == "Wonderland"


def test_parse_text_response(text_template):
    """Tests that a plain text response is parsed correctly."""
    form = CompletionForm(text_template)
    raw_response = "This is a summary."
    parsed = form.parse_response(raw_response)
    assert parsed["summary"] == "This is a summary."


def test_parse_text_response_with_thinking(text_template):
    """Tests parsing of a text response containing a <think> block."""
    form = CompletionForm(text_template)
    raw_response = "<think>I am thinking.</think>This is the actual summary."
    parsed = form.parse_response(raw_response)
    assert parsed["thinking"] == "I am thinking."
    assert parsed["summary"] == "This is the actual summary."


def test_validation_error_missing_keys(basic_template):
    """Tests that a validation error is raised for missing keys."""
    form = CompletionForm(basic_template)
    form.put("name", "Alice")
    with pytest.raises(FormValidationError, match="missing keys:.*'city'"):
        form.get_messages()


def test_invalid_template_errors():
    """Tests that appropriate errors are raised for invalid templates."""
    with pytest.raises(InvalidTemplateError, match="must include a 'user' key"):
        CompletionForm({"system": "s", "response": {}})
    with pytest.raises(InvalidTemplateError, match="must include a 'response' key"):
        CompletionForm({"system": "s", "user": "u"})
    with pytest.raises(ReservedKeyError, match="'thinking' is reserved"):
        CompletionForm({"system": "s", "user": "u", "response": {"thinking": {"type":"string"}}})


def test_invalid_put_calls(basic_template):
    """Tests that appropriate errors are raised for invalid `put` calls."""
    form = CompletionForm(basic_template)
    with pytest.raises(InvalidKeyError, match="Invalid key 'badkey'"):
        form.put("badkey", "value")
    with pytest.raises(InvalidValueError, match="Value must be a string"):
        form.put("name", 123)

def test_pprint_methods(basic_template, capsys):
    """Tests the pretty-printing utility methods."""
    form = CompletionForm(basic_template)
    form.put("name", "Bob").put("city", "Builderland")
    
    # Test raw message printing
    form.pprint_messages(raw=True)
    captured_raw = capsys.readouterr()
    assert "--- Raw Messages ---" in captured_raw.out
    assert "Hello, my name is {name} and I live in {city}." in captured_raw.out
    
    # Test formatted message printing
    form.pprint_messages(raw=False)
    captured_formatted = capsys.readouterr()
    assert "--- Formatted Messages ---" in captured_formatted.out
    assert "Hello, my name is Bob and I live in Builderland." in captured_formatted.out

    # Test response format printing
    form.pprint_response_format()
    captured_response = capsys.readouterr()
    assert "--- Response Format ---" in captured_response.out
    assert "'greeting':" in captured_response.out

def test_nested_form_from_file():
    """Tests that a form with a nested schema can be loaded and used."""
    form = CompletionForm.from_json_file(NESTED_FORM_PATH)
    assert "user_profile" in form.get_response_schema()
    assert "aliases" in form.get_response_schema()

    form.put("name", "test_user").put("age", "99")
    messages = form.get_messages()
    assert "Process my profile information." in messages[1]['content']
    assert "test_user" not in messages[1]['content']

    response_format = form.get_response_format()
    assert response_format is not None
    properties = response_format['json_schema']['schema']['properties']
    assert "user_profile" in properties
    assert "aliases" in properties
    assert properties['user_profile']['properties']['metadata']['properties']['source']['type'] == 'string' 