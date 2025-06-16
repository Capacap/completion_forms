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
    return {
        "system": "Summarize the following text.",
        "user": "{text_to_summarize}",
        "response": {
            "summary": {"type": "text"}
        }
    }


def test_form_instantiation_from_dict(basic_template):
    form = CompletionForm.from_dict(basic_template)
    assert isinstance(form, CompletionForm)
    assert sorted(form.keys) == ["city", "name"]


def test_form_instantiation_from_json():
    form = CompletionForm.from_json_file(BASIC_FORM_PATH)
    assert isinstance(form, CompletionForm)
    assert sorted(form.keys) == ["city", "name"]


def test_put_and_get_messages(basic_template):
    form = CompletionForm(basic_template)
    form.put("name", "Alice")
    form.put("city", "Wonderland")
    messages = form.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello, my name is Alice and I live in Wonderland."


def test_get_response_format_json(basic_template):
    form = CompletionForm(basic_template)
    response_format = form.get_response_format()
    assert response_format is not None
    assert response_format["type"] == "json_schema"
    assert "json_schema" in response_format


def test_get_response_format_text(text_template):
    form = CompletionForm(text_template)
    assert form.get_response_format() is None


def test_get_schemas(basic_template):
    form = CompletionForm(basic_template)
    response_schema = form.get_response_schema()
    messages_schema = form.get_messages_schema()

    assert response_schema == basic_template["response"]
    assert "user" in messages_schema
    assert "system" in messages_schema
    assert "response" not in messages_schema
    assert messages_schema["user"] == "Hello, my name is {name} and I live in {city}."


def test_parse_json_response(basic_template):
    form = CompletionForm(basic_template)
    raw_response = '{"greeting": "Hello Alice!", "location_echo": "Wonderland"}'
    parsed = form.parse_response(raw_response)
    assert parsed["greeting"] == "Hello Alice!"
    assert parsed["location_echo"] == "Wonderland"


def test_parse_text_response(text_template):
    form = CompletionForm(text_template)
    raw_response = "This is a summary."
    parsed = form.parse_response(raw_response)
    assert parsed["summary"] == "This is a summary."


def test_parse_text_response_with_thinking(text_template):
    form = CompletionForm(text_template)
    raw_response = "<think>I am thinking.</think>This is the actual summary."
    parsed = form.parse_response(raw_response)
    assert parsed["thinking"] == "I am thinking."
    assert parsed["summary"] == "This is the actual summary."


def test_validation_error_missing_keys(basic_template):
    form = CompletionForm(basic_template)
    form.put("name", "Alice")
    with pytest.raises(FormValidationError, match="missing keys:.*'city'"):
        form.get_messages()


def test_invalid_template_errors():
    with pytest.raises(InvalidTemplateError, match="must include a 'user' key"):
        CompletionForm({"system": "s", "response": {}})
    with pytest.raises(InvalidTemplateError, match="must include a 'response' key"):
        CompletionForm({"system": "s", "user": "u"})
    with pytest.raises(ReservedKeyError, match="The key 'thinking' is reserved"):
        CompletionForm({"system": "s", "user": "u", "response": {"thinking": {"type":"string"}}})


def test_invalid_put_calls(basic_template):
    form = CompletionForm(basic_template)
    with pytest.raises(InvalidKeyError, match="Invalid key 'badkey'"):
        form.put("badkey", "value")
    with pytest.raises(InvalidValueError, match="Value must be a string"):
        form.put("name", 123)

def test_pprint_methods(basic_template, capsys):
    form = CompletionForm(basic_template)
    form.put("name", "Bob")
    form.put("city", "Builderland")
    
    form.pprint_raw()
    captured_raw = capsys.readouterr()
    assert "--- Raw Request ---" in captured_raw.out
    assert "Hello, my name is {name} and I live in {city}." in captured_raw.out
    
    form.pprint_formatted()
    captured_formatted = capsys.readouterr()
    assert "--- Formatted Request ---" in captured_formatted.out
    assert "Hello, my name is Bob and I live in Builderland." in captured_formatted.out

def test_nested_form_from_file():
    form = CompletionForm.from_json_file(NESTED_FORM_PATH)
    assert "user_profile" in form.get_response_schema()
    assert "aliases" in form.get_response_schema()

    form.put("name", "test_user")
    form.put("age", "99")
    messages = form.get_messages()
    assert "Process my profile information." in messages[1]['content']
    assert "test_user" not in messages[1]['content']

    response_format = form.get_response_format()
    assert response_format is not None
    properties = response_format['json_schema']['schema']['properties']
    assert "user_profile" in properties
    assert "aliases" in properties
    assert properties['user_profile']['properties']['metadata']['properties']['source']['type'] == 'string' 