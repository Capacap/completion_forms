import os
import pytest
from completion_forms import CompletionForm
from completion_forms.exceptions import FormFileNotFoundError, InvalidValueError

# Define the path to the test file
TEST_FORM_PATH = os.path.join(os.path.dirname(__file__), "test_forms", "test_form.json")
NESTED_TEST_FORM_PATH = os.path.join(os.path.dirname(__file__), "test_forms", "nested_test_form.json")

def test_from_json_file_success():
    """Tests successful loading of a CompletionForm from a valid JSON file."""
    form = CompletionForm.from_json_file(TEST_FORM_PATH)
    assert isinstance(form, CompletionForm)
    # The keys should be 'name' and 'city' as defined in the user message
    assert set(form.keys) == {"name", "city"}

def test_from_json_file_not_found():
    """Tests that FormFileNotFoundError is raised for a non-existent file."""
    with pytest.raises(FormFileNotFoundError):
        CompletionForm.from_json_file("non_existent_form.json")

def test_from_json_file_invalid_path_type():
    """Tests that InvalidValueError is raised when the path is not a string."""
    with pytest.raises(InvalidValueError):
        # Pass an integer instead of a string path
        CompletionForm.from_json_file(12345)

def test_from_json_file_with_nested_properties():
    """Tests that a form with nested response properties is parsed correctly."""
    form = CompletionForm.from_json_file(NESTED_TEST_FORM_PATH)
    assert isinstance(form, CompletionForm)
    assert set(form.keys) == {"name", "age"}

    form.put("name", "Alice")
    form.put("age", "30")

    _, response_format, _ = form.format()

    assert response_format is not None
    
    expected_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {
                "type": "object",
                "properties": {
                    "user_profile": {
                        "type": "object",
                        "description": "The user's profile information.",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The user's name."
                            },
                            "age": {
                                "type": "integer",
                                "description": "The user's age."
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Additional metadata.",
                                "properties": {
                                    "source": {
                                        "type": "string",
                                        "description": "The source of the profile request."
                                    }
                                },
                                "required": ["source"]
                            }
                        },
                        "required": ["age", "metadata", "name"]
                    },
                    "aliases": {
                        "type": "array",
                        "description": "A list of aliases for the user.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "alias_name": {
                                    "type": "string"
                                },
                                "priority": {
                                    "type": "integer"
                                }
                            },
                            "required": ["alias_name", "priority"]
                        }
                    }
                },
                "required": ["aliases", "user_profile"]
            }
        }
    }
    assert response_format == expected_schema 