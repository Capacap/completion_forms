import pytest
import os
from completion_forms import CompletionClient, CompletionClientSettings, CompletionForm
from completion_forms.exceptions import ClientConfigurationError

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration

# Skip all tests in this file if OPENAI_API_KEY is not set
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    pytest.mark.skip("Skipping integration tests, OPENAI_API_KEY not set", allow_module_level=True)

# Use a fast and cheap model for testing
MODEL = "openai/gpt-4.1-nano" 

@pytest.fixture
def client():
    """Provides a configured CompletionClient for integration tests."""
    settings = CompletionClientSettings(
        model=MODEL,
        api_key=API_KEY,
        temperature=0.0, # For deterministic results
        max_tokens=150
    )
    return CompletionClient(settings)


@pytest.fixture
def json_form():
    """Provides a form that expects a JSON response."""
    template = {
        "system": "You are a helpful assistant that provides structured data in JSON format.",
        "user": "Extract the name and age from this sentence: Alice is 30 years old.",
        "response": {
            "name": {"type": "string", "description": "The person's name."},
            "age": {"type": "integer", "description": "The person's age."}
        }
    }
    form = CompletionForm(template)
    return form

@pytest.fixture
def text_form():
    """Provides a form that expects a text response."""
    template = {
        "system": "You are a helpful assistant. Be concise.",
        "user": "What is the capital of Japan?",
        "response": {"answer": {"type": "text"}}
    }
    form = CompletionForm(template)
    return form


def test_client_settings_validation():
    """Tests the validation logic within CompletionClientSettings."""
    with pytest.raises(ClientConfigurationError):
        CompletionClientSettings(model="")
    with pytest.raises(ClientConfigurationError):
        CompletionClientSettings(model="m", max_retries=-1)
    with pytest.raises(ClientConfigurationError):
        CompletionClientSettings(model="m", temperature=3.0)
    with pytest.raises(ClientConfigurationError):
        CompletionClientSettings(model="m", max_tokens=0)


def test_json_completion(client, json_form):
    """Tests a standard JSON-based completion request against the live API."""
    response = client.complete(json_form)
    
    assert "name" in response
    assert "age" in response
    assert isinstance(response["name"], str)
    assert isinstance(response["age"], int)
    assert "alice" in response["name"].lower()
    assert response["age"] == 30


def test_text_completion(client, text_form):
    """Tests a standard text-based completion request against the live API."""
    response = client.complete(text_form)

    assert "answer" in response
    assert "tokyo" in response["answer"].lower()


def test_streaming_completion(client, text_form):
    """Tests a streaming text-based completion request against the live API."""
    
    streamed_chunks = []
    def stream_handler(chunk):
        assert isinstance(chunk, str)
        streamed_chunks.append(chunk)

    response = client.complete(text_form, stream_handler=stream_handler)
    
    assert "answer" in response
    assert "tokyo" in response["answer"].lower()
    
    full_streamed_text = "".join(streamed_chunks)
    assert full_streamed_text.strip().lower() == response["answer"].strip().lower()
    assert len(streamed_chunks) > 1 # Ensure it actually streamed


def test_invalid_form_type_in_complete(client):
    """Tests that a TypeError is raised for an invalid form type."""
    with pytest.raises(TypeError, match="form must be an instance of CompletionForm"):
        client.complete("not a form")

def test_invalid_stream_handler_type(client, text_form):
    """Tests that a TypeError is raised for an invalid stream_handler type."""
    with pytest.raises(TypeError, match="stream_handler must be a callable or None"):
        client.complete(text_form, stream_handler="not a callable") 