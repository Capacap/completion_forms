# Completion Forms

`Completion Forms` is a lightweight Python utility for creating, validating, and managing structured requests for Large Language Models (LLMs). It simplifies the process of building complex, type-safe prompt templates and ensures that all required data is provided before formatting the final messages for an API call.

This library is designed for developers who need to reliably get structured data (like JSON) from LLMs, with a focus on clear, reusable, and self-documenting code.

## Installation

To install the library and its dependencies, clone the repository and use `pip`:

```bash
# For standard installation
pip install .

# For development (editable mode)
pip install -e .
```

## Key Features

*   **API Agnostic by Design**: While a convenient `CompletionClient` is included, the core `CompletionForm` is completely decoupled. Use `.get_messages()` and `.get_response_format()` to generate the exact payloads needed for any API client, including `anthropic`, `google-generativeai`, `cohere`, or your own custom solution. This gives you maximum flexibility to switch between model providers without rewriting your prompt logic.

*   **Single-File Prompt Management**: Define everything in one place. Each form, which can be a simple Python dictionary or a standalone `.json` file, encapsulates:
    *   The `system` prompt (the LLM's instructions).
    *   The `user` prompt template with placeholder `{keys}`.
    *   The complete `response` schema, defining the structure, types, and descriptions of the data you want back.
    *   This unified approach makes prompts self-documenting, easy to version control, and simple to share or reuse across a project.

## Core Components

The library is built around two main classes:

-   `CompletionForm`: Defines the structure of the prompt and the expected response. It's responsible for validating the template, populating it with user data, and parsing the final response from the LLM.
-   `CompletionClient`: A resilient client for sending requests to an OpenAI-compatible API. It handles authentication, retries with exponential backoff, and both standard and streaming responses.

## Quick Start

Here's a simple example of extracting structured data from a sentence.

```python
import os
from completion_forms import CompletionForm, CompletionClient, CompletionClientSettings

# 1. Define a template for your request
template = {
    "user": "Extract the name and age from this sentence: Alice is 30 years old.",
    "response": {
        "name": {"type": "string", "description": "The person's name."},
        "age": {"type": "integer", "description": "The person's age."}
    }
}

# 2. Create a form from the template
form = CompletionForm(template)

# 3. Configure the client
# Requires OPENAI_API_KEY environment variable
client = CompletionClient(
    CompletionClientSettings(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
)

# 4. Get the completion
try:
    with client:
        response = client.complete(form)
        print(response)
        # Expected output: {'name': 'Alice', 'age': 30}
except Exception as e:
    print(f"An error occurred: {e}")
```

## Usage Examples

### 1. Standard JSON Completion

This is the most common use case, where you define a JSON schema in the `response` part of your template. The library automatically instructs the LLM to return a valid JSON object.

```python
from completion_forms import CompletionForm, CompletionClient, CompletionClientSettings
import os

# The form defines the structure of the request and expected response
form = CompletionForm({
    "system": "You are a helpful travel agent. Extract the user's travel plans.",
    "user": "I want to fly from New York (JFK) to Los Angeles (LAX) on August 15th, 2024.",
    "response": {
        "departure_city": {"type": "string"},
        "destination_city": {"type": "string"},
        "departure_airport": {"type": "string"},
        "destination_airport": {"type": "string"},
        "date": {"type": "string", "description": "Date in YYYY-MM-DD format."}
    }
})

# Use the client in a 'with' block to ensure resources are managed
with CompletionClient(CompletionClientSettings(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))) as client:
    response = client.complete(form)
    print(response)
```

### 2. Streaming a Text Response

For long-form text generation, streaming provides a better user experience. To enable streaming, define a `response` field with a `type` of `"text"` and provide a `stream_handler` callable to the `complete` method.

```python
from completion_forms import CompletionForm, CompletionClient, CompletionClientSettings
import os

form = CompletionForm({
    "system": "You are a historian. Briefly explain the significance of the printing press.",
    "user": "Focus on its impact on communication and knowledge dissemination.",
    "response": {
        "explanation": {"type": "text"}
    }
})

def my_stream_handler(chunk: str):
    """A simple handler that prints each chunk of text as it arrives."""
    print(chunk, end="", flush=True)

with CompletionClient(CompletionClientSettings(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))) as client:
    print("AI Response: ", end="")
    response = client.complete(form, stream_handler=my_stream_handler)
    print("\n\n--- Final Parsed Object ---")
    print(response)
```

### 3. Local Inference with LM Studio

The client is compatible with any OpenAI-compatible server, such as the one provided by [LM Studio](https://lmstudio.ai/). This allows you to run completions on your local machine without an API key.

1.  In LM Studio, load a model and start the local server.
2.  Configure the `CompletionClientSettings` to point to your local server.

```python
from completion_forms import CompletionForm, CompletionClient, CompletionClientSettings

# Note: Model-forced JSON may not be supported by all local models.
# A text response is often more reliable.
form = CompletionForm({
    "user": "What is the capital of France?",
    "response": {"answer": {"type": "text"}}
})

# Configure the client for LM Studio
# The api_key is omitted, and the base_url is changed.
# The 'model' parameter is ignored by LM Studio but is still required by the client.
settings = CompletionClientSettings(
    model="local-model", # This can be any string
    base_url="http://localhost:1234/v1",
)

with CompletionClient(settings) as client:
    response = client.complete(form)
    print(response)
```

## Running Tests

To run the test suite, you will need an `OPENAI_API_KEY` set as an environment variable. The integration tests run against the live OpenAI API to ensure correctness.

```bash
# Make sure dev dependencies are installed
pip install -e .
pip install pytest

# Run the tests
pytest
``` 