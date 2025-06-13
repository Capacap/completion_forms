# Completion Forms

**Status: Work in Progress**

## About

`Completion Forms` is a lightweight Python utility for creating and managing structured prompts for language model completions. It simplifies the process of building complex prompt templates and ensures that all required data is provided before formatting the final messages for an API call.

## Installation

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

Here's a quick demonstration of how to use `CompletionForm` and `CompletionClient`:

```python
from template_form import CompletionForm, CompletionClient, CompletionClientSettings

# 1. Define your settings for the completion client
settings = CompletionClientSettings(
    model="your-model-name",  # e.g., "openai/gpt-4"
    base_url="http://your-api-endpoint/v1",
    api_key="your-api-key"
)
client = CompletionClient(settings)

# 2. Create a template for your prompt
template_data = {
    "system": "You are a helpful assistant who provides solutions.",
    "user": "Problem: {problem_description}",
    "response": {
        "solution": {"type": "string", "description": "A concise solution to the problem."}
    }
}

# 3. Create a form and populate it with data
completion_form = CompletionForm(template_data)
completion_form.put("problem_description", "The user's computer is unexpectedly shutting down.")

# 4. Get the completion
try:
    response = client.complete(completion_form)
    print("Solution:", response["solution"])
except Exception as e:
    print(f"An error occurred: {e}")

``` 