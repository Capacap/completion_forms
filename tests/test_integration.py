import json
import os
import argparse
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from completion_forms import (CompletionClient, CompletionClientSettings,
                              CompletionForm)


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
    if "thinking" in response:
        print("\n---THINKING---")
        print(response["thinking"].strip())
    print(response["summary"].strip())
    assert "summary" in response and len(response["summary"]) > 0


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
        if os.path.exists(test_json_file):
            os.remove(test_json_file)

def main():
    parser = argparse.ArgumentParser(description="Run integration tests for completion_forms.")
    parser.add_argument("--model", type=str, default="openai/gpt-4.1-nano", help="The model to use for testing.")
    parser.add_argument("--base_url", type=str, default=None, help="The base URL for the completion API.")
    parser.add_argument("--api_key", type=str, default=None, help="The API key for the completion API.")
    args = parser.parse_args()

    settings = CompletionClientSettings(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key
    )
    client = CompletionClient(settings)
    
    _test_json_completion(client)
    _test_text_completion(client)
    _test_form_validation()
    _test_streaming_completion(client)
    _test_factory_methods()

if __name__ == "__main__":
    main() 