import os
import sys
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from completion_forms import (
    CompletionClient,
    CompletionClientSettings,
    CompletionForm,
    CompletionRequest,
)


def _test_json_completion(client: CompletionClient):
    """Tests a standard JSON-based completion request."""
    print("--- TESTING JSON COMPLETION ---")
    json_template_data = {
        "system": "You are a helpful assistant that provides solutions.",
        "user": "I have a problem: {problem}",
        "response": {
            "solution": {"type": "string", "description": "Your detailed solution."}
        }
    }
    json_form = CompletionForm(json_template_data)
    json_form.put("problem", "There is a wild animal in my living room.")
    
    request = json_form.create_request()
    response = client.complete(request)
    
    print("\n--- SYSTEM ---")
    print(request.messages[0]["content"])
    print("\n--- USER ---")
    print(request.messages[1]["content"])
    print("\n--- RESPONSE FORMAT ---")
    print(request.response_format)
    print("\n--- LLM RESPONSE ---")
    print(response["solution"].strip())
    print("--- JSON COMPLETION TEST PASSED ---\n")


def _test_text_completion(client: CompletionClient):
    """Tests a text-based completion request."""
    print("--- TESTING TEXT COMPLETION ---")
    text_template_data = {
        "system": "You are a concise assistant.",
        "user": "Summarize this for me: {text}",
        "response": {"summary": {"type": "text"}}
    }
    text_form = CompletionForm(text_template_data)
    text_form.put("text", "The quick brown fox jumps over the lazy dog. It is a classic sentence.")
    
    request = text_form.create_request()
    response = client.complete(request)
 
    print("\n--- SYSTEM ---")
    print(request.messages[0]["content"])
    print("\n--- USER ---")
    print(request.messages[1]["content"])
    print("\n--- RESPONSE FORMAT ---")
    print(request.response_format) # Should be None for text responses
    print("\n--- LLM RESPONSE ---")
    if "thinking" in response:
        print("\n---THINKING---")
        print(response["thinking"].strip())
    print(response["summary"].strip())
    assert "summary" in response and len(response["summary"]) > 0
    print("--- TEXT COMPLETION TEST PASSED ---\n")


def _test_streaming_completion(client: CompletionClient):
    """Tests a streaming text-based completion request."""
    print("--- TESTING STREAMING COMPLETION ---")
    text_template_data = {
        "system": "You are a helpful storyteller.",
        "user": "Tell me a short story about a {animal}.",
        "response": {"story": {"type": "text"}}
    }
    streaming_text_form = CompletionForm(text_template_data)
    streaming_text_form.put("animal", "robot")

    request = streaming_text_form.create_request()

    print("Streaming content: ")
    streamed_response = client.complete(request, lambda chunk: print(chunk, end="", flush=True))
    print("\n\n--- FULL STREAMED RESPONSE ---")
    print(streamed_response)
    
    if "thinking" in streamed_response:
        print("\nFull streamed response (thinking):", streamed_response["thinking"].strip())

    assert "story" in streamed_response and len(streamed_response["story"]) > 0
    print("\n--- STREAMING COMPLETION TEST PASSED ---\n")


def main():
    parser = argparse.ArgumentParser(description="Run integration tests for completion_forms.")
    parser.add_argument("--model", type=str, default="openai/gpt-4.1-nano", help="The model to use for testing.")
    parser.add_argument("--base_url", type=str, default=None, help="The base URL for the completion API.")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="The API key for the completion API.")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable or pass it with --api_key.")

    settings = CompletionClientSettings(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0.1
    )
    client = CompletionClient(settings)
    
    _test_json_completion(client)
    _test_text_completion(client)
    _test_streaming_completion(client)

if __name__ == "__main__":
    main() 