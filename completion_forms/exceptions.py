"""Custom exceptions for the completion_forms library."""

class CompletionFormError(Exception):
    """Base exception for all errors raised by the completion_forms library."""

class FileError(CompletionFormError):
    """Base exception for file-related errors."""

class FormFileNotFoundError(FileError, FileNotFoundError):
    """Raised when a form template file is not found."""

class InvalidJSONError(FileError):
    """Raised when a form template file contains invalid JSON."""

class TemplateError(CompletionFormError):
    """Base exception for template-related errors."""

class InvalidTemplateError(TemplateError):
    """Raised when the template has a structural issue."""

class ReservedKeyError(TemplateError):
    """Raised when a reserved key (e.g., 'thinking') is used improperly."""

class InputError(CompletionFormError):
    """Base exception for errors related to user input for the form."""

class InvalidKeyError(InputError, KeyError):
    """Raised when an invalid key is provided to the form."""

class InvalidValueError(InputError, TypeError):
    """Raised when an invalid value is provided for a key."""

class FormValidationError(InputError):
    """Raised when form validation fails before formatting."""

class ClientError(CompletionFormError):
    """Base exception for client-related errors."""

class ClientConfigurationError(ClientError, ValueError):
    """Raised when the CompletionClient is configured with invalid settings."""

class CompletionError(ClientError):
    """Base exception for errors during the completion process."""

class MaxRetriesExceededError(CompletionError):
    """Raised when the completion request fails after all retries."""
    def __init__(self, message: str, last_exception: Exception | None):
        self.last_exception = last_exception
        super().__init__(f"{message}\nLast exception: {last_exception}")

class ResponseParsingError(CompletionError):
    """Raised when the response from the completion API cannot be parsed."""
    def __init__(self, message: str, raw_content: str | None):
        self.raw_content = raw_content
        super().__init__(f"{message}\nRaw content: '{raw_content}'") 