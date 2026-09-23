"""
SentrixException: wraps any error with the file and line where it happened.

    try:
        ...
    except Exception as e:
        raise SentrixException(e, sys) from e
"""

import sys


def _error_message_detail(error: Exception, error_detail: "sys") -> str:
    """Error message with file name, line number and the original error."""
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        # Raised manually without an active exception context
        return f"Error: {error}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error in script [{file_name}], line [{line_number}]: {error}"


class SentrixException(Exception):
    """Single exception type raised by every SENTRIX module."""

    def __init__(self, error: Exception, error_detail: "sys" = sys):
        self.message = _error_message_detail(error, error_detail)
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


if __name__ == "__main__":
    # Self-test: deliberately trigger and wrap an error
    try:
        result = 1 / 0
    except Exception as e:
        try:
            raise SentrixException(e, sys)
        except SentrixException as se:
            print("Exception self-test caught correctly:")
            print(se)
