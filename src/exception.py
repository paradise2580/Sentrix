"""
src/exception.py

Role
----
Wrap every caught error with the exact file and line number where it
occurred, so debugging a failure is instant instead of guesswork through
a generic traceback.

Usage pattern in every module
------------------------------
    from src.exception import SentrixException
    import sys

    try:
        df = load_data(path)
    except Exception as e:
        raise SentrixException(e, sys) from e
"""

import sys


def _error_message_detail(error: Exception, error_detail: "sys") -> str:
    """Build a precise message: which file, which line, what went wrong."""
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        # Raised manually without an active exception context
        return f"Error: {error}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error in script [{file_name}], line [{line_number}]: {error}"


class SentrixException(Exception):
    """
    Custom exception used across all SENTRIX modules.

    Every module catches its own low-level exceptions (SQL errors, HTTP
    errors, shape mismatches, etc.) and re-raises them as SentrixException,
    so the top-level caller — the API, a pipeline task, a test — always
    receives one consistent, traceable exception type.
    """

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
