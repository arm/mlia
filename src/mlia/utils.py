# Copyright 2021, Arm Ltd.
"""Utils module."""
import os
from contextlib import contextmanager
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from typing import Any
from typing import Generator


@contextmanager
def suppress_any_output() -> Generator[Any, Any, Any]:
    """Context manager for suppressing output."""
    with open(os.devnull, "w") as dev_null:
        with redirect_stderr(dev_null), redirect_stdout(dev_null):
            yield
