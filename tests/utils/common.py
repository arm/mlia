# Copyright 2021, Arm Ltd.
"""Common test utils module."""
from mlia.cli.common import ExecutionContext


class DummyContext(ExecutionContext):
    """Dummy context for testing purposes."""

    def __init__(self, tmpdir: str) -> None:
        """Init context."""
        super().__init__(working_dir=tmpdir)
