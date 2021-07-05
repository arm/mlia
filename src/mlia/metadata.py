# Copyright 2021, Arm Ltd.
"""Model's metadata module."""
from typing import List
from typing import NamedTuple
from typing import Tuple


class NpuSupported(NamedTuple):
    """Operation's npu supported attribute."""

    supported: bool
    reasons: List[Tuple[str, str]]


class Operation:
    """Model operation."""

    def __init__(self, name: str, op_type: str, run_on_npu: NpuSupported) -> None:
        """Init operation instance."""
        self.name = name
        self.op_type = op_type
        self.run_on_npu = run_on_npu
