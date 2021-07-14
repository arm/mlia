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


class Operations:
    """Model's operations."""

    def __init__(self, ops: List[Operation]) -> None:
        """Init operations instance."""
        self.ops = ops

    @property
    def npu_supported_ratio(self) -> float:
        """Return NPU supported ratio."""
        total = self.total_number
        npu_supported = self.npu_supported_number

        if total == 0 or npu_supported == 0:
            return 0

        return npu_supported / total

    @property
    def npu_unsupported_ratio(self) -> float:
        """Return NPU unsupported ratio."""
        return 1 - self.npu_supported_ratio

    @property
    def total_number(self) -> int:
        """Return total number of operators."""
        return len(self.ops)

    @property
    def npu_supported_number(self) -> int:
        """Return number of npu supported operators."""
        return sum(op.run_on_npu.supported for op in self.ops)
