# Copyright 2021, Arm Ltd.
"""Model's metadata module."""
from typing import List
from typing import NamedTuple
from typing import Tuple


class NpuSupported(NamedTuple):
    """Operator's npu supported attribute."""

    supported: bool
    reasons: List[Tuple[str, str]]


class Operator:
    """Model operator."""

    def __init__(self, name: str, op_type: str, run_on_npu: NpuSupported) -> None:
        """Init operation instance."""
        self.name = name
        self.op_type = op_type
        self.run_on_npu = run_on_npu

    @property
    def cpu_only(self) -> bool:
        """Return true if operator is CPU only."""
        npu_supported, reasons = self.run_on_npu
        return not npu_supported and reasons == [("CPU only operator", "")]


class Operators:
    """Model's operators."""

    def __init__(self, ops: List[Operator]) -> None:
        """Init operators instance."""
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
