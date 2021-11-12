# Copyright 2021, Arm Ltd.
"""Model's metadata module."""
from dataclasses import dataclass
from typing import List
from typing import Tuple


@dataclass
class NpuSupported:
    """Operator's npu supported attribute."""

    supported: bool
    reasons: List[Tuple[str, str]]


@dataclass
class Operator:
    """Model operator."""

    name: str
    op_type: str
    run_on_npu: NpuSupported

    @property
    def cpu_only(self) -> bool:
        """Return true if operator is CPU only."""
        cpu_only_reasons = [("CPU only operator", "")]
        return (
            not self.run_on_npu.supported
            and self.run_on_npu.reasons == cpu_only_reasons
        )


@dataclass
class Operators:
    """Model's operators."""

    ops: List[Operator]

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
