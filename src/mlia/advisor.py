"""Advisor module."""
from pathlib import Path
from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Union

from mlia.tools.vela_wrapper import VelaCompiler


class Operation(NamedTuple):
    """Operation details."""

    name: str
    op_type: str
    run_on_npu: bool
    reasons: List[Tuple[str, str]]


class Model:
    """Model metadata instance."""

    def __init__(self, op_list: List[Operation]) -> None:
        """Init model instance."""
        self.op_list = op_list

    def operations(self) -> List[Operation]:
        """Return list of operations."""
        return self.op_list


class MLAdvisor:
    """ML advisor class."""

    def inspect(self, model: Union[str, Path]) -> Model:
        """Read model's metadata."""
        compiler = VelaCompiler()
        model_metadata = compiler.read_model(model)

        op_list = [
            Operation(op.name(), op.type(), *op.run_on_npu())
            for op in model_metadata.operations()
        ]
        return Model(op_list)
