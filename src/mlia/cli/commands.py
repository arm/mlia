"""Cli commands module."""
from textwrap import fill

from mlia.advisor import MLAdvisor
from tabulate import tabulate
from typing_extensions import Annotated


class ParamAnnotation:
    """Parameter annotation."""

    def __init__(
        self, description: str = "", required: bool = False, positional: bool = False
    ):
        """Init param annotation."""
        self.description = description
        self.positional = positional
        self.required = required


def operators(
    model: Annotated[str, ParamAnnotation(description="TFLite model", positional=True)]
) -> None:
    """Print the model's operator list."""
    mladvisor = MLAdvisor()
    model_metadata = mladvisor.inspect(model)

    ops = model_metadata.operations()

    table_data = (
        (
            fill(name, 30),
            fill(op_type, 15),
            fill(str(run_on_npu), 20),
            tabulate(
                (
                    (fill(reason, 30), fill(description, 40))
                    for reason, description in reasons
                ),
                tablefmt="plain",
            ),
        )
        for name, op_type, run_on_npu, reasons in ops
    )

    print(
        tabulate(
            table_data,
            headers=["Operation name", "Operation type", "Supported on NPU", "Reason"],
            tablefmt="grid",
        )
    )
