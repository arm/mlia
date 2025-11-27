# SPDX-FileCopyrightText: Copyright 2022-2023, 2025, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TOSA compatibility module."""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast
from typing import Protocol

import mlia
import mlia.core.output_schema as schema
from mlia.backend.errors import BackendUnavailableError
from mlia.utils.filesystem import sha256
from mlia.utils.logging import capture_raw_output
from mlia.utils.misc import get_pkg_version

logger = logging.getLogger(__name__)


class TOSAChecker(Protocol):
    """TOSA checker protocol."""

    def is_tosa_compatible(self) -> bool:
        """Return true if model is TOSA compatible."""

    def _get_tosa_compatibility_for_ops(self) -> list[Any]:
        """Return list of operators."""


@dataclass
class Operator:
    """Operator's TOSA compatibility info."""

    location: str
    name: str
    is_tosa_compatible: bool


@dataclass
class TOSACompatibilityInfo:
    """Models' TOSA compatibility information."""

    tosa_compatible: bool
    operators: list[Operator]
    exception: Exception | None = None
    errors: list[str] | None = None
    std_out: list[str] | None = None

    def to_standardized_output(  # pylint: disable=too-many-locals,too-many-branches
        self,
        model_path: Path,
        run_id: str | None = None,
        timestamp: str | None = None,
        cli_arguments: list[str] | None = None,
        target_config: dict[str, Any] | None = None,
        backend_config: dict[str, Any] | None = None,
    ) -> Any:  # Returns StandardizedOutput but avoid circular import
        """Convert to standardized output format.

        Args:
            model_path: Path to the model file
            run_id: Optional run ID (will be generated if not provided)
            timestamp: Optional ISO 8601 timestamp (will be generated if not provided)
            cli_arguments: Optional CLI arguments used for the run
            target_config: Optional target configuration parameters
            backend_config: Optional backend configuration parameters

        Returns:
            StandardizedOutput object with TOSA compatibility results
        """
        # Generate run_id and timestamp if not provided
        if run_id is None:
            run_id = schema.StandardizedOutput.create_run_id()
        if timestamp is None:
            timestamp = schema.StandardizedOutput.create_timestamp()

        # Create tool info
        tool = schema.Tool(name="mlia", version=mlia.__version__)

        # Create backend with version
        try:
            backend_version = get_pkg_version("tosa-checker")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to get tosa-checker version: %s", exc)
            backend_version = "unknown"

        backend = schema.Backend(
            id="tosa-checker",
            name="TOSA Checker",
            version=backend_version,
            configuration=backend_config or {},
        )

        # Create target
        target = schema.Target(
            profile_name="tosa",
            target_type="tosa",
            components=[
                schema.Component(
                    type=schema.ComponentType.SPECIFICATION,
                    family="tosa",
                    model=None,
                    variant=None,
                    name="TOSA",
                )
            ],
            configuration=target_config or {},
            description=(
                "TOSA (Tensor Operator Set Architecture) " "specification compatibility"
            ),
        )

        # Create model info
        model_hash = sha256(model_path)
        model_size = model_path.stat().st_size if model_path.exists() else None
        model_format = (
            model_path.suffix.lstrip(".").lower() if model_path.suffix else "unknown"
        )

        model = schema.Model(
            name=model_path.name,
            format=model_format,
            hash=model_hash,
            size_bytes=model_size,
        )

        # Create context
        context = schema.Context(
            cli_arguments=cli_arguments or [],
            runtime_configuration=None,
            git=None,
            notes=None,
        )

        # Determine result status
        if self.exception:
            status = schema.ResultStatus.FAILED
        elif self.tosa_compatible:
            status = schema.ResultStatus.OK
        else:
            status = schema.ResultStatus.INCOMPATIBLE

        # Collect errors and warnings
        result_errors: list[str] = []
        result_warnings: list[str] = []

        if self.exception:
            result_errors.append(
                f"TOSA compatibility check failed: {repr(self.exception)}"
            )

        if self.errors:
            for err_line in self.errors:
                if err_line.strip():
                    result_errors.append(err_line.strip())

        # Create checks and entities for each operator
        checks: list[schema.Check] = []
        entities: list[schema.Entity] = []

        for idx, operator in enumerate(self.operators):
            check_status = (
                schema.CheckStatus.PASS
                if operator.is_tosa_compatible
                else schema.CheckStatus.FAIL
            )
            check = schema.Check(
                id=f"op_compat_{idx}",
                status=check_status,
                details={
                    "operator_name": operator.name,
                    "location": operator.location,
                    "compatible": operator.is_tosa_compatible,
                },
            )
            checks.append(check)

            entity = schema.Entity(
                scope=schema.OperatorScope.OPERATOR,
                name=operator.name,
                location=operator.location,
                placement="Unknown",
                id=f"op_{idx}",
                attributes={"tosa_compatible": operator.is_tosa_compatible},
            )
            entities.append(entity)

        # Create result
        result = schema.Result(
            kind=schema.ResultKind.COMPATIBILITY,
            status=status,
            producer=backend.id,
            warnings=result_warnings,
            errors=result_errors,
            checks=checks,
            entities=entities,
        )

        return schema.StandardizedOutput(
            schema_version=schema.SCHEMA_VERSION,
            run_id=run_id,
            timestamp=timestamp,
            tool=tool,
            target=target,
            model=model,
            context=context,
            backends=[backend],
            results=[result],
            extensions={},
        )


def get_tosa_compatibility_info(
    tflite_model_path: str | Path,
) -> TOSACompatibilityInfo:
    """Return list of the operators."""
    # Capture the possible exception in running get_tosa_checker
    try:
        with capture_raw_output(sys.stdout) as std_output_pkg, capture_raw_output(
            sys.stderr
        ) as stderr_output_pkg:
            checker = get_tosa_checker(tflite_model_path)
    except Exception as exc:  # pylint: disable=broad-except
        return TOSACompatibilityInfo(
            tosa_compatible=False,
            operators=[],
            exception=exc,
            errors=None,
            std_out=None,
        )

    # Capture the possible BackendUnavailableError when tosa-checker is not available
    if checker is None:
        raise BackendUnavailableError(
            "Backend tosa-checker is not available", "tosa-checker"
        )

    # Capture the possible exception when checking ops compatibility
    try:
        with capture_raw_output(sys.stdout) as std_output_ops, capture_raw_output(
            sys.stderr
        ) as stderr_output_ops:
            ops = [
                Operator(item.location, item.name, item.is_tosa_compatible)
                for item in checker._get_tosa_compatibility_for_ops()  # pylint: disable=protected-access
            ]
    except Exception as exc:  # pylint: disable=broad-except
        return TOSACompatibilityInfo(
            tosa_compatible=False,
            operators=[],
            exception=exc,
            errors=None,
            std_out=None,
        )

    # Concatenate all possbile stderr/stdout
    stderr_output = stderr_output_pkg + stderr_output_ops
    std_output = std_output_pkg + std_output_ops

    return TOSACompatibilityInfo(
        tosa_compatible=checker.is_tosa_compatible(),
        operators=ops,
        exception=None,
        errors=stderr_output,
        std_out=std_output,
    )


def get_tosa_checker(tflite_model_path: str | Path) -> TOSAChecker | None:
    """Return instance of the TOSA checker."""
    try:
        import tosa_checker as tc  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None

    checker = tc.TOSAChecker(str(tflite_model_path))  # pylint: disable=no-member
    return cast(TOSAChecker, checker)
