# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Small reusable helpers for pytest-native MLIA e2e cases."""

from __future__ import annotations

import glob
import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar, cast

from mlia.utils.types import is_list_of

MLIA_E2E_ARTIFACTS = "MLIA_E2E_ARTIFACTS"
MLIA_E2E_BACKENDS = "MLIA_E2E_BACKENDS"
MLIA_E2E_EXECUTIONS = "MLIA_E2E_EXECUTIONS"
MLIA_E2E_SHARD_INDEX = "MLIA_E2E_SHARD_INDEX"
MLIA_E2E_SHARD_COUNT = "MLIA_E2E_SHARD_COUNT"

E2E_COMPATIBILITY = "compatibility"
E2E_PERFORMANCE = "performance"

COMMON_PATTERNS = (
    r".*ML Inference Advisor started.*",
    r".*Target information.*",
    r".*Model Analysis.*",
    r".*Model Analysis Results.*",
    r".*Advice Generation.*",
)

COMPATIBILITY_PATTERNS = (r".*Operators(?: statistics)?:.*",)

PERFORMANCE_PATTERNS = (
    r".*Performance metrics:.*",
    r"│ Metric[ ]+│ Value[ ]+│ Unit[ ]+│.*",
    r".*IMPORTANT: The performance figures above refer to NPU only.*",
)

F = TypeVar("F", bound=Callable[..., object])


class E2EExecutionRuntimeError(RuntimeError):
    """Raised when e2e execution cannot be prepared or run."""


@dataclass(frozen=True)
class E2ECase:
    """One concrete e2e command execution."""

    command: str
    args: tuple[str, ...]

    def __str__(self) -> str:
        """Return the full `mlia` command for this case."""
        return shlex.join(("mlia", self.command, *self.args))


@dataclass(frozen=True)
class _ExecutionConfiguration:
    """One execution block from `MLIA_E2E_EXECUTIONS`."""

    command: str
    parameters: dict[str, list[list[str]]]

    @classmethod
    def from_dict(cls, execution: object) -> _ExecutionConfiguration:
        if not isinstance(execution, dict):
            raise E2EExecutionRuntimeError("Each execution must be a JSON object.")

        command = execution.get("command")
        parameters = execution.get("parameters")

        if not isinstance(command, str) or not command:
            raise E2EExecutionRuntimeError("Execution command is not defined.")
        if not isinstance(parameters, dict) or not parameters:
            raise E2EExecutionRuntimeError(f"Command {command} should have parameters.")
        if not all(
            isinstance(group_name, str)
            and is_list_of(group_values, list)
            and all(is_list_of(param_list, str) for param_list in group_values)
            for group_name, group_values in parameters.items()
        ):
            raise E2EExecutionRuntimeError(
                "Execution parameters must be a dictionary of list of list of strings."
            )

        return cls(command=command, parameters=parameters)

    @property
    def all_combinations(self) -> Iterable[list[str]]:
        """Generate command combinations within this execution block."""
        parameter_combinations = itertools.product(*self.parameters.values())
        return (
            [self.command, *itertools.chain.from_iterable(parameter_combination)]
            for parameter_combination in parameter_combinations
        )


def _load_backends() -> tuple[str, ...]:
    raw_value = os.environ.get(MLIA_E2E_BACKENDS, "")
    return tuple(value.strip() for value in raw_value.split(",") if value.strip())


def _normalize_backend_names(backends: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for backend in backends:
        name = backend.strip()
        if not name or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return tuple(normalized)


def _load_artifacts_dir() -> Path | None:
    raw_value = os.environ.get(MLIA_E2E_ARTIFACTS, "").strip()
    if raw_value:
        return Path(raw_value)
    if os.environ.get(MLIA_E2E_EXECUTIONS, "").strip():
        raise E2EExecutionRuntimeError(
            f"{MLIA_E2E_ARTIFACTS} must be set when e2e cases are configured."
        )
    return None


def _validate_prepared_artifacts_dir(prepared_root: Path) -> Path:
    """Return the prepared artifacts directory after validating it exists."""
    if not prepared_root.is_dir():
        raise E2EExecutionRuntimeError(
            "Prepared artifacts directory does not exist or is not a directory: "
            f"{prepared_root}."
        )
    return prepared_root.resolve()


def _resolve_prepared_artifact_path(
    prepared_root: Path, artifact_path: str | Path
) -> Path:
    """Resolve one prepared artifact and ensure it stays under the prepared root."""
    candidate = Path(artifact_path)
    if not candidate.is_absolute():
        candidate = prepared_root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(prepared_root)
    except ValueError as exc:
        raise E2EExecutionRuntimeError(
            "Prepared artifact path escapes prepared artifacts directory: "
            f"{artifact_path}."
        ) from exc
    return candidate


def _stage_prepared_artifacts(
    prepared_root: Path, artifact_paths: Sequence[str], workdir: Path
) -> None:
    prepared_root_resolved = _validate_prepared_artifacts_dir(prepared_root)
    workdir_resolved = workdir.resolve()

    for artifact_path in artifact_paths:
        source = _resolve_prepared_artifact_path(prepared_root_resolved, artifact_path)
        if not source.is_file():
            raise E2EExecutionRuntimeError(
                f"Prepared artifact does not exist: {artifact_path}."
            )
        destination = (
            workdir_resolved / source.relative_to(prepared_root_resolved)
        ).resolve()
        try:
            destination.relative_to(workdir_resolved)
        except ValueError as exc:
            raise E2EExecutionRuntimeError(
                f"Prepared artifact path escapes work directory: {artifact_path}."
            ) from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def emit_e2e_results(result: subprocess.CompletedProcess[str]) -> None:
    """Proxy captured e2e stdout/stderr to the active process streams."""
    if result.stdout:
        sys.stdout.write(result.stdout)
        if not result.stdout.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
    if result.stderr:
        sys.stderr.write(result.stderr)
        if not result.stderr.endswith("\n"):
            sys.stderr.write("\n")
        sys.stderr.flush()


def _default_backend_installer(argv: Sequence[str]) -> None:
    result = subprocess.run(
        list(argv),
        capture_output=True,
        check=False,
        cwd=Path.cwd(),
        text=True,
    )
    if result.returncode != 0:
        emit_e2e_results(result)
        raise E2EExecutionRuntimeError(
            "Backend installation failed:\n"
            f"command: {shlex.join(argv)}\n"
            f"return code: {result.returncode}\n"
            "--- stdout ---\n"
            f"{result.stdout.rstrip() or '<empty>'}\n"
            "--- stderr ---\n"
            f"{result.stderr.rstrip() or '<empty>'}"
        )


def _install_backend(backend: str) -> None:
    _default_backend_installer(
        [
            "mlia",
            "backend",
            "install",
            "--noninteractive",
            "--accept-eula",
            backend,
        ]
    )


def _installed_backend_names() -> set[str]:
    from mlia.api import list_backends

    installed: set[str] = set()
    for backend in list_backends():
        name = backend.get("name")
        if isinstance(name, str) and backend.get("installed") is True:
            installed.add(name)
    return installed


@cache
def install_requested_backends() -> tuple[str, ...]:
    """Install backends requested by the shared e2e environment."""
    backends = tuple(_load_backends())
    for backend in backends:
        _install_backend(backend)
    return backends


def ensure_backends_available(required_backends: Iterable[str]) -> tuple[str, ...]:
    """Install and verify the backends required by one e2e test case.

    This helper is intended for API e2e tests that have an explicit per-case
    backend list, for example a CLI/API parity case with
    ``backends=("vela", "corstone-300")``.

    The helper installs only the backends passed in ``required_backends``. It
    does not install every backend listed in ``MLIA_E2E_BACKENDS``. This avoids
    making one API e2e case depend on unrelated backends that may belong to a
    different case or suite.

    In GitHub Actions, ``MLIA_E2E_BACKENDS`` is treated as the e2e job's
    declared backend set. In that case, every backend in ``required_backends``
    must also be listed in ``MLIA_E2E_BACKENDS``. This check is performed even
    if the backend already appears to be installed, so cached runner state
    cannot hide a missing CI backend declaration.

    Args:
        required_backends: Backend names required by the current e2e case.

    Returns:
        A tuple of normalized backend names that were required and verified.
        Duplicate names and blank values are removed while preserving first
        occurrence order.

    Raises:
        E2EExecutionRuntimeError: A required backend is not declared in
            ``MLIA_E2E_BACKENDS``, backend installation fails, or a backend is
            still not reported as installed after installation.
    """
    required = _normalize_backend_names(required_backends)
    if not required:
        return required

    declared = set(_load_backends())
    if declared or os.environ.get("GITHUB_ACTIONS") == "true":
        undeclared = tuple(backend for backend in required if backend not in declared)
        if undeclared:
            configured = ", ".join(sorted(declared)) or "<empty>"
            raise E2EExecutionRuntimeError(
                "Required e2e backend(s) are not listed in "
                f"{MLIA_E2E_BACKENDS}: {', '.join(undeclared)}. "
                f"Configured backend(s): {configured}."
            )

    installed = _installed_backend_names()
    for backend in required:
        if backend not in installed:
            _install_backend(backend)
            installed = _installed_backend_names()

    missing = tuple(backend for backend in required if backend not in installed)
    if missing:
        raise E2EExecutionRuntimeError(
            "Required e2e backend(s) are not installed after installation: "
            f"{', '.join(missing)}."
        )
    return required


def prepared_artifact_path(artifact_path: str | Path) -> Path | None:
    """Resolve a prepared artifact path from the shared e2e artifacts root."""
    artifacts_dir = _load_artifacts_dir()
    if artifacts_dir is None:
        return None
    prepared_root = _validate_prepared_artifacts_dir(artifacts_dir)
    return _resolve_prepared_artifact_path(prepared_root, artifact_path)


def _load_execution_payload() -> list[dict[str, Any]]:
    raw_value = os.environ.get(MLIA_E2E_EXECUTIONS, "")
    if not raw_value.strip():
        return []
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise E2EExecutionRuntimeError(
            f"{MLIA_E2E_EXECUTIONS} must contain valid JSON: {exc.msg}."
        ) from exc
    if not is_list_of(payload, dict):
        raise E2EExecutionRuntimeError(
            f"{MLIA_E2E_EXECUTIONS} must contain a JSON list of objects."
        )
    return cast(list[dict[str, Any]], payload)


def _load_shard_selection() -> tuple[int, int] | None:
    raw_index = os.environ.get(MLIA_E2E_SHARD_INDEX, "").strip()
    raw_count = os.environ.get(MLIA_E2E_SHARD_COUNT, "").strip()

    if not raw_index and not raw_count:
        return None
    if not raw_index or not raw_count:
        raise E2EExecutionRuntimeError(
            f"{MLIA_E2E_SHARD_INDEX} and {MLIA_E2E_SHARD_COUNT} must both be set."
        )

    try:
        shard_index = int(raw_index)
        shard_count = int(raw_count)
    except ValueError as exc:
        raise E2EExecutionRuntimeError(
            f"{MLIA_E2E_SHARD_INDEX} and {MLIA_E2E_SHARD_COUNT} must be integers."
        ) from exc

    if shard_count < 1:
        raise E2EExecutionRuntimeError(
            f"{MLIA_E2E_SHARD_COUNT} must be greater than or equal to 1."
        )
    if shard_index < 0 or shard_index >= shard_count:
        raise E2EExecutionRuntimeError(
            f"{MLIA_E2E_SHARD_INDEX} must be in the range [0, {MLIA_E2E_SHARD_COUNT})."
        )

    return shard_index, shard_count


def _resolve_model_argument(argument: str, prepared_root: Path) -> tuple[str, ...]:
    prepared_root_resolved = _validate_prepared_artifacts_dir(prepared_root)
    filenames = sorted(
        glob.glob(str(prepared_root_resolved / argument), recursive=True)
    )
    if not filenames:
        raise E2EExecutionRuntimeError(f"Unable to resolve parameter {argument}")
    return tuple(
        str(
            _resolve_prepared_artifact_path(
                prepared_root_resolved, Path(filename)
            ).relative_to(prepared_root_resolved)
        )
        for filename in filenames
    )


def _resolve_command_arguments(
    args: Sequence[str], prepared_root: Path
) -> tuple[tuple[str, ...], ...]:
    for index, argument in enumerate(args):
        if not (argument.startswith("e2e_config/") and "*" in argument):
            continue
        return tuple(
            (*args[:index], resolved_argument, *args[index + 1 :])
            for resolved_argument in _resolve_model_argument(argument, prepared_root)
        )
    return (tuple(args),)


def _load_cases() -> tuple[E2ECase, ...]:
    executions = tuple(
        _ExecutionConfiguration.from_dict(item) for item in _load_execution_payload()
    )
    artifacts_dir = _load_artifacts_dir()
    if artifacts_dir is None:
        return ()
    if executions:
        _validate_prepared_artifacts_dir(artifacts_dir)

    cases: list[E2ECase] = []
    seen_commands: set[str] = set()

    for execution in executions:
        for combination in execution.all_combinations:
            command = combination[0]
            args = combination[1:]
            for resolved_args in _resolve_command_arguments(args, artifacts_dir):
                case = E2ECase(command=command, args=tuple(resolved_args))
                if str(case) in seen_commands:
                    raise E2EExecutionRuntimeError(f"Duplicate e2e command: {case}")
                seen_commands.add(str(case))
                cases.append(case)

    shard_selection = _load_shard_selection()
    if shard_selection is None:
        return tuple(cases)

    shard_index, shard_count = shard_selection
    return tuple(
        case for index, case in enumerate(cases) if index % shard_count == shard_index
    )


def _is_compatibility_case(case: E2ECase) -> bool:
    return case.command == "check" and "--compatibility" in case.args


def _is_performance_case(case: E2ECase) -> bool:
    return case.command == "check" and "--performance" in case.args


def _default_command_runner(
    argv: Sequence[str], workdir: Path
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("COLUMNS", "240")
    return subprocess.run(
        list(argv),
        cwd=workdir,
        capture_output=True,
        check=False,
        text=True,
        env=env,
    )


def _artifact_paths(args: Sequence[str], prepared_root: Path) -> tuple[str, ...]:
    prepared_root_resolved = _validate_prepared_artifacts_dir(prepared_root)
    artifact_paths: list[str] = []

    for argument in args:
        candidate = Path(argument)
        if candidate.is_absolute():
            continue

        source = _resolve_prepared_artifact_path(prepared_root_resolved, argument)
        if not source.is_file():
            continue

        artifact_paths.append(str(source.relative_to(prepared_root_resolved)))

    return tuple(dict.fromkeys(artifact_paths))


def run_case(case: E2ECase, *, workdir: Path) -> subprocess.CompletedProcess[str]:
    """Stage artifacts and run one e2e case."""
    install_requested_backends()
    artifacts_dir = _load_artifacts_dir()
    if artifacts_dir is None:
        raise E2EExecutionRuntimeError(
            f"{MLIA_E2E_ARTIFACTS} must be set when running e2e cases."
        )
    _stage_prepared_artifacts(
        artifacts_dir, _artifact_paths(case.args, artifacts_dir), workdir
    )
    argv = ("mlia", case.command, *case.args)
    return _default_command_runner(argv, workdir)


def parametrize(suite: str) -> Callable[[F], F]:
    """Return a pytest parametrization marker for one e2e suite."""
    import pytest

    cases = _load_cases()
    if suite == E2E_COMPATIBILITY:
        selected_cases = tuple(case for case in cases if _is_compatibility_case(case))
    elif suite == E2E_PERFORMANCE:
        selected_cases = tuple(case for case in cases if _is_performance_case(case))
    else:
        raise E2EExecutionRuntimeError(f"Unknown e2e suite: {suite}")

    return pytest.mark.parametrize(
        "case", selected_cases, ids=[str(case) for case in selected_cases]
    )
