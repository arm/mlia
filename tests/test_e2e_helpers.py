# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pytest-native MLIA e2e helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import ANY, MagicMock, call

import pytest

from mlia.testing.e2e import (
    MLIA_E2E_SHARD_COUNT,
    MLIA_E2E_SHARD_INDEX,
    E2ECase,
    E2EExecutionRuntimeError,
    _load_artifacts_dir,
    _load_backends,
    _load_cases,
    emit_e2e_results,
    ensure_backends_available,
    install_requested_backends,
    prepared_artifact_path,
    run_case,
)


@pytest.fixture(autouse=True)
def _clear_backend_install_cache() -> None:
    """Keep e2e helper cache state isolated between tests."""
    install_requested_backends.cache_clear()


def test_case_str_renders_full_mlia_command() -> None:
    """Case string rendering should match the final CLI invocation."""
    case = E2ECase(
        command="check",
        args=(
            "--performance",
            "--target-profile",
            "target-profile.toml",
            "e2e_config/model.tflite",
        ),
    )

    assert str(case) == (
        "mlia check --performance --target-profile target-profile.toml "
        "e2e_config/model.tflite"
    )


def test_load_cases_expands_execution_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model globs should expand into one case per matching artifact."""
    artifacts_dir = tmp_path / "prepared"
    keras_dir = artifacts_dir / "e2e_config" / "keras_models"
    keras_dir.mkdir(parents=True)
    (keras_dir / "baseline_a.h5").touch()
    (keras_dir / "baseline_b.h5").touch()

    monkeypatch.setenv("MLIA_E2E_ARTIFACTS", str(artifacts_dir))
    monkeypatch.setenv(
        "MLIA_E2E_EXECUTIONS",
        """[
            {
                "command": "check",
                "parameters": {
                    "category": [["--compatibility"]],
                    "target_profile": [["--target-profile", "target-profile.toml"]],
                    "models": [["e2e_config/keras_models/baseline_*.h5"]]
                }
            }
        ]""",
    )

    cases = _load_cases()

    assert tuple(str(case) for case in cases) == (
        (
            "mlia check --compatibility --target-profile target-profile.toml "
            "e2e_config/keras_models/baseline_a.h5"
        ),
        (
            "mlia check --compatibility --target-profile target-profile.toml "
            "e2e_config/keras_models/baseline_b.h5"
        ),
    )


def test_load_cases_rejects_missing_glob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown artifact globs should fail collection."""
    monkeypatch.setenv("MLIA_E2E_ARTIFACTS", str(tmp_path))
    monkeypatch.setenv(
        "MLIA_E2E_EXECUTIONS",
        """[
            {
                "command": "check",
                "parameters": {
                    "category": [["--performance"]],
                    "target_profile": [["--target-profile", "target-profile.toml"]],
                    "models": [["e2e_config/*.tflite"]]
                }
            }
        ]""",
    )

    with pytest.raises(E2EExecutionRuntimeError, match="Unable to resolve parameter"):
        _load_cases()


def test_load_cases_rejects_glob_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Glob-expanded artifacts must stay under the configured artifacts root."""
    artifacts_dir = tmp_path / "prepared"
    e2e_dir = artifacts_dir / "e2e_config"
    e2e_dir.mkdir(parents=True)
    (tmp_path / "outside-model.h5").touch()

    monkeypatch.setenv("MLIA_E2E_ARTIFACTS", str(artifacts_dir))
    monkeypatch.setenv(
        "MLIA_E2E_EXECUTIONS",
        """[
            {
                "command": "check",
                "parameters": {
                    "category": [["--performance"]],
                    "target_profile": [["--target-profile", "target-profile.toml"]],
                    "models": [["e2e_config/../../*.h5"]]
                }
            }
        ]""",
    )

    with pytest.raises(
        E2EExecutionRuntimeError,
        match=(
            "Prepared artifact path escapes prepared artifacts directory: "
            r".*outside-model\.h5"
        ),
    ):
        _load_cases()


def test_load_cases_rejects_missing_artifacts_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configured cases should fail clearly when artifacts root is invalid."""
    missing_dir = tmp_path / "missing-prepared"

    monkeypatch.setenv("MLIA_E2E_ARTIFACTS", str(missing_dir))
    monkeypatch.setenv(
        "MLIA_E2E_EXECUTIONS",
        """[
            {
                "command": "check",
                "parameters": {
                    "category": [["--compatibility"]],
                    "target_profile": [["--target-profile", "target-profile.toml"]],
                    "models": [["e2e_config/model.h5"]]
                }
            }
        ]""",
    )

    with pytest.raises(
        E2EExecutionRuntimeError,
        match=(
            "Prepared artifacts directory does not exist or is not a directory: "
            f"{missing_dir}"
        ),
    ):
        _load_cases()


def test_load_cases_does_not_touch_cli_command_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collection should normalize cases without touching CLI command setup."""
    artifacts_dir = tmp_path / "prepared"
    keras_dir = artifacts_dir / "e2e_config" / "keras_models"
    keras_dir.mkdir(parents=True)
    (keras_dir / "baseline.h5").touch()

    monkeypatch.setenv("MLIA_E2E_ARTIFACTS", str(artifacts_dir))
    monkeypatch.setenv(
        "MLIA_E2E_EXECUTIONS",
        """[
            {
                "command": "check",
                "parameters": {
                    "category": [["--compatibility"]],
                    "target_profile": [[
                        "--target-profile",
                        "target-profile.toml",
                        "--backend",
                        "dummy-backend"
                    ]],
                    "models": [["e2e_config/keras_models/baseline.h5"]]
                }
            }
        ]""",
    )

    cases = _load_cases()

    assert tuple(str(case) for case in cases) == (
        (
            "mlia check --compatibility --target-profile "
            "target-profile.toml --backend dummy-backend "
            "e2e_config/keras_models/baseline.h5"
        ),
    )


@pytest.fixture
def _configure_sharding_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure four deterministic cases for sharding tests."""
    artifacts_dir = tmp_path / "prepared"
    keras_dir = artifacts_dir / "e2e_config" / "keras_models"
    keras_dir.mkdir(parents=True)
    for suffix in range(4):
        (keras_dir / f"baseline_{suffix}.h5").touch()

    monkeypatch.setenv("MLIA_E2E_ARTIFACTS", str(artifacts_dir))
    monkeypatch.setenv(
        "MLIA_E2E_EXECUTIONS",
        """[
            {
                "command": "check",
                "parameters": {
                    "category": [["--compatibility"]],
                    "target_profile": [["--target-profile", "target-profile.toml"]],
                    "models": [["e2e_config/keras_models/baseline_*.h5"]]
                }
            }
        ]""",
    )


def test_load_cases_without_sharding_returns_all_cases(
    _configure_sharding_cases: None,
) -> None:
    """Unsharded collection should keep all resolved cases."""
    cases = _load_cases()

    assert tuple(str(case) for case in cases) == (
        (
            "mlia check --compatibility --target-profile target-profile.toml "
            "e2e_config/keras_models/baseline_0.h5"
        ),
        (
            "mlia check --compatibility --target-profile target-profile.toml "
            "e2e_config/keras_models/baseline_1.h5"
        ),
        (
            "mlia check --compatibility --target-profile target-profile.toml "
            "e2e_config/keras_models/baseline_2.h5"
        ),
        (
            "mlia check --compatibility --target-profile target-profile.toml "
            "e2e_config/keras_models/baseline_3.h5"
        ),
    )


def test_load_cases_shards_cases_for_shard_zero(
    _configure_sharding_cases: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shard zero should receive even-position cases."""
    monkeypatch.setenv(MLIA_E2E_SHARD_INDEX, "0")
    monkeypatch.setenv(MLIA_E2E_SHARD_COUNT, "2")

    cases = _load_cases()

    assert tuple(str(case) for case in cases) == (
        (
            "mlia check --compatibility --target-profile target-profile.toml "
            "e2e_config/keras_models/baseline_0.h5"
        ),
        (
            "mlia check --compatibility --target-profile target-profile.toml "
            "e2e_config/keras_models/baseline_2.h5"
        ),
    )


def test_load_cases_shards_cases_for_shard_one(
    _configure_sharding_cases: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shard one should receive odd-position cases."""
    monkeypatch.setenv(MLIA_E2E_SHARD_INDEX, "1")
    monkeypatch.setenv(MLIA_E2E_SHARD_COUNT, "2")

    cases = _load_cases()

    assert tuple(str(case) for case in cases) == (
        (
            "mlia check --compatibility --target-profile target-profile.toml "
            "e2e_config/keras_models/baseline_1.h5"
        ),
        (
            "mlia check --compatibility --target-profile target-profile.toml "
            "e2e_config/keras_models/baseline_3.h5"
        ),
    )


def test_load_cases_requires_both_shard_env_vars(
    _configure_sharding_cases: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial shard configuration should fail fast."""
    monkeypatch.setenv(MLIA_E2E_SHARD_INDEX, "0")

    with pytest.raises(
        E2EExecutionRuntimeError,
        match=f"{MLIA_E2E_SHARD_INDEX} and {MLIA_E2E_SHARD_COUNT} must both be set",
    ):
        _load_cases()


def test_load_cases_rejects_invalid_shard_index(
    _configure_sharding_cases: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out-of-range shard indexes should be rejected."""
    monkeypatch.setenv(MLIA_E2E_SHARD_INDEX, "2")
    monkeypatch.setenv(MLIA_E2E_SHARD_COUNT, "2")

    with pytest.raises(
        E2EExecutionRuntimeError,
        match=(
            f"{MLIA_E2E_SHARD_INDEX} must be in the range "
            f"\\[0, {MLIA_E2E_SHARD_COUNT}\\)"
        ),
    ):
        _load_cases()


def test_load_cases_rejects_invalid_shard_count(
    _configure_sharding_cases: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shard count must be at least one."""
    monkeypatch.setenv(MLIA_E2E_SHARD_INDEX, "0")
    monkeypatch.setenv(MLIA_E2E_SHARD_COUNT, "0")

    with pytest.raises(
        E2EExecutionRuntimeError,
        match=f"{MLIA_E2E_SHARD_COUNT} must be greater than or equal to 1",
    ):
        _load_cases()


def test_load_cases_detects_duplicates_before_sharding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate commands should fail before shard selection hides them."""
    artifacts_dir = tmp_path / "prepared"
    keras_dir = artifacts_dir / "e2e_config" / "keras_models"
    keras_dir.mkdir(parents=True)
    (keras_dir / "baseline.h5").touch()

    monkeypatch.setenv("MLIA_E2E_ARTIFACTS", str(artifacts_dir))
    monkeypatch.setenv(
        "MLIA_E2E_EXECUTIONS",
        """[
            {
                "command": "check",
                "parameters": {
                    "category": [["--compatibility"]],
                    "target_profile": [["--target-profile", "target-profile.toml"]],
                    "models": [["e2e_config/keras_models/baseline.h5"]]
                }
            },
            {
                "command": "check",
                "parameters": {
                    "category": [["--compatibility"]],
                    "target_profile": [["--target-profile", "target-profile.toml"]],
                    "models": [["e2e_config/keras_models/baseline.h5"]]
                }
            }
        ]""",
    )
    monkeypatch.setenv(MLIA_E2E_SHARD_INDEX, "0")
    monkeypatch.setenv(MLIA_E2E_SHARD_COUNT, "2")

    with pytest.raises(E2EExecutionRuntimeError, match="Duplicate e2e command"):
        _load_cases()


def test_run_case_installs_requested_backends_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backend installation should be cached across case execution."""
    prepared_root = tmp_path / "prepared"
    staged_dir = prepared_root / "e2e_config"
    staged_dir.mkdir(parents=True)
    (prepared_root / "target-profile.toml").write_text("target = 'dummy'")
    (staged_dir / "model.h5").write_text("model")

    case = E2ECase(
        command="check",
        args=(
            "--compatibility",
            "--target-profile",
            "target-profile.toml",
            "e2e_config/model.h5",
        ),
    )
    backend_installs: list[tuple[str, ...]] = []
    commands: list[tuple[str, ...]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if argv[0] == "mlia-backend":
            backend_installs.append(tuple(argv))
            return subprocess.CompletedProcess(argv, 0)
        commands.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("MLIA_E2E_ARTIFACTS", str(prepared_root))
    monkeypatch.setenv("MLIA_E2E_BACKENDS", "dummy-backend")

    run_case(case, workdir=tmp_path / "workdir-1")
    run_case(case, workdir=tmp_path / "workdir-2")

    assert backend_installs == [
        (
            "mlia-backend",
            "install",
            "--noninteractive",
            "--i-agree-to-the-contained-eula",
            "dummy-backend",
        )
    ]
    assert commands == [
        (
            "mlia",
            "check",
            "--compatibility",
            "--target-profile",
            "target-profile.toml",
            "e2e_config/model.h5",
        ),
        (
            "mlia",
            "check",
            "--compatibility",
            "--target-profile",
            "target-profile.toml",
            "e2e_config/model.h5",
        ),
    ]


def test_run_case_runs_command_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running a case should stage artifacts and execute one command."""
    prepared_root = tmp_path / "prepared"
    staged_dir = prepared_root / "e2e_config"
    staged_dir.mkdir(parents=True)
    (prepared_root / "target-profile.toml").write_text("target = 'dummy'")
    (staged_dir / "model.h5").write_text("model")

    case = E2ECase(
        command="check",
        args=(
            "--compatibility",
            "--target-profile",
            "target-profile.toml",
            "e2e_config/model.h5",
        ),
    )
    run_mock = MagicMock(
        return_value=subprocess.CompletedProcess(
            ["mlia", "check"],
            0,
            stdout="\n".join(
                [
                    "ML Inference Advisor started",
                    "Target information",
                    "Model Analysis",
                    "Model Analysis Results",
                    "Advice Generation",
                    "Operators:",
                ]
            ),
            stderr="",
        )
    )

    monkeypatch.setattr(subprocess, "run", run_mock)
    monkeypatch.setenv("MLIA_E2E_ARTIFACTS", str(prepared_root))

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    result = run_case(case, workdir=workdir)

    assert (workdir / "target-profile.toml").exists()
    assert (workdir / "e2e_config" / "model.h5").exists()
    assert run_mock.call_args_list == [
        call(
            [
                "mlia",
                "check",
                "--compatibility",
                "--target-profile",
                "target-profile.toml",
                "e2e_config/model.h5",
            ],
            cwd=workdir,
            capture_output=True,
            check=False,
            text=True,
            env=ANY,
        )
    ]
    assert run_mock.call_args.kwargs["env"]["COLUMNS"] == "240"
    assert result.returncode == 0


def test_run_case_preserves_explicit_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caller-provided COLUMNS should win over the helper default."""
    prepared_root = tmp_path / "prepared"
    staged_dir = prepared_root / "e2e_config"
    staged_dir.mkdir(parents=True)
    (prepared_root / "target-profile.toml").write_text("target = 'dummy'")
    (staged_dir / "model.h5").write_text("model")

    case = E2ECase(
        command="check",
        args=(
            "--compatibility",
            "--target-profile",
            "target-profile.toml",
            "e2e_config/model.h5",
        ),
    )

    run_mock = MagicMock(
        return_value=subprocess.CompletedProcess(
            ["mlia", "check"],
            0,
            stdout="ok",
            stderr="",
        )
    )

    monkeypatch.setattr(subprocess, "run", run_mock)
    monkeypatch.setenv("MLIA_E2E_ARTIFACTS", str(prepared_root))
    monkeypatch.setenv("COLUMNS", "321")

    workdir = tmp_path / "workdir-explicit-columns"
    workdir.mkdir()
    result = run_case(case, workdir=workdir)

    assert run_mock.call_args.kwargs["cwd"] == workdir
    assert run_mock.call_args.kwargs["capture_output"] is True
    assert run_mock.call_args.kwargs["check"] is False
    assert run_mock.call_args.kwargs["text"] is True
    assert run_mock.call_args.kwargs["env"]["COLUMNS"] == "321"
    assert result.returncode == 0


def test_run_case_rejects_artifact_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prepared artifact arguments must stay under the configured artifacts root."""
    prepared_root = tmp_path / "prepared"
    staged_dir = prepared_root / "e2e_config"
    staged_dir.mkdir(parents=True)
    (staged_dir / "model.h5").write_text("model")

    case = E2ECase(
        command="check",
        args=(
            "--compatibility",
            "--target-profile",
            "../target-profile.toml",
            "e2e_config/model.h5",
        ),
    )

    run_mock = MagicMock()
    monkeypatch.setattr(subprocess, "run", run_mock)
    monkeypatch.setenv("MLIA_E2E_ARTIFACTS", str(prepared_root))

    with pytest.raises(
        E2EExecutionRuntimeError,
        match=(
            "Prepared artifact path escapes prepared artifacts directory: "
            r"\.\./target-profile\.toml"
        ),
    ):
        run_case(case, workdir=tmp_path / "workdir-traversal")

    assert run_mock.call_count == 0


def test_load_backends_reads_csv_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend names should be parsed from a comma-separated env var."""
    monkeypatch.setenv("MLIA_E2E_BACKENDS", "backend-a, backend-b")

    assert _load_backends() == ("backend-a", "backend-b")


def test_load_artifacts_dir_requires_value_when_cases_are_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured executions without artifacts should fail immediately."""
    monkeypatch.delenv("MLIA_E2E_ARTIFACTS", raising=False)
    monkeypatch.setenv(
        "MLIA_E2E_EXECUTIONS",
        '[{"command": "check", "parameters": {"models": [["e2e_config/model.h5"]]}}]',
    )

    with pytest.raises(
        E2EExecutionRuntimeError,
        match="MLIA_E2E_ARTIFACTS must be set when e2e cases are configured",
    ):
        _load_artifacts_dir()


def test_prepared_artifact_path_returns_none_without_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unconfigured artifact roots should let optional e2e tests skip."""
    monkeypatch.delenv("MLIA_E2E_ARTIFACTS", raising=False)
    monkeypatch.delenv("MLIA_E2E_EXECUTIONS", raising=False)

    assert prepared_artifact_path("e2e_config/model.tflite") is None


def test_prepared_artifact_path_resolves_relative_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative artifacts should resolve under the configured prepared root."""
    prepared_root = tmp_path / "prepared"
    model = prepared_root / "e2e_config" / "model.tflite"
    model.parent.mkdir(parents=True)
    model.touch()
    monkeypatch.setenv("MLIA_E2E_ARTIFACTS", str(prepared_root))

    assert prepared_artifact_path("e2e_config/model.tflite") == model.resolve()


def test_prepared_artifact_path_rejects_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolved artifact paths must stay under the prepared root."""
    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    monkeypatch.setenv("MLIA_E2E_ARTIFACTS", str(prepared_root))

    with pytest.raises(
        E2EExecutionRuntimeError,
        match=r"Prepared artifact path escapes prepared artifacts directory",
    ):
        prepared_artifact_path("../outside-model.tflite")


def test_install_requested_backends_uses_backend_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured backends should be installed in declared order."""
    commands: list[list[str]] = []

    def fake_run(
        argv: list[str], *, capture_output: bool, check: bool, cwd: Path, text: bool
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert check is False
        assert text is True
        assert cwd == Path.cwd()
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    monkeypatch.setenv("MLIA_E2E_BACKENDS", "backend-a, backend-b")
    installed = install_requested_backends()

    assert installed == ("backend-a", "backend-b")
    assert commands == [
        [
            "mlia-backend",
            "install",
            "--noninteractive",
            "--i-agree-to-the-contained-eula",
            "backend-a",
        ],
        [
            "mlia-backend",
            "install",
            "--noninteractive",
            "--i-agree-to-the-contained-eula",
            "backend-b",
        ],
    ]


def test_install_requested_backends_streams_install_failure_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Install failures should echo captured output before asserting."""

    def fake_run(
        argv: list[str], *, capture_output: bool, check: bool, cwd: Path, text: bool
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert check is False
        assert text is True
        assert cwd == Path.cwd()
        return subprocess.CompletedProcess(
            argv,
            1,
            stdout="backend install stdout\nextra detail\n",
            stderr="backend install stderr\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("MLIA_E2E_BACKENDS", "backend-a")
    with pytest.raises(
        E2EExecutionRuntimeError, match="Backend installation failed"
    ) as exc_info:
        install_requested_backends()

    message = str(exc_info.value)
    captured = capsys.readouterr()
    assert (
        "mlia-backend install --noninteractive "
        "--i-agree-to-the-contained-eula backend-a" in message
    )
    assert "return code: 1" in message
    assert "backend install stdout" in message
    assert "backend install stderr" in message
    assert captured.out == "backend install stdout\nextra detail\n"
    assert captured.err == "backend install stderr\n"


def test_ensure_backends_available_installs_only_required_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-case setup should install only the required missing backends."""
    installed: set[str] = set()
    installs: list[tuple[str, ...]] = []

    def fake_list_backends() -> list[dict[str, object]]:
        return [
            {"name": "backend-a", "installed": "backend-a" in installed},
            {"name": "backend-b", "installed": "backend-b" in installed},
        ]

    def fake_installer(argv: list[str]) -> None:
        installs.append(tuple(argv))
        installed.add(argv[-1])

    monkeypatch.setattr("mlia.api.list_backends", fake_list_backends)
    monkeypatch.setattr("mlia.testing.e2e._default_backend_installer", fake_installer)
    monkeypatch.setenv("MLIA_E2E_BACKENDS", "backend-a, backend-b")

    verified = ensure_backends_available(("backend-b",))

    assert verified == ("backend-b",)
    assert installs == [
        (
            "mlia-backend",
            "install",
            "--noninteractive",
            "--i-agree-to-the-contained-eula",
            "backend-b",
        )
    ]


def test_ensure_backends_available_rejects_undeclared_backend_even_when_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured e2e backend drift should not be hidden by cached state."""
    installs: list[tuple[str, ...]] = []

    def fake_list_backends() -> list[dict[str, object]]:
        return [{"name": "backend-b", "installed": True}]

    def fake_installer(argv: list[str]) -> None:
        installs.append(tuple(argv))

    monkeypatch.setattr("mlia.api.list_backends", fake_list_backends)
    monkeypatch.setattr("mlia.testing.e2e._default_backend_installer", fake_installer)
    monkeypatch.setenv("MLIA_E2E_BACKENDS", "backend-a")

    with pytest.raises(
        E2EExecutionRuntimeError,
        match=(
            "Required e2e backend\\(s\\) are not listed in MLIA_E2E_BACKENDS: backend-b"
        ),
    ):
        ensure_backends_available(("backend-b",))

    assert installs == []


def test_ensure_backends_available_rejects_empty_backend_env_in_github_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CI runs should fail when required backends are not declared."""
    installs: list[tuple[str, ...]] = []

    def fake_list_backends() -> list[dict[str, object]]:
        return [{"name": "backend-a", "installed": True}]

    def fake_installer(argv: list[str]) -> None:
        installs.append(tuple(argv))

    monkeypatch.setattr("mlia.api.list_backends", fake_list_backends)
    monkeypatch.setattr("mlia.testing.e2e._default_backend_installer", fake_installer)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("MLIA_E2E_BACKENDS", "")

    with pytest.raises(
        E2EExecutionRuntimeError,
        match=(
            "Required e2e backend\\(s\\) are not listed in MLIA_E2E_BACKENDS: backend-a"
        ),
    ) as exc_info:
        ensure_backends_available(("backend-a",))

    assert "Configured backend(s): <empty>" in str(exc_info.value)
    assert installs == []


def test_ensure_backends_available_installs_required_backend_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local e2e runs may request explicit backends without an env allow-list."""
    installed: set[str] = set()
    installs: list[tuple[str, ...]] = []

    def fake_list_backends() -> list[dict[str, object]]:
        return [{"name": "backend-a", "installed": "backend-a" in installed}]

    def fake_installer(argv: list[str]) -> None:
        installs.append(tuple(argv))
        installed.add(argv[-1])

    monkeypatch.setattr("mlia.api.list_backends", fake_list_backends)
    monkeypatch.setattr("mlia.testing.e2e._default_backend_installer", fake_installer)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("MLIA_E2E_BACKENDS", raising=False)

    verified = ensure_backends_available(("backend-a",))

    assert verified == ("backend-a",)
    assert installs == [
        (
            "mlia-backend",
            "install",
            "--noninteractive",
            "--i-agree-to-the-contained-eula",
            "backend-a",
        )
    ]


def test_ensure_backends_available_refreshes_installed_state_after_each_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installing one backend may make another required backend available."""
    installed: set[str] = set()
    installs: list[tuple[str, ...]] = []

    def fake_list_backends() -> list[dict[str, object]]:
        return [
            {"name": "backend-a", "installed": "backend-a" in installed},
            {"name": "backend-b", "installed": "backend-b" in installed},
        ]

    def fake_installer(argv: list[str]) -> None:
        installs.append(tuple(argv))
        installed.update({"backend-a", "backend-b"})

    monkeypatch.setattr("mlia.api.list_backends", fake_list_backends)
    monkeypatch.setattr("mlia.testing.e2e._default_backend_installer", fake_installer)
    monkeypatch.setenv("MLIA_E2E_BACKENDS", "backend-a, backend-b")

    verified = ensure_backends_available(("backend-a", "backend-b"))

    assert verified == ("backend-a", "backend-b")
    assert installs == [
        (
            "mlia-backend",
            "install",
            "--noninteractive",
            "--i-agree-to-the-contained-eula",
            "backend-a",
        )
    ]


def test_ensure_backends_available_rejects_backend_missing_after_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend install success must be followed by visible installed state."""
    installs: list[tuple[str, ...]] = []

    def fake_list_backends() -> list[dict[str, object]]:
        return [{"name": "backend-a", "installed": False}]

    def fake_installer(argv: list[str]) -> None:
        installs.append(tuple(argv))

    monkeypatch.setattr("mlia.api.list_backends", fake_list_backends)
    monkeypatch.setattr("mlia.testing.e2e._default_backend_installer", fake_installer)
    monkeypatch.setenv("MLIA_E2E_BACKENDS", "backend-a")

    with pytest.raises(
        E2EExecutionRuntimeError,
        match="Required e2e backend\\(s\\) are not installed after installation",
    ):
        ensure_backends_available(("backend-a",))

    assert installs == [
        (
            "mlia-backend",
            "install",
            "--noninteractive",
            "--i-agree-to-the-contained-eula",
            "backend-a",
        )
    ]


def test_ensure_backends_available_normalizes_required_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required backend names should be stripped and de-duplicated."""

    def fake_list_backends() -> list[dict[str, object]]:
        return [{"name": "backend-a", "installed": True}]

    monkeypatch.setattr("mlia.api.list_backends", fake_list_backends)
    monkeypatch.setenv("MLIA_E2E_BACKENDS", "backend-a")

    assert ensure_backends_available((" backend-a ", "", "backend-a")) == ("backend-a",)


def test_emit_e2e_results_streams_stdout_and_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Captured subprocess output should be proxied to active streams."""
    emit_e2e_results(
        subprocess.CompletedProcess(
            ["mlia", "check"],
            0,
            stdout="stdout line",
            stderr="stderr line",
        )
    )

    captured = capsys.readouterr()

    assert captured.out == "stdout line\n"
    assert captured.err == "stderr line\n"
